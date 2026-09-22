"""
inference_spine.py

Deployment-style inference for the shared-encoder 3-head
multiplane spine model (coronal / sagittal / full_body_coronal).
Called as run_inference_spine(checkpoint, items, ...) from the pipeline.

  - NO ground-truth / Dice-IoU comparison (dropped — deployment, not eval)
  - DICOM input support (.dcm), incl. YBR_ICT/RCT/FULL/FULL_422 colour and
    VOI LUT / MONOCHROME1 handling
  - multi-format output: visualisation is written in the SAME format as
    the input (.jpg/.bmp/.tiff/.png); DICOM input -> PNG output
  - per-structure visibility assessment (local contrast / blur / confidence)
  - plane "Standard" / "Non-Standard" quality flag
  - a multi-candidate classify_plane(): all 3 heads are evaluated and
    scored by mandatory-structure match fraction (not just raw softmax
    confidence) to pick the best plane
  - contour overlays drawn directly onto the original image
  - per-structure JSON entry (title-cased names, qualitative labels)
  - per-image summary CSV AND per-structure CSV
  - per-structure individual mask PNGs (like head's save_per_structure_masks)

ASSUMPTIONS — flag these if wrong:
  1. STRUCTURE_THRESHOLDS reuses generic defaults (contrast=12, blur=15,
     conf=0.50) for every spine structure — no calibration data yet.
     Tune per-structure once you have real images to look at.
  2. full_body_coronal's mandatory list is 6 structures (Lungs, Heart, Stomach,
     Bladder, Liver, Bowel); provisional and unvalidated against real examples.
"""

import csv
from pathlib import Path
import traceback
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    _DICOM_AVAILABLE = True
except ImportError:
    _DICOM_AVAILABLE = False

import sys
sys.path.append('/mnt/data4tb/anusha/Prathicksha_Spine')

from config import Seg_Config
CONFIG = Seg_Config.Spine
from source_codes.seg.SPINE.model import build_model
from utils.seg_biom_results_json import write_segmentation_result


def _n_classes(p):
    return CONFIG.NUM_CLASSES_PER_PLANE["sagittal_exclusive"] if p == "Sagittal Spine" \
        else CONFIG.NUM_CLASSES_PER_PLANE[p]


def _structures(p):
    return CONFIG.STRUCTURES["sagittal_exclusive"] if p == "Sagittal Spine" \
        else CONFIG.STRUCTURES[p]


REGION_NAMES = CONFIG.STRUCTURES["sagittal_regions"]   # 4 sagittal region channels


# ─────────────────────────────────────────────────────────────────────────
#  Title-casing
# ─────────────────────────────────────────────────────────────────────────
_LOWERCASE_WORDS = {"of", "the", "and", "a", "an", "in", "on", "at", "to", "for"}
_UPPERCASE_TOKENS: set[str] = set()   # add spine abbreviations here if needed


def title_case_structure(name: str) -> str:
    tokens = name.strip().split()
    result = []
    for i, token in enumerate(tokens):
        inner = token.strip("()")
        has_open = token.startswith("(")
        has_close = token.endswith(")")

        if inner.upper() == inner and len(inner) <= 4 and inner.isalpha():
            formatted = inner.upper()
        elif inner.lower() in _UPPERCASE_TOKENS:
            formatted = inner.upper()
        elif i == 0:
            formatted = inner.capitalize()
        elif inner.lower() in _LOWERCASE_WORDS:
            formatted = inner.lower()
        elif inner.isdigit():
            formatted = inner
        else:
            formatted = inner.capitalize()

        if has_open:
            formatted = "(" + formatted
        if has_close:
            formatted = formatted + ")"
        result.append(formatted)
    return " ".join(result)


# ─────────────────────────────────────────────────────────────────────────
#  Colour palettes — auto-assigned from a 20-colour master list (wraps when exhausted)
# ─────────────────────────────────────────────────────────────────────────
_MASTER_COLORS = [
    (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 170, 255), (255, 0, 255),
    (255, 140, 0), (0, 255, 255), (170, 0, 255), (255, 255, 255), (255, 0, 140),
    (140, 255, 0), (0, 100, 255), (255, 215, 0), (0, 255, 140), (255, 70, 70),
    (180, 130, 255), (255, 180, 0), (0, 255, 210), (255, 60, 200), (150, 255, 255),
]

_color_cursor = 0


def _next_colors(n: int):
    global _color_cursor
    colors = [_MASTER_COLORS[(_color_cursor + i) % len(_MASTER_COLORS)] for i in range(n)]
    _color_cursor += n
    return colors


def _build_palette(n_classes: int):
    palette = [(0, 0, 0)]
    palette += _next_colors(n_classes - 1)
    return palette


def _bgr(rgb):
    return (rgb[2], rgb[1], rgb[0])


PALETTES_RGB = {p: _build_palette(_n_classes(p)) for p in CONFIG.PLANES}
PALETTES_BGR = {p: [_bgr(c) for c in PALETTES_RGB[p]] for p in CONFIG.PLANES}

PALETTES_REGIONS_RGB = _build_palette(len(REGION_NAMES) + 1)
PALETTES_REGIONS_BGR = [_bgr(c) for c in PALETTES_REGIONS_RGB]


# ─────────────────────────────────────────────────────────────────────────
#  Plane-quality (Standard / Non-Standard) — mandatory structures per plane
# ─────────────────────────────────────────────────────────────────────────
PLANE_MANDATORY: dict[str, list[str]] = CONFIG.MANDATORY_STRUCTURES
PLANE_STANDARD_THRESH = 0.90   # fraction of mandatory structures needed for "Standard"
PLANE_MATCH_THRESH = 0.60      # fraction needed to even be a plane *candidate* (classify_plane)

STRUCTURE_THRESHOLDS: dict[str, dict[str, float]] = {
    "_default": {"contrast": 12.0, "blur": 20.0, "conf": 0.50},
    "lungs": {"contrast": 3.0},
    "liver": {"contrast": 2.0},
    "bowel": {"contrast": 2.0},
}


def _get_thresh(name: str, key: str) -> float:
    return STRUCTURE_THRESHOLDS.get(name.lower(), {}).get(key) or STRUCTURE_THRESHOLDS["_default"][key]


def _qual_contrast(v, name):
    return "good" if v >= _get_thresh(name, "contrast") else "low"


def _qual_blur(v, name):
    return "sharp" if v >= _get_thresh(name, "blur") else "blurry"


def _qual_conf(v, name):
    return "high" if v >= _get_thresh(name, "conf") else "low"


# ─────────────────────────────────────────────────────────────────────────
#  Image I/O — DICOM support + multi-format output, ported from head
# ─────────────────────────────────────────────────────────────────────────
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".dcm", ".dicom"}

_EXT_PARAMS: dict[str, tuple[str, list]] = {
    ".jpg":   (".jpg",  [cv2.IMWRITE_JPEG_QUALITY, 95]),
    ".jpeg":  (".jpeg", [cv2.IMWRITE_JPEG_QUALITY, 95]),
    ".png":   (".png",  [cv2.IMWRITE_PNG_COMPRESSION, 3]),
    ".bmp":   (".bmp",  []),
    ".tiff":  (".tiff", []),
    ".tif":   (".tif",  []),
    ".dcm":   (".png",  [cv2.IMWRITE_PNG_COMPRESSION, 3]),   # DICOM -> PNG (lossless, viewable)
    ".dicom": (".png",  [cv2.IMWRITE_PNG_COMPRESSION, 3]),
}


def _output_ext_params(input_suffix: str) -> tuple[str, list]:
    return _EXT_PARAMS.get(input_suffix.lower(), (".png", [cv2.IMWRITE_PNG_COMPRESSION, 3]))


_YBR_PYDICOM_CONVERT = {"YBR_ICT", "YBR_RCT"}
_YBR_MANUAL_SWAP = {"YBR_FULL", "YBR_FULL_422"}


def _is_ybr_dicom(path: str) -> bool:
    if not _DICOM_AVAILABLE:
        return False
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        pi = getattr(ds, "PhotometricInterpretation", "").upper()
        return pi.startswith("YBR")
    except Exception:
        return False


def _load_dicom_as_gray_uint8(path: str) -> np.ndarray:
    if not _DICOM_AVAILABLE:
        raise ImportError("pydicom is required for DICOM input. Install with: pip install pydicom")
    ds = pydicom.dcmread(path)
    pi = getattr(ds, "PhotometricInterpretation", "").upper()

    if pi in _YBR_PYDICOM_CONVERT:
        arr = ds.pixel_array
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            arr = (0.2989 * arr[..., 0].astype(np.float32)
                 + 0.5870 * arr[..., 1].astype(np.float32)
                 + 0.1140 * arr[..., 2].astype(np.float32))
        return np.clip(arr, 0, 255).astype(np.uint8)

    arr = apply_voi_lut(ds.pixel_array.astype(np.float32), ds)

    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            arr = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2])
        elif arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr[arr.shape[0] // 2]
    elif arr.ndim != 2:
        raise ValueError(f"Unexpected DICOM pixel_array shape: {arr.shape}")

    if hasattr(ds, "PhotometricInterpretation") and ds.PhotometricInterpretation == "MONOCHROME1":
        arr = arr.max() - arr

    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
    else:
        arr = np.zeros_like(arr)

    return arr.astype(np.uint8)


def _load_dicom_as_bgr_uint8(path: str) -> np.ndarray | None:
    """Load a colour DICOM and return a BGR uint8 array. Handles YBR_ICT,
    YBR_RCT (JPEG-2000) and YBR_FULL / YBR_FULL_422. None if not colour."""
    if not _DICOM_AVAILABLE:
        return None
    try:
        ds = pydicom.dcmread(path)
        pi = getattr(ds, "PhotometricInterpretation", "").upper()

        if pi in _YBR_PYDICOM_CONVERT:
            arr = ds.pixel_array
            if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                lo, hi = arr.min(), arr.max()
                if hi > lo:
                    arr = ((arr.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)
                return cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2BGR)
            return None

        arr = apply_voi_lut(ds.pixel_array.astype(np.float32), ds)
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            lo, hi = arr.min(), arr.max()
            arr = ((arr - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
            if pi in _YBR_MANUAL_SWAP:
                arr = arr[..., [0, 2, 1]]
                return cv2.cvtColor(arr, cv2.COLOR_YCrCb2BGR)
            return cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2BGR)

        return None
    except Exception as e:
        print(f"  [WARN] Could not load colour DICOM {Path(path).name}: {e}")
        return None


def load_gray_image(image_path: str) -> np.ndarray | None:
    """Load any supported image (incl. DICOM) as a 2-D uint8 grayscale array."""
    suffix = Path(image_path).suffix.lower()
    if suffix in (".dcm", ".dicom"):
        try:
            return _load_dicom_as_gray_uint8(image_path)
        except Exception as e:
            print(f"  [WARN] Could not load DICOM {Path(image_path).name}: {e}")
            return None
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  [WARN] Could not load image: {Path(image_path).name}")
    return img


def load_bgr_image(image_path: str) -> np.ndarray | None:
    """Load any supported image as a 3-channel BGR array for visualisation.
    YBR DICOMs are returned in native colour; everything else grayscale→BGR."""
    suffix = Path(image_path).suffix.lower()
    if suffix in (".dcm", ".dicom"):
        if _is_ybr_dicom(image_path):
            bgr = _load_dicom_as_bgr_uint8(image_path)
            if bgr is not None:
                return bgr
        gray = load_gray_image(image_path)
        if gray is None:
            return None
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f"  [WARN] Cannot read image for visualisation: {image_path}")
    return bgr


def preprocess_from_array(gray: np.ndarray, H: int, W: int) -> torch.Tensor:
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (W, H)).astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)


# ─────────────────────────────────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────────────────────────────────

def load_model(checkpoint: str, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded checkpoint  epoch={ckpt.get('epoch', '?')}  mean_dice={ckpt.get('mean_dice', '?')}")
    return model


SMOOTH_STRUCTURES = {s.lower() for s in [
    "Liver", "Heart", "Stomach", "Bladder", "Gall Bladder",
    "Skin line",
    "Ossification Center - Arch of vertebra",
    "Ossification Center- Vertebral Body",
]}                                            # close gaps + fill holes + keep largest, matched in lowercase
MULTI_OK = set()                              # nothing exempt now

SMOOTH_KERNEL = {"liver": 41}          # gap-closing size; others use the default 25
SMOOTH_BLUR   = {"liver": 31}          # boundary blur size (odd); others use the default 15

def _keep_largest(cm, c, k=1):
    m = (cm == c).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1 + k:
        return
    keep = 1 + np.argsort(st[1:, cv2.CC_STAT_AREA])[::-1][:k]
    cm[(m == 1) & ~np.isin(lab, keep)] = 0


def _lungs_beside_heart(cm, lung_c, heart_c):
    """Keep max 2 lung blobs, on opposite sides of the heart."""
    m = (cm == lung_c).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    if n <= 2:                                   # 0 or 1 blob -> nothing to do
        return
    order = list(1 + np.argsort(st[1:, cv2.CC_STAT_AREA])[::-1][:4])
    heart = (cm == heart_c) if heart_c is not None else None
    if heart is not None and heart.any():
        ys, xs = np.nonzero(heart)
        h = np.array([xs.mean(), ys.mean()])
        best = None
        for i, a in enumerate(order):
            for b in order[i + 1:]:
                # opposite sides of heart = vectors heart->lung point in opposite directions
                if np.dot(cen[a] - h, cen[b] - h) < 0:
                    s = st[a, cv2.CC_STAT_AREA] + st[b, cv2.CC_STAT_AREA]
                    if best is None or s > best[0]:
                        best = (s, a, b)
        keep = list(best[1:]) if best else [order[0]]
    else:
        keep = order[:2]                         # no heart -> 2 largest
    cm[(m == 1) & ~np.isin(lab, keep)] = 0


def postprocess(class_map, probs, conf_thresh: float = 0.60, min_pixels: int = 50,
                plane: str | None = None):
    class_map_clean = class_map.copy()
    names = _structures(plane) if plane else None
    C = probs.shape[0]
    confidences = np.zeros(C, dtype=np.float32)
    for c in range(1, C):
        mask = (class_map_clean == c).astype(np.uint8)
        if mask.sum() == 0:
            continue
        class_probs = probs[c][class_map_clean == c]
        conf = float(np.percentile(class_probs, 95)) if class_probs.size else 0.0
        confidences[c] = conf
        if conf < conf_thresh:
            class_map_clean[class_map_clean == c] = 0
            continue
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        cleaned = np.zeros_like(mask)
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_pixels:
                cleaned[labels == lbl] = 1
        class_map_clean[(mask == 1) & (cleaned == 0)] = 0

        if not names:
            continue
        name = names[c - 1]
        if name == "Lungs" or name.lower() in [n.lower() for n in MULTI_OK]:
            continue                              # handled below / exempt

        if name.lower() in SMOOTH_STRUCTURES:
            m = (class_map_clean == c).astype(np.uint8)
            if m.any():
                ks = SMOOTH_KERNEL.get(name.lower(), 25)
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
                n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
                if n > 2:
                    m = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                filled = np.zeros_like(m)
                cv2.drawContours(filled, cnts, -1, 1, -1)
                bs = SMOOTH_BLUR.get(name.lower(), 15)
                filled = (cv2.GaussianBlur(filled.astype(np.float32), (bs, bs), 0) > 0.5).astype(np.uint8)
                write = (filled == 1) & ((class_map_clean == 0) | (class_map_clean == c))
                class_map_clean[class_map_clean == c] = 0
                class_map_clean[write] = c
        else:
            _keep_largest(class_map_clean, c, 1)   # 1 blob only

    # lungs last, so the final heart mask is available
    if names and "Lungs" in names:
        heart_c = names.index("Heart") + 1 if "Heart" in names else None
        _lungs_beside_heart(class_map_clean, names.index("Lungs") + 1, heart_c)

    return class_map_clean, confidences


def postprocess_regions(region_probs, conf_thresh: float = 0.3, min_pixels: int = 50):
    R, H, W = region_probs.shape
    best_r = np.argmax(region_probs, axis=0)
    best_p = np.max(region_probs, axis=0)
    region_mask_clean = np.zeros((R, H, W), dtype=np.uint8)
    for r in range(R):
        mask = ((best_r == r) & (best_p > conf_thresh)).astype(np.uint8)
        if mask.sum() == 0:
            continue
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        cleaned = np.zeros_like(mask)
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_pixels:
                cleaned[labels == lbl] = 1
        region_mask_clean[r] = cleaned
    return region_mask_clean


# ─────────────────────────────────────────────────────────────────────────
#  classify_plane — multi-candidate plane classification.
#  One forward pass, all 3 planes scored by mandatory-structure match
#  fraction. All 3 planes have a mandatory list. Falls back to raw
#  per-head confidence only if no plane clears PLANE_MATCH_THRESH.
# ─────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def classify_plane(model, tensor: torch.Tensor, device: torch.device,
                    conf_thresh: float = 0.60, min_pixels: int = 50,
                    region_conf_thresh: float = 0.3):
    use_amp = CONFIG.MIXED_PRECISION and device.type == "cuda"
    with autocast("cuda", enabled=use_amp):
        outputs = model(tensor.to(device))

    per_plane_predictions = {}
    plane_scores = {}
    candidates = []

    for p in CONFIG.PLANES:
        if p == "Sagittal Spine":
            logits_excl = outputs["sagittal_exclusive"]
            logits_region = outputs["sagittal_regions"]
            probs = F.softmax(logits_excl, dim=1)[0].cpu().numpy().astype(np.float32)
            region_probs = torch.sigmoid(logits_region)[0].cpu().numpy().astype(np.float32)
        else:
            logits = outputs[p]
            probs = F.softmax(logits, dim=1)[0].cpu().numpy().astype(np.float32)
            region_probs = None

        class_map_raw = np.argmax(probs, axis=0).astype(np.int32)
        class_map, confidences = postprocess(class_map_raw, probs, conf_thresh=conf_thresh, min_pixels=min_pixels, plane=p)
        region_mask = None
        if p == "Sagittal Spine" and region_probs is not None:
            region_mask = postprocess_regions(region_probs, conf_thresh=region_conf_thresh, min_pixels=min_pixels)

        per_plane_predictions[p] = (class_map, confidences, region_mask, region_probs)

        # raw per-head confidence score (fallback heuristic)
        maxp, cls = torch.from_numpy(probs).max(dim=0)
        fg = cls != 0
        raw_score = float(maxp[fg].mean()) * float(fg.float().mean()) if fg.any() else 0.0

        mandatory = PLANE_MANDATORY.get(p)
        if mandatory:
            structures = _structures(p)
            present, confs = [], []
            for s in mandatory:
                if s in structures:
                    c = structures.index(s) + 1
                    if (class_map == c).any():
                        present.append(s)
                        confs.append(float(confidences[c]))
                elif region_mask is not None and s in REGION_NAMES:
                    r = REGION_NAMES.index(s)
                    if region_mask[r].sum() >= 300:
                        present.append(s)
                        roi = region_probs[r][region_mask[r] > 0]
                        confs.append(float(roi.mean()) if roi.size else 0.0)
            match_frac = len(present) / len(mandatory)
            mean_conf = float(np.mean(confs)) if confs else 0.0
            plane_scores[p] = {"match_frac": round(match_frac, 4),
                    "mean_conf": round(mean_conf, 4),
                    "raw_score": round(raw_score, 4),
                    "present": present,
                    "missing": [s for s in mandatory if s not in present]}
            if match_frac >= PLANE_MATCH_THRESH:
                candidates.append((p, match_frac, mean_conf))
        else:
            plane_scores[p] = {"match_frac": None, "mean_conf": None,
                               "raw_score": round(raw_score, 4),
                               "present": None, "missing": None}

    if candidates:
        candidates.sort(key=lambda x: x[1] * x[2], reverse=True)
        best_plane = candidates[0][0]
    else:
        best_plane = max(plane_scores, key=lambda p: plane_scores[p]["raw_score"])

    return best_plane, per_plane_predictions, plane_scores


# ─────────────────────────────────────────────────────────────────────────
#  Structure visibility assessment — extended to also
#  cover the sagittal region channels.
# ─────────────────────────────────────────────────────────────────────────

def assess_structure_visibility(
    gray_orig: np.ndarray,
    class_map: np.ndarray,
    confidences: np.ndarray,
    plane: str,
    region_mask: np.ndarray | None = None,
    region_probs: np.ndarray | None = None,
    dilation_px: int = 21,
) -> tuple[list[dict], dict]:
    orig_H, orig_W = gray_orig.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px, dilation_px))
    structures, C = _structures(plane), _n_classes(plane)

    def _local_metrics(mask_orig, name, conf):
        num_pixels = int(mask_orig.sum())
        if num_pixels < 30:
            return {"name": name, "predicted": False, "vis_flag": "not_predicted",
                    "num_pixels": 0, "model_confidence": 0.0}
        dilated = cv2.dilate(mask_orig, kernel)
        bg_ring = ((dilated.astype(np.int32) - mask_orig.astype(np.int32)) > 0).astype(np.uint8)
        struct_px = gray_orig[mask_orig > 0].astype(np.float32)
        bg_px = gray_orig[bg_ring > 0].astype(np.float32) if bg_ring.sum() > 0 else struct_px
        local_contrast = abs(float(struct_px.mean()) - float(bg_px.mean()))
        x, y, w, h = cv2.boundingRect(mask_orig)
        pad = 10
        y1, y2 = max(0, y - pad), min(orig_H, y + h + pad)
        x1, x2 = max(0, x - pad), min(orig_W, x + w + pad)
        local_blur = min(float(cv2.Laplacian(gray_orig[y1:y2, x1:x2], cv2.CV_64F).var()), 300.0)

        if conf < _get_thresh(name, "conf"):
            vis_flag = "low_confidence"
        elif local_contrast < _get_thresh(name, "contrast"):
            vis_flag = "low_contrast"
        elif local_blur < _get_thresh(name, "blur"):
            vis_flag = "locally_blurry"
        else:
            vis_flag = "ok"

        return {"name": name, "predicted": True,
                "local_contrast": round(local_contrast, 2), "local_blur": round(local_blur, 2),
                "model_confidence": round(conf, 4), "num_pixels": num_pixels, "vis_flag": vis_flag}

    results = []
    for c in range(1, C):
        name = structures[c - 1]
        mask_orig = cv2.resize((class_map == c).astype(np.uint8), (orig_W, orig_H),
                                interpolation=cv2.INTER_NEAREST)
        res = _local_metrics(mask_orig, name, float(confidences[c]))
        if plane == "Full Body Coronal View" and name == "Lungs":
            n_lab, _ = cv2.connectedComponents((class_map == c).astype(np.uint8), connectivity=8)
            res["n_blobs"] = n_lab - 1          # number of separate lung pieces
        results.append(res)


    if region_mask is not None:
        for r, name in enumerate(REGION_NAMES):
            mask_orig = cv2.resize(region_mask[r], (orig_W, orig_H), interpolation=cv2.INTER_NEAREST)
            roi_probs = region_probs[r][region_mask[r] > 0] if region_probs is not None else None
            conf = float(roi_probs.mean()) if (roi_probs is not None and roi_probs.size) else 0.0
            results.append(_local_metrics(mask_orig, name, conf))

    predicted = [r for r in results if r.get("predicted")]
    n_pred = len(predicted)
    n_ok = sum(1 for r in predicted if r["vis_flag"] == "ok")
    overall = ("no_predictions" if n_pred == 0 else
               "ok" if n_ok >= n_pred * 0.7 else
               "partial" if n_ok >= n_pred * 0.6 else "poor")

    summary = {
        "vis_n_structures_predicted": n_pred,
        "vis_n_structures_ok": n_ok,
        "vis_n_structures_low_contrast": sum(1 for r in predicted if r["vis_flag"] == "low_contrast"),
        "vis_n_structures_low_conf": sum(1 for r in predicted if r["vis_flag"] == "low_confidence"),
        "vis_n_structures_locally_blurry": sum(1 for r in predicted if r["vis_flag"] == "locally_blurry"),
        "vis_overall_flag": overall,
    }
    return results, summary


def classify_plane_standard(struct_results: list[dict], plane: str) -> dict:
    mandatory = PLANE_MANDATORY.get(plane)
    if not mandatory:
        return {"predicted_plane": plane, "plane_match_frac": None,
                "plane_mean_conf": None, "plane_standard": "N/A"}
    by_name = {r["name"]: r for r in struct_results}
    present, confs = [], []
    for s in mandatory:
        r = by_name.get(s)
        if r and r.get("predicted") and r.get("vis_flag") != "not_predicted":
            present.append(s)
            confs.append(r.get("model_confidence", 0.0))
    match_frac = len(present) / len(mandatory)
    mean_conf = float(np.mean(confs)) if confs else 0.0

    two_lungs_ok = True
    if plane == "Full Body Coronal View":
        lungs = by_name.get("Lungs")
        two_lungs_ok = bool(lungs and lungs.get("n_blobs", 0) == 2)

    return {
        "predicted_plane": plane,
        "plane_match_frac": round(match_frac, 4),
        "plane_mean_conf": round(mean_conf, 4),
        "plane_standard": "Standard" if (match_frac >= PLANE_STANDARD_THRESH and two_lungs_ok)
                          else "Non-Standard",
    }


# ─────────────────────────────────────────────────────────────────────────
#  Mask saving — colour classmap + per-structure individual masks (head style)
# ─────────────────────────────────────────────────────────────────────────

def save_classmap_png(class_map: np.ndarray, plane: str, out_path: Path):
    H, W = class_map.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    palette = PALETTES_RGB[plane]
    for c in range(1, _n_classes(plane)):
        mask = (class_map == c)
        if mask.any():
            rgb[mask] = palette[c]
    cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def save_per_structure_masks(class_map: np.ndarray, plane: str, out_dir: Path, stem: str,
                              region_mask: np.ndarray | None = None):
    struct_dir = out_dir / stem
    struct_dir.mkdir(parents=True, exist_ok=True)
    structures = _structures(plane)
    for c in range(1, _n_classes(plane)):
        mask = (class_map == c).astype(np.uint8) * 255
        if mask.max() == 0:
            continue
        cv2.imwrite(str(struct_dir / f"{structures[c-1]}.png"), mask)
    if region_mask is not None:
        for r, name in enumerate(REGION_NAMES):
            mask = region_mask[r] * 255
            if mask.max() == 0:
                continue
            cv2.imwrite(str(struct_dir / f"{name.replace(' ', '_')}.png"), mask)


PER_STRUCTURE_CSV_FIELDNAMES = ["image_name", "structure_name", "mean_confidence", "num_pixels"]


def save_per_structure_csv(img_path: Path, struct_results: list[dict], csv_path: Path):
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_STRUCTURE_CSV_FIELDNAMES,
                                 extrasaction="ignore", restval="")
        if not file_exists:
            writer.writeheader()
        for r in struct_results:
            if not r.get("predicted"):
                continue
            writer.writerow({
                "image_name": img_path.name,
                "structure_name": title_case_structure(r["name"]),
                "mean_confidence": round(r.get("model_confidence", 0.0), 4),
                "num_pixels": r.get("num_pixels", 0),
            })

# ─────────────────────────────────────────────────────────────────────────
#  Polygon shapes (ported from head)
# ─────────────────────────────────────────────────────────────────────────

# per-structure (min_pts, max_pts). No calibration for spine yet, so all use the default.
STRUCTURE_POLY_POINTS: dict[str, tuple[int, int]] = {
    "_default": (5, 10),
    # example override:  "liver": (6, 12),
}

SHAPES_SKIP_REGION_CHANNELS = True   # matches visualise(): hides names containing "region"


def _poly_point_range(name: str) -> tuple[int, int]:
    key = name.lower().strip()
    if key in STRUCTURE_POLY_POINTS:
        return STRUCTURE_POLY_POINTS[key]
    return STRUCTURE_POLY_POINTS["_default"]


def _approx_to_point_range(cnt, min_pts, max_pts, epsilon_fraction: float = 0.01):
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return None
    approx = cv2.approxPolyDP(cnt, epsilon_fraction * perimeter, True)
    pts = approx.reshape(-1, 2)

    if len(pts) > max_pts:                       # sample DOWN
        idx = np.linspace(0, len(pts) - 1, max_pts).astype(int)
        pts = pts[idx]
    elif len(pts) < min_pts:                     # sample UP from raw contour
        raw = cnt.reshape(-1, 2)
        if len(raw) >= min_pts:
            idx = np.linspace(0, len(raw) - 1, min_pts).astype(int)
            pts = raw[idx]
    return pts.reshape(-1, 1, 2)


def mask_to_polygons(mask: np.ndarray, structure_name: str, min_pixels: int = 20):
    if mask.max() == 0:
        return []
    min_pts, max_pts = _poly_point_range(structure_name)
    polys = []
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < min_pixels:
            continue
        if len(cnt) < min_pts:
            pts = cnt.reshape(-1, 2)
        else:
            approx = _approx_to_point_range(cnt, min_pts, max_pts)
            if approx is None or len(approx) < 3:
                continue
            pts = approx.reshape(-1, 2)
        polys.append([{"x": float(x), "y": float(y)} for x, y in pts])
    return polys


def build_shapes(class_map: np.ndarray, plane: str, orig_H: int, orig_W: int,
                 region_mask: np.ndarray | None = None) -> list[dict]:
    shapes = []
    structures = _structures(plane)
    class_full = cv2.resize(class_map.astype(np.uint8), (orig_W, orig_H),
                            interpolation=cv2.INTER_NEAREST).astype(np.int32)

    for c in range(1, _n_classes(plane)):
        raw_name = structures[c - 1]
        if SHAPES_SKIP_REGION_CHANNELS and "region" in raw_name.lower():
            continue
        mask = (class_full == c).astype(np.uint8)
        if mask.max() == 0:
            continue
        display = title_case_structure(raw_name)
        for pts in mask_to_polygons(mask, raw_name):
            shapes.append({"points": pts, "structure": display, "type": "polygon"})

    # sagittal region channels (only if you want them; set flag above to False)
    if region_mask is not None and not SHAPES_SKIP_REGION_CHANNELS:
        for r, raw_name in enumerate(REGION_NAMES):
            mask_full = cv2.resize(region_mask[r].astype(np.uint8), (orig_W, orig_H),
                                   interpolation=cv2.INTER_NEAREST)
            if mask_full.max() == 0:
                continue
            display = title_case_structure(raw_name)
            for pts in mask_to_polygons(mask_full, raw_name):
                shapes.append({"points": pts, "structure": display, "type": "polygon"})

    return shapes

# ─────────────────────────────────────────────────────────────────────────
#  Visualisation — contours on the original image, output format matches
#  input format
# ─────────────────────────────────────────────────────────────────────────

def visualise(image_path: str, class_map: np.ndarray, plane: str, out_path: Path,
              input_suffix: str, plane_info: dict | None = None):
    bgr = load_bgr_image(image_path)
    if bgr is None:
        print(f"  [WARN] Cannot read image for visualisation: {image_path}")
        return
    orig_H, orig_W = bgr.shape[:2]
    palette_bgr = PALETTES_BGR[plane]
    structures = _structures(plane)

    class_map_full = cv2.resize(class_map.astype(np.uint8), (orig_W, orig_H),
                                 interpolation=cv2.INTER_NEAREST).astype(np.int32)
    overlay = bgr.copy()
    legend_items = []   # (label, bgr colour)
    for c in range(1, _n_classes(plane)):
        if "region" in structures[c - 1].lower():   # hide all region channels
            continue
        mask = (class_map_full == c).astype(np.uint8)
        if mask.max() == 0:
            continue
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, palette_bgr[c], 2)
        legend_items.append((title_case_structure(structures[c - 1]), palette_bgr[c]))

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = orig_W / 1000.0          # size reference = image WIDTH

    # ── Standard / Non-Standard banner (top-left of image) ──
    if plane_info is not None:
        label = plane_info["plane_standard"]
        colour = {"Standard": (0, 200, 0), "Non-Standard": (0, 0, 255)}.get(label, (160, 160, 160))
        bfs, bth = max(0.5, 1.0 * scale), max(1, int(round(2 * scale)))
        (tw, th_), bl = cv2.getTextSize(label, font, bfs, bth)
        p = max(4, int(10 * scale))
        cv2.rectangle(overlay, (0, 0), (tw + 2 * p, th_ + bl + 2 * p), (0, 0, 0), -1)
        cv2.putText(overlay, label, (p, th_ + p), font, bfs, colour, bth, cv2.LINE_AA)

    # ── Legend panel BELOW the image, one label per row ──
    canvas = overlay
    if legend_items:
        pad = max(6, int(12 * scale))
        sw = max(10, int(18 * scale))
        row_h = max(int(30 * scale), sw + 6)
        lfs = max(0.4, 0.6 * scale)
        lth = max(1, int(round(1.5 * scale)))

        # shrink font until the longest label fits inside the image width
        avail = orig_W - 3 * pad - sw
        while lfs > 0.3:
            max_tw = max(cv2.getTextSize(n, font, lfs, lth)[0][0] for n, _ in legend_items)
            if max_tw <= avail:
                break
            lfs -= 0.05

        panel_h = pad * 2 + row_h * len(legend_items)
        canvas = np.zeros((orig_H + panel_h, orig_W, 3), dtype=np.uint8)
        canvas[:orig_H] = overlay
        for i, (name, col) in enumerate(legend_items):
            y = orig_H + pad + i * row_h
            cv2.rectangle(canvas, (pad, y), (pad + sw, y + sw), col, -1)
            (_, th_), _ = cv2.getTextSize(name, font, lfs, lth)
            cv2.putText(canvas, name, (pad * 2 + sw, y + (sw + th_) // 2),
                        font, lfs, (255, 255, 255), lth, cv2.LINE_AA)

    out_ext, imwrite_params = _output_ext_params(input_suffix)
    cv2.imwrite(str(out_path.parent / (out_path.name + out_ext)), canvas, imwrite_params)

# ─────────────────────────────────────────────────────────────────────────
#  CSV / JSON builders
# ─────────────────────────────────────────────────────────────────────────

CSV_FIELDNAMES = ["image_path", "predicted_plane", "plane_standard", "image_quality"]


def _build_csv_row(img_path: Path, plane_info: dict, struct_summary: dict) -> dict:
    return {
        "image_path": str(img_path),
        "predicted_plane": plane_info["predicted_plane"],
        "plane_standard": plane_info["plane_standard"],
        "image_quality": struct_summary["vis_overall_flag"],
    }

def _present_missing(struct_results, plane):
    by_name = {r["name"]: r for r in struct_results}
    present, missing = [], []
    for s in PLANE_MANDATORY[plane]:
        r = by_name.get(s)
        ok = r and r.get("predicted") and r.get("vis_flag") != "not_predicted"
        (present if ok else missing).append(s)
    return present, missing


def build_plane_mandatory_structures_json(resolved_plane, struct_results, plane_scores=None):
    out = {}
    for plane, required in PLANE_MANDATORY.items():
        if plane == resolved_plane:
            present, missing = _present_missing(struct_results, plane)
            frac = len(present) / len(required)
        elif plane_scores and plane_scores.get(plane, {}).get("present") is not None:
            present = plane_scores[plane]["present"]
            missing = plane_scores[plane]["missing"]
            frac = plane_scores[plane]["match_frac"]
        else:  # plane not evaluated (explicit plane given)
            out[plane] = {"required": required, "present": [], "missing": [],
                          "detected_fraction": None, "anchor_pass": False}
            continue
        out[plane] = {"required": required, "present": present, "missing": missing,
                      "detected_fraction": round(frac, 6),
                      "anchor_pass": frac >= PLANE_MATCH_THRESH}
    return out


def build_plane_candidates(resolved_plane, mandatory_json):
    cands = [p for p, v in mandatory_json.items()
             if v["detected_fraction"] is not None and v["detected_fraction"] >= PLANE_MATCH_THRESH]
    if resolved_plane not in cands:
        cands.append(resolved_plane)
    cands.sort(key=lambda p: mandatory_json[p]["detected_fraction"] or 0.0, reverse=True)
    return cands


def compute_mean_structure_confidence(struct_results):
    vals = [r.get("model_confidence", 0.0) for r in struct_results if r.get("predicted")]
    return float(np.mean(vals)) if vals else 0.0


def _build_json_entry(img_path: Path, struct_results: list[dict],
                      struct_summary: dict, plane_info: dict,
                      shapes: list[dict] | None = None,
                      plane_scores: dict | None = None,
                      vis_path: Path | None = None,
                      gt_metrics: dict | None = None) -> dict:
    structures = []
    for r in struct_results:
        if not r.get("predicted"):
            continue
        raw_name = r["name"]
        name = title_case_structure(raw_name)
        conf = r.get("model_confidence", 0.0)
        contrast = r.get("local_contrast", 0.0)
        blur = r.get("local_blur", 0.0)
        entry = {
            "structure_name": name,
            "segmentation_confidence": {
                "value": round(conf, 4), "qualitative": _qual_conf(conf, raw_name),
                "threshold": _get_thresh(raw_name, "conf"),
            },
            "local_contrast": {
                "value": round(contrast, 2), "qualitative": _qual_contrast(contrast, raw_name),
                "threshold": _get_thresh(raw_name, "contrast"),
            },
            "local_blur": {
                "value": round(blur, 2), "qualitative": _qual_blur(blur, raw_name),
                "threshold": _get_thresh(raw_name, "blur"),
            },
        }
        if "n_blobs" in r:
            entry["n_blobs"] = r["n_blobs"]
        structures.append(entry)

    plane = plane_info["predicted_plane"]
    mandatory_json = build_plane_mandatory_structures_json(plane, struct_results, plane_scores)
    return {
        "image_path": str(img_path),
        "vis_path": str(vis_path) if vis_path else None,
        "input_filename": img_path.name,
        "plane": plane,
        "plane_quality": plane_info["plane_standard"],
        "mean_structure_confidence": round(compute_mean_structure_confidence(struct_results), 6),
        "plane_candidates": build_plane_candidates(plane, mandatory_json),
        "plane_mandatory_structures": mandatory_json,
        "polygons": shapes or [],
        "gt_metrics_available": gt_metrics is not None,
        "gt_metrics": gt_metrics,
        "image_quality_summary": {
            "n_structures_predicted":      struct_summary["vis_n_structures_predicted"],
            "n_structures_ok":             struct_summary["vis_n_structures_ok"],
            "n_structures_low_contrast":   struct_summary["vis_n_structures_low_contrast"],
            "n_structures_low_conf":       struct_summary["vis_n_structures_low_conf"],
            "n_structures_locally_blurry": struct_summary["vis_n_structures_locally_blurry"],
            "overall_image_quality":       struct_summary["vis_overall_flag"],
        },
        "structures": structures,
        "quality_reasons": [],
    }


# ─────────────────────────────────────────────────────────────────────────
#  write_segmentation_result — writes the result straight to disk.
#  Called ONCE PER IMAGE (not batched at the end), so a
#  crash partway through a run doesn't lose already-processed images:
#    - appends one row to <met_dir>/inference_summary.csv
#    - appends one entry to <met_dir>/structure_details.json
# ─────────────────────────────────────────────────────────────────────────

# def write_segmentation_result(met_dir: Path, csv_row: dict, json_entry: dict):
#     met_dir.mkdir(parents=True, exist_ok=True)

#     # CSV — append a row, writing the header only the first time
#     csv_path = met_dir / "inference_summary.csv"
#     file_exists = csv_path.exists()
#     with open(csv_path, "a", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore", restval="")
#         if not file_exists:
#             writer.writeheader()
#         writer.writerow(csv_row)

#     # JSON — load what's already on disk, append this entry, write back.
#     # (Re-reading each time is a bit more I/O than a pure append-log, but
#     # keeps structure_details.json valid JSON at every step.)
#     json_path = met_dir / "structure_details.json"
#     existing_entries = []
#     if json_path.exists():
#         try:
#             with open(json_path, "r") as f:
#                 existing_entries = json.load(f)
#             if not isinstance(existing_entries, list):
#                 existing_entries = []
#         except (json.JSONDecodeError, OSError):
#             existing_entries = []
#     existing_entries.append(json_entry)
#     with open(json_path, "w") as f:
#         json.dump(existing_entries, f, indent=2)

_MODEL_CACHE = {}


def get_model(checkpoint: str, device: torch.device):
    key = (str(Path(checkpoint).resolve()), device.type)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_model(checkpoint, device)
    return _MODEL_CACHE[key]


# ─────────────────────────────────────────────────────────────────────────
#  Main deployment loop
# ─────────────────────────────────────────────────────────────────────────


def run_inference_spine(
    checkpoint: str,
    items,
    conf_thresh: float = CONFIG.INFER_CONF_THRESH,
    min_pixels: int = 50,
    region_conf_thresh: float = 0.3,
    plane: str | None = None,          # unused (kept for caller compatibility); plane is always auto-detected
    device_str: str = "cuda",
    progress_callback=None,
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    H, W = CONFIG.IMAGE_SIZE
    model = get_model(checkpoint, device)
    total = len(items)
    results = []

    for idx, item in enumerate(items):
        try:
            img_path = Path(item["image_path"])
            print(f"  Processing: {img_path.name}")

            raw_gray = load_gray_image(str(img_path))
            if raw_gray is None:
                raise ValueError("could not load image")
            tensor = preprocess_from_array(raw_gray, H, W)

            resolved_plane, per_plane_predictions, plane_scores = classify_plane(
                model, tensor, device, conf_thresh=conf_thresh,
                min_pixels=min_pixels, region_conf_thresh=region_conf_thresh)
            class_map, confidences, region_mask, region_probs = per_plane_predictions[resolved_plane]

            struct_results, struct_summary = assess_structure_visibility(
                raw_gray, class_map, confidences, resolved_plane,
                region_mask=region_mask, region_probs=region_probs)
            plane_info = classify_plane_standard(struct_results, resolved_plane)


            shapes = build_shapes(class_map, resolved_plane,
                      raw_gray.shape[0], raw_gray.shape[1], region_mask)
            json_entry = _build_json_entry(img_path, struct_results, struct_summary,
                                           plane_info, shapes, plane_scores)
            write_segmentation_result(item, json_entry)

            results.append({
                "status": "standard" if plane_info["plane_standard"] == "Standard" else "non-standard",
                "predicted_plane": plane_info["predicted_plane"],
                "plane_match_frac": plane_info["plane_match_frac"],
                "plane_mean_conf": plane_info["plane_mean_conf"],
            })
        except Exception as e:
            traceback.print_exc()
            results.append({"status": "error", "predicted_plane": None,
                            "plane_match_frac": None, "plane_mean_conf": None,
                            "error": str(e)})
        finally:
            if progress_callback:
                progress_callback(idx + 1, total)

    return results