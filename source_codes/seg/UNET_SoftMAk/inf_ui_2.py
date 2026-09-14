import argparse
import csv
import json
import os
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast


try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut, convert_color_space
    _DICOM_AVAILABLE = True
except ImportError:
    _DICOM_AVAILABLE = False

from seg.UNET_SoftMAk.config import Config
from seg.UNET_SoftMAk.model import build_model


_LOWERCASE_WORDS  = {"of", "the", "and", "a", "an", "in", "on", "at", "to", "for"}
_UPPERCASE_TOKENS = {"lv", "csp"}


def title_case_structure(name: str) -> str:
    tokens = name.strip().split()
    result = []
    for i, token in enumerate(tokens):
        inner     = token.strip("()")
        has_open  = token.startswith("(")
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



STRUCTURE_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    'thalamus 2':                             (255,   0,   0),
    'hemisphere 1':                           (  0,   0, 255),
    'hemisphere 2':                           (  0, 255, 255),
    'inner calvarium':                        (255, 140,   0),
    'midline falx':                           (128,   0, 128),
    'arrow sign':                             (  0, 100,   0),
    'anterior midline falx':                  ( 50, 205,  50),
    'thalamus 1':                             (255, 255,   0),
    'lateral sulcus 1':                       (255,  20, 147),
    'outer calvarium':                        (255, 105, 180),
    'cavum septum pellucidum':                (  0, 128, 128),
    'csp (cavum septum pellucidum)':          (  0, 128, 128),
    'lateral sulcus 2':                       (220,  20,  60),
    'hippocampal gyrus 1':                    (139,  69,  19),
    'hippocampal gyrus 2':                    (238, 130, 238),
    'anterior horn of lateral ventricles 2':  (  0, 255,   0),
    'anterior horn of lateral ventricles 1':  (255, 215,   0),
    'pillars of fornix':                      (255,   0, 255),
    'posterior horn of lv 1':                 (160,  82,  45),
    'posterior horn of lv 2':                 (218, 112, 214),
    'lateral ventricle 2':                    ( 75,   0, 130),
    'choroid plexus 1':                       ( 70, 130, 180),
    'choroid plexus 2':                       (  0,   0, 128),
    'lv measurement':                         ( 60, 179, 113),
    'cerebellum':                             (233, 150, 122),
    'cerebral vermis':                        (255, 127,  80),
    'cerebellar vermis':                      (255, 127,  80),
    'petrous part of temporal bone 1':        (244, 164,  96),
    'nuchal fold':                            (255,   0, 255),
    'petrous part of temporal bone 2':        (154, 205,  50),
    'cisternae magna':                        (173, 255,  47),
    'dural fold':                             (153,  50, 204),
    'cerebral peduncle 1':                    (128,   0,   0),
    'cerebral peduncle 2':                    (178,  34,  34),
}


def _rgb_for_structure(name: str, class_idx: int) -> tuple[int, int, int]:
    name_lower = name.lower().strip()
    if name_lower in STRUCTURE_COLORS_RGB:
        return STRUCTURE_COLORS_RGB[name_lower]
    for key, col in STRUCTURE_COLORS_RGB.items():
        if key in name_lower or name_lower in key:
            return col
    import matplotlib.colors as mc
    rgb = mc.hsv_to_rgb(((class_idx - 1) / max(1, Config.NUM_CLASSES - 1), 0.82, 0.92))
    return tuple(int(v * 255) for v in rgb)


def _bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return (rgb[2], rgb[1], rgb[0])


PALETTE_RGB = [(0, 0, 0)] + [
    _rgb_for_structure(Config.STRUCTURES[c - 1], c)
    for c in range(1, Config.NUM_CLASSES)
]
PALETTE_BGR     = [_bgr(c) for c in PALETTE_RGB]
CALV_COLORS_RGB = [
    STRUCTURE_COLORS_RGB.get('inner calvarium', (255, 140,   0)),
    STRUCTURE_COLORS_RGB.get('outer calvarium', (255, 105, 180)),
]
CALV_COLORS_BGR = [_bgr(c) for c in CALV_COLORS_RGB]
CALV_NAMES      = ["inner calvarium", "outer calvarium"]

VISUALISE_SKIP_STRUCTURES = {"hemisphere 1", "hemisphere 2"}
JAGGEDNESS_MAX            = 1.181



STRUCTURE_THRESHOLDS: dict[str, dict[str, float]] = {
    "inner calvarium":               {"contrast": 30.0, "blur": 25.0, "conf": 0.95},
    "outer calvarium":               {"contrast": 30.0, "blur": 25.0, "conf": 0.95},
    "csp (cavum septum pellucidum)": {"contrast": 10.0, "blur": 15.0, "conf": 0.65},
    "thalamus 1":                    {"contrast": 10.0, "blur": 15.0, "conf": 0.50},
    "thalamus 2":                    {"contrast": 10.0, "blur": 15.0, "conf": 0.50},
    "choroid plexus 2":              {"contrast": 15.0, "blur": 15.0, "conf": 0.50},
    "hemisphere 1":                  {"contrast": 10.0, "blur": 10.0, "conf": 0.50},
    "hemisphere 2":                  {"contrast": 10.0, "blur": 10.0, "conf": 0.50},
    "lateral ventricle 2":           {"contrast":  5.0, "blur": 10.0, "conf": 0.50},
    "cerebellum":                    {"contrast": 12.0, "blur": 15.0, "conf": 0.50},
    "cerebellar vermis":             {"contrast": 12.0, "blur": 15.0, "conf": 0.50},
    "cisternae magna":               {"contrast": 10.0, "blur": 10.0, "conf": 0.50},
    "_default":                      {"contrast": 12.0, "blur": 15.0, "conf": 0.50},
}

STRUCT_VIS_COLUMNS = [
    "vis_n_structures_predicted",
    "vis_n_structures_ok",
    "vis_n_structures_low_contrast",
    "vis_n_structures_low_conf",
    "vis_n_structures_locally_blurry",
    "vis_overall_flag",
]


def _get_thresh(name: str, key: str) -> float:
    return (
        STRUCTURE_THRESHOLDS.get(name, {}).get(key)
        or STRUCTURE_THRESHOLDS["_default"][key]
    )


PLANE_MATCH_THRESH    = 0.60
PLANE_STANDARD_THRESH = 0.90

PLANE_MANDATORY: dict[str, list[str]] = {
    "Transthalamic": [
        "thalamus 1", "thalamus 2", "arrow sign",
        "csp (cavum septum pellucidum)", "midline falx",
        "anterior horn of lateral ventricles 2", "inner calvarium", "outer calvarium"
    ],
    "Transventricular": [
        "lateral ventricle 2", "csp (cavum septum pellucidum)",
        "choroid plexus 2", "posterior horn of lateral ventricle 2",
        "midline falx", "anterior horn of lateral ventricles 2", "inner calvarium", "outer calvarium"
    ],
    "Transcerebellar": [
        "cerebellum", "cerebellar vermis", "cisternae magna",
        "csp (cavum septum pellucidum)", "cerebral peduncle 1",
        "cerebral peduncle 2", "anterior horn of lateral ventricles 2", "inner calvarium", "outer calvarium"
    ],
}



def classify_plane(struct_results: list[dict]) -> dict:
    result_by_name: dict[str, dict] = {r["name"]: r for r in struct_results}
    scores: list[tuple[str, float, float]] = []

    for plane, mandatory in PLANE_MANDATORY.items():
        present, confs = [], []
        for s in mandatory:
            r = result_by_name.get(s)
            if r and r.get("predicted") and r.get("vis_flag") != "not_predicted":
                present.append(s)
                confs.append(r.get("mean_confidence") or r.get("model_confidence") or 0.0)
        match_frac = len(present) / max(len(mandatory), 1)
        mean_conf  = float(np.mean(confs)) if confs else 0.0
        scores.append((plane, match_frac, mean_conf))

    passing = [(p, f, c) for p, f, c in scores if f >= PLANE_MATCH_THRESH]

    if not passing:
        return {
            "predicted_plane":  "Unknown",
            "plane_match_frac": round(max(f for _, f, _ in scores), 4) if scores else 0.0,
            "plane_mean_conf":  0.0,
            "plane_standard":   "Non-Standard",
            "plane_candidates": "",
        }

    passing.sort(key=lambda x: (x[1], x[2]), reverse=True)
    best_plane, best_frac, best_conf = passing[0]
    plane_standard = "Standard" if best_frac >= PLANE_STANDARD_THRESH else "Non-Standard"

    return {
        "predicted_plane":  best_plane,
        "plane_match_frac": round(best_frac, 4),
        "plane_mean_conf":  round(best_conf, 4),
        "plane_standard":   plane_standard,
        "plane_candidates": ", ".join(p for p, _, _ in passing),
    }


def _qual_contrast(value: float, name: str) -> str:
    return "good" if value >= _get_thresh(name, "contrast") else "low"

def _qual_blur(value: float, name: str) -> str:
    return "sharp" if value >= _get_thresh(name, "blur") else "blurry"

def _qual_conf(value: float, name: str) -> str:
    return "high" if value >= _get_thresh(name, "conf") else "low"


def assess_structure_visibility(
    gray_orig:   np.ndarray,
    inner_map:   np.ndarray,
    calv_bin:    np.ndarray,
    confidences: np.ndarray,
    calv_probs:  np.ndarray,
    dilation_px: int = 21,
) -> tuple[list[dict], dict]:
    orig_H, orig_W = gray_orig.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px, dilation_px))

    def _local_metrics(mask_orig, name, conf, raw_probs=None):
        num_pixels = int(mask_orig.sum())
        if num_pixels < 30:
            return {
                "name": name, "predicted": False, "vis_flag": "not_predicted",
                "num_pixels": 0, "mean_confidence": 0.0,
            }
        mean_conf  = float(raw_probs.mean()) if (raw_probs is not None and raw_probs.size) else conf
        dilated    = cv2.dilate(mask_orig, kernel)
        bg_ring    = ((dilated.astype(np.int32) - mask_orig.astype(np.int32)) > 0).astype(np.uint8)
        struct_px  = gray_orig[mask_orig > 0].astype(np.float32)
        bg_px      = (gray_orig[bg_ring > 0].astype(np.float32)
                      if bg_ring.sum() > 0 else struct_px)
        struct_mean    = float(struct_px.mean())
        bg_mean        = float(bg_px.mean())
        local_contrast = abs(struct_mean - bg_mean)
        x, y, w, h = cv2.boundingRect(mask_orig)
        pad = 10
        y1 = max(0, y - pad); y2 = min(orig_H, y + h + pad)
        x1 = max(0, x - pad); x2 = min(orig_W, x + w + pad)
        local_blur = float(cv2.Laplacian(gray_orig[y1:y2, x1:x2], cv2.CV_64F).var())
        c_thresh  = _get_thresh(name, "contrast")
        b_thresh  = _get_thresh(name, "blur")
        cf_thresh = _get_thresh(name, "conf")
        if conf < cf_thresh:
            vis_flag = "low_confidence"
        elif local_contrast < c_thresh:
            vis_flag = "low_contrast"
        elif local_blur < b_thresh:
            vis_flag = "locally_blurry"
        else:
            vis_flag = "ok"
        return {
            "name":             name,
            "predicted":        True,
            "local_contrast":   round(local_contrast, 2),
            "local_blur":       round(local_blur, 2),
            "model_confidence": round(conf, 4),
            "mean_confidence":  round(mean_conf, 4),
            "num_pixels":       num_pixels,
            "struct_mean":      round(struct_mean, 2),
            "bg_mean":          round(bg_mean, 2),
            "vis_flag":         vis_flag,
        }

    results = []
    C = Config.NUM_CLASSES

    for c in range(1, C):
        name       = Config.STRUCTURES[c - 1]
        mask_model = (inner_map == c).astype(np.uint8)
        mask_orig  = cv2.resize(mask_model, (orig_W, orig_H), interpolation=cv2.INTER_NEAREST)
        conf       = float(confidences[c])
        results.append(_local_metrics(mask_orig, name, conf))

    for ch, name in enumerate(CALV_NAMES):
        mask_model = calv_bin[ch]
        mask_orig  = cv2.resize(mask_model, (orig_W, orig_H), interpolation=cv2.INTER_NEAREST)
        roi_probs  = calv_probs[ch][calv_bin[ch] > 0]
        conf       = float(roi_probs.mean()) if roi_probs.size else 0.0
        results.append(_local_metrics(mask_orig, name, conf,
                                      raw_probs=roi_probs if roi_probs.size else None))

    predicted = [r for r in results if r.get("predicted")]
    n_pred    = len(predicted)
    n_ok      = sum(1 for r in predicted if r["vis_flag"] == "ok")
    n_lc      = sum(1 for r in predicted if r["vis_flag"] == "low_contrast")
    n_lconf   = sum(1 for r in predicted if r["vis_flag"] == "low_confidence")
    n_blur    = sum(1 for r in predicted if r["vis_flag"] == "locally_blurry")

    if n_pred == 0:
        overall = "no_predictions"
    elif n_ok >= n_pred * 0.7:
        overall = "ok"
    elif n_ok >= n_pred * 0.6:
        overall = "partial"
    else:
        overall = "poor"

    summary = {
        "vis_n_structures_predicted":      n_pred,
        "vis_n_structures_ok":             n_ok,
        "vis_n_structures_low_contrast":   n_lc,
        "vis_n_structures_low_conf":       n_lconf,
        "vis_n_structures_locally_blurry": n_blur,
        "vis_overall_flag":                overall,
    }
    return results, summary


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif",
                  ".dcm", ".dicom"}

_EXT_PARAMS: dict[str, tuple[str, list]] = {
    ".jpg":   (".jpg",  [cv2.IMWRITE_JPEG_QUALITY, 95]),
    ".jpeg":  (".jpeg", [cv2.IMWRITE_JPEG_QUALITY, 95]),
    ".png":   (".png",  [cv2.IMWRITE_PNG_COMPRESSION, 3]),
    ".bmp":   (".bmp",  []),
    ".tiff":  (".tiff", []),
    ".tif":   (".tif",  []),
    ".dcm":   (".png",  [cv2.IMWRITE_PNG_COMPRESSION, 3]),
    ".dicom": (".png",  [cv2.IMWRITE_PNG_COMPRESSION, 3]),
}

def _output_ext_params(input_suffix: str) -> tuple[str, list]:
    return _EXT_PARAMS.get(input_suffix.lower(), (".png", [cv2.IMWRITE_PNG_COMPRESSION, 3]))


_YBR_PYDICOM_CONVERT = {"YBR_ICT", "YBR_RCT"}   # pydicom handles these
_YBR_MANUAL_SWAP     = {"YBR_FULL", "YBR_FULL_422"}


def _is_ybr_dicom(path: str) -> bool:
    """Return True for any YBR photometric interpretation."""
    if not _DICOM_AVAILABLE:
        return False
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        pi = getattr(ds, "PhotometricInterpretation", "").upper()
        return pi.startswith("YBR")
    except Exception:
        return False


def _load_dicom_as_gray_uint8(path: str) -> np.ndarray:
    """Read any DICOM and return a guaranteed 2-D uint8 grayscale array."""
    if not _DICOM_AVAILABLE:
        raise ImportError(
            "pydicom is required for DICOM input. Install with:  pip install pydicom"
        )
    ds  = pydicom.dcmread(path)
    pi  = getattr(ds, "PhotometricInterpretation", "").upper()

    # For YBR_ICT / YBR_RCT let pydicom convert to RGB first, then we
    # extract luminance — avoids manual DCT/wavelet inversion.
    if pi in _YBR_PYDICOM_CONVERT:
        arr = ds.pixel_array          # pydicom returns uint8 RGB for J2K YBR
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            arr = (0.2989 * arr[..., 0].astype(np.float32)
                 + 0.5870 * arr[..., 1].astype(np.float32)
                 + 0.1140 * arr[..., 2].astype(np.float32))
        return np.clip(arr, 0, 255).astype(np.uint8)

    arr = apply_voi_lut(ds.pixel_array.astype(np.float32), ds)

    # Collapse to 2-D
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            arr = (0.2989 * arr[..., 0]
                 + 0.5870 * arr[..., 1]
                 + 0.1140 * arr[..., 2])
        elif arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr[arr.shape[0] // 2]
    elif arr.ndim != 2:
        raise ValueError(f"Unexpected DICOM pixel_array shape: {arr.shape}")

    if hasattr(ds, "PhotometricInterpretation"):
        if ds.PhotometricInterpretation == "MONOCHROME1":
            arr = arr.max() - arr

    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
    else:
        arr = np.zeros_like(arr)

    return arr.astype(np.uint8)


def _load_dicom_as_bgr_uint8(path: str) -> np.ndarray | None:
    """
    Load a colour DICOM and return a BGR uint8 array for visualisation.
    Handles YBR_ICT, YBR_RCT (JPEG-2000) and YBR_FULL / YBR_FULL_422.
    Returns None if the image is not colour or cannot be read.
    """
    if not _DICOM_AVAILABLE:
        return None
    try:
        ds = pydicom.dcmread(path)
        pi = getattr(ds, "PhotometricInterpretation", "").upper()

        # ── JPEG-2000 YBR (ICT / RCT) ────────────────────────────────────
        # pydicom's J2K handler already does the inverse colour transform and
        # returns a uint8 RGB array — we just need to flip R↔B for OpenCV.
        if pi in _YBR_PYDICOM_CONVERT:
            arr = ds.pixel_array          # uint8, H×W×3, RGB
            if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                lo, hi = arr.min(), arr.max()
                if hi > lo:
                    arr = ((arr.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)
                bgr = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2BGR)
                return bgr
            return None

        # ── YBR_FULL / YBR_FULL_422 ──────────────────────────────────────
        arr = apply_voi_lut(ds.pixel_array.astype(np.float32), ds)
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            lo, hi = arr.min(), arr.max()
            arr = ((arr - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
            if pi in _YBR_MANUAL_SWAP:
                # YBR_FULL  channel order: Y, Cb, Cr
                # OpenCV YCrCb order:       Y, Cr, Cb  → swap ch 1 & 2
                arr = arr[..., [0, 2, 1]]
                bgr = cv2.cvtColor(arr, cv2.COLOR_YCrCb2BGR)
            else:
                bgr = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2BGR)
            return bgr

        return None

    except Exception as e:
        print(f"  [WARN] Could not load colour DICOM {Path(path).name}: {e}")
        return None


def load_gray_image(image_path: str) -> np.ndarray | None:
    """Load any supported image as a 2-D uint8 grayscale array."""
    suffix = Path(image_path).suffix.lower()
    if suffix in (".dcm", ".dicom"):
        try:
            return _load_dicom_as_gray_uint8(image_path)
        except Exception as e:
            print(f"  [WARN] Could not load DICOM {Path(image_path).name}: {e}")
            return None
    else:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [WARN] Could not load image: {Path(image_path).name}")
        return img


def load_bgr_image(image_path: str) -> np.ndarray | None:
    """Load any supported image as a 3-channel BGR array for visualisation.
    YBR DICOMs (ICT, RCT, FULL, FULL_422) are returned in native colour;
    all others as grayscale→BGR."""
    suffix = Path(image_path).suffix.lower()
    if suffix in (".dcm", ".dicom"):
        if _is_ybr_dicom(image_path):
            bgr = _load_dicom_as_bgr_uint8(image_path)
            if bgr is not None:
                return bgr
            # colour load failed — fall through to grayscale
        gray = load_gray_image(image_path)
        if gray is None:
            return None
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv2.imread(image_path)
        if bgr is None:
            print(f"  [WARN] Cannot read image for visualisation: {image_path}")
        return bgr


def load_model(checkpoint: str, device: torch.device):
    ckpt  = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded checkpoint  epoch={ckpt.get('epoch', '?')}  "
          f"inner_mDice={ckpt.get('mean_dice_inner', '?')}  "
          f"inner_calv_dice={ckpt.get('inner_calv_dice', '?')}  "
          f"outer_calv_dice={ckpt.get('outer_calv_dice', '?')}")
    return model


def preprocess_from_array(gray: np.ndarray, H: int, W: int) -> torch.Tensor:
    """Build a [1, 1, H, W] model-ready tensor from a 2-D uint8 grayscale array."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (W, H)).astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)


def preprocess(image_path: str, H: int, W: int) -> torch.Tensor:
    """Load + preprocess from file path (kept for backwards compatibility)."""
    gray = load_gray_image(image_path)
    if gray is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return preprocess_from_array(gray, H, W)


@torch.no_grad()
def predict(model, tensor: torch.Tensor, device: torch.device,
            calv_threshold: float = 0.5):
    use_amp = Config.MIXED_PRECISION and device.type == "cuda"
    with autocast("cuda", enabled=use_amp):
        out = model(tensor.to(device))
    inner_probs     = F.softmax(out["inner"], dim=1)[0].cpu().numpy().astype(np.float32)
    inner_class_map = np.argmax(inner_probs, axis=0).astype(np.int32)
    calv_probs      = torch.sigmoid(out["calv"])[0].cpu().numpy().astype(np.float32)
    calv_binary     = (calv_probs >= calv_threshold).astype(np.uint8)
    return inner_class_map, inner_probs, calv_binary, calv_probs


def postprocess_inner(
    class_map:   np.ndarray,
    probs:       np.ndarray,
    conf_thresh: float = 0.60,
    min_pixels:  int   = 100,
) -> tuple[np.ndarray, np.ndarray]:
    class_map_clean = class_map.copy()
    C               = probs.shape[0]
    confidences     = np.zeros(C, dtype=np.float32)
    hemi_classes    = {
        Config.STRUCTURE_TO_IDX["hemisphere 1"],
        Config.STRUCTURE_TO_IDX["hemisphere 2"],
    }

    for c in range(1, C):
        mask = (class_map_clean == c).astype(np.uint8)
        if mask.sum() == 0:
            continue
        class_probs    = probs[c][class_map_clean == c]
        conf           = float(np.percentile(class_probs, 95)) if class_probs.size else 0.0
        confidences[c] = conf

        if conf < conf_thresh:
            class_map_clean[class_map_clean == c] = 0
            confidences[c] = 0.0
            continue

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        cleaned = np.zeros_like(mask)
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_pixels:
                cleaned[labels == lbl] = 1
        class_map_clean[(mask == 1) & (cleaned == 0)] = 0
        mask = (class_map_clean == c).astype(np.uint8)
        if mask.sum() == 0:
            confidences[c] = 0.0
            continue

        if c in hemi_classes:
            n_lbls, lbls, sts, _ = cv2.connectedComponentsWithStats(mask, 8)
            if n_lbls > 2:
                largest_lbl = int(np.argmax(sts[1:, cv2.CC_STAT_AREA])) + 1
                class_map_clean[(lbls != largest_lbl) & (lbls != 0)] = 0
                mask = (class_map_clean == c).astype(np.uint8)
            if mask.sum() == 0:
                confidences[c] = 0.0
                continue
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                class_map_clean[class_map_clean == c] = 0
                confidences[c] = 0.0
                continue
            cnt           = max(contours, key=cv2.contourArea)
            perimeter_raw = cv2.arcLength(cnt, True)
            if perimeter_raw == 0:
                class_map_clean[class_map_clean == c] = 0
                confidences[c] = 0.0
                continue
            epsilon          = 0.01 * perimeter_raw
            cnt_smooth       = cv2.approxPolyDP(cnt, epsilon, True)
            perimeter_smooth = cv2.arcLength(cnt_smooth, True)
            if perimeter_smooth == 0:
                class_map_clean[class_map_clean == c] = 0
                confidences[c] = 0.0
                continue
            if perimeter_raw / perimeter_smooth > JAGGEDNESS_MAX:
                class_map_clean[class_map_clean == c] = 0
                confidences[c] = 0.0

    return class_map_clean, confidences


def postprocess_calv(calv_binary: np.ndarray, min_pixels: int = 100) -> np.ndarray:
    cleaned        = np.zeros_like(calv_binary)
    JAGGEDNESS_MAX = 1.18
    for ch in range(calv_binary.shape[0]):
        mask = calv_binary[ch]
        if mask.max() == 0:
            continue
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] < min_pixels:
                continue
            blob    = (labels == lbl).astype(np.uint8)
            cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt           = max(cnts, key=cv2.contourArea)
            perimeter_raw = cv2.arcLength(cnt, True)
            if perimeter_raw == 0:
                continue
            epsilon          = 0.01 * perimeter_raw
            cnt_smooth       = cv2.approxPolyDP(cnt, epsilon, True)
            perimeter_smooth = cv2.arcLength(cnt_smooth, True)
            if perimeter_smooth == 0:
                continue
            if perimeter_raw / perimeter_smooth <= JAGGEDNESS_MAX:
                cleaned[ch][blob == 1] = 1
    return cleaned



def save_inner_classmap_png(class_map: np.ndarray, out_path: Path):
    H, W = class_map.shape
    rgb  = np.zeros((H, W, 3), dtype=np.uint8)
    for c in range(1, Config.NUM_CLASSES):
        if (class_map == c).any():
            rgb[class_map == c] = PALETTE_RGB[c]
    cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def save_calv_png(calv_binary: np.ndarray, out_path: Path):
    H, W = calv_binary.shape[1], calv_binary.shape[2]
    rgb  = np.zeros((H, W, 3), dtype=np.uint8)
    for ch in range(2):
        if calv_binary[ch].any():
            rgb[calv_binary[ch].astype(bool)] = np.array(CALV_COLORS_RGB[ch], dtype=np.uint8)
    cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def save_per_structure_masks(
    class_map:   np.ndarray,
    calv_binary: np.ndarray,
    out_dir:     Path,
    stem:        str,
):
    struct_dir = out_dir / stem
    struct_dir.mkdir(parents=True, exist_ok=True)
    for c in range(1, Config.NUM_CLASSES):
        mask = (class_map == c).astype(np.uint8) * 255
        if mask.max() == 0:
            continue
        cv2.imwrite(str(struct_dir / f"{Config.STRUCTURES[c-1]}.png"), mask)
    for ch, name in enumerate(CALV_NAMES):
        mask = calv_binary[ch] * 255
        if mask.max() == 0:
            continue
        cv2.imwrite(str(struct_dir / f"{name.replace(' ', '_')}.png"), mask)


def _draw_inner_contours(bgr: np.ndarray, class_map: np.ndarray) -> np.ndarray:
    overlay = bgr.copy()
    for c in range(1, Config.NUM_CLASSES):
        name = Config.STRUCTURES[c - 1]
        mask = (class_map == c).astype(np.uint8)
        if mask.max() == 0:
            continue
        color_bgr = _bgr(_rgb_for_structure(name, c))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, color_bgr, 2)
    return overlay


def _draw_calv_contours(bgr: np.ndarray, calv_binary: np.ndarray) -> np.ndarray:
    overlay = bgr.copy()
    for ch in range(2):
        mask = calv_binary[ch]
        if mask.max() == 0:
            continue
        if mask.shape != bgr.shape[:2]:
            mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        color_bgr = CALV_COLORS_BGR[ch]
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, color_bgr, 2)
    return overlay


def visualise(
    image_path:     str,
    inner_map:      np.ndarray,
    calv_binary:    np.ndarray,
    out_path:       Path,
    input_suffix:   str,
    struct_results: list[dict] | None = None,
    struct_summary: dict       | None = None,
    plane_info:     dict       | None = None,
):
    bgr = load_bgr_image(image_path)
    if bgr is None:
        print(f"  [WARN] Cannot read image for visualisation: {image_path}")
        return

    orig_H, orig_W = bgr.shape[:2]

    inner_map_full = cv2.resize(
        inner_map.astype(np.uint8), (orig_W, orig_H),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int32)

    calv_full = np.stack([
        cv2.resize(calv_binary[ch], (orig_W, orig_H),
                   interpolation=cv2.INTER_NEAREST)
        for ch in range(2)
    ])

    result = _draw_inner_contours(bgr, inner_map_full)
    result = _draw_calv_contours(result, calv_full)

    out_ext, imwrite_params = _output_ext_params(input_suffix)
    final_path = out_path.with_suffix(out_ext)
    cv2.imwrite(str(final_path), result, imwrite_params)

CSV_FIELDNAMES = [
    "image_path", "predicted_plane", "plane_standard", "image_quality",
]


def _build_csv_row(img_path: Path, plane_info: dict, struct_summary: dict) -> dict:
    return {
        "image_path":      str(img_path),
        "predicted_plane": plane_info["predicted_plane"],
        "plane_standard":  plane_info["plane_standard"],
        "image_quality":   struct_summary["vis_overall_flag"],
    }


def _build_json_entry(
    img_path:       Path,
    struct_results: list[dict],
    struct_summary: dict,
) -> dict:
    structures = []
    for r in struct_results:
        if not r.get("predicted"):
            continue
        raw_name = r["name"]
        name     = title_case_structure(raw_name)
        contrast = r.get("local_contrast", 0.0)
        blur     = r.get("local_blur", 0.0)
        conf     = r.get("model_confidence", 0.0)
        structures.append({
            "structure_name": name,
            "segmentation_confidence": {
                "value":       round(conf, 4),
                "qualitative": _qual_conf(conf, raw_name),
                "threshold":   _get_thresh(raw_name, "conf"),
            },
            "local_contrast": {
                "value":       round(contrast, 2),
                "qualitative": _qual_contrast(contrast, raw_name),
                "threshold":   _get_thresh(raw_name, "contrast"),
            },
            "local_blur": {
                "value":       round(blur, 2),
                "qualitative": _qual_blur(blur, raw_name),
                "threshold":   _get_thresh(raw_name, "blur"),
            },
        })

    return {
        "image_path": str(img_path),
        "summary": {
            "n_structures_predicted":      struct_summary["vis_n_structures_predicted"],
            "n_structures_ok":             struct_summary["vis_n_structures_ok"],
            "n_structures_low_contrast":   struct_summary["vis_n_structures_low_contrast"],
            "n_structures_low_conf":       struct_summary["vis_n_structures_low_conf"],
            "n_structures_locally_blurry": struct_summary["vis_n_structures_locally_blurry"],
            "overall_image_quality":       struct_summary["vis_overall_flag"],
        },
        "structures": structures,
    }


PER_STRUCTURE_CSV_FIELDNAMES = [
    "image_name",
    "structure_name",
    "mean_confidence",
    "num_pixels",
]


def save_per_structure_csv(
    img_path:       Path,
    struct_results: list[dict],
    met_dir:        Path,
    csv_path:       Path | None = None,
) -> str:
    csv_path    = csv_path or (met_dir / "structure_confidence.csv")
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
                "image_name":      img_path.name,
                "structure_name":  title_case_structure(r["name"]),
                "mean_confidence": round(r.get("mean_confidence") or r.get("model_confidence", 0.0), 4),
                "num_pixels":      r.get("num_pixels", 0),
            })

    return str(csv_path)



def run_inference(
    checkpoint:        str,
    image_dir:         str,
    output_dir:        str,
    conf_thresh:       float = 0.60,
    min_pixels:        int   = 100,
    calv_threshold:    float = 0.85,
    calv_min_px:       int   = 1000,
    device_str:        str   = "cuda",
    progress_callback         = None,
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    H, W   = Config.IMAGE_SIZE

    out     = Path(output_dir)
    vis_dir = out / "visualisations"
    msk_dir = out / "masks"
    met_dir = out / "metrics"
    for d in [vis_dir, msk_dir, met_dir]:
        d.mkdir(parents=True, exist_ok=True)

    model = load_model(checkpoint, device)

    img_paths = sorted(
        p for p in Path(image_dir).rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )

    print(f"\nFound {len(img_paths)} images in {image_dir}\n")

    csv_rows:         list[dict] = []
    json_entries:     list[dict] = []
    class_csv_str     = ""
    structure_csv_str = ""
    json_str          = ""

    for img_path in img_paths:
        stem         = img_path.stem
        input_suffix = img_path.suffix
        print(f"  Processing: {stem}{input_suffix}")

        raw_gray = load_gray_image(str(img_path))
        if raw_gray is None:
            continue

        tensor = preprocess_from_array(raw_gray, H, W)
        inner_map_raw, inner_probs, calv_bin_raw, calv_probs = predict(
            model, tensor, device, calv_threshold=calv_threshold,
        )

        inner_map, confidences = postprocess_inner(
            inner_map_raw, inner_probs,
            conf_thresh=conf_thresh, min_pixels=min_pixels,
        )
        calv_bin = postprocess_calv(calv_bin_raw, min_pixels=calv_min_px)

        struct_results, struct_summary = assess_structure_visibility(
            gray_orig   = raw_gray,
            inner_map   = inner_map,
            calv_bin    = calv_bin,
            confidences = confidences,
            calv_probs  = calv_probs,
        )

        plane_info   = classify_plane(struct_results)
        is_stnd      = plane_info["plane_standard"] == "Standard"
        img_path_str = str(img_path)

        if progress_callback is not None:
            print(f"    Progress callback: {img_path.name} → {plane_info['plane_standard']}")
            progress_callback(img_path_str, is_stnd)

        csv_rows.append(_build_csv_row(img_path, plane_info, struct_summary))
        json_entries.append(_build_json_entry(img_path, struct_results, struct_summary))
        structure_csv_str = save_per_structure_csv(img_path, struct_results, met_dir)

        save_inner_classmap_png(inner_map, msk_dir / f"{stem}_inner.png")
        save_calv_png(calv_bin,            msk_dir / f"{stem}_calv.png")
        save_per_structure_masks(inner_map, calv_bin, out / "masks_per_structure", stem)

        visualise(
            str(img_path), inner_map, calv_bin,
            vis_dir / stem,
            input_suffix=input_suffix,
        )

    if csv_rows:
        class_csv_path = met_dir / "inference_summary.csv"
        with open(class_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES,
                                    extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n  [CSV]  inference_summary.csv  →  {met_dir}")
        class_csv_str = str(class_csv_path)

    if json_entries:
        json_path = met_dir / "structure_details.json"
        json_str  = str(json_path)
        with open(json_path, "w") as f:
            json.dump(json_entries, f, indent=2)
        print(f"  [JSON] structure_details.json  →  {met_dir}")

    return vis_dir, class_csv_str, structure_csv_str, json_str



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dual-head segmentation model")
    parser.add_argument("--checkpoint",     default=Config.INFER_CHECKPOINT)
    parser.add_argument("--image_dir",      default=Config.INFER_IMAGE_DIR)
    parser.add_argument("--output_dir",     default=Config.INFER_OUTPUT_DIR)
    parser.add_argument("--conf_thresh",    type=float, default=0.60)
    parser.add_argument("--min_pixels",     type=int,   default=100)
    parser.add_argument("--calv_threshold", type=float, default=0.88)
    parser.add_argument("--calv_min_px",    type=int,   default=2000)
    parser.add_argument("--device",         default="cuda")
    args = parser.parse_args()

    run_inference(
        checkpoint     = args.checkpoint,
        image_dir      = args.image_dir,
        output_dir     = args.output_dir,
        conf_thresh    = args.conf_thresh,
        min_pixels     = args.min_pixels,
        calv_threshold = args.calv_threshold,
        calv_min_px    = args.calv_min_px,
        device_str     = args.device,
    )