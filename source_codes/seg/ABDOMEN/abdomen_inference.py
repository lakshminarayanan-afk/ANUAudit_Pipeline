from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.morphology import skeletonize

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    _DICOM_AVAILABLE = True
except ImportError:
    _DICOM_AVAILABLE = False

from source_codes.seg.ABDOMEN.config import Config_abd
from source_codes.seg.ABDOMEN.model import build_model
from source_codes.seg.ABDOMEN.organ_postprocessing import (
    masks_to_exclusive_label_map,
    postprocess_structures,
)

from config import PALETTE_RGB_ABDOMEN
from utils.extract_panels import extract_panel
from utils.seg_biom_results_json import write_segmentation_result


# ═════════════════════════════════════════════════════════════════════════════
#  IMAGE I/O  (generic framework, borrowed from limbs_inference.py: PNG/JPEG/
#  BMP/TIFF + DICOM. Never touches abdomen preprocessing - only produces a
#  plain grayscale uint8 array that predict() consumes exactly as before.)
# ═════════════════════════════════════════════════════════════════════════════

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".dcm", ".dicom"}

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
    """
    Robust DICOM -> grayscale uint8 loader (VOI LUT aware, MONOCHROME1-inverted,
    YBR-aware). Output is a plain grayscale uint8 array, identical in kind to what
    cv2.IMREAD_GRAYSCALE gives for PNG/JPEG, so it feeds the SAME predict() used for every
    other format - this function only ever *produces* a gray array; it never touches the
    abdomen model's resize/normalise logic.
    """
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
    """Colour DICOM -> BGR uint8, used only for the visualisation background. Returns None
    if the DICOM isn't colour or can't be decoded as colour; caller falls back to grayscale."""
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
        logging.warning(f"Could not load colour DICOM {Path(path).name}: {e}")
        return None


def load_gray_image(image_path: str) -> np.ndarray | None:
    """Single entry point for grayscale loading (model input). Identical behaviour to the
    previous cv2.imread(..., IMREAD_GRAYSCALE) for every non-DICOM format; DICOM is
    additionally supported."""
    suffix = Path(image_path).suffix.lower()
    if suffix in (".dcm", ".dicom"):
        try:
            return _load_dicom_as_gray_uint8(image_path)
        except Exception as e:
            logging.warning(f"Could not load DICOM {Path(image_path).name}: {e}")
            return None
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logging.warning(f"Could not load image: {Path(image_path).name}")
    return img


def load_bgr_image(image_path: str) -> np.ndarray | None:
    """Load image in BGR specifically for visualisation."""
    suffix = Path(image_path).suffix.lower()

    if suffix in (".dcm", ".dicom"):
        # Try to preserve actual colour information first
        bgr = _load_dicom_as_bgr_uint8(image_path)
        if bgr is not None:
            return bgr

        # Only use grayscale for genuinely monochrome DICOMs
        gray = load_gray_image(image_path)
        if gray is None:
            return None

        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # PNG/JPEG/TIFF/etc.
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if bgr is None:
        logging.warning(f"Cannot read image for visualisation: {image_path}")

    return bgr


def discover_images(image_dir: str) -> list[Path]:
    """Robust recursive discovery of every supported image/DICOM file under image_dir."""
    return sorted(
        p for p in Path(image_dir).rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )


# ═════════════════════════════════════════════════════════════════════════════
#  ABDOMEN CONSTANTS  (unchanged from the existing abdomen infer.py)
# ═════════════════════════════════════════════════════════════════════════════

N_STRUCT     = Config_abd.NUM_STRUCTURES          # 12  (channels 0-11)
SKIN_CH      = Config_abd.NUM_CHANNELS - 1        # 12  (channel 12)
STRUCT_NAMES: list[str] = list(Config_abd.STRUCTURES)
SKIN_NAME    = Config_abd.SKIN_LINE_NAME
EXCLUDED_STRUCTURE_NAMES: set[str] = {"Bladder", "Gall Bladder"}
EXCLUDED_STRUCTURE_LABELS: list[int] = [
    ch + 1 for ch, name in enumerate(STRUCT_NAMES) if name in EXCLUDED_STRUCTURE_NAMES
]

# ── Post-processing: structure blobs ─────────────────────────────────────────
# FIX 1: Per-channel minimum component pixel count.
MIN_COMPONENT_PIXELS_DEFAULT = 150

_MIN_COMP_OVERRIDE: dict[str, int] = {
    "Renal Pelvis":   20,
    "Umbilical Vein": 50,
    "Aorta":          40,
    "IVC":            10,
    "Cord Insertion": 10,
}

MIN_COMP_PER_CHANNEL: list[int] = [
    _MIN_COMP_OVERRIDE.get(name, MIN_COMPONENT_PIXELS_DEFAULT)
    for name in STRUCT_NAMES
]

# ── FIX 2: Per-channel sigmoid detection threshold ───────────────────────────
_THRESHOLD_OVERRIDE: dict[str, float] = {
    "Renal Pelvis":   0.35,
    "Umbilical Vein": 0.35,
    "Aorta":          0.35,
    "IVC":            0.35,
    "Cord Insertion": 0.35,
}

# ── FIX 3: Priority channels - re-stamp after argmax ─────────────────────────
PRIORITY_CHANNEL_NAMES: set[str] = {
    "Renal Pelvis",
    "Umbilical Vein",
    "Cord Insertion",
}

# ── Post-processing: skin line ───────────────────────────────────────────────
SKIN_CLOSE_KSIZE    = 15
SKIN_MIN_PIXELS     = 1000
SKIN_MIN_ARC_FRAC   = 0.23
SKIN_JAGGEDNESS_MAX = 1.15

SKINLINE_MIN_CIRCULARITY = 0.80
SKINLINE_MAX_AXIS_RATIO  = 3.0

UV_MIN_ASPECT_RATIO   = 2.0
UV_MIN_SKELETON_LEN   = 30

_VIS_CONF_DEFAULT = 0.70

_VIS_CONF_OVERRIDE: dict[str, float] = {
    "Aorta":          0.40,
    "IVC":            0.40,
    "Adrenal":        0.40,
    "Renal Pelvis":   0.30,
    "Umbilical Vein": 0.35,
    "Bladder":        0.70,
}

VIS_CONF_PER_CHANNEL: list[float] = [
    _VIS_CONF_OVERRIDE.get(name, _VIS_CONF_DEFAULT)
    for name in STRUCT_NAMES
]

VIS_CONFIDENCE_THRESHOLD = _VIS_CONF_DEFAULT


# ═════════════════════════════════════════════════════════════════════════════
#  Plane Classification  (unchanged from the existing abdomen infer.py)
# ═════════════════════════════════════════════════════════════════════════════

PLANE_MANDATORY_STRUCTURES: dict[str, list[str]] = {
    "Abdominal Circumference": [
        "Umbilical Vein",
        "Skin Line",
        "Vertebrae",
        "Stomach",
        "Adrenal",
    ],
    "Cord Insertion": [
        "Cord Insertion",
        "Skin Line",
        "Vertebrae",
    ],
    "Transverse Kidneys": [
        "Kidney Cortex",
        "Renal Pelvis",
        "Vertebrae",
    ],
}

PLANE_ANCHOR_STRUCTURES: dict[str, list[str]] = {
    "Abdominal Circumference": ["Skin Line"],
    "Cord Insertion": ["Cord Insertion"],
    "Transverse Kidneys": ["Kidney Cortex"],
}

PLANE_DETECT_THRESHOLD   = 0.50   # >= 50% -> candidate plane
PLANE_STANDARD_THRESHOLD = 0.90   # >= 90% -> Standard quality


def classify_plane(
    label_map:        np.ndarray,
    skin_bin:         np.ndarray,
    skin_circularity: float = float("nan"),
    skin_axis_ratio:  float = float("nan"),
) -> dict:
    """
    Classify the scan plane and assess image quality. UNCHANGED from the existing abdomen
    infer.py - see that module's original docstring for full behavioural detail (always
    assigns a plane except the one all-zero edge case; Standard/Non-Standard quality gated
    by anchor + 90% mandatory-structure checks; Abdominal Circumference-only Umbilical Vein shape constraint).
    """
    struct_present: dict[str, bool] = {}
    for ch, name in enumerate(STRUCT_NAMES):
        struct_present[name] = bool((label_map == ch + 1).any())
    struct_present["Skin Line"] = bool(skin_bin.any())

    scores:          dict[str, float]      = {}
    anchor_pass:     dict[str, bool]       = {}
    present:         dict[str, list[str]]  = {}
    missing:         dict[str, list[str]]  = {}
    disqualified_by: dict[str, list[str]]  = {}

    for plane, mandatory in PLANE_MANDATORY_STRUCTURES.items():
        present_structs = [s for s in mandatory if struct_present.get(s, False)]
        missing_structs = [s for s in mandatory if not struct_present.get(s, False)]
        fraction        = len(present_structs) / len(mandatory)

        scores[plane]  = fraction
        present[plane] = present_structs
        missing[plane] = missing_structs

        anchors         = PLANE_ANCHOR_STRUCTURES.get(plane, [])
        missing_anchors = [a for a in anchors if not struct_present.get(a, False)]
        anchor_pass[plane]     = len(missing_anchors) == 0
        disqualified_by[plane] = missing_anchors

    uv_standard, uv_metrics = is_uv_standard(label_map)

    priority_order = ["Abdominal Circumference", "Cord Insertion", "Transverse Kidneys"]

    if all(fraction <= 0.0 for fraction in scores.values()):
        return {
            "plane":           "Unclassified",
            "quality":         "N/A",
            "scores":          scores,
            "anchor_pass":     anchor_pass,
            "present":         present,
            "missing":         missing,
            "disqualified_by": disqualified_by,
            "uv_standard":     uv_standard,
            "uv_metrics":      uv_metrics,
            "extra_reasons":   [],
        }

    best_plane = max(
        scores,
        key=lambda p: (scores[p], -priority_order.index(p)),
    )
    best_score = scores[best_plane]

    if anchor_pass[best_plane] and best_score >= PLANE_STANDARD_THRESHOLD:
        quality = "Standard"
    else:
        quality = "Non-Standard"

    extra_reasons: list[str] = []
    if best_plane == "Abdominal Circumference" and not uv_standard:
        quality = "Non-Standard"
        extra_reasons.append("Umbilical Vein shape constraint failed")

    return {
        "plane":           best_plane,
        "quality":         quality,
        "scores":          scores,
        "anchor_pass":     anchor_pass,
        "present":         present,
        "missing":         missing,
        "disqualified_by": disqualified_by,
        "uv_standard":     uv_standard,
        "uv_metrics":      uv_metrics,
        "extra_reasons":   extra_reasons,
    }


def print_classification(stem: str, result: dict) -> None:
    """Log the classification result. Unchanged content from the existing abdomen infer.py,
    routed through `logging` instead of print() to match the generic framework's logging
    style."""
    plane, quality  = result["plane"], result["quality"]
    scores          = result["scores"]
    anchor_pass     = result["anchor_pass"]
    present         = result["present"]
    missing         = result["missing"]
    disqualified_by = result["disqualified_by"]

    logging.info(f"    -- Plane Classification -- Plane: {plane}   Quality: {quality}")
    for p in ["Abdominal Circumference", "Cord Insertion", "Transverse Kidneys"]:
        pct  = scores[p] * 100
        tag  = " <- selected" if p == plane else ""
        anchor_tag = "[anchor OK]" if anchor_pass[p] else f"[ANCHOR FAIL: {', '.join(disqualified_by[p])}]"
        logging.info(
            f"      {p}  {pct:5.1f}%  {anchor_tag}{tag}  "
            f"present=[{', '.join(present[p]) or 'none'}]  missing=[{', '.join(missing[p]) or 'none'}]"
        )

    uvm, uv_ok = result["uv_metrics"], result["uv_standard"]
    logging.info(
        f"    [UV shape] area={uvm['area']:.1f}  aspect_ratio={uvm['aspect_ratio']:.3f}  "
        f"skeleton_length={uvm['skeleton_length']}  UV={'STANDARD' if uv_ok else 'NON-STANDARD'}"
    )
    if result.get("extra_reasons"):
        logging.info(f"    [Abdominal Circumference reason] {'; '.join(result['extra_reasons'])}")


# ═════════════════════════════════════════════════════════════════════════════
#  Colour palette (BGR for OpenCV) - unchanged
# ═════════════════════════════════════════════════════════════════════════════

PALETTE_RGB = PALETTE_RGB_ABDOMEN

# SKIN_PALETTE_IDX = len(PALETTE_RGB_ABDOMEN)
# Palette index 0 = background
# Palette indices 1-12 = 12 structures
# Palette index 13 = Skin Line
SKIN_PALETTE_IDX = Config_abd.NUM_CHANNELS


def _bgr(rgb_row: np.ndarray) -> tuple[int, int, int]:
    return (int(rgb_row[2]), int(rgb_row[1]), int(rgb_row[0]))


# ═════════════════════════════════════════════════════════════════════════════
#  GT loading - unchanged from the existing abdomen infer.py
# ═════════════════════════════════════════════════════════════════════════════

_MAX_RAW = max(Config_abd.RAW_LABEL_TO_CHANNEL.keys())
_RAW2CH  = np.full(_MAX_RAW + 1, -1, dtype=np.int8)
for _rl, _ch in Config_abd.RAW_LABEL_TO_CHANNEL.items():
    _RAW2CH[_rl] = _ch

_SKIN_RAW_LABEL: int = next(
    rl for rl, ch in Config_abd.RAW_LABEL_TO_CHANNEL.items()
    if ch == Config_abd.NUM_CHANNELS - 1
)


def _merge_raw_to_13ch(raw_channels: np.ndarray) -> np.ndarray:
    """Identical to dataset.py::_merge_to_13ch(). Unchanged."""
    N_RAW, H, W = raw_channels.shape
    out = np.zeros((Config_abd.NUM_CHANNELS, H, W), dtype=np.uint8)
    for raw_idx in range(N_RAW):
        raw_label = raw_idx + 1
        if raw_label > _MAX_RAW:
            continue
        out_ch = int(_RAW2CH[raw_label])
        if out_ch < 0:
            continue
        out[out_ch] |= raw_channels[raw_idx]
    return out


def _load_npz_13ch(path: Path) -> np.ndarray | None:
    """Load any supported NPZ and return (13, H, W) uint8 {0, 255}. Mirrors
    dataset.py::_load_masks() exactly. Returns None on failure - the caller treats that
    exactly like "no GT found" (never crashes the run)."""
    try:
        npz  = np.load(str(path), allow_pickle=False)
        keys = set(npz.files)
    except Exception as exc:
        logging.warning(f"Cannot load NPZ {path.name}: {exc}")
        return None

    if "mask_structures" in keys and "mask_skinline" in keys:
        structures: np.ndarray = npz["mask_structures"]
        skinline:   np.ndarray = npz["mask_skinline"]
        N_RAW_STRUCT = structures.shape[0]

        if N_RAW_STRUCT == Config_abd.NUM_STRUCTURES:
            out = np.zeros((Config_abd.NUM_CHANNELS,) + structures.shape[1:], dtype=np.uint8)
            out[:Config_abd.NUM_STRUCTURES] = structures
            out[Config_abd.NUM_STRUCTURES]  = skinline[0]
            return out
        else:
            combined = np.concatenate([structures, skinline], axis=0)
            return _merge_raw_to_13ch(combined)

    if "mask" in keys:
        return _merge_raw_to_13ch(npz["mask"])

    if npz.files:
        arr = npz[npz.files[0]]
        if arr.ndim == 3:
            logging.warning(f"Unknown NPZ format in {path.name}; trying first key '{npz.files[0]}'")
            return _merge_raw_to_13ch(arr)

    logging.warning(f"Unrecognised NPZ format in {path.name}. Found keys: {sorted(keys)}")
    return None


def find_gt_mask_path(masks_dir: Path | None, stem: str) -> Path | None:
    """Filename-stem GT resolution, mirroring the existing abdomen NPZ naming convention
    (with the same '_image' suffix tolerance the abdomen loader already had)."""
    if masks_dir is None:
        return None
    candidates = [stem]
    if stem.endswith("_image"):
        candidates.append(stem[: -len("_image")])
    else:
        candidates.append(stem + "_image")
    for cand in candidates:
        p = masks_dir / f"{cand}.npz"
        if p.exists():
            return p
    return None


def load_gt(masks_dir: Path, stem: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Find the NPZ for this image stem and return:
        gt_label  (H, W)  int32   - 1-based structure label map (0 = background)
        gt_skin   (H, W)  uint8   - binary skin line {0, 1}
    Returns (None, None) if no NPZ is found - unchanged behaviour from the existing abdomen
    infer.py, now routed through find_gt_mask_path() for the generic optional-GT flow.
    """
    npz_path = find_gt_mask_path(masks_dir, stem)
    if npz_path is None:
        return None, None

    arr13 = _load_npz_13ch(npz_path)
    if arr13 is None:
        return None, None

    H, W = arr13.shape[1], arr13.shape[2]
    gt_label = np.zeros((H, W), dtype=np.int32)
    for ch in range(N_STRUCT):
        gt_label[arr13[ch] > 0] = ch + 1
    gt_skin = (arr13[SKIN_CH] > 0).astype(np.uint8)
    return gt_label, gt_skin


# ═════════════════════════════════════════════════════════════════════════════
#  Model loading - unchanged from the existing abdomen infer.py
# ═════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint: str, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    ckpt_state        = ckpt["model"]
    ckpt_num_channels = ckpt_state["head.9.weight"].shape[0]

    if ckpt_num_channels != Config_abd.NUM_CHANNELS:
        logging.warning(
            f"Checkpoint has {ckpt_num_channels} output channels but "
            f"Config_abd.NUM_CHANNELS={Config_abd.NUM_CHANNELS}. Building model with "
            f"{ckpt_num_channels} channels to match checkpoint."
        )
        _orig = Config_abd.NUM_CHANNELS
        Config_abd.NUM_CHANNELS = ckpt_num_channels
        model = build_model().to(device)
        Config_abd.NUM_CHANNELS = _orig
    else:
        model = build_model().to(device)

    model.load_state_dict(ckpt_state)
    model.eval()
    logging.info(
        f"Loaded  epoch={ckpt.get('epoch', '?')}  best_val_dice={ckpt.get('best_val_dice', '?')}  "
        f"channels={ckpt_num_channels}"
    )
    return model


# ═════════════════════════════════════════════════════════════════════════════
#  Post-processing - structures  (unchanged from the existing abdomen infer.py)
# ═════════════════════════════════════════════════════════════════════════════

def _remove_small_components(binary_mask: np.ndarray, min_pixels: int = MIN_COMPONENT_PIXELS_DEFAULT) -> np.ndarray:
    """Remove connected components smaller than min_pixels. Returns uint8 mask. Unchanged."""
    if min_pixels <= 0 or not binary_mask.any():
        return binary_mask
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_pixels:
            cleaned[labels == lbl] = 1
    return cleaned


def _postprocess_skinline(
    raw_mask:       np.ndarray,
    close_ksize:    int   = SKIN_CLOSE_KSIZE,
    min_pixels:     int   = SKIN_MIN_PIXELS,
    min_arc_frac:   float = SKIN_MIN_ARC_FRAC,
    jaggedness_max: float = SKIN_JAGGEDNESS_MAX,
) -> np.ndarray:
    """Unchanged from the existing abdomen infer.py."""
    if not raw_mask.any():
        return raw_mask

    H, W    = raw_mask.shape
    diag    = float(np.hypot(H, W))
    min_arc = min_arc_frac * diag

    k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, k)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    cleaned = np.zeros_like(closed)

    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] < min_pixels:
            continue

        component = (labels == lbl).astype(np.uint8)
        cnts, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)

        arc_raw = cv2.arcLength(cnt, closed=True)
        if arc_raw < min_arc:
            continue
        if arc_raw == 0:
            continue
        epsilon    = 0.01 * arc_raw
        cnt_smooth = cv2.approxPolyDP(cnt, epsilon, closed=True)
        arc_smooth = cv2.arcLength(cnt_smooth, closed=True)
        if arc_smooth == 0:
            continue
        jaggedness = arc_raw / arc_smooth
        if jaggedness > jaggedness_max:
            continue

        cleaned[labels == lbl] = 1

    return cleaned


def compute_skinline_shape_metrics(mask: np.ndarray) -> tuple[float, float]:
    """Unchanged from the existing abdomen infer.py."""
    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        return float("nan"), float("nan")

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return float("nan"), float("nan")

    largest_contour = max(contours, key=cv2.contourArea)
    area      = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, closed=True)

    circularity = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else float("nan")

    axis_ratio = float("nan")
    if len(largest_contour) >= 5:
        (_center, (axis_a, axis_b), _angle) = cv2.fitEllipse(largest_contour)
        minor_axis, major_axis = min(axis_a, axis_b), max(axis_a, axis_b)
        if minor_axis > 0:
            axis_ratio = major_axis / minor_axis

    return circularity, axis_ratio


def is_valid_skinline(
    circularity:     float,
    axis_ratio:      float,
    min_circularity: float = SKINLINE_MIN_CIRCULARITY,
    max_axis_ratio:  float = SKINLINE_MAX_AXIS_RATIO,
) -> bool:
    """Unchanged from the existing abdomen infer.py."""
    if np.isnan(circularity) or np.isnan(axis_ratio):
        return False
    return (circularity >= min_circularity) and (axis_ratio <= max_axis_ratio)


def is_uv_standard(
    label_map:        np.ndarray,
    min_aspect_ratio: float = UV_MIN_ASPECT_RATIO,
    min_skeleton_len: int   = UV_MIN_SKELETON_LEN,
) -> tuple[bool, dict]:
    """Unchanged from the existing abdomen infer.py."""
    uv_ch    = STRUCT_NAMES.index("Umbilical Vein")
    uv_label = uv_ch + 1
    uv_mask  = (label_map == uv_label).astype(np.uint8)

    metrics = {"area": 0.0, "aspect_ratio": float("nan"), "skeleton_length": 0}

    if not uv_mask.any():
        return False, metrics

    contours, _ = cv2.findContours(uv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return False, metrics

    largest = max(contours, key=cv2.contourArea)
    area    = cv2.contourArea(largest)

    (_center, (rw, rh), _angle) = cv2.minAreaRect(largest)
    short_side, long_side = min(rw, rh), max(rw, rh)
    aspect_ratio = long_side / (short_side + 1e-6)

    skeleton        = skeletonize(uv_mask.astype(bool))
    skeleton_length = int(skeleton.sum())

    metrics["area"]            = round(float(area), 4)
    metrics["aspect_ratio"]    = round(float(aspect_ratio), 4)
    metrics["skeleton_length"] = skeleton_length

    uv_standard = (aspect_ratio > min_aspect_ratio) and (skeleton_length > min_skeleton_len)
    return uv_standard, metrics


# ═════════════════════════════════════════════════════════════════════════════
#  Prediction  (unchanged from the existing abdomen infer.py)
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict(
    model,
    gray:      np.ndarray,
    device:    torch.device,
    threshold: float = 0.5,
    plane:     str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    """Unchanged from the existing abdomen infer.py. `gray` may originate from PNG/JPEG/BMP/
    TIFF/DICOM - the resize+normalise below is byte-for-byte the same regardless of source."""
    mH, mW = Config_abd.IMAGE_SIZE

    resized = cv2.resize(gray, (mW, mH), interpolation=cv2.INTER_LINEAR)
    img_f   = resized.astype(np.float32) / 255.0
    t       = torch.from_numpy(img_f).unsqueeze(0).unsqueeze(0).to(device)

    logits: torch.Tensor = model(t)
    probs   = torch.sigmoid(logits)[0].cpu().numpy()

    struct_probs = probs[:N_STRUCT]
    ch_masks = postprocess_structures(struct_probs, STRUCT_NAMES, plane=plane)

    priority_indices = [ch for ch, name in enumerate(STRUCT_NAMES) if name in PRIORITY_CHANNEL_NAMES]
    label_map = masks_to_exclusive_label_map(ch_masks, struct_probs, priority_indices=priority_indices)

    # Model was never trained on these structures - drop any spurious predictions for them
    # right here, before anything downstream (confidence, polygons, plane classification,
    # visualisation) ever sees the label map.
    for _excluded_label in EXCLUDED_STRUCTURE_LABELS:
        label_map[label_map == _excluded_label] = 0

    skin_raw = (probs[SKIN_CH] >= threshold).astype(np.uint8)
    skin_bin = _postprocess_skinline(skin_raw)

    skin_circularity, skin_axis_ratio = compute_skinline_shape_metrics(skin_bin)

    if not is_valid_skinline(skin_circularity, skin_axis_ratio):
        skin_bin = np.zeros_like(skin_bin)

    return label_map, skin_bin, probs, (skin_circularity, skin_axis_ratio)


# ═════════════════════════════════════════════════════════════════════════════
#  Metrics  (unchanged formulas from the existing abdomen infer.py)
# ═════════════════════════════════════════════════════════════════════════════

def _safe_div(n: float, d: float) -> float | None:
    return round(n / d, 6) if d > 0 else None


def _pixel_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray) -> dict:
    p, g = pred_bin.astype(bool), gt_bin.astype(bool)
    tp = int((p &  g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p &  g).sum())
    tn = int((~p & ~g).sum())
    return {
        "dice":        _safe_div(2 * tp, 2 * tp + fp + fn),
        "iou":         _safe_div(tp, tp + fp + fn),
        "sensitivity": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def compute_metrics(label_map: np.ndarray, skin_bin: np.ndarray,
                     gt_label: np.ndarray, gt_skin: np.ndarray) -> list[dict]:
    """Per-structure + skin-line pixel metrics. Unchanged formulas."""
    rows = []
    for ch, name in enumerate(STRUCT_NAMES):
        label = ch + 1
        pred  = (label_map == label).astype(np.uint8)
        gt    = (gt_label  == label).astype(np.uint8)
        m     = _pixel_metrics(pred, gt)
        rows.append({"structure": name, "gt_present": int(gt.any()), "pred_present": int(pred.any()), **m})
    m = _pixel_metrics(skin_bin, gt_skin)
    rows.append({"structure": SKIN_NAME, "gt_present": int(gt_skin.any()), "pred_present": int(skin_bin.any()), **m})
    return rows


def _mean(vals: list) -> float | None:
    v = [x for x in vals if x is not None]
    return round(sum(v) / len(v), 6) if v else None


def gt_metrics_rows_to_dict(rows: list[dict]) -> dict:
    """Reshape compute_metrics()'s row list into the {structure_name: {...}} dict used in the
    per-image JSON entry's "gt_metrics" field."""
    return {
        r["structure"]: {
            "dice":         r["dice"],
            "iou":          r["iou"],
            "sensitivity":  r["sensitivity"],
            "specificity":  r["specificity"],
            "gt_present":   bool(r["gt_present"]),
            "pred_present": bool(r["pred_present"]),
        }
        for r in rows
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Segmentation-derived confidence  (spec section 12: from the FINAL mask's
#  own sigmoid probabilities, never from plane/mandatory-structure scores)
# ═════════════════════════════════════════════════════════════════════════════

def _channel_confidence(probs: np.ndarray, ch: int, bin_mask: np.ndarray,
                         model_size: tuple[int, int]) -> float:
    """Mean sigmoid probability over the foreground pixels of channel ch. Unchanged from the
    existing abdomen infer.py."""
    mH, mW = model_size
    if bin_mask.shape != (mH, mW):
        bin_mask = cv2.resize(bin_mask, (mW, mH), interpolation=cv2.INTER_NEAREST)
    fg_pixels = probs[ch][bin_mask > 0]
    return float(fg_pixels.mean()) if len(fg_pixels) > 0 else 0.0


def compute_mean_structure_confidence(
    label_map: np.ndarray, skin_bin: np.ndarray, probs: np.ndarray, model_size: tuple[int, int],
) -> float:
    """
    mean_structure_confidence = mean, over every structure (+ skin line) actually present in
    the FINAL post-processed mask, of that channel's own mean sigmoid probability restricted
    to exactly the pixels the final mask assigns to it. Never plane-percentage, never
    mandatory-structure percentage - purely the segmentation head's own confidence in what it
    kept, generalising the limbs-inference concept (there: one predicted class; here: the set
    of structures the final mask actually contains).
    """
    confidences: list[float] = []
    for ch in range(N_STRUCT):
        bin_mask = (label_map == ch + 1).astype(np.uint8)
        if bin_mask.any():
            confidences.append(_channel_confidence(probs, ch, bin_mask, model_size))
    if skin_bin.any():
        confidences.append(_channel_confidence(probs, SKIN_CH, skin_bin, model_size))
    return float(np.mean(confidences)) if confidences else 0.0


# ═════════════════════════════════════════════════════════════════════════════
#  Polygons  (spec section 11: FINAL mask only, in ORIGINAL image coordinates)
# ═════════════════════════════════════════════════════════════════════════════

def mask_to_polygons(mask_orig: np.ndarray, structure_name: str,
                      min_pixels: int = 20, epsilon_fraction: float = 0.01) -> list[dict]:
    """Extracts one polygon entry per connected component of `structure_name`, from a mask
    already in original-image coordinates. Multiple disconnected components each become their
    own polygon entry. Style borrowed from limbs_inference.py's mask_to_polygons()."""
    polygons = []
    binary = mask_orig.astype(np.uint8)
    if binary.max() == 0:
        return polygons
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < min_pixels:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        epsilon = epsilon_fraction * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        pts = approx.reshape(-1, 2)
        if len(pts) < 3:
            pts = cnt.reshape(-1, 2)
        polygons.append({
            "structure": structure_name,
            "type": "polygon",
            "points": [{"x": float(x), "y": float(y)} for x, y in pts],
        })
    return polygons


def build_polygons_from_final_mask(label_map: np.ndarray, skin_bin: np.ndarray,
                                    orig_h: int, orig_w: int) -> list[dict]:
    """
    Builds the full polygons list for one image from the FINAL abdomen mask (never raw
    logits). The abdomen model works at Config_abd.IMAGE_SIZE resolution (resize, not padding),
    so - unlike limbs, which only pads - the final label_map/skin_bin must be resized back up
    to the ORIGINAL image resolution before contour extraction, using the same nearest-
    neighbour upscale the abdomen visualisation overlay already performs. This is purely a
    coordinate transform for reporting; it does not touch the mask values or any
    post-processing decision.
    """
    label_map_orig = cv2.resize(
        label_map.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
    ).astype(np.int32)
    skin_bin_orig = cv2.resize(skin_bin, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    polygons: list[dict] = []
    for ch, name in enumerate(STRUCT_NAMES):
        class_mask = (label_map_orig == ch + 1).astype(np.uint8)
        polygons.extend(mask_to_polygons(class_mask, name))
    polygons.extend(mask_to_polygons(skin_bin_orig, SKIN_NAME))
    return polygons


# ═════════════════════════════════════════════════════════════════════════════
#  Visualisation  (unchanged abdomen visualisation content; borrowed I/O only)
# ═════════════════════════════════════════════════════════════════════════════

def _label_to_colour_mask(label_map: np.ndarray, skin_bin: np.ndarray) -> np.ndarray:
    """Filled colour mask (H, W, 3) BGR. Unchanged from the existing abdomen infer.py."""
    H, W   = label_map.shape
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for ch in range(N_STRUCT):
        label = ch + 1
        mask  = label_map == label
        if mask.any():
            canvas[mask] = _bgr(PALETTE_RGB[label])
    skin_mask = skin_bin > 0
    if skin_mask.any():
        canvas[skin_mask] = _bgr(PALETTE_RGB[SKIN_PALETTE_IDX])
    return canvas


def _draw_overlay(
    bgr: np.ndarray, label_map: np.ndarray, skin_bin: np.ndarray, probs: np.ndarray,
    conf_per_ch: list[float] | None = None, contour_t: int = 2,
    vis_confidence: float = _VIS_CONF_DEFAULT,
) -> np.ndarray:
    """Draw structure contours + skin line on a copy of bgr. Unchanged from the existing
    abdomen infer.py."""
    H, W          = bgr.shape[:2]
    out           = bgr.copy()
    mH_lm, mW_lm = label_map.shape

    if (mH_lm, mW_lm) != (H, W):
        label_map_draw = cv2.resize(
            label_map.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
        ).astype(np.int32)
        skin_bin_draw = cv2.resize(skin_bin, (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        label_map_draw = label_map
        skin_bin_draw  = skin_bin

    for ch in range(N_STRUCT):
        label    = ch + 1
        bin_mask = (label_map_draw == label).astype(np.uint8)
        if not bin_mask.any():
            continue

        gate = conf_per_ch[ch] if conf_per_ch is not None else vis_confidence
        conf = _channel_confidence(probs, ch, bin_mask, (mH_lm, mW_lm))
        if conf < gate:
            continue

        cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, _bgr(PALETTE_RGB[label]), contour_t)

    if skin_bin_draw.any():
        cnts, _ = cv2.findContours(skin_bin_draw.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, _bgr(PALETTE_RGB[SKIN_PALETTE_IDX]), 2)

    return out


def _add_legend(img: np.ndarray, present_chs: list[int], skin_present: bool) -> np.ndarray:
    """Unchanged from the existing abdomen infer.py."""
    entries: list[tuple[tuple[int, int, int], str]] = []
    for ch in present_chs:
        entries.append((_bgr(PALETTE_RGB[ch + 1]), STRUCT_NAMES[ch]))
    if skin_present:
        entries.append((_bgr(PALETTE_RGB[SKIN_PALETTE_IDX]), SKIN_NAME))
    if not entries:
        return img

    H          = img.shape[0]
    line_h     = 22
    pad        = 8
    swatch_w   = 16
    swatch_h   = 14
    font       = cv2.FONT_HERSHEY_SIMPLEX
    panel_w    = 185

    panel = np.full((H, panel_w, 3), 25, dtype=np.uint8)
    cv2.putText(panel, "Legend", (pad, 20), font, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.line(panel, (pad, 25), (panel_w - pad, 25), (80, 80, 80), 1)

    for i, (bgr_c, name) in enumerate(entries):
        y = 35 + i * line_h
        if y + line_h > H - pad:
            break
        cv2.rectangle(panel, (pad, y), (pad + swatch_w, y + swatch_h), bgr_c, -1)
        cv2.rectangle(panel, (pad, y), (pad + swatch_w, y + swatch_h), (180, 180, 180), 1)
        cv2.putText(panel, name[:20], (pad + swatch_w + 6, y + swatch_h - 1),
                    font, 0.42, (230, 230, 230), 1, cv2.LINE_AA)

    return np.concatenate([img, panel], axis=1)


def _add_title(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((32, img.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(bar, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


def _add_plane_banner(img: np.ndarray, plane: str, quality: str, uv_standard: bool | None = None) -> np.ndarray:
    """Unchanged from the existing abdomen infer.py."""
    _, W = img.shape[:2]
    bar_h, uv_bar_h = 36, 28

    if quality == "Standard":
        bar_colour, text_colour = (20, 140, 20), (220, 255, 220)
    elif quality == "Non-Standard":
        bar_colour, text_colour = (0, 120, 200), (255, 240, 200)
    else:
        bar_colour, text_colour = (60, 60, 60), (180, 180, 180)

    bar = np.full((bar_h, W, 3), bar_colour, dtype=np.uint8)
    cv2.putText(bar, f"Plane: {plane}   Quality: {quality}", (10, bar_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, text_colour, 1, cv2.LINE_AA)

    uv_bar = np.full((uv_bar_h, W, 3), (35, 35, 35), dtype=np.uint8)
    if uv_standard is True:
        uv_text, uv_text_color = "UV: STANDARD", (140, 255, 140)
    elif uv_standard is False:
        uv_text, uv_text_color = "UV: NON-STANDARD", (140, 200, 255)
    else:
        uv_text, uv_text_color = "UV: N/A", (170, 170, 170)
    cv2.putText(uv_bar, uv_text, (10, uv_bar_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.50, uv_text_color, 1, cv2.LINE_AA)

    return np.concatenate([img, bar, uv_bar], axis=0)


def make_3tile(
    image_path: str, label_map: np.ndarray, skin_bin: np.ndarray, probs: np.ndarray,
    gt_label: np.ndarray | None, gt_skin: np.ndarray | None, out_path: Path,
    plane_result: dict | None = None, conf_per_ch: list[float] | None = None,
    vis_confidence: float = _VIS_CONF_DEFAULT,
) -> Path | None:
    """
    Save a 3-tile image: Input | Ground Truth | Prediction, with the plane/quality/UV banner
    below the prediction tile. Same abdomen-specific visualisation content as before; the only
    framework change is using the generic load_bgr_image() (DICOM-aware) instead of a bare
    cv2.imread(), so DICOM inputs get a visualisation background too.
    """
    bgr = load_bgr_image(str(image_path))
    if bgr is None:
        return None
    H, W = bgr.shape[:2]

    def _scale_to_height(tile: np.ndarray, h: int) -> np.ndarray:
        ratio = h / tile.shape[0]
        return cv2.resize(tile, (int(tile.shape[1] * ratio), h))

    tile_raw = _add_title(bgr.copy(), "Input Image")

    if gt_label is not None and gt_skin is not None:
        gt_l_rs = cv2.resize(gt_label.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        gt_s_rs = cv2.resize(gt_skin, (W, H), interpolation=cv2.INTER_NEAREST)

        gt_overlay = _draw_overlay(
            bgr, gt_l_rs, gt_s_rs,
            probs=np.ones((Config_abd.NUM_CHANNELS, H, W), dtype=np.float32),
            conf_per_ch=[0.0] * N_STRUCT, vis_confidence=0.0,
        )
        gt_chs     = [ch for ch in range(N_STRUCT) if (gt_l_rs == ch + 1).any()]
        gt_overlay = _add_legend(gt_overlay, gt_chs, bool(gt_s_rs.any()))
        tile_gt    = _add_title(gt_overlay, "Ground Truth")
    else:
        placeholder = np.full((H, W, 3), 50, dtype=np.uint8)
        cv2.putText(placeholder, "No GT available", (W // 2 - 100, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 1)
        tile_gt = _add_title(placeholder, "Ground Truth")

    pred_overlay = _draw_overlay(bgr, label_map, skin_bin, probs=probs,
                                  conf_per_ch=conf_per_ch, vis_confidence=vis_confidence)
    # pred_chs     = [ch for ch in range(N_STRUCT) if (label_map == ch + 1).any()]
    # pred_overlay = _add_legend(pred_overlay, pred_chs, bool(skin_bin.any()))
    # tile_pred    = _add_title(pred_overlay, "Prediction")

    # if plane_result is not None:
    #     tile_pred = _add_plane_banner(tile_pred, plane_result["plane"], plane_result["quality"],
    #                                    plane_result.get("uv_standard"))

    # target_h = max(tile_raw.shape[0], tile_gt.shape[0], tile_pred.shape[0])
    # # tile_raw  = _scale_to_height(tile_raw,  target_h)
    # # tile_gt   = _scale_to_height(tile_gt,   target_h)
    # tile_pred = _scale_to_height(tile_pred, target_h)

    # sep      = np.full((target_h, 4, 3), 60, dtype=np.uint8)
    # combined = np.concatenate([tile_raw, sep, tile_gt, sep, tile_pred], axis=1)
    # cv2.imwrite(str(out_path), combined)
    # return out_path

    cv2.imwrite(str(out_path), pred_overlay)
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
#  JSON export  (spec sections 6-9: ONE aggregate JSON, one "images" list)
# ═════════════════════════════════════════════════════════════════════════════

def build_plane_mandatory_structures_json(plane_result: dict) -> dict:
    """
    Builds the "plane_mandatory_structures" block directly from classify_plane()'s own
    present/missing/scores/anchor_pass dicts (never a second, independent classifier), one
    {"required", "present", "missing", "detected_fraction", "anchor_pass"} entry per plane.
    """
    out: dict = {}
    for plane, required in PLANE_MANDATORY_STRUCTURES.items():
        out[plane] = {
            "required":          required,
            "present":           plane_result["present"][plane],
            "missing":           plane_result["missing"][plane],
            "detected_fraction": round(plane_result["scores"][plane], 6),
            "anchor_pass":       bool(plane_result["anchor_pass"][plane]),
        }
    return out


def build_plane_candidates(plane_result: dict) -> list[str]:
    """
    Candidate planes from the EXISTING abdomen plane-classification scores, using the
    EXISTING PLANE_DETECT_THRESHOLD constant (>= 50% mandatory structures detected) - no new
    candidate-selection algorithm. The winning plane is always included (it is, by
    definition, the highest-scoring plane) even in the rare case its own score sits under the
    detect threshold; candidates are sorted by score, descending.
    """
    scores = plane_result["scores"]
    if plane_result["plane"] == "Unclassified":
        return []
    candidates = [p for p, s in scores.items() if s >= PLANE_DETECT_THRESHOLD]
    if plane_result["plane"] not in candidates:
        candidates.append(plane_result["plane"])
    candidates.sort(key=lambda p: scores[p], reverse=True)
    return candidates


def build_per_image_json_entry(
    image_path: Path,
    plane_result: dict,
    mean_structure_confidence: float,
    polygons: list[dict],
    gt_metrics: dict | None,
    skin_circularity: float,
    skin_axis_ratio: float,
) -> dict:
    """
    Builds ONE image's entry for the single aggregate inference_results.json. Field names
    follow spec section 7 exactly (plane / plane_quality / mean_structure_confidence /
    plane_candidates / plane_mandatory_structures / polygons / gt_metrics_available /
    gt_metrics); a few additional abdomen-specific diagnostics are appended (uv_standard/
    uv_metrics, skin_line_shape, quality_reasons) since classify_plane()/is_uv_standard()
    already compute them and they are useful context - purely additive.
    """
    return {
        "image_path": str(image_path),
        "input_filename": image_path.name,
        "plane": plane_result["plane"],
        "plane_quality": plane_result["quality"],
        "mean_structure_confidence": round(mean_structure_confidence, 6),
        "plane_candidates": build_plane_candidates(plane_result),
        "plane_mandatory_structures": build_plane_mandatory_structures_json(plane_result),
        "polygons": polygons,
        "gt_metrics_available": gt_metrics is not None,
        "gt_metrics": gt_metrics,
        "uv_standard": plane_result["uv_standard"],
        "uv_metrics": plane_result["uv_metrics"],
        "skin_line_shape": {
            "circularity": None if np.isnan(skin_circularity) else round(skin_circularity, 6),
            "axis_ratio":  None if np.isnan(skin_axis_ratio) else round(skin_axis_ratio, 6),
        },
        "quality_reasons": plane_result.get("extra_reasons", []),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN INFERENCE LOOP  (generic framework, borrowed from limbs_inference.py)
# ═════════════════════════════════════════════════════════════════════════════

def run_inference_abdomen(
    checkpoint:     str,
    items,
    masks_dir:      str | None = None,
    threshold:      float      = 0.5,
    vis_confidence: float      = _VIS_CONF_DEFAULT,
    device_str:     str        = "cuda",
) -> dict:

    # output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logging.info(f"Device         : {device}")
    logging.info(f"Threshold      : {threshold}  (global sigmoid gate; overridden per channel)")
    logging.info(f"NUM_CHANNELS   : {Config_abd.NUM_CHANNELS}")

    logging.info("Plane classification settings:")
    logging.info(f"  Detect threshold   : >= {PLANE_DETECT_THRESHOLD*100:.0f}% mandatory structures")
    logging.info(f"  Standard threshold : >= {PLANE_STANDARD_THRESHOLD*100:.0f}% mandatory structures")
    for plane, structs in PLANE_MANDATORY_STRUCTURES.items():
        anchors = ", ".join(PLANE_ANCHOR_STRUCTURES.get(plane, []))
        logging.info(f"    {plane:<3}  anchors=[{anchors}]  mandatory=[{', '.join(structs)}]")

    # ── Flat output layout directly under --output_dir. No per-plane subdirectories:
    # plane is decided PER IMAGE by classify_plane(), so every image's visualisation and
    # metrics land in the SAME two folders regardless of which plane (Abdominal Circumference/Cord Insertion/Transverse Kidneys) it was
    # classified as. No masks/ folder - the coloured mask PNG export has been removed. ──
    # out     = Path(output_dir)
    # out.mkdir(parents= True, exist_ok = True)
    # vis_dir = out / "visualisations"
    # vis_dir.mkdir(parents= True, exist_ok = True)
    # met_dir = out / "metrics"
    # for d in (vis_dir, met_dir):
    #     d.mkdir(parents=True, exist_ok=True)

    gt_dir: Path | None = Path(masks_dir) if masks_dir else None
    if gt_dir is not None and not gt_dir.is_dir():
        logging.warning(f"masks_dir '{masks_dir}' not found - GT metrics will be skipped for every image.")
        gt_dir = None

    model = load_model(checkpoint, device)

    # img_paths = discover_images(image_dir)
    # logging.info(f"\nFound {len(img_paths)} image(s) under {image_dir}\n{'-' * 50}")

    mH, mW = Config_abd.IMAGE_SIZE
    json_entries: list[dict] = []
    gt_dice_all: list[float] = []
    n_with_gt = 0
    gt_metrics_csv_rows: list[dict] = []
    summary_csv_rows: list[dict] = []
    results = []

    for item in items:
        img_path = Path(item["image_path"])
        panel = item["panel"]
        logging.info(f"Processing: {img_path.name}")
        gray = load_gray_image(str(img_path))
        if gray is None:
            results.append({
                "image_path": str(img_path),
                "status": "unknown",
                "result": {}
            })
            continue

        orig_h, orig_w = gray.shape[:2]

        panel_gray = extract_panel(
            gray,
            panel
        )


        # ── Predict (EXISTING abdomen model + EXISTING abdomen post-processing, untouched) ──
        label_map, skin_bin, probs, (skin_circularity, skin_axis_ratio) = predict(
            model, panel_gray, device, threshold
        )

        # ── GT (optional; never crashes the run) ──
        gt_metrics: dict | None = None
        image_mean_struct_dice: float | None = None
        if gt_dir is not None:
            gt_label_orig, gt_skin_orig = load_gt(gt_dir, stem)
            if gt_label_orig is not None:
                gt_label = cv2.resize(
                    gt_label_orig.astype(np.uint8), (mW, mH), interpolation=cv2.INTER_NEAREST
                ).astype(np.int32)
                gt_skin_arr = (
                    cv2.resize(gt_skin_orig, (mW, mH), interpolation=cv2.INTER_NEAREST)
                    if gt_skin_orig is not None else np.zeros((mH, mW), dtype=np.uint8)
                )
                rows = compute_metrics(label_map, skin_bin, gt_label, gt_skin_arr)
                gt_metrics = gt_metrics_rows_to_dict(rows)
                n_with_gt += 1

                for r in rows:
                    gt_metrics_csv_rows.append({
                        "input_filename": img_path.name,
                        "structure":       r["structure"],
                        "dice":            r["dice"],
                        "iou":             r["iou"],
                        "sensitivity":     r["sensitivity"],
                        "specificity":     r["specificity"],
                        "gt_present":      bool(r["gt_present"]),
                        "pred_present":    bool(r["pred_present"]),
                    })

                struct_dices = [r["dice"] for r in rows if r["structure"] != SKIN_NAME
                                and r["gt_present"] and r["dice"] is not None]
                if struct_dices:
                    image_mean_struct_dice = float(np.mean(struct_dices))
                    gt_dice_all.append(image_mean_struct_dice)
                    logging.info(f"    mean_struct_dice={image_mean_struct_dice:.4f}")
            else:
                gt_label = gt_skin_arr = None
                logging.info(f"    [GT] no NPZ found for '{stem}' - gt_metrics_available=false")
        else:
            gt_label = gt_skin_arr = None

        # ── Plane classification (EXISTING abdomen logic, untouched) ──
        plane_result = classify_plane(label_map, skin_bin, skin_circularity, skin_axis_ratio)
        # print_classification(stem, plane_result)

        is_valid = is_valid_skinline(skin_circularity, skin_axis_ratio)
        logging.info(
            f"    [Skin-line shape] circularity={skin_circularity:.4f}  axis_ratio={skin_axis_ratio:.4f}  "
            f"valid_shape={is_valid}  plane={plane_result['plane']}  quality={plane_result['quality']}"
        )

        # ── Segmentation confidence (FINAL mask only) ──
        mean_structure_confidence = compute_mean_structure_confidence(label_map, skin_bin, probs, (mH, mW))

        # ── Polygons (FINAL mask only, original image coordinates) ──
        polygons = build_polygons_from_final_mask(label_map, skin_bin, orig_h, orig_w)

        # from ui.utils.dcm_png_utils import anonymized_png_for_dicom

        # ── Visualisation (same final mask; DICOM-aware background loader) ──
        # png_path = anonymized_png_for_dicom(img_path)
        # if png_path:
        #     vis_path = make_3tile(
        #         str(png_path), label_map, skin_bin, probs, gt_label, gt_skin_arr,
        #         vis_dir / f"{stem}_vis.png", plane_result=plane_result,
        #         conf_per_ch=VIS_CONF_PER_CHANNEL, vis_confidence=vis_confidence,
        #     )
        # else:
        #     vis_path = make_3tile(
        #         str(img_path), label_map, skin_bin, probs, gt_label, gt_skin_arr,
        #         vis_dir / f"{stem}_vis.png", plane_result=plane_result,
        #         conf_per_ch=VIS_CONF_PER_CHANNEL, vis_confidence=vis_confidence,
        #     )

        
        # if vis_path:
        #     logging.info(f"    [VIS] saved -> {vis_path}")

        # ── One JSON entry, appended to the SHARED aggregate list (never a per-image file) ──
        entry = build_per_image_json_entry(
            img_path, plane_result, mean_structure_confidence, polygons, gt_metrics,
            skin_circularity, skin_axis_ratio,
        )
        if plane_result["quality"] == "Standard":
            status = "standard"
        else:
            status = "non-standard"
        results.append({
            "image_path": str(img_path),
            "status": status,
            "result": entry,
        })
        write_segmentation_result(item, entry)
        json_entries.append(entry)

        # # ── One inference_summary.csv row per image ──
        # summary_csv_rows.append({
        #     "input_filename":             img_path.name,
        #     "plane":                      plane_result["plane"],
        #     "plane_quality":              plane_result["quality"],
        #     "mean_structure_confidence":  round(mean_structure_confidence, 6),
        #     "uv_standard":                plane_result["uv_standard"],
        #     "skin_circularity":           None if np.isnan(skin_circularity) else round(skin_circularity, 6),
        #     "skin_axis_ratio":            None if np.isnan(skin_axis_ratio) else round(skin_axis_ratio, 6),
        #     "gt_metrics_available":       gt_metrics is not None,
        #     "mean_struct_dice":           round(image_mean_struct_dice, 6) if image_mean_struct_dice is not None else None,
        # })

        # if progress_callback is not None:
        #     img_path_str = str(img_path)

        #     # True when the selected plane is Standard
        #     is_stnd = plane_result["quality"] == "Standard"

        #     progress_callback(img_path_str, is_stnd)

    # ── ONE aggregate JSON for the whole run - flat under metrics/, no per-plane dir.
    # APPEND mode: if inference_results.json already exists (e.g. this is a second/third
    # run against the same --output_dir for a different plane's --image_dir), its existing
    # "images" entries are merged in rather than overwritten, so results from every plane
    # accumulate in the same file instead of the last run clobbering the previous ones. ──
    # results_json_path = out / "inference_results.json"
    # existing_entries: list[dict] = []
    # existing_n_with_gt = 0
    # if results_json_path.exists():
    #     try:
    #         with open(results_json_path, "r") as f:
    #             prev = json.load(f)
    #         existing_entries = prev.get("images", [])
    #         existing_n_with_gt = prev.get("run_summary", {}).get("n_images_with_gt", 0)
    #         logging.info(f"[JSON] found existing {results_json_path} with {len(existing_entries)} "
    #                      f"image(s) - appending this run's results to it")
    #     except Exception as exc:
    #         logging.warning(f"Could not read existing {results_json_path} ({exc}) - starting fresh.")

    # all_entries = existing_entries + json_entries
    # run_summary = {
    #     "n_images":           len(all_entries),
    #     "n_images_with_gt":   existing_n_with_gt + n_with_gt,
    #     "mean_structure_dice_over_images_with_gt": None,  # filled in below, after gt_metrics.csv is written
    # }
    # with open(results_json_path, "w") as f:
    #     json.dump({"run_summary": run_summary, "images": all_entries}, f, indent=2)
    # logging.info(f"\n[JSON] wrote {results_json_path}  ({len(all_entries)} images total)")

    # ── gt_metrics.csv - one row per (image, structure), only for images with GT.
    # APPEND mode: header written only the first time the file is created. ──
    # gt_metrics_csv_path = met_dir / "gt_metrics.csv"
    # gt_metrics_fields = [
    #     "input_filename", "structure", "dice", "iou", "sensitivity", "specificity",
    #     "gt_present", "pred_present",
    # ]
    # write_header = not gt_metrics_csv_path.exists()
    # with open(gt_metrics_csv_path, "a", newline="") as f:
    #     writer = csv.DictWriter(f, fieldnames=gt_metrics_fields)
    #     if write_header:
    #         writer.writeheader()
    #     writer.writerows(gt_metrics_csv_rows)
    # logging.info(f"[CSV]  appended {len(gt_metrics_csv_rows)} row(s) -> {gt_metrics_csv_path}")

    # ── Recompute the cumulative mean_structure_dice_over_images_with_gt from the FULL
    # (just-appended) gt_metrics.csv, so it reflects every plane's run so far, not only
    # this run's images - then rewrite inference_results.json with that corrected value. ──

    cumulative_dice_by_image: dict[str, list[float]] = {}
    # with open(gt_metrics_csv_path, "r", newline="") as f:
    #     for row in csv.DictReader(f):
    #         if row["structure"] == SKIN_NAME:
    #             continue
    #         if row["gt_present"] != "True":
    #             continue
    #         if row["dice"] in ("", "None"):
    #             continue
    #         cumulative_dice_by_image.setdefault(row["input_filename"], []).append(float(row["dice"]))

    # cumulative_image_means = [float(np.mean(v)) for v in cumulative_dice_by_image.values() if v]
    # run_summary["mean_structure_dice_over_images_with_gt"] = _mean(cumulative_image_means)
    # with open(results_json_path, "w") as f:
    #     json.dump({"run_summary": run_summary, "images": all_entries}, f, indent=2)

    # ── inference_summary.csv - one row per image. APPEND mode, same as above. ──
    # inference_summary_csv_path = met_dir / "inference_summary.csv"
    # summary_fields = [
    #     "input_filename", "plane", "plane_quality", "mean_structure_confidence",
    #     "uv_standard", "skin_circularity", "skin_axis_ratio",
    #     "gt_metrics_available", "mean_struct_dice",
    # ]
    # write_header = not inference_summary_csv_path.exists()
    # with open(inference_summary_csv_path, "a", newline="") as f:
    #     writer = csv.DictWriter(f, fieldnames=summary_fields)
    #     if write_header:
    #         writer.writeheader()
    #     writer.writerows(summary_csv_rows)
    # logging.info(f"[CSV]  appended {len(summary_csv_rows)} row(s) -> {inference_summary_csv_path}")

    # logging.info(f"{'-' * 50}\nDone.  Output -> {out.resolve()}")
    # logging.info(f"  Visualisations -> {vis_dir}")
    # logging.info(f"  Metrics        -> {met_dir}")

    # return str(vis_dir), str(results_json_path)
        # "metrics_dir": str(met_dir),
        
        # "gt_metrics_csv_path": str(gt_metrics_csv_path),
        # "inference_summary_csv_path": str(inference_summary_csv_path),
        # "n_images": len(json_entries),
        # "n_images_with_gt": n_with_gt,


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generic-framework inference for the 13-channel sigmoid abdomen UNet "
                     "(Abdominal Circumference / Cord Insertion / Transverse Kidneys plane classification). Supports PNG/JPEG/BMP/TIFF/DICOM "
                     "input. Plane is decided per image by the existing classify_plane() "
                     "logic - there is no separate per-plane input folder."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a best.pth checkpoint.")
    parser.add_argument("--image_dir", required=True,
                         help="Folder to scan recursively for images/DICOM.")
    parser.add_argument("--output_dir", required=True,
                         help="Where visualisations/ and metrics/ for this run go "
                              "(flat layout, no per-plane subdirectories, no masks/).")
    parser.add_argument("--masks_dir", default=None,
                         help="Optional folder of GT .npz masks, matched to images by "
                              "filename stem. Omit to run without GT metrics.")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Global per-pixel sigmoid threshold (default 0.5). Per-channel "
                              "overrides take precedence.")
    parser.add_argument("--vis_threshold", type=float, default=VIS_CONFIDENCE_THRESHOLD,
                         help=f"Global fallback visualisation confidence gate "
                              f"(default {VIS_CONFIDENCE_THRESHOLD}). Per-structure overrides "
                              f"take precedence.")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    run_inference_abdomen(
        checkpoint=args.checkpoint,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        masks_dir=args.masks_dir,
        threshold=args.threshold,
        vis_confidence=args.vis_threshold,
        device_str=args.device,
    )