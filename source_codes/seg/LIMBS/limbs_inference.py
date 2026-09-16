
import os
import json
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    _DICOM_AVAILABLE = True
except ImportError:
    _DICOM_AVAILABLE = False

from seg.LIMBS.config import Config
from seg.LIMBS.model import MultiTaskSwinUNet
from seg.LIMBS.dataset import load_bone_mask_from_npz, LABEL_MAP, NUM_SEG_CLASSES, IGNORE_INDEX
from utils.extract_panels import extract_panel
from utils.seg_biom_results_json import write_segmentation_result

# from LIMBS.config import Config
# from LIMBS.model import MultiTaskSwinUNet
# from LIMBS.dataset import load_bone_mask_from_npz, LABEL_MAP, NUM_SEG_CLASSES, IGNORE_INDEX


# ═══════════════════════════════════════════════════════════════════════
#  IMAGE I/O  (generic framework: PNG/JPEG/BMP/TIFF + DICOM)
# ═══════════════════════════════════════════════════════════════════════

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".dcm", ".dicom"}

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
    Robust DICOM -> grayscale uint8 loader (VOI LUT aware, MONOCHROME1-inverted, YBR-aware).
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
    """Colour DICOM -> BGR uint8, used only for the visualisation background."""
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
    """Single entry point for grayscale loading, used for BOTH model input and GT-shape
    validation. Identical behaviour to cv2.imread(..., IMREAD_GRAYSCALE) for every
    non-DICOM format; DICOM is additionally supported."""
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
    """Loader for the visualisation background only - never used for model input."""
    suffix = Path(image_path).suffix.lower()
    if suffix in (".dcm", ".dicom"):
        if _is_ybr_dicom(image_path):
            bgr = _load_dicom_as_bgr_uint8(image_path)
            if bgr is not None:
                return bgr
        gray = load_gray_image(image_path)
        return None if gray is None else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    bgr = cv2.imread(image_path)
    if bgr is None:
        logging.warning(f"Cannot read image for visualisation: {image_path}")
    return bgr


def discover_images(image_dir: str) -> list[Path]:
    """Robust recursive discovery of every supported image/DICOM file under image_dir."""
    return sorted(
        p for p in Path(image_dir).rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )


# ═══════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════

def build_model(device: torch.device) -> MultiTaskSwinUNet:
    """Exact same constructor call the existing limbs training/inference code uses."""
    return MultiTaskSwinUNet(
        in_chans=Config.IN_CHANNELS,
        num_seg_classes=Config.NUM_SEG_CLASSES,
        num_bone_classes=Config.NUM_BONE_CLASSES,
        embed_dim=Config.EMBED_DIM,
        depths=Config.DEPTHS,
        num_heads=Config.NUM_HEADS,
        window_size=Config.WINDOW_SIZE,
        fpn_channels=Config.FPN_CHANNELS,
        proj_dim=Config.PROJ_DIM,
        ccdc_proj_dim=Config.CCDC_PROJ_DIM,
    ).to(device)


def load_checkpoint_strict(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    """
    Strict checkpoint verification: reports missing/unexpected/shape-mismatched keys
    and ABORTS rather than silently partially loading.
    """
    if not os.path.exists(checkpoint_path):
        raise SystemExit(f"[ABORT] Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not (isinstance(ckpt, dict) and "model" in ckpt):
        raise SystemExit(
            f"[ABORT] Checkpoint at {checkpoint_path} is not a dict with a 'model' key - "
            f"refusing to guess how to load it."
        )
    state_dict = ckpt["model"]
    epoch = ckpt.get("epoch", None)
    best_mean_dice = ckpt.get("best_mean_dice", None)

    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    ckpt_keys = set(state_dict.keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    shape_mismatches = []
    for k in (model_keys & ckpt_keys):
        if model_state[k].shape != state_dict[k].shape:
            shape_mismatches.append((k, tuple(state_dict[k].shape), tuple(model_state[k].shape)))

    logging.info("=== Checkpoint verification (STRICT) ===")
    logging.info(f"  checkpoint path        : {checkpoint_path}")
    logging.info(f"  checkpoint epoch       : {epoch}")
    logging.info(f"  checkpoint best_mean_dice (bookkeeping value at save time): {best_mean_dice}")

    if missing or unexpected or shape_mismatches:
        logging.error(
            f"  RESULT: INCOMPLETE / MISMATCHED ({len(missing)} missing, {len(unexpected)} "
            f"unexpected, {len(shape_mismatches)} shape mismatches) - inference will NOT run."
        )
        for k in missing:
            logging.error(f"    missing: {k}")
        for k in unexpected:
            logging.error(f"    unexpected: {k}")
        for k, ck_shape, m_shape in shape_mismatches:
            logging.error(f"    shape mismatch: {k}: checkpoint {ck_shape} vs model {m_shape}")
        raise SystemExit(
            "[ABORT] Refusing to run inference with partially random weights."
        )

    model.load_state_dict(state_dict, strict=True)
    logging.info(f"  RESULT: all {len(model_keys)}/{len(model_keys)} parameters restored exactly.")
    return epoch, best_mean_dice


# ═══════════════════════════════════════════════════════════════════════
#  PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════

PAD_MULTIPLE = Config.PAD_MULTIPLE  # 32


def pad_to_multiple_image(image: np.ndarray, multiple: int = PAD_MULTIPLE) -> tuple[np.ndarray, int, int]:
    h, w = image.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return image, 0, 0
    padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    return padded, pad_h, pad_w


def pad_to_multiple_mask(mask: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    if pad_h == 0 and pad_w == 0:
        return mask
    return cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=IGNORE_INDEX)


def preprocess_gray_to_tensor(gray: np.ndarray) -> tuple[torch.Tensor, int, int]:
    """gray uint8 (H,W) -> normalized (1,1,H',W') tensor, plus the (pad_h, pad_w) applied."""
    padded, pad_h, pad_w = pad_to_multiple_image(gray)
    img = padded.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    return tensor, pad_h, pad_w


def derive_gt_bone_label(mask: np.ndarray) -> int | None:
    """Same dominant-class logic as BoneUSDataset._derive_bone_label()."""
    vals, counts = np.unique(mask[mask != IGNORE_INDEX], return_counts=True)
    fg = [(v, c) for v, c in zip(vals, counts) if v != 0]
    if not fg:
        return None
    dominant_val = max(fg, key=lambda vc: vc[1])[0]
    return int(dominant_val) - 1  # femur=0, humerus=1, radius_ulna=2, tibia_fibula=3


def find_gt_mask_path(masks_dir: Path | None, stem: str) -> Path | None:
    """
    Resolves GT by filename-STEM matching against masks_dir. Returns None (not an error) if
    masks_dir is None, doesn't exist, or no match is found - GT metrics are simply skipped.
    """
    if masks_dir is None:
        return None
    candidate = masks_dir / f"{stem}.npz"
    return candidate if candidate.exists() else None


def load_gt_for_image(masks_dir: Path | None, stem: str, pad_h: int, pad_w: int
                       ) -> tuple[np.ndarray | None, int | None]:
    """
    GT is entirely OPTIONAL. If masks_dir is None (no --masks_dir given), or the folder
    doesn't exist, or no matching <stem>.npz is found, this returns (None, None) and the
    rest of the pipeline (metrics, JSON, visualisation) proceeds in inference-only mode -
    it never raises and never blocks a prediction from being produced/visualised.
    """
    npz_path = find_gt_mask_path(masks_dir, stem)
    if npz_path is None:
        return None, None
    gt_mask = load_bone_mask_from_npz(str(npz_path))
    gt_mask = pad_to_multiple_mask(gt_mask, pad_h, pad_w)
    gt_bone_idx = derive_gt_bone_label(gt_mask)
    return gt_mask, gt_bone_idx


@torch.no_grad()
def run_model(model: torch.nn.Module, tensor: torch.Tensor, device: torch.device):
    """ONE forward pass. seg_logits and bone_logits are both taken from this single call and
    reused for every downstream computation."""
    outputs = model(tensor.to(device))
    return outputs["seg_logits"], outputs["bone_logits"]


# ═══════════════════════════════════════════════════════════════════════
#  LIMB POST-PROCESSING (unchanged math)
# ═══════════════════════════════════════════════════════════════════════

BONE_CLASS_ORDER = ("femur", "humerus", "radius_ulna", "tibia_fibula")
BONE_LABEL_TO_SEG_CLASS = {"femur": 1, "humerus": 2, "radius_ulna": 3, "tibia_fibula": 4}

MIN_COMPONENT_THRESHOLDS = {
    "femur": 750,
    "humerus": 750,
    "radius_ulna": 750,
    "tibia_fibula": 500,
}

MORPHOLOGY_CONFIG = {
    "femur": {"kernel_size": 7, "max_hole_area": 1000},
    "humerus": {"kernel_size": 7, "max_hole_area": 1000},
    "radius_ulna": {"kernel_size": 5, "max_hole_area": 700},
    "tibia_fibula": {"kernel_size": 7, "max_hole_area": 1000},
}

CLASS_COLORS_BGR = {
    1: (60, 60, 255),      # Femur RGB = (255, 60, 60)
    2: (0, 165, 255),      # Humerus RGB = (255, 165, 0)
    3: (60, 210, 60),      # Radius RGB = (60, 210, 60)
    4: (255, 144, 30),     # Tibia RGB = (30, 144, 255)
}

METRIC_NAMES = ("dice", "iou", "sensitivity", "specificity")


def filter_small_components(pred_mask, thresholds=MIN_COMPONENT_THRESHOLDS):
    filtered = pred_mask.copy()
    for label, seg_class in BONE_LABEL_TO_SEG_CLASS.items():
        threshold = thresholds.get(label)
        if threshold is None:
            continue
        binary = (pred_mask == seg_class).astype(np.uint8)
        if binary.sum() == 0:
            continue
        num_labels, components, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for comp_id in range(1, num_labels):
            if stats[comp_id, cv2.CC_STAT_AREA] < threshold:
                filtered[components == comp_id] = 0
    return filtered


def replace_small_wrong_class_components(pred_mask, min_ratio=1.0, connectivity=8):
    cleaned = pred_mask.copy()
    class_areas = {
        seg_class: int((pred_mask == seg_class).sum())
        for seg_class in BONE_LABEL_TO_SEG_CLASS.values()
    }
    nonzero = {c: a for c, a in class_areas.items() if a > 0}
    if not nonzero:
        return cleaned
    dominant_class = max(nonzero, key=nonzero.get)
    minority_pixels = (pred_mask > 0) & (pred_mask != dominant_class)
    cleaned[minority_pixels] = dominant_class
    return cleaned


def remove_edge_protrusions(pred_mask, opening_sizes=None):
    if opening_sizes is None:
        opening_sizes = {"femur": 5, "humerus": 5, "radius_ulna": 3, "tibia_fibula": 5}
    cleaned = pred_mask.copy()
    for label, seg_class in BONE_LABEL_TO_SEG_CLASS.items():
        kernel_size = int(opening_sizes.get(label, 0))
        if kernel_size <= 0:
            continue
        binary = (pred_mask == seg_class).astype(np.uint8)
        if binary.sum() == 0:
            continue
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        removed = (binary == 1) & (opened == 0)
        cleaned[removed] = 0
    return cleaned


def enforce_largest_component(pred_mask, min_area_ratio=0.15, connectivity=8):
    filtered = pred_mask.copy()
    for label, seg_class in BONE_LABEL_TO_SEG_CLASS.items():
        binary = (pred_mask == seg_class).astype(np.uint8)
        if binary.sum() == 0:
            continue
        num_labels, components, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=connectivity)
        if num_labels <= 1:
            continue
        areas = [(cid, stats[cid, cv2.CC_STAT_AREA]) for cid in range(1, num_labels)]
        largest_area = max(a for _, a in areas)
        for comp_id, area in areas:
            if area < min_area_ratio * largest_area:
                filtered[components == comp_id] = 0
    return filtered


def repair_bone_mask(pred_mask_filtered, seg_logits, morphology_config=None):
    if morphology_config is None:
        morphology_config = MORPHOLOGY_CONFIG

    seg_probs = torch.softmax(seg_logits.detach().float(), dim=0).cpu().numpy()  # (C,H,W)
    bg_prob = seg_probs[0]
    repaired = pred_mask_filtered.copy()

    for label, seg_class in BONE_LABEL_TO_SEG_CLASS.items():
        cfg = morphology_config.get(label)
        if not cfg:
            continue
        kernel_size = int(cfg.get("kernel_size", 0))
        max_hole_area = int(cfg.get("max_hole_area", 0))
        if kernel_size <= 0:
            continue

        binary = (pred_mask_filtered == seg_class).astype(np.uint8)
        if binary.sum() == 0:
            continue

        class_prob = seg_probs[seg_class]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        gap_candidates = (closed == 1) & (binary == 0)

        inv_bg = (1 - closed).astype(np.uint8)
        num_bg_labels, bg_components, bg_stats, _ = cv2.connectedComponentsWithStats(inv_bg, connectivity=4)
        border_ids = (set(bg_components[0, :].tolist()) | set(bg_components[-1, :].tolist())
                      | set(bg_components[:, 0].tolist()) | set(bg_components[:, -1].tolist()))

        hole_candidates = np.zeros_like(closed, dtype=bool)
        for comp_id in range(1, num_bg_labels):
            if comp_id in border_ids:
                continue
            area = bg_stats[comp_id, cv2.CC_STAT_AREA]
            if area <= max_hole_area:
                hole_candidates |= (bg_components == comp_id)

        proposed = gap_candidates | hole_candidates
        if not proposed.any():
            continue

        accepted = proposed & (class_prob >= bg_prob + 0.10)
        writable = accepted & (repaired == 0)
        repaired[writable] = seg_class

    return repaired


def run_limb_postprocessing(pred_mask_raw: np.ndarray, seg_logits_single: torch.Tensor
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stage = replace_small_wrong_class_components(pred_mask_raw)
    stage = filter_small_components(stage, MIN_COMPONENT_THRESHOLDS)
    pred_mask_filtered = stage
    stage = remove_edge_protrusions(stage)
    stage = enforce_largest_component(stage, min_area_ratio=0.15, connectivity=8)
    pred_mask_final = repair_bone_mask(stage, seg_logits=seg_logits_single, morphology_config=MORPHOLOGY_CONFIG)
    return pred_mask_filtered, pred_mask_final, pred_mask_raw


# ═══════════════════════════════════════════════════════════════════════
#  BONE CLASSIFICATION (from bone_logits only, never from the seg mask)
# ═══════════════════════════════════════════════════════════════════════

def extract_bone_classification(bone_logits_single: torch.Tensor) -> tuple[dict, dict]:
    logits = bone_logits_single.detach().float().cpu()
    probs = torch.softmax(logits, dim=0)
    logits_dict = {label: float(logits[i].item()) for i, label in enumerate(BONE_CLASS_ORDER)}
    probs_dict = {label: float(probs[i].item()) for i, label in enumerate(BONE_CLASS_ORDER)}
    return probs_dict, logits_dict


# ═══════════════════════════════════════════════════════════════════════
#  METRICS (only ever computed when GT happens to be available)
# ═══════════════════════════════════════════════════════════════════════

def dice_per_class_from_mask(pred_mask, gt_mask, num_classes, ignore_index=IGNORE_INDEX, eps=1e-6):
    valid = gt_mask != ignore_index
    dices = []
    for c in range(num_classes):
        p = (pred_mask == c) & valid
        t = (gt_mask == c) & valid
        inter = np.logical_and(p, t).sum()
        union = p.sum() + t.sum()
        dices.append(float((2 * inter + eps) / (union + eps)))
    return dices


def compute_detailed_class_metrics(pred, gt, cls, ignore_index=IGNORE_INDEX, eps=1e-6):
    valid = gt != ignore_index
    p = (pred == cls)[valid]
    t = (gt == cls)[valid]
    tp = np.logical_and(p, t).sum()
    fp = np.logical_and(p, ~t).sum()
    fn = np.logical_and(~p, t).sum()
    tn = np.logical_and(~p, ~t).sum()
    class_present = t.any()
    if class_present:
        dice = float((2 * tp + eps) / (2 * tp + fp + fn + eps))
        iou = float((tp + eps) / (tp + fp + fn + eps))
        sensitivity = float((tp + eps) / (tp + fn + eps))
    else:
        dice = iou = sensitivity = None
    specificity = float((tn + eps) / (tn + fp + eps)) if (tn + fp) > 0 else None
    return {"dice": dice, "iou": iou, "sensitivity": sensitivity, "specificity": specificity,
            "present": bool(class_present)}


def compute_training_comparable_mean_fg_dice(rows: list[dict]) -> float | None:
    per_class_means = []
    for c in range(1, NUM_SEG_CLASSES):
        label = LABEL_MAP[c]
        vals = [float(r[f"dice_{label}"]) for r in rows if r[f"dice_{label}"] != "NA"]
        if vals:
            per_class_means.append(float(np.mean(vals)))
    return float(np.mean(per_class_means)) if per_class_means else None


def compute_all_class_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, dict]:
    return {
        LABEL_MAP[c]: compute_detailed_class_metrics(pred_mask, gt_mask, c)
        for c in range(1, NUM_SEG_CLASSES)
    }


# ═══════════════════════════════════════════════════════════════════════
#  SEGMENTATION-DERIVED PLANE / CONFIDENCE / POLYGONS (FINAL mask only)
#  NOTE: "plane" here just means "which of the 4 bone classes was segmented" -
#  it has nothing to do with input folder organisation.
# ═══════════════════════════════════════════════════════════════════════

PLANE_CODE_FOR_LABEL = {"femur": "Femur", "humerus": "Humerus", "radius_ulna": "Radius and Ulna", "tibia_fibula": "Tibia and Fibula"}
DISPLAY_NAME_FOR_LABEL = {"femur": "Femur", "humerus": "Humerus",
                          "radius_ulna": "Radius_Ulna", "tibia_fibula": "Tibia_Fibula"}


def determine_final_bone(pred_mask_final: np.ndarray) -> tuple[int | None, str | None]:
    areas = {seg_class: int((pred_mask_final == seg_class).sum())
             for seg_class in BONE_LABEL_TO_SEG_CLASS.values()}
    nonzero = {c: a for c, a in areas.items() if a > 0}
    if not nonzero:
        return None, None
    dominant_class = max(nonzero, key=nonzero.get)
    label = LABEL_MAP[dominant_class]
    return dominant_class, label


def compute_mean_structure_confidence(seg_logits_single: torch.Tensor, pred_mask_final: np.ndarray,
                                       predicted_class: int | None) -> float:
    if predicted_class is None:
        return 0.0
    seg_probs = torch.softmax(seg_logits_single.detach().float(), dim=0).cpu().numpy()  # (C,H,W)
    class_pixels = (pred_mask_final == predicted_class)
    if not class_pixels.any():
        return 0.0
    return float(seg_probs[predicted_class][class_pixels].mean())


def compute_plane_candidates(probs_dict: dict, threshold: float = 0.10) -> list[str]:
    candidates = [(label, p) for label, p in probs_dict.items() if p >= threshold]
    candidates.sort(key=lambda lp: lp[1], reverse=True)
    return [label for label, _ in candidates]


def mask_to_polygons(mask_orig: np.ndarray, structure_name: str, min_pixels: int = 20,
                      epsilon_fraction: float = 0.01) -> list[dict]:
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
            "structure": DISPLAY_NAME_FOR_LABEL.get(structure_name, structure_name),
            "type": "polygon",
            "points": [{"x": float(x), "y": float(y)} for x, y in pts],
        })
    return polygons


def build_polygons_from_final_mask(pred_mask_final: np.ndarray, orig_h: int, orig_w: int) -> list[dict]:
    mask_orig = crop_to_original(pred_mask_final, orig_h, orig_w)
    polygons: list[dict] = []
    for label, seg_class in BONE_LABEL_TO_SEG_CLASS.items():
        class_mask = (mask_orig == seg_class).astype(np.uint8)
        polygons.extend(mask_to_polygons(class_mask, label))
    return polygons


# ═══════════════════════════════════════════════════════════════════════
#  VISUALISATION (works fully in inference-only mode; GT panel just omitted)
# ═══════════════════════════════════════════════════════════════════════

def overlay_mask(base_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.35,
                  contour_thickness: int = 2) -> np.ndarray:
    overlay = base_bgr.copy().astype(np.float32)
    for cls_idx, color in CLASS_COLORS_BGR.items():
        binary = (mask == cls_idx).astype(np.uint8)
        if binary.sum() == 0:
            continue
        color_arr = np.array(color, dtype=np.float32)
        m = binary.astype(bool)
        overlay[m] = overlay[m] * (1 - alpha) + color_arr * alpha
    overlay = overlay.astype(np.uint8)
    for cls_idx, color in CLASS_COLORS_BGR.items():
        binary = (mask == cls_idx).astype(np.uint8)
        if binary.sum() == 0:
            continue
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, contour_thickness, lineType=cv2.LINE_AA)
    return overlay


def add_title_bar(img_bgr: np.ndarray, text: str, bar_height: int = 32,
                   bg_color=(30, 30, 30), text_color=(255, 255, 255)) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    bar = np.full((bar_height, w, 3), bg_color, dtype=np.uint8)
    cv2.putText(bar, text, (10, bar_height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)
    return np.vstack([bar, img_bgr])


def add_legend(img_bgr: np.ndarray, bar_height: int = 32,
                bg_color=(245, 245, 245), text_color=(20, 20, 20)) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    bar = np.full((bar_height, w, 3), bg_color, dtype=np.uint8)
    x = 14
    for cls_idx in range(1, NUM_SEG_CLASSES):
        color = CLASS_COLORS_BGR[cls_idx]
        cv2.rectangle(bar, (x, bar_height // 2 - 9), (x + 18, bar_height // 2 + 9), color, -1)
        label = LABEL_MAP[cls_idx]
        cv2.putText(bar, label, (x + 24, bar_height // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, text_color, 1, cv2.LINE_AA)
        x += 24 + len(label) * 9 + 24
    return np.vstack([img_bgr, bar])


def crop_to_original(arr: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
    oh = min(orig_h, arr.shape[0])
    ow = min(orig_w, arr.shape[1])
    return arr[:oh, :ow]


def visualise(image_path: str, pred_mask_final: np.ndarray, gt_mask: np.ndarray | None,
              orig_h: int, orig_w: int, out_path: Path, input_suffix: str,
              gt_bone_name: str, pred_bone_name: str, alpha: float = 0.35) -> Path | None:
    """
    Prediction overlay is ALWAYS produced. The side-by-side GT panel is added only when
    gt_mask is not None (i.e. GT happened to be found) - it is never required.
    """
    bgr = load_bgr_image(image_path)
    if bgr is None:
        return None
    bgr = crop_to_original(bgr, orig_h, orig_w)
    pred_crop = crop_to_original(pred_mask_final, orig_h, orig_w)

    pred_overlay = overlay_mask(bgr, pred_crop, alpha=alpha)
    # pred_tile = add_title_bar(pred_overlay, f"Prediction  |  bone: {pred_bone_name}")

    if gt_mask is not None:
        gt_crop = crop_to_original(gt_mask, orig_h, orig_w)
        gt_overlay = overlay_mask(bgr, gt_crop, alpha=alpha)
        gt_tile = add_title_bar(gt_overlay, f"Ground Truth  |  bone: {gt_bone_name}")
        divider = np.full((gt_tile.shape[0], 6, 3), 200, dtype=np.uint8)
        # combined = np.hstack([gt_tile, divider, pred_tile])
        combined = np.hstack([
            gt_tile,
            divider,
            pred_overlay
        ])
    else:
        # combined = pred_tile
        combined = pred_overlay

    # combined = add_legend(combined)

    out_ext, imwrite_params = _output_ext_params(input_suffix)
    final_path = out_path.with_suffix(out_ext)
    cv2.imwrite(str(final_path), combined, imwrite_params)
    return final_path


# ═══════════════════════════════════════════════════════════════════════
#  JSON EXPORT
# ═══════════════════════════════════════════════════════════════════════

def build_per_image_json_entry(
    image_path: Path, vis_path: Path | None, seg_plane: str | None, seg_predicted_label: str | None,
    mean_structure_confidence: float, plane_candidates: list[str],
    probs_dict: dict, polygons: list[dict],
    class_metrics: dict[str, dict] | None,
) -> dict:
    return {
        "image_path": str(image_path),
        "vis_path": str(vis_path),
        "input_filename": image_path.name,
        "plane": seg_plane if seg_plane is not None else "unknown",
        "predicted_bone_class": seg_predicted_label.upper() if seg_predicted_label else None,
        "mean_structure_confidence": round(mean_structure_confidence, 6),
        "plane_candidates": plane_candidates,
        "bone_classification": {
            "probabilities": probs_dict,
        },
        "polygons": polygons,
        "gt_metrics_available": class_metrics is not None,
        "gt_metrics": (
            {
                label: {
                    "dice": m["dice"], "iou": m["iou"],
                    "sensitivity": m["sensitivity"], "specificity": m["specificity"],
                    "present": m["present"],
                }
                for label, m in class_metrics.items()
            }
            if class_metrics is not None else None
        ),
    }


from ui.utils.dcm_png_utils import anonymized_png_for_dicom

# ═══════════════════════════════════════════════════════════════════════
#  MAIN INFERENCE LOOP  (single flat image_dir, no plane organisation)
# ═══════════════════════════════════════════════════════════════════════

def _process_images(
    model: torch.nn.Module,
    device: torch.device,
    image_dir: str,
    masks_dir: str | None,
    vis_dir: Path,
    alpha: float,
    accum: dict,
) -> dict:
    """
    Runs inference over every image found (recursively) under image_dir. masks_dir is fully
    OPTIONAL: pass None to run pure inference - predictions + visualisations + JSON are still
    produced for every image, GT-dependent fields just come back empty/null.
    """
    n_images = 0
    n_with_gt = 0

    masks_path = Path(masks_dir) if masks_dir else None
    if masks_path is not None and not masks_path.is_dir():
        logging.info(f"masks_dir '{masks_dir}' does not exist - "
                     f"running in INFERENCE-ONLY mode (no GT metrics).")
        masks_path = None
    elif masks_path is None:
        logging.info("No masks_dir given - running in INFERENCE-ONLY mode "
                      "(predictions + visualisations only, no GT metrics).")

    img_paths = discover_images(image_dir)
    logging.info(f"Found {len(img_paths)} images in {image_dir}")
    logging.info(f"GT masks directory: {masks_path if masks_path else 'None (GT metrics disabled)'}")

    for n_done, img_path in enumerate(img_paths, start=1):
        stem = img_path.stem
        input_suffix = img_path.suffix
        logging.info(f"[{n_done}/{len(img_paths)}] Processing: {stem}{input_suffix}")

        raw_gray = load_gray_image(str(img_path))
        if raw_gray is None:
            continue
        orig_h, orig_w = raw_gray.shape[:2]

        tensor, pad_h, pad_w = preprocess_gray_to_tensor(raw_gray)

        # ── single forward pass; seg_logits + bone_logits reused for everything below ──
        seg_logits, bone_logits = run_model(model, tensor, device)

        pred_mask_raw = torch.argmax(seg_logits[0], dim=0).cpu().numpy()

        pred_mask_filtered, pred_mask_final, _ = run_limb_postprocessing(pred_mask_raw, seg_logits[0])

        # ── GT (fully optional; None whenever masks_path is None - never blocks inference) ──
        gt_mask, gt_bone_idx = load_gt_for_image(masks_path, stem, pad_h, pad_w)
        gt_bone_name = LABEL_MAP[gt_bone_idx + 1] if gt_bone_idx is not None else None
        class_metrics = None
        if gt_mask is not None:
            n_with_gt += 1
            accum["n_with_gt"] += 1
            class_metrics = compute_all_class_metrics(pred_mask_final, gt_mask)

        # ── bone classification (from bone_logits only) ──
        probs_dict, logits_dict = extract_bone_classification(bone_logits[0])
        pred_bone_idx = int(torch.argmax(bone_logits[0]).item())
        pred_bone_name = LABEL_MAP[pred_bone_idx + 1]
        if gt_bone_idx is not None:
            accum["bone_total"] += 1
            accum["bone_correct"] += int(pred_bone_idx == gt_bone_idx)

        # ── visualisation: ALWAYS produced, regardless of GT availability. Filenames use the
        #    plain stem since there's only one shared folder - use rglob-safe stem (no plane
        #    prefix needed since images all come from one tree, but if two files share a stem
        #    in different subfolders this will overwrite; rare in flat datasets) ──
        png_path = anonymized_png_for_dicom(img_path)

        if png_path:
            vis_path = visualise(
                str(png_path), pred_mask_final, gt_mask, orig_h, orig_w,
                vis_dir / f"{stem}_vis", input_suffix,
                gt_bone_name=(gt_bone_name.upper() if gt_bone_name else "NA"),
                pred_bone_name=pred_bone_name.upper(), alpha=alpha,
            )
        else:
            vis_path = visualise(
                str(img_path), pred_mask_final, gt_mask, orig_h, orig_w,
                vis_dir / f"{stem}_vis", input_suffix,
                gt_bone_name=(gt_bone_name.upper() if gt_bone_name else "NA"),
                pred_bone_name=pred_bone_name.upper(), alpha=alpha,
            )

        if vis_path:
            logging.info(f"    [VIS] saved -> {vis_path}")
        else:
            logging.warning(f"    [VIS] FAILED to save visualisation for {stem}{input_suffix}")

        # ── segmentation-derived plane / confidence / polygons (FINAL mask only) ──
        seg_class_idx, seg_label = determine_final_bone(pred_mask_final)
        seg_plane_code = PLANE_CODE_FOR_LABEL[seg_label] if seg_label is not None else None
        mean_structure_confidence = compute_mean_structure_confidence(
            seg_logits[0], pred_mask_final, seg_class_idx
        )
        if seg_label is None:
            plane_candidates = []
            polygons = []
        else:
            plane_candidates = compute_plane_candidates(probs_dict)
            polygons = build_polygons_from_final_mask(pred_mask_final, orig_h, orig_w)

        # ── per-image entry, appended into the SHARED aggregate json_entries list ──
        entry = build_per_image_json_entry(
            img_path, vis_path, seg_plane_code, seg_label, mean_structure_confidence, plane_candidates,
            probs_dict, polygons, class_metrics,
        )
        accum["json_entries"].append(entry)

        n_images += 1

        logging.info(f"    GT={gt_bone_name.upper() if gt_bone_name else 'NA'}  "
                     f"Pred={pred_bone_name.upper()}  "
                     f"Femur={probs_dict['femur']:.4f} Humerus={probs_dict['humerus']:.4f} "
                     f"Radius_Ulna={probs_dict['radius_ulna']:.4f} Tibia_Fibula={probs_dict['tibia_fibula']:.4f}")

    return {"n_images": n_images, "n_images_with_gt": n_with_gt}


def _cumulative_training_comparable_mean_fg_dice(all_entries: list[dict]) -> float | None:
    per_class_means = []
    for label in BONE_CLASS_ORDER:
        vals = []
        for entry in all_entries:
            gt_metrics = entry.get("gt_metrics")
            if not gt_metrics:
                continue
            m = gt_metrics.get(label)
            if m and m.get("present") and m.get("dice") is not None:
                vals.append(float(m["dice"]))
        if vals:
            per_class_means.append(float(np.mean(vals)))
    return float(np.mean(per_class_means)) if per_class_means else None


def run_inference_limbs(
    checkpoint: str,
    items,
    masks_dir: str | None = None,
    device_str: str = "cuda",
    alpha: float = 0.35,
    min_component_thresholds: dict | None = None,
    morphology_config: dict | None = None,
) -> tuple[Path, Path]:
    """
    Single-folder limb-bone segmentation + classification inference. No plane_dirs, no FE/HU/
    RU/TF concept - point it at ONE folder of images and it predicts every image's bone type
    automatically from the model itself.

    masks_dir is entirely OPTIONAL - leave it as None to run pure inference: predictions +
    visualisations + JSON are produced exactly the same either way, GT-dependent fields simply
    come back as null/"NA".
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    global MIN_COMPONENT_THRESHOLDS, MORPHOLOGY_CONFIG
    thresholds = min_component_thresholds or MIN_COMPONENT_THRESHOLDS
    morph_cfg = morphology_config or MORPHOLOGY_CONFIG

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # if not Path(image_dir).is_dir():
    #     raise SystemExit(f"[ABORT] image_dir '{image_dir}' does not exist.")

    # out = Path(output_dir)
    # out.mkdir(parents= True, exist_ok = True)

    # vis_dir = out / "visualisations"
    # met_dir = out / "metrics"
    # for d in (vis_dir, met_dir):
    #     d.mkdir(parents=True, exist_ok=True)

    model = build_model(device)
    epoch, best_mean_dice = load_checkpoint_strict(model, checkpoint, device)
    model.eval()

    logging.info(f"Checkpoint: {checkpoint}  (epoch={epoch}, best_mean_dice={best_mean_dice})")
    # logging.info(f"Image dir: {image_dir}")

    accum = {
        "json_entries": [],
        "bone_correct": 0,
        "bone_total": 0,
        "n_with_gt": 0,
    }

    stats = _process_images(
        model=model,
        device=device,
        image_dir=image_dir,
        masks_dir=masks_dir,
        vis_dir=vis_dir,
        alpha=alpha,
        accum=accum,
    )
    logging.info(f"Finished: {stats['n_images']} images processed ({stats['n_images_with_gt']} with GT)")

    json_entries = accum["json_entries"]
    if not json_entries:
        logging.warning("No images processed - nothing to write. Double-check --image_dir "
                         "actually exists and contains supported files "
                         "(.png/.jpg/.jpeg/.bmp/.tiff/.tif/.dcm/.dicom).")
        return vis_dir, out / "inference_results.json"

    # ── ONE aggregate JSON. APPEND mode: if inference_results.json already exists, its
    #    existing "images" entries are merged in rather than overwritten. ──
    results_json_path = out / "inference_results.json"
    existing_entries: list[dict] = []
    existing_bone_correct = 0
    existing_bone_total = 0
    if results_json_path.exists():
        try:
            with open(results_json_path, "r") as f:
                prev = json.load(f)
            existing_entries = prev.get("images", [])
            prev_summary = prev.get("run_summary", {})
            existing_bone_correct = prev_summary.get("bone_classification_correct", 0) or 0
            existing_bone_total = prev_summary.get("bone_classification_total", 0) or 0
            logging.info(f"[JSON] found existing {results_json_path} with {len(existing_entries)} "
                         f"image(s) - appending this run's results to it")
        except Exception as exc:
            logging.warning(f"Could not read existing {results_json_path} ({exc}) - starting fresh.")

    all_entries = existing_entries + json_entries
    total_bone_correct = existing_bone_correct + accum["bone_correct"]
    total_bone_total = existing_bone_total + accum["bone_total"]
    n_with_gt = sum(1 for e in all_entries if e.get("gt_metrics_available"))
    bone_acc = (total_bone_correct / total_bone_total) if total_bone_total > 0 else None

    run_summary = {
        "n_images": len(all_entries),
        "n_images_with_gt": n_with_gt,
        "mean_structure_dice_over_images_with_gt": _cumulative_training_comparable_mean_fg_dice(all_entries),
        "bone_classification_accuracy": bone_acc,
        "bone_classification_correct": total_bone_correct,
        "bone_classification_total": total_bone_total,
    }
    with open(results_json_path, "w") as f:
        json.dump({"run_summary": run_summary, "images": all_entries}, f, indent=2)
    logging.info(f"\n[JSON] wrote {results_json_path}  ({len(all_entries)} images total)")

    return str(vis_dir), str(results_json_path)


# ─────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Single-folder limb-bone segmentation + classification INFERENCE "
            "(femur / humerus / radius_ulna / tibia_fibula), no plane_dirs. "
            "Point --image_dir at ONE folder (scanned recursively) and every image's bone "
            "type is detected automatically by the model - PNG/JPEG/BMP/TIFF/DICOM supported. "
            "--masks_dir is entirely OPTIONAL: omit it to run pure inference (predictions + "
            "visualisations + JSON, no dice/iou/etc metrics)."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default="/mnt/data4tb/anusha/Limbs_Model_copy/checkpoints_full_agreement_head_v2/best.pt",
        help="Path to train.py-style checkpoint."
    )
    parser.add_argument(
        "--output_dir",
        default="/mnt/data4tb/anusha/Limbs_Model_copy/test/output",
        help="Where visualisations/ and metrics/ go."
    )
    parser.add_argument(
        "--image_dir",
        default = "/mnt/data4tb/anusha/Limbs_Model_copy/Online_data",
        help="Folder to scan recursively for images/DICOM. Required."
    )
    parser.add_argument(
        "--masks_dir",
        default=None,
        help="OPTIONAL: folder of GT .npz masks, matched to images by filename stem. "
             "Leave unset to run pure inference (no GT metrics)."
    )
    parser.add_argument("--device", default="cuda", help="Device to use for inference, e.g. cuda or cpu.")
    parser.add_argument("--alpha", type=float, default=0.35, help="Overlay blend alpha for visualisation.")

    args = parser.parse_args()

    run_inference_limbs(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        masks_dir=args.masks_dir,
        device_str=args.device,
        alpha=args.alpha,
    )