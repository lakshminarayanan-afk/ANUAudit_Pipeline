"""
infer/infer.py
==============

Complete Cls -> YOLO26m Det -> Binary Swin-UNet Seg inference.

Pipeline
--------
1. Frozen existing hierarchical ConvNeXt classifier
2. YOLO26m detector
3. Per-bone confidence extraction
4. 60% classifier + 40% YOLO weighted fusion
5. Select fused target bone
6. Use ALL spatially separate YOLO boxes as ROIs
   (YOLO class is NOT a segmentation-class gate)
7. Binary Swin-UNet segments the FUSED bone inside every YOLO ROI
8. Strong smooth post-processing:
      - remove tiny blobs
      - elliptical opening
      - elliptical closing
      - fill holes
      - final small-component filtering
      - intensity-guided boundary recovery
      - Gaussian boundary smoothing
      - NO largest-component filtering
9. Restore masks to original image size
10. Standard / Non-Standard / UNKNOWN classification
11. Visualization:
      LEFT  = original image
      RIGHT = prediction
      TOP   = classifier / YOLO / fused confidence
      BOTTOM = Standard / Non-Standard / UNKNOWN + short reason
12. inference_results.json (aggregate run_summary + images)
13. inference_summary.json (backward-compatible image-list copy)

Classification rule
-------------------
UNKNOWN
    -> no segmentation mask was produced

NON-STANDARD
    -> a mask exists, but its geometry does not satisfy
       the Standard requirements

STANDARD
    -> a mask exists and its geometry satisfies
       the Standard requirements
"""

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Optional DICOM support.  DICOM files are converted to the same grayscale
# uint8 image representation used by PNG/JPEG/TIFF before entering the
# existing classifier -> detector -> segmenter pipeline.
try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    _DICOM_AVAILABLE = True
except ImportError:
    pydicom = None
    apply_voi_lut = None
    _DICOM_AVAILABLE = False

from config.config import Config
from model.model import (
    HierarchicalUltrasoundModel,
    SegmentationModel,
    load_yolo_detector,
)


# ============================================================
# CONSTANTS
# ============================================================

BONE_NAMES = {
    1: "Femur",
    2: "Humerus",
    3: "Radius/Ulna",
    4: "Tibia/Fibula",
}

YOLO_CLASS_TO_BONE = {
    0: 1,   # YOLO Femur       -> final Femur
    1: 2,   # YOLO Humerus     -> final Humerus
    2: 3,   # YOLO Radius/Ulna -> final Radius/Ulna
    3: 4,   # YOLO Tibia/Fibula-> final Tibia/Fibula
}

# Recovered from the existing classifier label mapping.
# DO NOT change these.
CLASSIFIER_PLANE_TO_BONE = {
    9: 1,    # Femur
    11: 2,   # Humerus
    20: 3,   # Radius/Ulna
    22: 4,   # Tibia/Fibula
}

CLASSIFIER_WEIGHT = 0.60
DETECTOR_WEIGHT = 0.40

# Only suppress boxes that are effectively duplicates.
# Separate split-screen boxes remain.
DUPLICATE_IOU_THRESHOLD = 0.70

# Segmentation cleanup.
MIN_COMPONENT_AREA = 150

# Stronger morphology for smooth bone boundaries.
OPEN_KERNEL = (5, 5)
CLOSE_KERNEL = (9, 9)
FINAL_CLOSE_KERNEL = (5, 5)

# ------------------------------------------------------------
# Additional boundary smoothing
# ------------------------------------------------------------
#
# This is applied AFTER the existing morphology/intensity
# recovery. It smooths jagged / pixelated segmentation edges
# without selecting only the largest component.
#
SMOOTH_BLUR_KERNEL = (5, 5)
SMOOTH_BLUR_SIGMA = 1.2
SMOOTH_THRESHOLD = 0.50

# ------------------------------------------------------------
# Intensity-guided boundary recovery
# ------------------------------------------------------------

INTENSITY_GROW_KERNEL = (5, 5)
INTENSITY_THRESHOLD = 28.0
INTENSITY_GROW_ITERATIONS = 2

# ------------------------------------------------------------
# Standard / Non-Standard geometry
# ------------------------------------------------------------

FE_HU_MAX_ANGLE = 20.0
RU_TF_MAX_ANGLE = 40.0

# Minimum geometry for a component to be considered a bone.
MIN_BONE_AREA = 150
MIN_BONE_LENGTH = 40.0
MIN_BONE_ELONGATION = 2.0

# For RU/TF normal-view pair detection.
# Two components need to be reasonably close and similarly oriented.
MAX_PAIR_CENTER_DISTANCE_RATIO = 0.45
MAX_PAIR_ANGLE_DIFFERENCE = 25.0

# Multiple spatially separated boxes/components are treated as evidence
# of split-screen. This is deliberately conservative.
SPLIT_SCREEN_X_GAP_RATIO = 0.15

# ------------------------------------------------------------
# Generic image/DICOM input framework
# ------------------------------------------------------------
# Mirrors the HEAD inference's generic input conventions: a single
# shared extension set used both for recursive collection and for
# dispatching DICOM vs. raster loading, plus the transfer-syntax
# photometric interpretations pydicom already decodes to RGB itself
# (so VOI LUT windowing must NOT be re-applied on top of that RGB
# output, matching the HEAD implementation).

_RASTER_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_DICOM_EXTS = {".dcm", ".dicom"}
SUPPORTED_EXTS = _RASTER_EXTS | _DICOM_EXTS

# Photometric interpretations that pydicom's own pixel handler already
# decodes to RGB (JPEG-2000 lossy/lossless colour transforms). VOI LUT
# windowing is meant for raw stored pixel values, not this decoded RGB,
# so it is skipped for these — same behavior as the HEAD reference.
_YBR_PYDICOM_CONVERT = {"YBR_ICT", "YBR_RCT"}


# ============================================================
# DEVICE
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# IMAGE LOADING
# ============================================================

def _load_dicom_as_gray_uint8(path):
    """
    Robust DICOM -> grayscale uint8 loader.

    Handles:
      - standard monochrome DICOM
      - VOI LUT / windowing when available
      - MONOCHROME1 inversion
      - RGB / YBR-like multi-channel pixel arrays by grayscale conversion
      - multi-frame DICOM by using the first frame

    The returned array is always HxW uint8, so the existing classifier,
    YOLO and segmentation preprocessing remain unchanged.
    """
    if not _DICOM_AVAILABLE:
        raise ImportError(
            "DICOM input requires pydicom. Install with: pip install pydicom"
        )

    ds = pydicom.dcmread(str(path))
    photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
    arr = ds.pixel_array

    # ----------------------------------------------------------------
    # HEAD-style early return for JPEG-2000 YBR transfer syntaxes.
    # ----------------------------------------------------------------
    # For YBR_ICT / YBR_RCT, pydicom's own pixel handler has already
    # performed the inverse colour transform and returns RGB-like uint8
    # data. Applying VOI LUT / windowing on top of that (as the generic
    # path below does for raw stored values) would double-process the
    # pixel data, so — exactly as in the HEAD reference implementation —
    # this case is handled separately and returned early.
    if photometric in _YBR_PYDICOM_CONVERT:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            arr = (
                0.2989 * arr[..., 0]
                + 0.5870 * arr[..., 1]
                + 0.1140 * arr[..., 2]
            )
        return np.clip(arr, 0, 255).astype(np.uint8)

    # Multi-frame: use first frame.  A normal ultrasound DICOM is usually
    # single-frame, but this keeps the inference loop robust.
    if arr.ndim == 4:
        arr = arr[0]

    # Colour DICOM -> grayscale.
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3].astype(np.float32)
        arr = (
            0.2989 * arr[..., 0]
            + 0.5870 * arr[..., 1]
            + 0.1140 * arr[..., 2]
        )
    elif arr.ndim == 3:
        # Single-channel multi-frame DICOM: middle frame (matches the
        # HEAD reference's multi-frame convention).
        arr = arr[arr.shape[0] // 2]

    arr = np.asarray(arr, dtype=np.float32)

    # Apply modality/VOI transformation when pydicom supports it.
    try:
        arr_voi = apply_voi_lut(arr, ds)
        arr = np.asarray(arr_voi, dtype=np.float32)
    except Exception:
        pass

    # MONOCHROME1 means low stored values are displayed as bright.
    if photometric == "MONOCHROME1":
        arr = arr.max() - arr

    # Robust per-image conversion to uint8.
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)

    valid = arr[finite]
    lo = float(valid.min())
    hi = float(valid.max())

    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr = np.zeros_like(arr)

    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(arr, 0, 255).astype(np.uint8)


def load_grayscale_image(path):
    """
    Load PNG/JPEG/BMP/TIFF or DICOM as the grayscale uint8 representation
    expected by the existing inference pipeline.
    """
    path = Path(path)

    if path.suffix.lower() in _DICOM_EXTS:
        return _load_dicom_as_gray_uint8(path)

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {path}"
        )

    return image


# ============================================================
# CLASSIFIER
# ============================================================

def load_classifier(device):
    """
    IMPORTANT:
    This is the OLD WORKING classifier loader.

    Do not replace the constructor arguments with guessed names.
    The checkpoint is byte-compatible with this architecture.
    """

    with open(
        Config.CLASSIFIER_LABEL_META,
        "r",
    ) as f:
        meta = json.load(f)

    idx2plane = {
        int(k): v
        for k, v in meta["idx2plane"].items()
    }

    num_anatomies = len(
        meta["anatomy2idx"]
    )

    num_planes = len(
        meta["plane2idx"]
    )

    classifier = HierarchicalUltrasoundModel(
        num_anatomies=num_anatomies,
        num_planes=num_planes,
        backbone_name=Config.CLASSIFIER_BACKBONE,
        pretrained=False,
        dropout=Config.CLASSIFIER_DROPOUT,
    ).to(device)

    ckpt = torch.load(
        Config.CLASSIFIER_CHECKPOINT,
        map_location=device,
    )

    classifier.load_state_dict(
        ckpt
    )

    classifier.eval()

    return classifier, idx2plane


# ============================================================
# YOLO DETECTOR
# ============================================================

def load_detector(
    device,
    ckpt_path,
):
    return load_yolo_detector(
        ckpt_path,
        device=device,
    )


# ============================================================
# SEGMENTER
# ============================================================

def load_segmenter(
    device,
    ckpt_path,
):
    """
    OLD WORKING binary Swin-UNet loader.
    """

    segmenter = SegmentationModel(
        in_chans=Config.IN_CHANNELS,
        num_seg_classes=Config.NUM_SEG_CLASSES,
        embed_dim=Config.EMBED_DIM,
        depths=Config.DEPTHS,
        num_heads=Config.NUM_HEADS,
        window_size=Config.WINDOW_SIZE,
        fpn_channels=Config.FPN_CHANNELS,
    ).to(device)

    ckpt = torch.load(
        ckpt_path,
        map_location=device,
    )

    segmenter.load_state_dict(
        ckpt["model"]
    )

    segmenter.eval()

    return segmenter


# ============================================================
# CLASSIFIER PREPROCESSING
# ============================================================

def prepare_classifier_input(
    image_gray,
    device,
):
    """
    Existing classifier preprocessing:

        grayscale
        -> [0,1]
        -> resize
        -> 3 channels
        -> ImageNet normalization
    """

    image = (
        image_gray.astype(
            np.float32
        )
        / 255.0
    )

    size = getattr(
        Config,
        "CLASSIFIER_IMG_SIZE",
        224,
    )

    image = cv2.resize(
        image,
        (size, size),
        interpolation=cv2.INTER_LINEAR,
    )

    image = np.stack(
        [
            image,
            image,
            image,
        ],
        axis=0,
    )

    mean = np.array(
        [
            0.485,
            0.456,
            0.406,
        ],
        dtype=np.float32,
    ).reshape(
        3,
        1,
        1,
    )

    std = np.array(
        [
            0.229,
            0.224,
            0.225,
        ],
        dtype=np.float32,
    ).reshape(
        3,
        1,
        1,
    )

    image = (
        image - mean
    ) / std

    tensor = torch.from_numpy(
        image
    ).unsqueeze(
        0
    ).float().to(
        device
    )

    return tensor


# ============================================================
# CLASSIFIER INFERENCE
# ============================================================

@torch.no_grad()
def run_classifier(
    image_gray,
    classifier,
    idx2plane,
    device,
):
    """
    Returns per-bone classifier probabilities.

    Multiple classifier plane classes can map to the same bone,
    so their probabilities are aggregated.
    """

    x = prepare_classifier_input(
        image_gray,
        device,
    )

    output = classifier(x)

    if not isinstance(
        output,
        dict,
    ):
        raise RuntimeError(
            "Classifier output must be a dict containing "
            "'plane' logits."
        )

    plane_logits = output["plane"]

    plane_probs = torch.softmax(
        plane_logits,
        dim=1,
    )[0]

    bone_probs = {
        1: 0.0,
        2: 0.0,
        3: 0.0,
        4: 0.0,
    }

    for (
        plane_idx,
        probability,
    ) in enumerate(
        plane_probs.detach().cpu().numpy()
    ):

        if plane_idx not in CLASSIFIER_PLANE_TO_BONE:
            continue

        bone_id = (
            CLASSIFIER_PLANE_TO_BONE[
                plane_idx
            ]
        )

        bone_probs[bone_id] += float(
            probability
        )

    predicted_bone = max(
        bone_probs,
        key=bone_probs.get,
    )

    confidence = bone_probs[
        predicted_bone
    ]

    top_plane_idx = int(
        torch.argmax(
            plane_probs
        ).item()
    )

    top_plane_name = idx2plane.get(
        top_plane_idx,
        str(top_plane_idx),
    )

    return {
        "bone_probs": bone_probs,
        "predicted_bone": predicted_bone,
        "confidence": float(
            confidence
        ),
        "top_plane_idx": top_plane_idx,
        "top_plane_name": top_plane_name,
    }


# ============================================================
# YOLO INFERENCE
# ============================================================

@torch.no_grad()
def run_detector(
    image_gray,
    detector,
    device,
):
    """
    Run YOLO on the full image.

    Returns:
        list of:
        {
            bbox,
            confidence,
            yolo_class,
            bone_id
        }
    """

    image_rgb = cv2.cvtColor(
        image_gray,
        cv2.COLOR_GRAY2RGB,
    )

    imgsz = getattr(
        Config,
        "YOLO_IMGSZ",
        512,
    )

    conf = getattr(
        Config,
        "YOLO_CONF_THRESHOLD",
        0.25,
    )

    iou = getattr(
        Config,
        "YOLO_IOU_THRESHOLD",
        0.45,
    )

    max_det = getattr(
        Config,
        "YOLO_MAX_DET",
        20,
    )

    results = detector.predict(
        source=image_rgb,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        verbose=False,
        device=(
            0
            if device.type == "cuda"
            else "cpu"
        ),
    )

    if (
        results is None
        or len(results) == 0
    ):
        return []

    result = results[0]

    if result.boxes is None:
        return []

    boxes = result.boxes

    xyxy = (
        boxes.xyxy
        .detach()
        .cpu()
        .numpy()
    )

    confidences = (
        boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    classes = (
        boxes.cls
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )

    detections = []

    for (
        bbox,
        confidence,
        cls_id,
    ) in zip(
        xyxy,
        confidences,
        classes,
    ):

        bone_id = (
            YOLO_CLASS_TO_BONE.get(
                int(cls_id)
            )
        )

        if bone_id is None:
            continue

        x1, y1, x2, y2 = [
            float(v)
            for v in bbox
        ]

        detections.append(
            {
                "bbox": [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
                "confidence": float(
                    confidence
                ),
                "yolo_class": int(
                    cls_id
                ),
                "bone_id": int(
                    bone_id
                ),
            }
        )

    return detections


# ============================================================
# YOLO PER-BONE CONFIDENCES
# ============================================================

def get_detector_bone_confidences(
    detections,
):
    """
    Maximum YOLO confidence for each bone.
    """

    bone_probs = {
        1: 0.0,
        2: 0.0,
        3: 0.0,
        4: 0.0,
    }

    for detection in detections:

        bone_id = detection[
            "bone_id"
        ]

        bone_probs[bone_id] = max(
            bone_probs[bone_id],
            detection["confidence"],
        )

    return bone_probs


# ============================================================
# BOX IOU
# ============================================================

def box_iou(
    box_a,
    box_b,
):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(
        ax1,
        bx1,
    )

    iy1 = max(
        ay1,
        by1,
    )

    ix2 = min(
        ax2,
        bx2,
    )

    iy2 = min(
        ay2,
        by2,
    )

    iw = max(
        0.0,
        ix2 - ix1,
    )

    ih = max(
        0.0,
        iy2 - iy1,
    )

    intersection = (
        iw * ih
    )

    area_a = (
        max(
            0.0,
            ax2 - ax1,
        )
        *
        max(
            0.0,
            ay2 - ay1,
        )
    )

    area_b = (
        max(
            0.0,
            bx2 - bx1,
        )
        *
        max(
            0.0,
            by2 - by1,
        )
    )

    union = (
        area_a
        +
        area_b
        -
        intersection
    )

    if union <= 0:
        return 0.0

    return (
        intersection
        /
        union
    )


# ============================================================
# REMOVE DUPLICATE TARGET BOXES
# ============================================================

def filter_target_detections(
    detections,
    target_bone,
    duplicate_iou=0.70,
):
    """
    Return ALL spatially separate YOLO boxes.

    IMPORTANT:
        The fused bone decides WHAT the binary Swin-UNet must segment.
        YOLO class does NOT decide the segmentation class.
        YOLO boxes only provide WHERE segmentation is allowed to run.

    Therefore, if fusion says Humerus but YOLO detects a Femur box, the
    Humerus segmenter is still run inside that Femur box. If YOLO produces
    no boxes at all, segmentation is skipped.

    ``target_bone`` is retained in the signature for compatibility with the
    existing caller and downstream code. It is deliberately NOT used to
    filter detections by YOLO class.

    Only highly overlapping duplicate boxes are suppressed; spatially
    separate boxes are all kept for split-screen support.
    """

    if not detections:
        return []

    # The fused target is the segmentation identity, not a YOLO-class gate.
    _ = target_bone

    candidates = sorted(
        detections,
        key=lambda x: x["confidence"],
        reverse=True,
    )

    kept = []

    for candidate in candidates:
        duplicate = False

        for existing in kept:
            if (
                box_iou(
                    candidate["bbox"],
                    existing["bbox"],
                )
                >= duplicate_iou
            ):
                duplicate = True
                break

        if not duplicate:
            kept.append(candidate)

    kept.sort(
        key=lambda d: (
            d["bbox"][0],
            d["bbox"][1],
        )
    )

    return kept


# ============================================================
# SEGMENTER OUTPUT EXTRACTION
# ============================================================

def extract_segmentation_logits(
    output,
):
    """
    Supports tensor or dict-style segmenter outputs.
    """

    if torch.is_tensor(output):
        return output

    if isinstance(
        output,
        dict,
    ):

        for key in [
            "seg_logits",
            "segmentation",
            "logits",
            "out",
        ]:

            if key in output:
                return output[key]

    raise RuntimeError(
        "Could not find segmentation logits "
        "in segmenter output."
    )


# ============================================================
# SMALL COMPONENT REMOVAL
# ============================================================

def remove_small_components(
    binary,
    min_area=MIN_COMPONENT_AREA,
):
    binary = (
        binary > 0
    ).astype(
        np.uint8
    )

    if binary.sum() == 0:
        return binary

    (
        num_labels,
        labels,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    cleaned = np.zeros_like(
        binary
    )

    for component_id in range(
        1,
        num_labels,
    ):

        area = stats[
            component_id,
            cv2.CC_STAT_AREA,
        ]

        if area >= min_area:

            cleaned[
                labels == component_id
            ] = 1

    return cleaned


# ============================================================
# INTENSITY-GUIDED BOUNDARY RECOVERY
# ============================================================

def intensity_guided_growth(
    mask,
    image_gray,
):
    """
    Recover boundary pixels that are close in intensity to the existing
    segmented bone.

    IMPORTANT:
    This only grows directly around an already existing mask.
    It does not search the entire image.

    Therefore it is intended to recover missed bone-border pixels rather
    than create completely new detections.
    """

    binary = (
        mask > 0
    ).astype(
        np.uint8
    )

    if binary.sum() == 0:
        return binary

    image = image_gray.astype(
        np.float32
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        INTENSITY_GROW_KERNEL,
    )

    for _ in range(
        INTENSITY_GROW_ITERATIONS
    ):

        bone_pixels = image[
            binary > 0
        ]

        if bone_pixels.size < 10:
            break

        # Median is more stable than mean for ultrasound.
        reference_intensity = float(
            np.median(
                bone_pixels
            )
        )

        dilated = cv2.dilate(
            binary,
            kernel,
            iterations=1,
        )

        boundary = (
            (dilated > 0)
            &
            (binary == 0)
        )

        if not boundary.any():
            break

        intensity_difference = np.abs(
            image
            -
            reference_intensity
        )

        candidate = (
            boundary
            &
            (
                intensity_difference
                <= INTENSITY_THRESHOLD
            )
        )

        if not candidate.any():
            break

        binary[
            candidate
        ] = 1

    return binary.astype(
        np.uint8
    )


# ============================================================
# GAUSSIAN BOUNDARY SMOOTHING
# ============================================================

def gaussian_boundary_smoothing(
    binary,
):
    """
    Smooth jagged / stair-stepped mask boundaries.

    The binary mask is temporarily converted to a floating-point
    image, blurred, and thresholded again.

    This smooths the contour while preserving separate spatial
    components. There is deliberately NO largest-component selection.
    """

    binary = (
        binary > 0
    ).astype(
        np.uint8
    )

    if binary.sum() == 0:
        return binary

    blurred = cv2.GaussianBlur(
        binary.astype(
            np.float32
        ),
        SMOOTH_BLUR_KERNEL,
        SMOOTH_BLUR_SIGMA,
    )

    smoothed = (
        blurred
        >= SMOOTH_THRESHOLD
    ).astype(
        np.uint8
    )

    return smoothed


# ============================================================
# SMOOTH BINARY MASK
# ============================================================

def smooth_binary_mask(
    mask,
    min_area=MIN_COMPONENT_AREA,
    image_gray=None,
):
    """
    Strong smoothing for clean bone boundaries.

    Processing:

        1. Remove tiny isolated components
        2. Elliptical opening
        3. Elliptical closing
        4. Fill holes
        5. Remove tiny components
        6. Intensity-guided boundary recovery
        7. Gaussian boundary smoothing
        8. Final elliptical closing
        9. Final small-component removal

    IMPORTANT:
        - No largest-component selection.
        - Separate split-screen bones are preserved.
        - Small amoeba-like blobs are removed.
        - Similar-intensity pixels immediately around the boundary
          may be recovered.
    """

    binary = (
        mask > 0
    ).astype(
        np.uint8
    )

    if binary.sum() == 0:
        return np.zeros_like(
            binary,
            dtype=np.uint8,
        )

    # --------------------------------------------------------
    # 1. Remove tiny isolated components
    # --------------------------------------------------------

    binary = remove_small_components(
        binary,
        min_area=min_area,
    )

    if binary.sum() == 0:
        return np.zeros_like(
            binary,
            dtype=np.uint8,
        )

    # --------------------------------------------------------
    # 2. Elliptical opening
    # --------------------------------------------------------

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        OPEN_KERNEL,
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1,
    )

    if binary.sum() == 0:
        return np.zeros_like(
            binary,
            dtype=np.uint8,
        )

    # --------------------------------------------------------
    # 3. Elliptical closing
    # --------------------------------------------------------

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        CLOSE_KERNEL,
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1,
    )

    # --------------------------------------------------------
    # 4. Fill holes
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    filled = np.zeros_like(
        binary
    )

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area >= min_area:

            cv2.drawContours(
                filled,
                [contour],
                -1,
                1,
                thickness=cv2.FILLED,
            )

    binary = filled

    # --------------------------------------------------------
    # 5. Remove tiny components again
    # --------------------------------------------------------

    binary = remove_small_components(
        binary,
        min_area=min_area,
    )

    if binary.sum() == 0:
        return np.zeros_like(
            binary,
            dtype=np.uint8,
        )

    # --------------------------------------------------------
    # 6. Intensity-guided boundary recovery
    # --------------------------------------------------------

    if image_gray is not None:

        binary = intensity_guided_growth(
            binary,
            image_gray,
        )

        binary = remove_small_components(
            binary,
            min_area=min_area,
        )

    if binary.sum() == 0:
        return np.zeros_like(
            binary,
            dtype=np.uint8,
        )

    # --------------------------------------------------------
    # 7. Gaussian boundary smoothing
    # --------------------------------------------------------

    binary = gaussian_boundary_smoothing(
        binary
    )

    if binary.sum() == 0:
        return np.zeros_like(
            binary,
            dtype=np.uint8,
        )

    # --------------------------------------------------------
    # 8. Final light elliptical closing
    # --------------------------------------------------------

    final_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        FINAL_CLOSE_KERNEL,
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        final_kernel,
        iterations=1,
    )

    # --------------------------------------------------------
    # 9. Final small-component filtering
    # --------------------------------------------------------

    binary = remove_small_components(
        binary,
        min_area=min_area,
    )

    return binary.astype(
        np.uint8
    )


# ============================================================
# SEGMENT ONE BOX
# ============================================================

@torch.no_grad()
def segment_box(
    image_gray,
    bbox,
    segmenter,
    device,
):
    """
    Crop one YOLO ROI and run binary Swin-UNet.

    DS preprocessing:
        ROI / 255
        resize
        (x - 0.5) / 0.5
    """

    h, w = image_gray.shape[:2]

    x1, y1, x2, y2 = bbox

    x1 = max(
        0,
        min(
            w - 1,
            int(round(x1)),
        ),
    )

    y1 = max(
        0,
        min(
            h - 1,
            int(round(y1)),
        ),
    )

    x2 = max(
        x1 + 1,
        min(
            w,
            int(round(x2)),
        ),
    )

    y2 = max(
        y1 + 1,
        min(
            h,
            int(round(y2)),
        ),
    )

    roi = image_gray[
        y1:y2,
        x1:x2,
    ]

    if roi.size == 0:
        return None

    original_h, original_w = (
        roi.shape
    )

    segmenter_size = getattr(
        Config,
        "SEGMENTER_IMG_SIZE",
        512,
    )

    # --------------------------------------------------------
    # DS preprocessing
    # --------------------------------------------------------

    roi_float = (
        roi.astype(
            np.float32
        )
        / 255.0
    )

    roi_resized = cv2.resize(
        roi_float,
        (
            segmenter_size,
            segmenter_size,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    roi_norm = (
        roi_resized
        -
        0.5
    ) / 0.5

    tensor = torch.from_numpy(
        roi_norm
    ).unsqueeze(
        0
    ).unsqueeze(
        0
    ).float()

    tensor = tensor.to(
        device
    )

    # --------------------------------------------------------
    # Segmentation
    # --------------------------------------------------------

    output = segmenter(
        tensor
    )

    logits = extract_segmentation_logits(
        output
    )

    if logits.ndim != 4:
        raise RuntimeError(
            "Unexpected segmentation logits shape: "
            f"{tuple(logits.shape)}"
        )

    if logits.shape[1] == 1:

        probability = torch.sigmoid(
            logits
        )

        pred = (
            probability[0, 0]
            >= 0.5
        ).float()

    else:

        pred = torch.argmax(
            logits,
            dim=1,
        )[0]

        pred = (
            pred > 0
        ).float()

    pred = (
        pred.detach()
        .cpu()
        .numpy()
        .astype(
            np.uint8
        )
    )

    # --------------------------------------------------------
    # Restore original ROI dimensions
    # --------------------------------------------------------

    pred = cv2.resize(
        pred,
        (
            original_w,
            original_h,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    # --------------------------------------------------------
    # Smooth + intensity recovery
    # --------------------------------------------------------

    pred = smooth_binary_mask(
        pred,
        min_area=MIN_COMPONENT_AREA,
        image_gray=roi,
    )

    # --------------------------------------------------------
    # Put ROI mask into full image
    # --------------------------------------------------------

    full_mask = np.zeros(
        (
            h,
            w,
        ),
        dtype=np.uint8,
    )

    full_mask[
        y1:y2,
        x1:x2
    ] = pred

    return {
        "mask": full_mask,
        "bbox": [
            x1,
            y1,
            x2,
            y2,
        ],
        "area": int(
            pred.sum()
        ),
    }


# ============================================================
# SEGMENT ALL TARGET BOXES
# ============================================================

def segment_target_boxes(
    image_gray,
    target_boxes,
    target_bone,
    segmenter,
    device,
):
    """
    Segment EVERY YOLO ROI using the FUSED bone identity.

    YOLO class labels are intentionally ignored here. The binary segmenter
    is conditioned by ``target_bone`` from classifier+YOLO fusion.
    """

    h, w = image_gray.shape[:2]

    final_mask = np.zeros(
        (
            h,
            w,
        ),
        dtype=np.int64,
    )

    successful = []
    failed = []

    for index, detection in enumerate(
        target_boxes,
        start=1,
    ):

        result = segment_box(
            image_gray=image_gray,
            bbox=detection["bbox"],
            segmenter=segmenter,
            device=device,
        )

        if result is None:
            failed.append(index)
            continue

        box_mask = result["mask"]

        if box_mask.sum() == 0:
            failed.append(index)
            continue

        final_mask[
            box_mask > 0
        ] = target_bone

        successful.append(
            {
                "index": index,
                "bbox": result["bbox"],
                "detector_confidence": detection[
                    "confidence"
                ],
                "area": result["area"],
            }
        )

    # --------------------------------------------------------
    # Final whole-image smoothing.
    # --------------------------------------------------------

    if final_mask.any():

        binary = (
            final_mask == target_bone
        ).astype(
            np.uint8
        )

        binary = smooth_binary_mask(
            binary,
            min_area=MIN_COMPONENT_AREA,
            image_gray=image_gray,
        )

        final_mask[:] = 0

        final_mask[
            binary > 0
        ] = target_bone

    return (
        final_mask,
        successful,
        failed,
    )


# ============================================================
# FUSION
# ============================================================

def fuse_bone_confidences(
    classifier_result,
    detector_bone_probs,
):
    classifier_probs = (
        classifier_result[
            "bone_probs"
        ]
    )

    fused = {}

    for bone_id in BONE_NAMES:

        cls_conf = classifier_probs.get(
            bone_id,
            0.0,
        )

        yolo_conf = detector_bone_probs.get(
            bone_id,
            0.0,
        )

        fused[bone_id] = (
            CLASSIFIER_WEIGHT
            *
            cls_conf
            +
            DETECTOR_WEIGHT
            *
            yolo_conf
        )

    target_bone = max(
        fused,
        key=fused.get,
    )

    return (
        fused,
        target_bone,
    )


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def get_component_geometry(
    component_mask,
):
    """
    Calculate geometry of a connected segmentation component.

    Angle:
        principal axis relative to horizontal image axis.

    Therefore:
        0°   = horizontal
        20°  = acceptable FE/HU limit
        40°  = acceptable RU/TF limit
    """

    ys, xs = np.where(
        component_mask > 0
    )

    area = len(xs)

    if area < 2:
        return None

    points = np.column_stack(
        [
            xs.astype(
                np.float32
            ),
            ys.astype(
                np.float32
            ),
        ]
    )

    center = points.mean(
        axis=0
    )

    centered = (
        points
        -
        center
    )

    covariance = np.cov(
        centered.T
    )

    if covariance.shape != (
        2,
        2,
    ):
        return None

    (
        eigenvalues,
        eigenvectors,
    ) = np.linalg.eigh(
        covariance
    )

    order = np.argsort(
        eigenvalues
    )[::-1]

    eigenvalues = (
        eigenvalues[
            order
        ]
    )

    eigenvectors = (
        eigenvectors[
            :,
            order
        ]
    )

    major_vector = (
        eigenvectors[
            :,
            0
        ]
    )

    major_variance = max(
        float(
            eigenvalues[0]
        ),
        1e-6,
    )

    minor_variance = max(
        float(
            eigenvalues[1]
        ),
        1e-6,
    )

    elongation = (
        major_variance
        /
        minor_variance
    )

    angle = np.degrees(
        np.arctan2(
            major_vector[1],
            major_vector[0],
        )
    )

    # PCA direction is equivalent after a 180-degree flip.
    while angle > 90:
        angle -= 180

    while angle < -90:
        angle += 180

    angle = abs(
        float(angle)
    )

    projections = (
        centered
        @
        major_vector
    )

    length = float(
        projections.max()
        -
        projections.min()
    )

    x, y, bw, bh = cv2.boundingRect(
        component_mask.astype(
            np.uint8
        )
    )

    return {
        "area": int(area),
        "length": length,
        "elongation": float(
            elongation
        ),
        "angle": angle,
        "center_x": float(
            center[0]
        ),
        "center_y": float(
            center[1]
        ),
        "bbox": [
            int(x),
            int(y),
            int(bw),
            int(bh),
        ],
    }


def get_bone_components(
    segmentation_mask,
    target_bone,
):
    """
    Extract connected components belonging to the selected bone.
    """

    binary = (
        segmentation_mask == target_bone
    ).astype(
        np.uint8
    )

    if binary.sum() == 0:
        return []

    (
        num_labels,
        labels,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    components = []

    for component_id in range(
        1,
        num_labels,
    ):

        area = stats[
            component_id,
            cv2.CC_STAT_AREA,
        ]

        if area < MIN_BONE_AREA:
            continue

        component_mask = (
            labels == component_id
        ).astype(
            np.uint8
        )

        geometry = get_component_geometry(
            component_mask
        )

        if geometry is None:
            continue

        geometry[
            "component_id"
        ] = component_id

        components.append(
            geometry
        )

    return components


def is_valid_bone_component(
    component,
    max_angle,
):
    """
    Decide whether a connected component looks sufficiently like a
    segmented long bone.
    """

    if (
        component["area"]
        <
        MIN_BONE_AREA
    ):
        return False

    if (
        component["length"]
        <
        MIN_BONE_LENGTH
    ):
        return False

    if (
        component["elongation"]
        <
        MIN_BONE_ELONGATION
    ):
        return False

    if (
        component["angle"]
        >
        max_angle
    ):
        return False

    return True


def components_can_form_pair(
    first,
    second,
    image_width,
    image_height,
):
    """
    Check whether two valid components plausibly represent the paired
    bones in a normal RU/TF image.

    This avoids counting two tiny/unrelated fragments as the two bones.
    """

    dx = (
        first["center_x"]
        -
        second["center_x"]
    )

    dy = (
        first["center_y"]
        -
        second["center_y"]
    )

    distance = np.sqrt(
        dx * dx
        +
        dy * dy
    )

    diagonal = np.sqrt(
        image_width ** 2
        +
        image_height ** 2
    )

    distance_ratio = (
        distance
        /
        max(
            diagonal,
            1.0,
        )
    )

    if (
        distance_ratio
        >
        MAX_PAIR_CENTER_DISTANCE_RATIO
    ):
        return False

    angle_difference = abs(
        first["angle"]
        -
        second["angle"]
    )

    angle_difference = min(
        angle_difference,
        180.0 - angle_difference,
    )

    if (
        angle_difference
        >
        MAX_PAIR_ANGLE_DIFFERENCE
    ):
        return False

    return True


def has_split_screen_layout(
    components,
    image_width,
    target_boxes,
):
    """
    Determine whether the available segmentation/detection geometry gives
    evidence of a split-screen image.

    Evidence used:
        1. multiple target YOLO boxes with a large horizontal separation
        OR
        2. multiple segmented components with a large horizontal separation

    We do NOT force a split-screen decision when the image contains only
    one component/box because there is not enough geometric evidence.
    """

    centers = []

    # Use successful target boxes.
    for detection in target_boxes:

        x1, _, x2, _ = (
            detection["bbox"]
        )

        centers.append(
            (x1 + x2)
            /
            2.0
        )

    # Also use segmentation component centers.
    for component in components:

        centers.append(
            component["center_x"]
        )

    if len(centers) < 2:
        return False

    centers = sorted(
        centers
    )

    largest_gap = 0.0

    for i in range(
        len(centers) - 1
    ):

        gap = (
            centers[i + 1]
            -
            centers[i]
        )

        largest_gap = max(
            largest_gap,
            gap,
        )

    gap_ratio = (
        largest_gap
        /
        max(
            float(image_width),
            1.0,
        )
    )

    return (
        gap_ratio
        >= SPLIT_SCREEN_X_GAP_RATIO
    )


# ============================================================
# STANDARD / NON-STANDARD / UNKNOWN CLASSIFICATION
# ============================================================

def classify_standard_nonstandard(
    image_gray,
    segmentation_mask,
    target_bone,
    target_boxes,
):
    """
    Final Standard / Non-Standard / UNKNOWN decision.

    UNKNOWN
    -------
    No segmentation mask/component exists.

    FE / HU
    -------
    STANDARD if at least one valid bone is segmented with
    angle <= 20°.

    RU / TF
    -------
    Normal view:
        STANDARD only if two compatible bone components are segmented,
        both with angle <= 40°.

    Split-screen:
        STANDARD if at least one valid bone component is segmented
        with angle <= 40°.

    Returns:
        standard_status
        standard_reason
        geometry_details
    """

    image_height, image_width = (
        image_gray.shape[:2]
    )

    bone_name = BONE_NAMES[
        target_bone
    ]

    # --------------------------------------------------------
    # No mask at all
    # --------------------------------------------------------

    if (
        segmentation_mask is None
        or
        np.count_nonzero(
            segmentation_mask
        ) == 0
    ):

        return (
            "UNKNOWN",
            (
                f"UNKNOWN: no {bone_name} "
                f"segmentation mask was produced."
            ),
            {
                "components": [],
                "valid_components": [],
                "split_screen": False,
            },
        )

    components = get_bone_components(
        segmentation_mask,
        target_bone,
    )

    # --------------------------------------------------------
    # Mask exists but no valid components
    # --------------------------------------------------------

    if not components:

        return (
            "UNKNOWN",
            (
                f"UNKNOWN: a segmentation result was produced, "
                f"but no measurable {bone_name} component "
                f"remained after mask processing."
            ),
            {
                "components": [],
                "valid_components": [],
                "split_screen": False,
            },
        )

    # --------------------------------------------------------
    # FE / HU
    # --------------------------------------------------------

    if target_bone in (
        1,
        2,
    ):

        valid_components = [
            component
            for component in components
            if is_valid_bone_component(
                component,
                FE_HU_MAX_ANGLE,
            )
        ]

        if valid_components:

            best = max(
                valid_components,
                key=lambda x: x["area"],
            )

            if len(
                valid_components
            ) == 1:

                reason = (
                    f"STANDARD: one valid "
                    f"{bone_name} was segmented "
                    f"with angle "
                    f"{best['angle']:.1f}°, "
                    f"within the "
                    f"{FE_HU_MAX_ANGLE:.0f}° "
                    f"limit."
                )

            else:

                reason = (
                    f"STANDARD: "
                    f"{len(valid_components)} "
                    f"valid {bone_name} "
                    f"structures were segmented; "
                    f"at least one valid bone "
                    f"has angle "
                    f"{best['angle']:.1f}°, "
                    f"within the "
                    f"{FE_HU_MAX_ANGLE:.0f}° "
                    f"limit."
                )

            return (
                "STANDARD",
                reason,
                {
                    "components": components,
                    "valid_components": valid_components,
                    "split_screen": has_split_screen_layout(
                        valid_components,
                        image_width,
                        target_boxes,
                    ),
                },
            )

        # ----------------------------------------------------
        # Mask exists, but geometry failed.
        # ----------------------------------------------------

        largest = max(
            components,
            key=lambda x: x["area"],
        )

        if (
            largest["angle"]
            >
            FE_HU_MAX_ANGLE
        ):

            reason = (
                f"NON-STANDARD: segmented "
                f"{bone_name} angle is "
                f"{largest['angle']:.1f}°, "
                f"which exceeds the "
                f"{FE_HU_MAX_ANGLE:.0f}° "
                f"limit."
            )

        else:

            reason = (
                f"NON-STANDARD: segmented "
                f"{bone_name} does not meet "
                f"the required long-bone "
                f"size/shape criteria."
            )

        return (
            "NON-STANDARD",
            reason,
            {
                "components": components,
                "valid_components": [],
                "split_screen": False,
            },
        )

    # --------------------------------------------------------
    # RU / TF
    # --------------------------------------------------------

    valid_components = [
        component
        for component in components
        if is_valid_bone_component(
            component,
            RU_TF_MAX_ANGLE,
        )
    ]

    if not valid_components:

        reason = (
            f"NON-STANDARD: no valid "
            f"{bone_name} structure was "
            f"segmented within the "
            f"{RU_TF_MAX_ANGLE:.0f}° "
            f"angle limit."
        )

        return (
            "NON-STANDARD",
            reason,
            {
                "components": components,
                "valid_components": [],
                "split_screen": False,
            },
        )

    # --------------------------------------------------------
    # Split-screen
    # --------------------------------------------------------

    split_screen = has_split_screen_layout(
        valid_components,
        image_width,
        target_boxes,
    )

    if split_screen:

        best = max(
            valid_components,
            key=lambda x: x["area"],
        )

        reason = (
            f"STANDARD: split-screen layout "
            f"detected; at least one clearly "
            f"segmented {bone_name} structure "
            f"is valid with angle "
            f"{best['angle']:.1f}°, within the "
            f"{RU_TF_MAX_ANGLE:.0f}° limit."
        )

        return (
            "STANDARD",
            reason,
            {
                "components": components,
                "valid_components": valid_components,
                "split_screen": True,
            },
        )

    # --------------------------------------------------------
    # Normal RU / TF:
    # Both bones required.
    # --------------------------------------------------------

    compatible_pair = None

    for i in range(
        len(valid_components)
    ):

        for j in range(
            i + 1,
            len(valid_components),
        ):

            first = valid_components[i]
            second = valid_components[j]

            if components_can_form_pair(
                first,
                second,
                image_width,
                image_height,
            ):

                compatible_pair = (
                    first,
                    second,
                )

                break

        if compatible_pair is not None:
            break

    if compatible_pair is not None:

        first, second = (
            compatible_pair
        )

        reason = (
            f"STANDARD: two distinct "
            f"{bone_name} structures were "
            f"segmented with angles "
            f"{first['angle']:.1f}° and "
            f"{second['angle']:.1f}°, "
            f"both within the "
            f"{RU_TF_MAX_ANGLE:.0f}° limit."
        )

        return (
            "STANDARD",
            reason,
            {
                "components": components,
                "valid_components": valid_components,
                "split_screen": False,
                "pair": [
                    first,
                    second,
                ],
            },
        )

    # --------------------------------------------------------
    # Only one valid bone OR components don't form a pair.
    # --------------------------------------------------------

    if len(
        valid_components
    ) == 1:

        reason = (
            f"NON-STANDARD: only one "
            f"valid {bone_name} structure "
            f"was segmented; both paired "
            f"bones are required in a "
            f"normal view."
        )

    else:

        reason = (
            f"NON-STANDARD: multiple "
            f"{bone_name} structures were "
            f"segmented, but they do not "
            f"form a valid paired-bone "
            f"configuration."
        )

    return (
        "NON-STANDARD",
        reason,
        {
            "components": components,
            "valid_components": valid_components,
            "split_screen": False,
        },
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

@torch.no_grad()
def run_pipeline(
    image_gray,
    classifier,
    idx2plane,
    detector,
    segmenter,
    device,
):

    # --------------------------------------------------------
    # 1. CLASSIFIER
    # --------------------------------------------------------

    classifier_result = run_classifier(
        image_gray,
        classifier,
        idx2plane,
        device,
    )

    # --------------------------------------------------------
    # 2. YOLO
    # --------------------------------------------------------

    detections = run_detector(
        image_gray,
        detector,
        device,
    )

    detector_bone_probs = (
        get_detector_bone_confidences(
            detections
        )
    )

    yolo_predicted_bone = None

    if detections:

        yolo_predicted_bone = max(
            detector_bone_probs,
            key=detector_bone_probs.get,
        )

    yolo_confidence = (
        detector_bone_probs[
            yolo_predicted_bone
        ]
        if yolo_predicted_bone is not None
        else 0.0
    )

    # --------------------------------------------------------
    # 3. FUSION
    # --------------------------------------------------------

    (
        fused_probs,
        target_bone,
    ) = fuse_bone_confidences(
        classifier_result,
        detector_bone_probs,
    )

    fused_confidence = fused_probs[
        target_bone
    ]

    # --------------------------------------------------------
    # 4. TARGET BOXES
    # --------------------------------------------------------

    target_boxes = (
        filter_target_detections(
            detections,
            target_bone,
            duplicate_iou=(
                DUPLICATE_IOU_THRESHOLD
            ),
        )
    )

    # --------------------------------------------------------
    # 5. NO YOLO
    # --------------------------------------------------------

    if not detections:

        return {
            "segmentation_mask": None,
            "classifier_result": classifier_result,
            "detections": detections,
            "detector_bone_probs": detector_bone_probs,
            "yolo_predicted_bone": yolo_predicted_bone,
            "yolo_confidence": yolo_confidence,
            "fused_probs": fused_probs,
            "target_bone": target_bone,
            "fused_confidence": fused_confidence,
            "target_boxes": [],
            "successful_boxes": [],
            "failed_boxes": [],
            "prediction_found": False,
            "status": "NO PREDICTION",
            "reason": (
                "YOLO produced no detections"
            ),
            "standard_status": "UNKNOWN",
            "standard_reason": (
                f"UNKNOWN: no "
                f"{BONE_NAMES[target_bone]} "
                f"region was detected for "
                f"segmentation."
            ),
            "standard_details": {},
        }

    # --------------------------------------------------------
    # 6. FUSED TARGET HAS NO YOLO BOX
    # --------------------------------------------------------

    if not target_boxes:

        return {
            "segmentation_mask": None,
            "classifier_result": classifier_result,
            "detections": detections,
            "detector_bone_probs": detector_bone_probs,
            "yolo_predicted_bone": yolo_predicted_bone,
            "yolo_confidence": yolo_confidence,
            "fused_probs": fused_probs,
            "target_bone": target_bone,
            "fused_confidence": fused_confidence,
            "target_boxes": [],
            "successful_boxes": [],
            "failed_boxes": [],
            "prediction_found": False,
            "status": "NO PREDICTION",
            "reason": (
                f"Fusion selected "
                f"{BONE_NAMES[target_bone]}, "
                f"but YOLO produced no boxes"
            ),
            "standard_status": "UNKNOWN",
            "standard_reason": (
                f"UNKNOWN: no YOLO region was "
                f"available for {BONE_NAMES[target_bone]} "
                f"segmentation."
            ),
            "standard_details": {},
        }

    # --------------------------------------------------------
    # 7. SEGMENT ALL TARGET BOXES
    # --------------------------------------------------------

    (
        segmentation_mask,
        successful_boxes,
        failed_boxes,
    ) = segment_target_boxes(
        image_gray=image_gray,
        target_boxes=target_boxes,
        target_bone=target_bone,
        segmenter=segmenter,
        device=device,
    )

    # --------------------------------------------------------
    # 8. NO VALID SEGMENTATION
    # --------------------------------------------------------

    if (
        not successful_boxes
        or
        segmentation_mask is None
        or
        np.count_nonzero(
            segmentation_mask
        ) == 0
    ):

        return {
            "segmentation_mask": None,
            "classifier_result": classifier_result,
            "detections": detections,
            "detector_bone_probs": detector_bone_probs,
            "yolo_predicted_bone": yolo_predicted_bone,
            "yolo_confidence": yolo_confidence,
            "fused_probs": fused_probs,
            "target_bone": target_bone,
            "fused_confidence": fused_confidence,
            "target_boxes": target_boxes,
            "successful_boxes": [],
            "failed_boxes": failed_boxes,
            "prediction_found": False,
            "status": "NO PREDICTION",
            "reason": (
                f"{len(target_boxes)} target box(es) "
                f"found, but segmentation produced "
                f"empty masks"
            ),
            "standard_status": "UNKNOWN",
            "standard_reason": (
                f"UNKNOWN: no valid "
                f"{BONE_NAMES[target_bone]} "
                f"segmentation mask was produced."
            ),
            "standard_details": {},
        }

    # --------------------------------------------------------
    # 9. PIPELINE STATUS
    # --------------------------------------------------------

    if failed_boxes:

        status = "PARTIAL"

        reason = (
            f"{len(successful_boxes)} of "
            f"{len(target_boxes)} target boxes "
            f"were successfully segmented"
        )

    else:

        status = "SUCCESS"

        reason = (
            f"{len(successful_boxes)} "
            f"{BONE_NAMES[target_bone]} "
            f"box(es) successfully segmented"
        )

    # --------------------------------------------------------
    # 10. STANDARD / NON-STANDARD / UNKNOWN
    # --------------------------------------------------------

    (
        standard_status,
        standard_reason,
        standard_details,
    ) = classify_standard_nonstandard(
        image_gray=image_gray,
        segmentation_mask=segmentation_mask,
        target_bone=target_bone,
        target_boxes=target_boxes,
    )

    return {
        "segmentation_mask": segmentation_mask,
        "classifier_result": classifier_result,
        "detections": detections,
        "detector_bone_probs": detector_bone_probs,
        "yolo_predicted_bone": yolo_predicted_bone,
        "yolo_confidence": yolo_confidence,
        "fused_probs": fused_probs,
        "target_bone": target_bone,
        "fused_confidence": fused_confidence,
        "target_boxes": target_boxes,
        "successful_boxes": successful_boxes,
        "failed_boxes": failed_boxes,
        "prediction_found": True,
        "status": status,
        "reason": reason,
        "standard_status": standard_status,
        "standard_reason": standard_reason,
        "standard_details": standard_details,
    }


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def draw_text_with_background(
    image,
    text,
    position,
    font_scale=0.65,
    thickness=2,
):
    font = cv2.FONT_HERSHEY_SIMPLEX

    x, y = position

    (
        tw,
        th,
    ), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    cv2.rectangle(
        image,
        (
            x - 6,
            y - th - 8,
        ),
        (
            x + tw + 6,
            y + baseline + 5,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        image,
        text,
        (
            x,
            y,
        ),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def create_prediction_panel(
    image_gray,
    mask,
    result,
):
    """
    Create prediction visualization.

    IMPORTANT:
    YOLO bounding boxes are NOT drawn.
    """

    panel = cv2.cvtColor(
        image_gray,
        cv2.COLOR_GRAY2BGR,
    )

    # --------------------------------------------------------
    # Segmentation overlay
    # --------------------------------------------------------

    if (
        mask is not None
        and
        np.count_nonzero(mask) > 0
    ):

        target_bone = result[
            "target_bone"
        ]

        if target_bone == 1:
            color = (
                255,
                0,
                0,
            )

        elif target_bone == 2:
            color = (
                0,
                255,
                0,
            )

        elif target_bone == 3:
            color = (
                0,
                165,
                255,
            )

        else:
            color = (
                255,
                0,
                255,
            )

        binary = (
            mask == target_bone
        ).astype(
            np.uint8
        )

        overlay = panel.copy()

        overlay[
            binary > 0
        ] = color

        panel = cv2.addWeighted(
            panel,
            0.55,
            overlay,
            0.45,
            0,
        )

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            panel,
            contours,
            -1,
            color,
            2,
            cv2.LINE_AA,
        )

    # --------------------------------------------------------
    # NO YOLO BOX DRAWING HERE
    # --------------------------------------------------------

    return panel


def create_visualization(
    image_gray,
    mask,
    result,
):
    """
    Prediction-only visualization.

    Layout:
        TOP    = Standard / Non-Standard / UNKNOWN + target bone
        BODY   = prediction image with segmentation overlay

    IMPORTANT:
        - No original-image panel.
        - No classifier / YOLO / fused confidence text.
        - No box-count text.
        - No segmented-count text.
        - No classification reason text.
        - Existing prediction/segmentation visualization is preserved.
    """

    right = create_prediction_panel(
        image_gray,
        mask,
        result,
    )

    standard_status = result.get(
        "standard_status",
        "UNKNOWN",
    )

    target_bone = result.get(
        "target_bone",
    )

    target_name = BONE_NAMES.get(
        target_bone,
        "Unknown",
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    header_h = 80

    prediction_canvas = cv2.copyMakeBorder(
        right,
        header_h,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(
            30,
            30,
            30,
        ),
    )

    # Standard / Non-Standard / UNKNOWN
    cv2.putText(
        prediction_canvas,
        standard_status,
        (
            20,
            32,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (
            255,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    # Target bone
    cv2.putText(
        prediction_canvas,
        f"Target: {target_name}",
        (
            20,
            62,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (
            220,
            220,
            220,
        ),
        1,
        cv2.LINE_AA,
    )

    return prediction_canvas


# ============================================================
# IMAGE COLLECTION
# ============================================================

def collect_images(
    images_dir,
):
    """
    Recursively collect images.

    This supports:

        unseen_data/
            FE/
            HU/
            RU/
            TF/
    """

    root = Path(
        images_dir
    )

    paths = [
        p
        for p in root.rglob("*")
        if (
            p.is_file()
            and
            p.suffix.lower()
            in SUPPORTED_EXTS
        )
    ]

    return sorted(
        paths,
        key=lambda p: str(p),
    )


# ============================================================
# POLYGON / HEAD-STYLE JSON HELPERS
# ============================================================

def mask_to_polygons(
    segmentation_mask,
    target_bone,
):
    """Convert the final fused-bone mask to head-style polygon objects."""

    if (
        segmentation_mask is None
        or target_bone is None
    ):
        return []

    binary = (
        segmentation_mask == int(target_bone)
    ).astype(np.uint8)

    if int(binary.sum()) == 0:
        return []

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    polygon_items = []

    for contour in contours:
        area = float(cv2.contourArea(contour))

        if area < MIN_COMPONENT_AREA:
            continue

        perimeter = float(
            cv2.arcLength(contour, True)
        )

        epsilon = (
            0.002 * perimeter
            if perimeter > 0
            else 0.0
        )

        approx = cv2.approxPolyDP(
            contour,
            epsilon,
            True,
        )

        points = [
            {
                "x": float(point[0][0]),
                "y": float(point[0][1]),
            }
            for point in approx
        ]

        if len(points) < 3:
            continue

        polygon_items.append({
            "points": points,
            "structure": BONE_NAMES.get(
                int(target_bone),
                str(target_bone),
            ),
            "type": "polygon",
            "_area": area,
        })

    polygon_items.sort(
        key=lambda item: item["_area"],
        reverse=True,
    )

    # Keep the head-inference JSON shape exactly: points/structure/type.
    for item in polygon_items:
        item.pop("_area", None)

    return polygon_items


def _top_plane_candidates(
    classifier_result,
    idx2plane,
    limit=3,
):
    """Return the top classifier plane names for head-style JSON."""

    # The current classifier result already contains the full plane index
    # only for the top plane, so derive candidate bone names from bone probs
    # while preserving a useful classifier-plane field.
    bone_probs = classifier_result.get(
        "bone_probs",
        {},
    )

    ordered = sorted(
        bone_probs.items(),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    names = [
        BONE_NAMES.get(
            int(bone_id),
            str(bone_id),
        )
        for bone_id, _ in ordered[:limit]
    ]

    top_plane_name = classifier_result.get(
        "top_plane_name",
        "unknown",
    )

    if not names and top_plane_name:
        names = [str(top_plane_name)]

    return ", ".join(names)


# ============================================================
# SUMMARY
# ============================================================

# Same visibility thresholds used by the HEAD JSON format.
STRUCTURE_THRESHOLDS = {
    "_default": {
        "contrast": 12.0,
        "blur": 15.0,
        "conf": 0.50,
    },
}


def _get_thresh(name, metric):
    """Return the HEAD-style threshold for a limb structure."""
    cfg = STRUCTURE_THRESHOLDS.get(
        str(name),
        STRUCTURE_THRESHOLDS["_default"],
    )
    return float(
        cfg.get(
            metric,
            STRUCTURE_THRESHOLDS["_default"][metric],
        )
    )


def _qual_contrast(value, name):
    return (
        "good"
        if float(value) >= _get_thresh(name, "contrast")
        else "low"
    )


def _qual_blur(value, name):
    return (
        "sharp"
        if float(value) >= _get_thresh(name, "blur")
        else "blurry"
    )


def _qual_conf(value, name):
    return (
        "high"
        if float(value) >= _get_thresh(name, "conf")
        else "low"
    )


def title_case_structure(name):
    """
    Match the HEAD JSON naming style.

    Limb names such as radius/ulna and tibia/fibula are written in
    title case while preserving the slash-separated form.
    """
    if name is None:
        return "Unknown"

    text = str(name).strip()
    if not text:
        return "Unknown"

    return "/".join(
        part.strip().title()
        for part in text.split("/")
    )


def _calculate_limb_visibility(
    img_path,
    segmentation_mask,
    confidence,
    structure_name,
):
    """
    Calculate the HEAD-style local visibility values for a limb mask.

    This is JSON metadata only. It does not modify the segmentation mask
    or any part of the classifier -> YOLO -> fusion -> segmenter pipeline.
    """
    try:
        gray_orig = load_grayscale_image(
            Path(img_path)
        )

        if gray_orig is None:
            return {
                "local_contrast": 0.0,
                "local_blur": 0.0,
            }

        mask_orig = np.asarray(
            segmentation_mask
        )

        if mask_orig.ndim > 2:
            mask_orig = np.squeeze(mask_orig)

        if mask_orig.shape != gray_orig.shape:
            mask_orig = cv2.resize(
                mask_orig.astype(np.uint8),
                (
                    gray_orig.shape[1],
                    gray_orig.shape[0],
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        mask_orig = (
            mask_orig > 0
        ).astype(np.uint8)

        if np.count_nonzero(mask_orig) < 30:
            return {
                "local_contrast": 0.0,
                "local_blur": 0.0,
            }

        dilation_px = 21
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation_px, dilation_px),
        )

        dilated = cv2.dilate(
            mask_orig,
            kernel,
        )

        bg_ring = (
            (
                dilated.astype(np.int32)
                - mask_orig.astype(np.int32)
            ) > 0
        ).astype(np.uint8)

        struct_px = gray_orig[
            mask_orig > 0
        ].astype(np.float32)

        bg_px = (
            gray_orig[
                bg_ring > 0
            ].astype(np.float32)
            if bg_ring.sum() > 0
            else struct_px
        )

        struct_mean = float(
            struct_px.mean()
        )
        bg_mean = float(
            bg_px.mean()
        )

        local_contrast = abs(
            struct_mean - bg_mean
        )

        x, y, w, h = cv2.boundingRect(
            mask_orig
        )

        pad = 10

        y1 = max(
            0,
            y - pad,
        )
        y2 = min(
            gray_orig.shape[0],
            y + h + pad,
        )
        x1 = max(
            0,
            x - pad,
        )
        x2 = min(
            gray_orig.shape[1],
            x + w + pad,
        )

        crop = gray_orig[
            y1:y2,
            x1:x2,
        ]

        local_blur = float(
            cv2.Laplacian(
                crop,
                cv2.CV_64F,
            ).var()
        )

        return {
            "local_contrast": round(
                local_contrast,
                2,
            ),
            "local_blur": round(
                local_blur,
                2,
            ),
        }

    except Exception as e:
        print(
            f"  [WARN] Could not calculate limb "
            f"visibility metrics for "
            f"{Path(img_path).name}: {e}"
        )

        return {
            "local_contrast": 0.0,
            "local_blur": 0.0,
        }



def _build_json_entry(
    img_path: Path,
    result: dict,
    visualization_file: Path,
) -> dict:
    """
    Build the per-image JSON entry in the same style as HEAD inference.

    Existing limbs caller is kept unchanged:
        _build_json_entry(img_path, result, visualization_file)

    JSON contains only:
      - image_path
      - summary
      - structures
    """

    structures = []

    # ------------------------------------------------------------
    # Build structures from the existing limbs inference result.
    # ------------------------------------------------------------
    target_bone = result.get("target_bone")
    fused_conf = result.get("fused_confidence", 0.0)

    if result.get("prediction_found") and target_bone:
        structures.append({
            "structure_name": str(target_bone),
            "segmentation_confidence": {
                "value": round(float(fused_conf), 4),
                "qualitative": _qual_conf(
                    float(fused_conf),
                    str(target_bone),
                ),
                "threshold": _get_thresh(
                    str(target_bone),
                    "conf",
                ),
            },
            "local_contrast": {
                "value": 0.0,
                "qualitative": _qual_contrast(
                    0.0,
                    str(target_bone),
                ),
                "threshold": _get_thresh(
                    str(target_bone),
                    "contrast",
                ),
            },
            "local_blur": {
                "value": 0.0,
                "qualitative": _qual_blur(
                    0.0,
                    str(target_bone),
                ),
                "threshold": _get_thresh(
                    str(target_bone),
                    "blur",
                ),
            },
        })

    # ------------------------------------------------------------
    # HEAD-style summary.
    # ------------------------------------------------------------
    n_predicted = len(structures)

    return {
        "image_path": str(img_path),
        "summary": {
            "n_structures_predicted": n_predicted,
            "n_structures_ok": n_predicted,
            "n_structures_low_contrast": 0,
            "n_structures_low_conf": 0,
            "n_structures_locally_blurry": 0,
            "overall_image_quality": (
                "GOOD" if n_predicted > 0 else "UNKNOWN"
            ),
            "plane_standard": {
                "stnd": result.get(
                    "standard_status",
                    "UNKNOWN",
                ),
                "confidence": round(
                    float(
                        result.get(
                            "fused_confidence",
                            0.0,
                        )
                    ),
                    4,
                ),
                "predicted_plane": result.get(
                    "predicted_plane",
                    None,
                ),
            },
        },
        "structures": structures,
    }

@torch.no_grad()
def run_inference_limbs(
    checkpoint: str,
    items,
    detector_checkpoint: str,
    segmenter_checkpoint: str,
    output_dir: str,
    images_dir: str | None = None,
    device_str: str = "cuda",
    progress_callback=None,
):
    """
    Main limb inference function, intentionally following the structure and
    naming convention of the HEAD reference's run_inference_head().

    The HEAD-specific segmentation logic is not used. The existing limb
    classifier -> YOLO -> fusion -> binary Swin-UNet -> Standard/Non-Standard/
    UNKNOWN pipeline remains unchanged inside run_pipeline().
    """

    device = torch.device(
        device_str if torch.cuda.is_available() else "cpu"
    )

    output_root = Path(output_dir)
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Model loading — same ordering as the HEAD inference function.
    # --------------------------------------------------------
    classifier, idx2plane = load_classifier(device)
    detector = load_detector(
        device,
        detector_checkpoint,
    )
    segmenter = load_segmenter(
        device,
        segmenter_checkpoint,
    )

    csv_rows: list[dict] = []
    json_entries: list[dict] = []
    results = []

    for item in items:

        img_path = item["image_path"]
        if not isinstance(img_path, Path):
            img_path = Path(img_path)

        stem = img_path.stem
        input_suffix = img_path.suffix

        print(
            f"  Processing: {stem}{input_suffix}"
        )

        try:
            # HEAD-style image loading entry point. The limb loader already
            # supports PNG/JPEG/BMP/TIFF and DICOM (.dcm/.dicom).
            image_gray = load_grayscale_image(
                img_path
            )

            if image_gray is None:
                raise ValueError(
                    f"Could not load image: {img_path}"
                )

            result = run_pipeline(
                image_gray=image_gray,
                classifier=classifier,
                idx2plane=idx2plane,
                detector=detector,
                segmenter=segmenter,
                device=device,
            )

            # ------------------------------------------------
            # Existing output organisation is retained.
            # ------------------------------------------------
            relative_path = img_path
            try:
                relative_path = img_path.relative_to(
                    Path(images_dir) if images_dir else img_path.parent
                )
            except Exception:
                relative_path = Path(img_path.name)

            plane = None
            for part in relative_path.parts[:-1]:
                upper_part = str(part).upper()
                if upper_part in {"FE", "HU", "RU", "TF"}:
                    plane = upper_part
                    break

            if plane is None:
                target_bone = result.get("target_bone")
                target_to_plane = {
                    1: "FE",
                    2: "HU",
                    3: "RU",
                    4: "TF",
                }
                plane = target_to_plane.get(
                    target_bone,
                    "UNKNOWN",
                )

            plane_root = output_root / plane
            masks_dir = plane_root / "masks"
            visualization_dir = plane_root / "visualization"

            masks_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
            visualization_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            mask = result.get(
                "segmentation_mask"
            )

            if mask is None:
                mask = np.zeros(
                    image_gray.shape,
                    dtype=np.int64,
                )

            mask_binary = (
                mask > 0
            ).astype(np.uint8) * 255

            mask_file = (
                masks_dir
                / f"{relative_path.stem}_mask.png"
            )

            cv2.imwrite(
                str(mask_file),
                mask_binary,
            )

            visualization = create_visualization(
                image_gray=image_gray,
                mask=mask,
                result=result,
            )

            visualization_file = (
                visualization_dir
                / f"{relative_path.stem}_visualization.png"
            )

            cv2.imwrite(
                str(visualization_file),
                visualization,
            )

            # ------------------------------------------------
            # HEAD-style per-image JSON construction.
            # ------------------------------------------------
            json_entry = _build_json_entry(
                img_path,
                result,
                visualization_file,
            )

            json_entries.append(json_entry)
            results.append(result)

            csv_rows.append(
                _build_csv_row(
                    img_path,
                    result,
                )
            )

            if progress_callback is not None:
                progress_callback(
                    len(json_entries),
                    len(items),
                    json_entry,
                )

        except Exception as exc:

            print(
                f"  [ERROR] {exc}"
            )

            error_result = {
                "path": str(img_path),
                "image_path": str(img_path),
                "status": "ERROR",
                "reason": str(exc),
                "standard_status": "UNKNOWN",
                "standard_reason": (
                    "UNKNOWN: inference failed before "
                    "Standard/Non-Standard classification "
                    "could be completed."
                ),
                "prediction_found": False,
                "target_bone": None,
                "fused_confidence": 0.0,
                "classifier_result": {},
                "detections": [],
                "yolo_predicted_bone": None,
                "yolo_confidence": 0.0,
                "target_boxes": [],
                "successful_boxes": [],
                "failed_boxes": [],
                "standard_details": {},
                "segmentation_mask": None,
            }

            json_entry = _build_json_entry(
                img_path,
                error_result,
                output_root / "UNKNOWN" / "visualization" / f"{stem}_visualization.png",
            )

            json_entries.append(json_entry)
            results.append(error_result)

            csv_rows.append({
                "image_path": str(img_path),
                "predicted_plane": "unknown",
                "plane_standard": "UNKNOWN",
                "image_quality": "ERROR",
            })

    # --------------------------------------------------------
    # HEAD-style aggregate JSON output.
    # --------------------------------------------------------
    inference_results = {
        "run_summary": {
            "input_directory": str(
                Path(images_dir).resolve()
            ) if images_dir else "",
            "detector_checkpoint": str(
                Path(detector_checkpoint).resolve()
            ),
            "segmenter_checkpoint": str(
                Path(segmenter_checkpoint).resolve()
            ),
            "classifier_checkpoint": str(
                Path(checkpoint).resolve()
            ),
            "classifier_weight": CLASSIFIER_WEIGHT,
            "detector_weight": DETECTOR_WEIGHT,
            "n_images": len(json_entries),
            "n_predictions": sum(
                1 for r in json_entries
                if r.get("prediction_found") is True
            ),
            "n_standard": sum(
                1 for r in json_entries
                if r.get("standard_status") == "STANDARD"
            ),
            "n_non_standard": sum(
                1 for r in json_entries
                if r.get("standard_status") == "NON-STANDARD"
            ),
            "n_unknown": sum(
                1 for r in json_entries
                if r.get("standard_status") == "UNKNOWN"
            ),
            "n_errors": sum(
                1 for r in json_entries
                if r.get("status") == "ERROR"
            ),
            "dicom_support": _DICOM_AVAILABLE,
        },
        "images": json_entries,
    }

    results_path = output_root / "inference_results.json"
    with open(results_path, "w") as f:
        json.dump(
            inference_results,
            f,
            indent=2,
        )

    # Keep the older filename as a compatibility copy.
    summary_path = output_root / "inference_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            json_entries,
            f,
            indent=2,
        )

    save_aggregate_csv(
        csv_rows,
        output_root,
    )

    print(
        f"\nJSON saved to: {results_path}"
    )
    print(
        f"Summary JSON saved to: {summary_path}"
    )

    return results


# ------------------------------------------------------------
# HEAD-style aggregate summary CSV
# ------------------------------------------------------------
# Same generic convention as the HEAD reference: one row per image with
# image_path / predicted_plane / plane_standard / image_quality. The
# HEAD field names are generic structural conventions, not head-specific
# fields, so they are reused as-is; the values populated underneath are
# entirely limb-specific (fused bone name, Standard/Non-Standard/UNKNOWN,
# and the limb pipeline's own per-image status flag).

CSV_FIELDNAMES = [
    "image_path", "predicted_plane", "plane_standard", "image_quality",
]


def _build_csv_row(image_path, result):
    """
    Build one HEAD-style aggregate CSV row for an image.

    Field mapping (generic convention -> limb-specific value):
        image_path      -> input image path
        predicted_plane -> fused target bone name
        plane_standard  -> Standard / Non-Standard / UNKNOWN
        image_quality   -> per-image pipeline status
                           (SUCCESS / PARTIAL / NO PREDICTION / ERROR)
    """

    target_bone = result.get("target_bone")

    return {
        "image_path": str(image_path),
        "predicted_plane": BONE_NAMES.get(
            target_bone,
            "unknown",
        ),
        "plane_standard": result.get(
            "standard_status",
            "UNKNOWN",
        ),
        "image_quality": result.get(
            "status",
            "UNKNOWN",
        ),
    }


def save_aggregate_csv(csv_rows, output_dir, csv_path=None):
    """
    Append aggregate summary rows to a shared CSV file.

    Mirrors the HEAD reference's aggregate-CSV convention: creates the
    file with a header on first write, and appends rows on subsequent
    runs so results from multiple invocations accumulate in one file.
    """

    if not csv_rows:
        return None

    csv_path = csv_path or (
        Path(output_dir) / "inference_summary.csv"
    )

    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDNAMES,
            extrasaction="ignore",
            restval="",
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(csv_rows)

    return str(csv_path)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Classifier -> YOLO26m -> "
            "Swin-UNet inference"
        )
    )

    parser.add_argument(
        "--images_dir",
        type=str,
        default=(
            "/mnt/data4tb/anusha/"
            "Limbs_Model_Copy/unseen_data"
        ),
    )

    parser.add_argument(
        "--detector_checkpoint",
        type=str,
        default=str(
            Config.DETECTOR_CHECKPOINT
        ),
    )

    parser.add_argument(
        "--segmenter_checkpoint",
        type=str,
        default=str(
            Config.SEGMENTER_CHECKPOINT
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=(
            "./cds_inference_outputs_val"
        ),
    )

    args = parser.parse_args()

    image_paths = collect_images(
        args.images_dir
    )

    print("=" * 70)
    print(
        "Cls -> YOLO26m -> Swin-UNet inference"
    )
    print("=" * 70)

    device = get_device()
    print(f"Device: {device}")
    print(
        f"DICOM support: "
        f"{'ENABLED' if _DICOM_AVAILABLE else 'DISABLED (pip install pydicom)'}"
    )
    print(
        f"Classifier weight: {CLASSIFIER_WEIGHT:.2f}"
    )
    print(
        f"YOLO weight:       {DETECTOR_WEIGHT:.2f}"
    )
    print(
        f"Duplicate IoU:     {DUPLICATE_IOU_THRESHOLD:.2f}"
    )
    print(
        f"Min component:     {MIN_COMPONENT_AREA}px"
    )
    print(
        f"FE/HU angle limit: {FE_HU_MAX_ANGLE:.0f}°"
    )
    print(
        f"RU/TF angle limit: {RU_TF_MAX_ANGLE:.0f}°"
    )
    print(
        f"Mask smoothing:    Gaussian "
        f"{SMOOTH_BLUR_KERNEL}, sigma={SMOOTH_BLUR_SIGMA}"
    )
    print("No mask status:    UNKNOWN")
    print("=" * 70)

    if not image_paths:
        print(
            "ERROR: No images found."
        )
        return

    print(
        f"\nFound {len(image_paths)} images."
    )
    print(
        f"Output directory: {args.output_dir}"
    )

    # HEAD-style `items` input. The limb pipeline has no head-specific
    # `panel` field, so each item only carries its image path plus the input
    # root used for output-plane organisation.
    items = [
        {
            "image_path": image_path,
        }
        for image_path in image_paths
    ]

    run_inference_limbs(
        checkpoint=str(
            Config.CLASSIFIER_CHECKPOINT
        ),
        items=items,
        detector_checkpoint=args.detector_checkpoint,
        segmenter_checkpoint=args.segmenter_checkpoint,
        output_dir=args.output_dir,
        images_dir=args.images_dir,
        device_str=str(device),
    )


if __name__ == "__main__":
    main()