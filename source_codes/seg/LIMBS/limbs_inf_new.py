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

def make_summary_row(
    image_path,
    result,
    output_path,
):
    classifier_result = result.get(
        "classifier_result",
        {},
    )

    target_bone = result.get(
        "target_bone"
    )

    target_name = BONE_NAMES.get(
        target_bone,
        "Unknown",
    )

    segmentation_mask = result.get(
        "segmentation_mask"
    )

    shapes = mask_to_polygons(
        segmentation_mask,
        target_bone,
    )

    standard_status = result.get(
        "standard_status",
        "UNKNOWN",
    )

    standard_details = result.get(
        "standard_details",
        {},
    ) or {}

    components = standard_details.get(
        "components",
        [],
    ) or []

    valid_components = standard_details.get(
        "valid_components",
        [],
    ) or []

    bone_ok = bool(
        segmentation_mask is not None
        and np.count_nonzero(segmentation_mask) > 0
    )

    geometry_ok = bool(
        len(valid_components) > 0
    )

    shape_ok = bool(
        len(shapes) > 0
    )

    # Head-style JSON first, followed by all of the existing CDS fields.
    row = {
        "image_path": str(image_path),

        "predicted_plane": target_name,

        "plane_info": {
            "predicted_plane": target_name,
            "classifier_plane": classifier_result.get(
                "top_plane_name",
                "unknown",
            ),
            "plane_match_frac": float(
                classifier_result.get(
                    "bone_probs",
                    {},
                ).get(
                    target_bone,
                    0.0,
                )
            ) if target_bone is not None else 0.0,
            "plane_mean_conf": float(
                classifier_result.get(
                    "bone_probs",
                    {},
                ).get(
                    target_bone,
                    0.0,
                )
            ) if target_bone is not None else 0.0,
            "plane_standard": standard_status,
            "plane_candidates": _top_plane_candidates(
                classifier_result,
                {},
                limit=3,
            ),
        },

        "quality": {
            "bone_ok": bone_ok,
            "geometry_ok": geometry_ok,
            "shape_ok": shape_ok,
            "overall": standard_status,
            "details": {
                "bone": {
                    "present": bone_ok,
                    "component_count": len(components),
                    "valid_component_count": len(valid_components),
                    "components": components,
                },
                "standard_reason": result.get(
                    "standard_reason",
                    "",
                ),
            },
        },

        "final_verdict": standard_status,

        "final_verdict_detail": {
            "geometry_ok": geometry_ok,
            "shape_ok": shape_ok,
            "component_count": len(components),
            "valid_component_count": len(valid_components),
            "split_screen": bool(
                standard_details.get(
                    "split_screen",
                    False,
                )
            ),
        },

        "shapes": shapes,

        # ----------------------------------------------------
        # Existing CDS summary fields preserved.
        # ----------------------------------------------------
        "path": str(image_path),

        "classifier_bone": BONE_NAMES.get(
            classifier_result.get(
                "predicted_bone"
            ),
            "unknown",
        ),

        "classifier_confidence": float(
            classifier_result.get(
                "confidence",
                0.0,
            )
        ),

        "yolo_bone": (
            BONE_NAMES.get(
                result.get(
                    "yolo_predicted_bone"
                ),
                "none",
            )
            if result.get(
                "yolo_predicted_bone"
            ) is not None
            else "none"
        ),

        "yolo_confidence": float(
            result.get(
                "yolo_confidence",
                0.0,
            )
        ),

        "fused_bone": target_name,

        "fused_confidence": float(
            result.get(
                "fused_confidence",
                0.0,
            )
        ),

        "num_yolo_detections": len(
            result.get(
                "detections",
                [],
            )
        ),

        "num_target_boxes": len(
            result.get(
                "target_boxes",
                [],
            )
        ),

        "num_segmented_boxes": len(
            result.get(
                "successful_boxes",
                [],
            )
        ),

        "failed_boxes": result.get(
            "failed_boxes",
            [],
        ),

        "prediction_found": bool(
            result.get(
                "prediction_found",
                False,
            )
        ),

        "status": result.get(
            "status",
            "UNKNOWN",
        ),

        "reason": result.get(
            "reason",
            "",
        ),

        "standard_status": standard_status,
        "standard_reason": result.get(
            "standard_reason",
            "",
        ),
        "standard_details": standard_details,

        "output_image": str(output_path),
    }

    return row


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

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    device = get_device()

    print("=" * 70)
    print(
        "Cls -> YOLO26m -> Swin-UNet inference"
    )
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    print(
        f"DICOM support: {'ENABLED' if _DICOM_AVAILABLE else 'DISABLED (pip install pydicom)'}"
    )

    print(
        f"Classifier weight: "
        f"{CLASSIFIER_WEIGHT:.2f}"
    )

    print(
        f"YOLO weight:       "
        f"{DETECTOR_WEIGHT:.2f}"
    )

    print(
        f"Duplicate IoU:     "
        f"{DUPLICATE_IOU_THRESHOLD:.2f}"
    )

    print(
        f"Min component:     "
        f"{MIN_COMPONENT_AREA}px"
    )

    print(
        f"FE/HU angle limit: "
        f"{FE_HU_MAX_ANGLE:.0f}°"
    )

    print(
        f"RU/TF angle limit: "
        f"{RU_TF_MAX_ANGLE:.0f}°"
    )

    print(
        f"Mask smoothing:    "
        f"Gaussian "
        f"{SMOOTH_BLUR_KERNEL}, "
        f"sigma={SMOOTH_BLUR_SIGMA}"
    )

    print(
        "No mask status:    UNKNOWN"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    print(
        "\nLoading classifier..."
    )

    # EXACT ORIGINAL LOADER.
    classifier, idx2plane = (
        load_classifier(
            device
        )
    )

    print(
        "Classifier loaded."
    )

    print(
        "\nLoading YOLO detector..."
    )

    detector = load_detector(
        device,
        args.detector_checkpoint,
    )

    print(
        "YOLO detector loaded."
    )

    print(
        "\nLoading segmenter..."
    )

    segmenter = load_segmenter(
        device,
        args.segmenter_checkpoint,
    )

    print(
        "Segmenter loaded."
    )

    # --------------------------------------------------------
    # COLLECT IMAGES
    # --------------------------------------------------------

    image_paths = collect_images(
        args.images_dir
    )

    print(
        f"\nFound {len(image_paths)} images."
    )

    if not image_paths:

        print(
            "ERROR: No images found."
        )

        return

    print(
        f"Output directory: "
        f"{args.output_dir}"
    )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    summary_rows = []
    csv_rows = []

    for (
        index,
        image_path,
    ) in enumerate(
        image_paths,
        start=1,
    ):

        print(
            f"\n[{index}/{len(image_paths)}] "
            f"{image_path}"
        )

        try:

            image_gray = load_grayscale_image(
                image_path
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
            # Console information
            # ------------------------------------------------

            cls_result = (
                result[
                    "classifier_result"
                ]
            )

            cls_name = BONE_NAMES.get(
                cls_result[
                    "predicted_bone"
                ],
                "unknown",
            )

            yolo_name = (
                BONE_NAMES.get(
                    result[
                        "yolo_predicted_bone"
                    ],
                    "none",
                )
                if result[
                    "yolo_predicted_bone"
                ] is not None
                else "none"
            )

            fused_name = BONE_NAMES.get(
                result[
                    "target_bone"
                ],
                "unknown",
            )

            print(
                f"  Classifier : "
                f"{cls_name} "
                f"{cls_result['confidence']:.3f}"
            )

            print(
                f"  YOLO       : "
                f"{yolo_name} "
                f"{result['yolo_confidence']:.3f}"
            )

            print(
                f"  Fused      : "
                f"{fused_name} "
                f"{result['fused_confidence']:.3f}"
            )

            print(
                f"  Target boxes: "
                f"{len(result['target_boxes'])}"
            )

            print(
                f"  Segmented   : "
                f"{len(result['successful_boxes'])}"
            )

            print(
                f"  Status      : "
                f"{result['status']}"
            )

            print(
                f"  Reason      : "
                f"{result['reason']}"
            )

            # ------------------------------------------------
            # STANDARD CLASSIFICATION
            # ------------------------------------------------

            print(
                f"  Standard    : "
                f"{result.get('standard_status', 'UNKNOWN')}"
            )

            print(
                f"  Std reason  : "
                f"{result.get('standard_reason', '')}"
            )

            # ------------------------------------------------
            # Mask
            # ------------------------------------------------

            mask = result[
                "segmentation_mask"
            ]

            if mask is None:

                mask = np.zeros(
                    image_gray.shape,
                    dtype=np.int64,
                )

            # ------------------------------------------------
            # OUTPUT ORGANIZATION
            # ------------------------------------------------
            # Every plane gets its own folder containing:
            #
            #   FE/
            #     masks/          -> all FE masks
            #     visualization/  -> all FE visualizations
            #
            # Same structure for HU, RU and TF.

            relative_path = (
                image_path.relative_to(
                    Path(args.images_dir)
                )
            )

            plane = None

            # Prefer the actual input-folder plane when available.
            for part in relative_path.parts[:-1]:
                upper_part = str(part).upper()
                if upper_part in {"FE", "HU", "RU", "TF"}:
                    plane = upper_part
                    break

            # Fallback to the fused prediction if the input directory
            # does not contain FE/HU/RU/TF in its path.
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

            plane_root = (
                Path(args.output_dir) / plane
            )

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

            # ------------------------------------------------
            # SAVE MASK
            # ------------------------------------------------
            # Binary mask: foreground = 255, background = 0.
            # The mask contains only the final fused target-bone
            # segmentation; YOLO boxes are not written into it.

            mask_binary = (
                mask > 0
            ).astype(np.uint8) * 255

            mask_file = (
                masks_dir
                /
                f"{relative_path.stem}_mask.png"
            )

            cv2.imwrite(
                str(mask_file),
                mask_binary,
            )

            # ------------------------------------------------
            # SAVE VISUALIZATION
            # ------------------------------------------------

            visualization = create_visualization(
                image_gray=image_gray,
                mask=mask,
                result=result,
            )

            visualization_file = (
                visualization_dir
                /
                f"{relative_path.stem}_visualization.png"
            )

            cv2.imwrite(
                str(visualization_file),
                visualization,
            )

            # ------------------------------------------------
            # Summary
            # ------------------------------------------------

            summary_rows.append(
                make_summary_row(
                    image_path,
                    result,
                    visualization_file,
                )
            )

            csv_rows.append(
                _build_csv_row(
                    image_path,
                    result,
                )
            )

        except Exception as exc:

            print(
                f"  [ERROR] {exc}"
            )

            summary_rows.append(
                {
                    "path": str(
                        image_path
                    ),
                    "status": "ERROR",
                    "reason": str(
                        exc
                    ),
                    "standard_status": "UNKNOWN",
                    "standard_reason": (
                        "UNKNOWN: inference "
                        "failed before "
                        "Standard/Non-Standard "
                        "classification could "
                        "be completed."
                    ),
                }
            )

            csv_rows.append(
                {
                    "image_path": str(
                        image_path
                    ),
                    "predicted_plane": "unknown",
                    "plane_standard": "UNKNOWN",
                    "image_quality": "ERROR",
                }
            )

    # --------------------------------------------------------
    # WRITE JSON — SAME AGGREGATE STYLE AS THE HEAD/LIMB
    # GENERIC INFERENCE FRAMEWORK
    # --------------------------------------------------------
    # One JSON file for the entire run.
    #
    # {
    #   "run_summary": {...},
    #   "images": [ ... one entry per image ... ]
    # }
    #
    # Keep inference_summary.json as a compatibility copy of the image
    # entries, while inference_results.json is the canonical aggregate file.

    output_root = Path(args.output_dir)

    inference_results = {
        "run_summary": {
            "input_directory": str(Path(args.images_dir).resolve()),
            "detector_checkpoint": str(Path(args.detector_checkpoint).resolve()),
            "segmenter_checkpoint": str(Path(args.segmenter_checkpoint).resolve()),
            "classifier_checkpoint": str(Path(Config.CLASSIFIER_CHECKPOINT).resolve()),
            "classifier_weight": CLASSIFIER_WEIGHT,
            "detector_weight": DETECTOR_WEIGHT,
            "n_images": len(summary_rows),
            "n_predictions": sum(1 for r in summary_rows if r.get("prediction_found") is True),
            "n_standard": sum(1 for r in summary_rows if r.get("standard_status") == "STANDARD"),
            "n_non_standard": sum(1 for r in summary_rows if r.get("standard_status") == "NON-STANDARD"),
            "n_unknown": sum(1 for r in summary_rows if r.get("standard_status") == "UNKNOWN"),
            "n_errors": sum(1 for r in summary_rows if r.get("status") == "ERROR"),
            "dicom_support": _DICOM_AVAILABLE,
        },
        "images": summary_rows,
    }

    results_path = output_root / "inference_results.json"
    with open(results_path, "w") as f:
        json.dump(inference_results, f, indent=2)

    # Backward-compatible filename used by the older CDS inference.
    summary_path = output_root / "inference_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)

    # HEAD-style aggregate CSV (image_path / predicted_plane /
    # plane_standard / image_quality), appended across runs.
    csv_path = save_aggregate_csv(csv_rows, output_root)

    print(f"\nJSON saved to: {results_path}")
    print(f"Compatibility JSON saved to: {summary_path}")

    if csv_path:
        print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()