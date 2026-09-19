"""
measure.py — Combined fetal biometry measurement script.

Usage:
    python measure.py --measurement lv   --image <path>
    python measure.py --measurement tcd  --image <path>
    python measure.py --measurement cm   --image <path>

    python measure.py --measurement lv   --folder <dir>
    python measure.py --measurement tcd  --folder <dir>
    python measure.py --measurement cm   --folder <dir>

Input images may be plain image files (png/jpg/jpeg/bmp/tiff) or DICOM
files (.dcm/.dicom). Measurements are written to the CSV in millimeters
for DICOM input (converted via the DICOM's PixelSpacing/ImagerPixelSpacing
tag) and in raw pixel units for plain images. A "units" column in the
output CSV records which was used for each row.
"""

import argparse
import csv
import math
import os
import sys
import cv2
import numpy as np
import pydicom
import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms.functional as TF

# -------------------------------------------------------------------------
# Optional heavy imports
# -------------------------------------------------------------------------
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
# _BASE_DIR   = os.path.abspath(os.path.join(_SCRIPT_DIR,  '..'))

sys.path.insert(0, os.path.join(_SCRIPT_DIR, 'model', 'head', 'head_unet_new'))
# sys.path.insert(0, os.path.join(_BASE_DIR, 'model', 'head', 'head_medsam'))

# UNet — required for TT plane (BPD / OFD / HC)
try:
    from model import UNet
    UNET_AVAILABLE = True
except ImportError as e:
    UNET_AVAILABLE = False
    print(f"[WARN] UNet not available: {e}")

# MedSAM — required for TC / TV planes (TCD / CM / LV)
# try:
#     from segment_anything import sam_model_registry
#     # from model_with_classifier import MedSAMFineTune
#     # from config import Config
#     # from load_test_data import CerebellumDataset
#     # from utils import postprocess_masks
#     INFERENCE_AVAILABLE = True
# except ImportError as e:
#     INFERENCE_AVAILABLE = False
#     print(f"[WARN] MedSAM not available: {e}")


# =========================================================================
# TARGET STRUCTURES
# =========================================================================

MEASUREMENT_TARGETS = {

    "lv": [
        "Choroid Plexus 2",
        "Lateral ventricle 2",
    ],

    "tcd": [
        "Cerebellum",
        "Midline Falx",
    ],

    "cm": [
        "cisternae magna",
        "Midline Falx",
    ],
}


SUPPORTED_IMAGE_EXTS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tiff",
    ".tif",
)

# ── DICOM support ─────────────────────────────────────────────────────────
DICOM_EXTS = (
    ".dcm",
    ".dicom",
)

# Combined set used when scanning input folders for files to process
ALL_SUPPORTED_EXTS = SUPPORTED_IMAGE_EXTS + DICOM_EXTS

# ── UNet (BPD/OFD) configuration ─────────────────────────────────────────
# UNET_CHECKPOINT_PATH = os.path.abspath(
#     os.path.join(_BASE_DIR, 'weights', 'head', 'best_model.pth')
# )
UNET_DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNET_INPUT_SIZE      = (512, 512)   # (width, height) for PIL resize

MASK_COLORS_HEAD = {
    'outer_calvarium': (255, 200,   0),
    'inner_calvarium': (  0, 255, 180),
    'midline_falx'   : (200,   0, 255),
}


# =========================================================================
# GENERIC IMAGE / DICOM LOADER
# =========================================================================

def is_dicom_file(path):
    """True if *path* has a DICOM extension (.dcm / .dicom)."""
    return os.path.splitext(path)[1].lower() in DICOM_EXTS


def load_image_generic(path):
    """
    Load either a plain image (png/jpg/...) or a DICOM (.dcm/.dicom) file.

    Returns:
        img_bgr        : np.ndarray, uint8 BGR image ready for cv2/PIL use
        pixel_spacing  : (row_mm_per_px, col_mm_per_px) — (1.0, 1.0) for
                          plain images since there is no physical spacing
        is_dicom       : bool
    """
    if is_dicom_file(path):
        ds     = pydicom.dcmread(path)
        pixels = ds.pixel_array.astype(float)
        pixels = (pixels - np.min(pixels)) / (np.max(pixels) - np.min(pixels) + 1e-8) * 255.0
        pixels = pixels.astype(np.uint8)

        if pixels.ndim == 2:
            img_bgr = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

        # Standard tags (Voluson), else fall back to US Regions sequence (Sonoscape)
        spacing = ds.get("PixelSpacing") or ds.get("ImagerPixelSpacing")
        if not spacing:
            us_regions = ds.get((0x0018, 0x6011))
            if us_regions:
                r  = us_regions[0]
                dx = r.get((0x0018, 0x602C))
                dy = r.get((0x0018, 0x602E))
                spacing = (
                    [abs(float(dy.value)) * 10.0, abs(float(dx.value)) * 10.0]
                    if dx and dy else [1.0, 1.0]
                )
            else:
                spacing = [1.0, 1.0]

        pixel_spacing = (float(spacing[0]), float(spacing[1]))
        print(f"  [DICOM] {os.path.basename(path)} | PixelSpacing: {pixel_spacing} mm/px")
        return img_bgr, pixel_spacing, True

    else:
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise IOError(f"Could not read image: {path}")
        pixel_spacing = (1.0, 1.0)
        return img_bgr, pixel_spacing, False


def px_to_output_value(value_px, pixel_spacing, is_dicom, ndigits=2):
    """
    Convert a pixel measurement to its stored/CSV value.

    - DICOM input  -> value in mm (using mean of row/col pixel spacing), units="mm"
    - Plain image  -> raw pixel value unchanged,                        units="px"

    Returns (value, units). value is None if value_px is None.
    """
    if value_px is None:
        return None, ("mm" if is_dicom else "px")

    if is_dicom:
        mean_spacing = (pixel_spacing[0] + pixel_spacing[1]) / 2.0
        return round(float(value_px) * mean_spacing, ndigits), "mm"

    return round(float(value_px), ndigits), "px"


# =========================================================================
# SAVE MASK OVERLAY
# =========================================================================

def save_mask_overlay(
    image,
    masks,
    measurement,
    output_path
):
    """
    *image* is the already-loaded BGR np.ndarray (works for both plain
    images and DICOM-derived arrays) — no re-reading from disk here so
    this works uniformly regardless of source file type.
    """

    if image is None:
        return

    overlay = image.copy()

    if measurement == "tcd":

        target_structures = [
            ("Cerebellum", (0, 255, 0))
        ]

    elif measurement == "cm":

        target_structures = [
            ("cisternae magna", (0, 255, 0))
        ]

    elif measurement == "lv":

        target_structures = [
            ("Lateral ventricle 2", (0, 255, 0)),
            ("Choroid Plexus 2", (0, 0, 255)),
        ]

    else:
        return

    for structure_name, color in target_structures:

        mask = masks.get(structure_name)

        if mask is None:
            continue

        binary = (
            mask > 127
        ).astype(np.uint8)

        colored_mask = np.zeros_like(image)

        colored_mask[:, :, 0] = binary * color[0]
        colored_mask[:, :, 1] = binary * color[1]
        colored_mask[:, :, 2] = binary * color[2]

        overlay = cv2.addWeighted(
            overlay,
            1.0,
            colored_mask,
            0.4,
            0
        )

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            overlay,
            contours,
            -1,
            color,
            2
        )

    cv2.imwrite(
        output_path,
        overlay
    )


# =========================================================================
# GENERIC LINE
# =========================================================================

def get_line_inside_mask(
    binary_mask,
    center_x,
    center_y,
    direction
):

    dx = float(direction[0])
    dy = float(direction[1])

    norm = np.sqrt(dx**2 + dy**2)

    dx /= norm
    dy /= norm

    x1 = float(center_x)
    y1 = float(center_y)

    last_x1 = x1
    last_y1 = y1

    for _ in range(1000):

        nx = x1 - dx
        ny = y1 - dy

        ix = int(round(nx))
        iy = int(round(ny))

        if (
            ix < 0
            or iy < 0
            or ix >= binary_mask.shape[1]
            or iy >= binary_mask.shape[0]
        ):
            break

        if binary_mask[iy, ix] == 0:
            break

        last_x1 = nx
        last_y1 = ny

        x1 = nx
        y1 = ny

    x2 = float(center_x)
    y2 = float(center_y)

    last_x2 = x2
    last_y2 = y2

    for _ in range(1000):

        nx = x2 + dx
        ny = y2 + dy

        ix = int(round(nx))
        iy = int(round(ny))

        if (
            ix < 0
            or iy < 0
            or ix >= binary_mask.shape[1]
            or iy >= binary_mask.shape[0]
        ):
            break

        if binary_mask[iy, ix] == 0:
            break

        last_x2 = nx
        last_y2 = ny

        x2 = nx
        y2 = ny

    pt1 = (
        int(round(last_x1)),
        int(round(last_y1))
    )

    pt2 = (
        int(round(last_x2)),
        int(round(last_y2))
    )

    length = int(
        np.linalg.norm(
            np.array(pt1) -
            np.array(pt2)
        )
    )

    return pt1, pt2, length


# =========================================================================
# FIND LONGEST LINE
# =========================================================================

def find_longest_perpendicular_line(
    binary_mask,
    perp_direction,
    falx_direction,
    step_size=1
):

    h, w = binary_mask.shape

    def _unit(v):

        n = np.sqrt(v[0] ** 2 + v[1] ** 2)

        return np.array(
            [v[0] / n, v[1] / n],
            dtype=np.float64
        )

    pd = _unit(perp_direction)

    fd = _unit(falx_direction)

    ys, xs = np.where(binary_mask > 0)

    if len(xs) == 0:
        return (0, 0), (0, 0), 0

    mask_center = np.array(
        [float(np.mean(xs)), float(np.mean(ys))]
    )

    pts = np.column_stack((xs, ys)).astype(np.float64)

    projections = pts @ fd

    proj_min = projections.min()
    proj_max = projections.max()

    best_pt1 = (0, 0)
    best_pt2 = (0, 0)
    best_length = 0

    ref = mask_center

    t = proj_min

    while t <= proj_max:

        seed = ref + (t - (ref @ fd)) * fd

        seed_x = float(seed[0])
        seed_y = float(seed[1])

        ix = int(round(seed_x))
        iy = int(round(seed_y))

        if (
            0 <= ix < w
            and 0 <= iy < h
            and binary_mask[iy, ix] > 0
        ):

            pt1, pt2, length = get_line_inside_mask(
                binary_mask,
                seed_x,
                seed_y,
                pd
            )

            if length > best_length:

                best_length = length
                best_pt1 = pt1
                best_pt2 = pt2

        t += step_size

    return best_pt1, best_pt2, best_length


# =========================================================================
# FALX ORIENTATION
# =========================================================================

def compute_falx_orientation(falx_mask):

    ys, xs = np.where(falx_mask > 0)

    if len(xs) < 2:
        return None, None, None

    pts = np.column_stack(
        (xs, ys)
    ).astype(np.float32)

    mean, eigenvectors = cv2.PCACompute(
        pts,
        mean=None
    )

    center = mean[0]

    direction = eigenvectors[0]

    angle_rad = np.arctan2(
        direction[1],
        direction[0]
    )

    angle_deg = np.degrees(angle_rad)

    return (
        angle_deg,
        center,
        direction
    )


# =========================================================================
# LV MEASUREMENT
# =========================================================================

# def measure_lv(
#     image_path,
#     lv_mask,
#     cp_mask,
#     output_path
# ):

#     image = cv2.imread(image_path)

#     if image is None:
#         return None

#     _, lv_bin = cv2.threshold(
#         lv_mask,
#         127,
#         255,
#         cv2.THRESH_BINARY
#     )

#     _, cp_bin = cv2.threshold(
#         cp_mask,
#         127,
#         255,
#         cv2.THRESH_BINARY
#     )

#     lv_contours, _ = cv2.findContours(
#         lv_bin,
#         cv2.RETR_EXTERNAL,
#         cv2.CHAIN_APPROX_SIMPLE
#     )

#     cp_contours, _ = cv2.findContours(
#         cp_bin,
#         cv2.RETR_EXTERNAL,
#         cv2.CHAIN_APPROX_SIMPLE
#     )

#     if not lv_contours or not cp_contours:
#         return None

#     lv_contour = max(
#         lv_contours,
#         key=cv2.contourArea
#     )

#     cp_contour = max(
#         cp_contours,
#         key=cv2.contourArea
#     )

#     cv2.drawContours(
#         image,
#         [lv_contour],
#         -1,
#         (0, 255, 0),
#         2
#     )

#     x, y, w, h = cv2.boundingRect(
#         cp_contour
#     )

#     cx = x + w // 2

#     ys = np.where(lv_bin[:, cx] > 0)[0]

#     if len(ys) == 0:
#         return None

#     top_y = ys[0]
#     bottom_y = ys[-1]

#     pixel_value = bottom_y - top_y

#     cv2.line(
#         image,
#         (cx, top_y),
#         (cx, bottom_y),
#         (0, 0, 255),
#         2
#     )

#     cv2.imwrite(output_path, image)

#     return pixel_value


def measure_lv(
    image,
    lv_mask,
    cp_mask
):
    """
    *image* is the already-loaded BGR np.ndarray (works uniformly for
    plain images and DICOM-derived arrays).
    """

    if image is None:
        return None

    image = image.copy()

    _, lv_bin = cv2.threshold(
        lv_mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    _, cp_bin = cv2.threshold(
        cp_mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    # Combine LV and CP masks
    combined_mask = cv2.bitwise_or(
        lv_bin,
        cp_bin
    )

    contours, _ = cv2.findContours(
        combined_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contour = max(
        contours,
        key=cv2.contourArea
    )

    cv2.drawContours(
        image,
        [contour],
        -1,
        (0, 255, 0),
        2
    )

    # CP bounding box defines search region
    cp_contours, _ = cv2.findContours(
        cp_bin,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not cp_contours:
        return None

    cp_contour = max(
        cp_contours,
        key=cv2.contourArea
    )

    x, y, w, h = cv2.boundingRect(
        cp_contour
    )

    best_length = 0
    best_top = None
    best_bottom = None

    # Search all columns inside CP region
    for cx in range(x, x + w):

        ys = np.where(
            combined_mask[:, cx] > 0
        )[0]

        if len(ys) < 2:
            continue

        top_y = ys[0]
        bottom_y = ys[-1]

        length = bottom_y - top_y

        if length > best_length:
            best_length = length
            best_top = (cx, top_y)
            best_bottom = (cx, bottom_y)

    if best_top is None:
        return None

    cv2.line(
        image,
        best_top,
        best_bottom,
        (0, 0, 255),
        2
    )

    cv2.circle(
        image,
        best_top,
        4,
        (255, 0, 0),
        -1
    )

    cv2.circle(
        image,
        best_bottom,
        4,
        (255, 0, 0),
        -1
    )

    return best_length



# =========================================================================
# TCD MEASUREMENT
# =========================================================================

# def measure_tcd(
#     image_path,
#     cerebellum_mask,
#     falx_mask,
#     output_path
# ):

#     original = cv2.imread(image_path)

#     if original is None:
#         return None

#     mask = cerebellum_mask

#     if mask.shape[:2] != original.shape[:2]:

#         mask = cv2.resize(
#             mask,
#             (
#                 original.shape[1],
#                 original.shape[0]
#             ),
#             interpolation=cv2.INTER_NEAREST
#         )

#     _, binary = cv2.threshold(
#         mask,
#         127,
#         255,
#         cv2.THRESH_BINARY
#     )

#     overlay = original.copy()

#     contours, _ = cv2.findContours(
#         binary,
#         cv2.RETR_EXTERNAL,
#         cv2.CHAIN_APPROX_NONE
#     )

#     if not contours:
#         return None

#     largest = max(
#         contours,
#         key=cv2.contourArea
#     )

#     cv2.drawContours(
#         overlay,
#         [largest],
#         -1,
#         (0, 255, 0),
#         2
#     )

#     ys, xs = np.where(binary > 0)

#     if len(xs) < 2:
#         return None

#     pts = np.column_stack(
#         (xs, ys)
#     ).astype(np.float32)

#     mean, eigenvectors = cv2.PCACompute(
#         pts,
#         mean=None
#     )

#     principal_direction = eigenvectors[0]

#     px = float(principal_direction[0])
#     py = float(principal_direction[1])

#     norm = np.sqrt(px**2 + py**2)

#     if norm == 0:
#         return None

#     px /= norm
#     py /= norm

#     measurement_direction = np.array(
#         [px, py]
#     )

#     sweep_direction = np.array(
#         [-py, px]
#     )

#     pt1, pt2, line_length_px = (
#         find_longest_perpendicular_line(
#             binary,
#             perp_direction=measurement_direction,
#             falx_direction=sweep_direction,
#             step_size=1
#         )
#     )

#     cv2.line(
#         overlay,
#         pt1,
#         pt2,
#         (0, 0, 255),
#         2
#     )

#     cv2.circle(
#         overlay,
#         pt1,
#         4,
#         (255, 0, 0),
#         -1
#     )

#     cv2.circle(
#         overlay,
#         pt2,
#         4,
#         (255, 0, 0),
#         -1
#     )

#     cv2.imwrite(
#         output_path,
#         overlay
#     )

#     return line_length_px


def measure_tcd(
    original,
    cerebellum_mask,
    falx_mask,
    # output_path
):
    """
    *original* is the already-loaded BGR np.ndarray (works uniformly for
    plain images and DICOM-derived arrays).
    """

    if original is None:
        return None

    original = original.copy()

    mask = cerebellum_mask

    if mask.shape[:2] != original.shape[:2]:
        mask = cv2.resize(
            mask,
            (
                original.shape[1],
                original.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

    _, binary = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    overlay = original.copy()

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return None

    largest = max(
        contours,
        key=cv2.contourArea
    )

    # Draw contour
    cv2.drawContours(
        overlay,
        [largest],
        -1,
        (0, 255, 0),
        2
    )

    # Get contour points
    contour_pts = largest[:, 0, :]  # shape (N, 2)

    # Topmost contour point (minimum y)
    top_point = tuple(
        contour_pts[np.argmin(contour_pts[:, 1])]
    )

    # Bottommost contour point (maximum y)
    bottom_point = tuple(
        contour_pts[np.argmax(contour_pts[:, 1])]
    )

    top_point = (
        int(top_point[0]),
        int(top_point[1])
    )

    bottom_point = (
        int(bottom_point[0]),
        int(bottom_point[1])
    )

    # Calculate distance
    line_length_px = np.sqrt(
        (bottom_point[0] - top_point[0]) ** 2 +
        (bottom_point[1] - top_point[1]) ** 2
    )

    # Draw measurement line (red)
    cv2.line(
        overlay,
        top_point,
        bottom_point,
        (0, 0, 255),
        2
    )

    # Draw endpoints (blue)
    cv2.circle(
        overlay,
        top_point,
        4,
        (255, 0, 0),
        -1
    )

    cv2.circle(
        overlay,
        bottom_point,
        4,
        (255, 0, 0),
        -1
    )

    # cv2.imwrite(
    #     output_path,
    #     overlay
    # )

    return line_length_px


# =========================================================================
# CM MEASUREMENT
# =========================================================================

def measure_cm(
    original,
    cm_mask,
    falx_mask,
    # output_path
):
    """
    *original* is the already-loaded BGR np.ndarray (works uniformly for
    plain images and DICOM-derived arrays).
    """

    if original is None:
        return None

    original = original.copy()

    mask = cm_mask

    if mask.shape[:2] != original.shape[:2]:

        mask = cv2.resize(
            mask,
            (
                original.shape[1],
                original.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

    _, binary = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return None

    overlay = original.copy()

    largest = max(
        contours,
        key=cv2.contourArea
    )

    cv2.drawContours(
        overlay,
        [largest],
        -1,
        (0, 255, 0),
        2
    )

    ys, xs = np.where(binary > 0)

    if len(xs) == 0:
        return None

    center_x = int(np.mean(xs))
    center_y = int(np.mean(ys))

    direction = np.array([1.0, 0.0])

    if falx_mask is not None:

        if falx_mask.shape[:2] != original.shape[:2]:

            falx_mask = cv2.resize(
                falx_mask,
                (
                    original.shape[1],
                    original.shape[0]
                ),
                interpolation=cv2.INTER_NEAREST
            )

        (
            angle_deg,
            center,
            falx_direction
        ) = compute_falx_orientation(
            falx_mask
        )

        if falx_direction is not None:

            direction = falx_direction

    pt1, pt2, cm_length = (
        get_line_inside_mask(
            binary,
            center_x,
            center_y,
            direction
        )
    )

    cv2.line(
        overlay,
        pt1,
        pt2,
        (0, 0, 255),
        2
    )

    cv2.circle(
        overlay,
        pt1,
        4,
        (255, 0, 0),
        -1
    )

    cv2.circle(
        overlay,
        pt2,
        4,
        (255, 0, 0),
        -1
    )

    # cv2.imwrite(
    #     output_path,
    #     overlay
    # )

    return {
        "line_length": cm_length
    }



# =========================================================================
# UNET HEAD MODEL SETUP
# =========================================================================

def setup_unet_model(tc_tv_model_ckpt):
    """Load the Dual-Head UNet from UNET_CHECKPOINT_PATH."""
    print(f"_SCRIPT_DIR:{_SCRIPT_DIR}")
    model      = UNet()
    checkpoint = torch.load(tc_tv_model_ckpt, map_location=UNET_DEVICE)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    return model.to(UNET_DEVICE).eval()


# =========================================================================
# HEAD MASK OVERLAY (bio_ga_final_12 style)
# =========================================================================

def save_mask_overlay_head(img_bgr, masks_dict, save_path, alpha=0.45):
    """Blend coloured masks onto img_bgr and save."""
    overlay = img_bgr.copy().astype(np.float32)
    for label, (mask_np, color_bgr) in masks_dict.items():
        if mask_np.shape[:2] != img_bgr.shape[:2]:
            mask_np = cv2.resize(
                mask_np.astype(np.uint8),
                (img_bgr.shape[1], img_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        bool_mask = mask_np.astype(bool)
        overlay[bool_mask] = (
            (1 - alpha) * overlay[bool_mask]
            + alpha * np.array(color_bgr, dtype=np.float32)
        )
    result = np.clip(overlay, 0, 255).astype(np.uint8)
    cv2.imwrite(save_path, result)
    print(f"  Mask overlay saved : {save_path}")
    return result


# =========================================================================
# GEOMETRY HELPERS (ported from bio_ga_final_12.py)
# =========================================================================

def compute_pca_line(mask_np):
    """PCA on the largest contour of a binary mask; returns (mean, axis, angle_deg)."""
    kernel  = np.ones((5, 5), np.uint8)
    mask_np = cv2.morphologyEx(mask_np.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Midline not found")
    pts  = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    mean = np.mean(pts, axis=0)
    _, eig = np.linalg.eigh(np.cov(pts - mean, rowvar=False))
    axis = eig[:, 1]
    return mean, axis, math.degrees(math.atan2(axis[1], axis[0]))


def line_contour_intersection(p1, p2, contour):
    """Return list of (x, y) intersection points between segment p1-p2 and a contour."""
    if contour is None:
        return []
    ints = []
    for i in range(len(contour)):
        seg_p3 = tuple(contour[i][0])
        seg_p4 = tuple(contour[(i + 1) % len(contour)][0])
        x1, y1 = p1;  x2, y2 = p2
        x3, y3 = seg_p3; x4, y4 = seg_p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) > 1e-10:
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
            if 0 <= t <= 1 and 0 <= u <= 1:
                ints.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
    return ints


def line_intersection_2pt(p1, p2, p3, p4):
    """Return intersection of two infinite lines, or None if parallel."""
    x1, y1 = p1; x2, y2 = p2
    x3, y3 = p3; x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
    return (px, py)


def get_clinical_symmetry_bpd(image_np, ofd_pts, principal_axis, outer_cont, inner_cont):
    """Compute BPD as perpendicular to OFD at its midpoint, with sub-pixel refinement."""
    if len(ofd_pts) < 2:
        return 0, {'top': None, 'bottom': None}

    mid_x = (ofd_pts[0][0] + ofd_pts[-1][0]) / 2
    mid_y = (ofd_pts[0][1] + ofd_pts[-1][1]) / 2
    perp_vec  = np.array([-principal_axis[1], principal_axis[0]])
    perp_vec /= np.linalg.norm(perp_vec)

    p1 = (int(mid_x - perp_vec[0] * 1024), int(mid_y - perp_vec[1] * 1024))
    p2 = (int(mid_x + perp_vec[0] * 1024), int(mid_y + perp_vec[1] * 1024))

    o_ints = line_contour_intersection(p1, p2, outer_cont)
    i_ints = line_contour_intersection(p1, p2, inner_cont)

    if not (o_ints and i_ints):
        return 0, {'top': None, 'bottom': None}

    top_pt         = min(o_ints, key=lambda p: p[1])
    bot_pt_initial = max(i_ints, key=lambda p: p[1])

    # Gradient refinement on the bottom point
    refined_bot_y = bot_pt_initial[1]
    gray          = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    best_gradient = -1
    for offset in range(-5, 15):
        y = int(bot_pt_initial[1] + offset)
        x = int(bot_pt_initial[0])
        if 0 <= y + 1 < gray.shape[0]:
            gradient = int(gray[y + 1, x]) - int(gray[y, x])
            if gradient > best_gradient:
                best_gradient = gradient
                refined_bot_y = y
    bot_pt = (bot_pt_initial[0], refined_bot_y)

    dist = np.linalg.norm(np.array(top_pt) - np.array(bot_pt))
    return dist, {'top': top_pt, 'bottom': bot_pt}


def save_visualization_head(orig_img, ofd_pts, bpd_pts, midline_mask, save_path):
    """Draw OFD/BPD lines on orig_img (PIL RGB) and save to save_path."""
    img     = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
    overlay = img.copy()
    if midline_mask is not None:
        overlay[midline_mask > 0] = [128, 0, 128]
        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
    if len(ofd_pts) >= 2:
        cv2.line(
            img,
            tuple(map(int, ofd_pts[0])),
            tuple(map(int, ofd_pts[-1])),
            (255, 255, 0), 2,
        )
    if bpd_pts['top'] and bpd_pts['bottom']:
        p1 = tuple(map(int, bpd_pts['top']))
        p2 = tuple(map(int, bpd_pts['bottom']))
        cv2.line(img, p1, p2, (0, 255, 0), 2)
        cv2.circle(img, p1, 5, (0, 0, 255), -1)
        cv2.circle(img, p2, 5, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)
    print(f"  Measurement overlay saved : {save_path}")
    return img


# =========================================================================
# PROCESS SINGLE TT IMAGE  (BPD / OFD / HC via UNet)
# =========================================================================

def process_single_image_tt(image_path, model_unet, out_mask_dir, out_meas_dir):
    """
    Run UNet inference on one TT-plane image, measure BPD and OFD, derive HC,
    and save mask_overlay + measurement_overlay.

    Handles both plain images and DICOM (.dcm/.dicom) input transparently.

    Returns (ofd_value, bpd_value, hc_value, units) on success, None on
    failure/rejection. units is "mm" for DICOM input (converted using
    PixelSpacing) or "px" for plain images (raw pixel value).
    """
    DEVIATION_THRESHOLD = 20

    try:
        img_bgr, pixel_spacing, is_dicom = load_image_generic(image_path)
    except IOError as e:
        print(f"  [WARN] Cannot read: {image_path} ({e})")
        return None

    orig_h, orig_w = img_bgr.shape[:2]

    # ── Preprocess: BGR → grayscale PIL → 512×512 tensor ─────────────
    gray_pil   = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    img_resized = gray_pil.resize(UNET_INPUT_SIZE, resample=Image.BILINEAR)
    img_tensor  = TF.to_tensor(img_resized).unsqueeze(0).to(UNET_DEVICE)

    with torch.no_grad():
        outputs     = model_unet(img_tensor)
        inner_probs = torch.softmax(outputs["inner"], dim=1)
        inner_pred  = torch.argmax(inner_probs, dim=1).squeeze(0).cpu().numpy()
        calv_probs  = torch.sigmoid(outputs["calv"]).squeeze(0).cpu().numpy()

    # ── Binary masks at 512×512 → resize to original ─────────────────
    mid_512   = (inner_pred == 3).astype(np.uint8)
    inner_512 = (calv_probs[0] > 0.5).astype(np.uint8)
    outer_512 = (calv_probs[1] > 0.5).astype(np.uint8)

    def _resize(m):
        return cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    outer = _resize(outer_512)
    inner = _resize(inner_512)
    mid   = _resize(mid_512)

    # ── PCA on midline falx ───────────────────────────────────────────
    try:
        mean_pt, axis, angle = compute_pca_line(mid)
    except ValueError:
        print(f"  [WARN] Midline not detected: {os.path.basename(image_path)}")
        return None

    def _get_contour(m):
        cnts, _ = cv2.findContours(
            m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return max(cnts, key=cv2.contourArea) if cnts else None

    outer_c = _get_contour(outer)
    inner_c = _get_contour(inner)

    # ── OFD: intersection of PCA axis with outer calvarium ───────────
    p1 = (int(mean_pt[0] - axis[0] * 1024), int(mean_pt[1] - axis[1] * 1024))
    p2 = (int(mean_pt[0] + axis[0] * 1024), int(mean_pt[1] + axis[1] * 1024))

    ofd_pts = sorted(
        line_contour_intersection(p1, p2, outer_c),
        key=lambda p: p[0],
    )

    ofd_px = 0.0
    if len(ofd_pts) >= 2:
        ofd_px = np.linalg.norm(np.array(ofd_pts[0]) - np.array(ofd_pts[-1]))
    ofd_px = round(ofd_px, 2)

    # ── BPD: perpendicular to OFD at its midpoint ────────────────────
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    bpd_px, bpd_pts = get_clinical_symmetry_bpd(
        img_rgb, ofd_pts, axis, outer_c, inner_c
    )
    bpd_px = round(bpd_px, 2)

    # ── Deviation filter ─────────────────────────────────────────────
    if (
        len(ofd_pts) < 2
        or bpd_pts['top'] is None
        or bpd_pts['bottom'] is None
    ):
        print(f"  [WARN] Insufficient points: {os.path.basename(image_path)}")
        return None

    intersect_pt = line_intersection_2pt(
        ofd_pts[0], ofd_pts[-1],
        bpd_pts['top'], bpd_pts['bottom'],
    )
    if intersect_pt is None:
        print(f"  [WARN] OFD & BPD parallel: {os.path.basename(image_path)}")
        return None

    mid_bpd     = (
        (bpd_pts['top'][0] + bpd_pts['bottom'][0]) / 2,
        (bpd_pts['top'][1] + bpd_pts['bottom'][1]) / 2,
    )
    deviation   = round(
        float(np.linalg.norm(np.array(intersect_pt) - np.array(mid_bpd))), 2
    )

    if deviation > DEVIATION_THRESHOLD:
        print(
            f"  [REJECT] {os.path.basename(image_path)} "
            f"deviation={deviation} > {DEVIATION_THRESHOLD}"
        )
        return None

    # ── HC derived from BPD + OFD ────────────────────────────────────
    hc_px = round(math.pi * (bpd_px + ofd_px) / 2, 2)

    print(
        f"  [ACCEPT] {os.path.basename(image_path)} "
        f"BPD={bpd_px}px  OFD={ofd_px}px  HC={hc_px}px  dev={deviation}"
    )

    stem = os.path.splitext(os.path.basename(image_path))[0]

    # ── Save mask overlay ─────────────────────────────────────────────
    mask_overlay_path = os.path.join(out_mask_dir, f"{stem}_mask_overlay.png")
    save_mask_overlay_head(
        img_bgr,
        {
            'outer_calvarium': (outer, MASK_COLORS_HEAD['outer_calvarium']),
            'inner_calvarium': (inner, MASK_COLORS_HEAD['inner_calvarium']),
            'midline_falx'   : (mid,   MASK_COLORS_HEAD['midline_falx']),
        },
        mask_overlay_path,
    )

    # ── Save measurement overlay ──────────────────────────────────────
    meas_overlay_path = os.path.join(out_meas_dir, f"{stem}_measurement_overlay.png")
    orig_pil = Image.fromarray(img_rgb)
    save_visualization_head(orig_pil, ofd_pts, bpd_pts, mid, meas_overlay_path)

    ofd_val, units = px_to_output_value(ofd_px, pixel_spacing, is_dicom)
    bpd_val, _     = px_to_output_value(bpd_px, pixel_spacing, is_dicom)
    hc_val, _      = px_to_output_value(hc_px,  pixel_spacing, is_dicom)

    return ofd_val, bpd_val, hc_val, units


# =========================================================================
# UNET STRUCTURE INDICES  (from Config.STRUCTURE_TO_IDX, 1-based)
# =========================================================================

_UNET_STRUCT_IDX = {
    "Midline Falx"       : 3,
    "Choroid Plexus 2"   : 12,
    "Lateral ventricle 2": 13,
    "Cerebellum"         : 18,
    "cisternae magna"    : 20,
}


def _unet_run(img_bgr, model_unet):
    """Forward pass; return inner_pred class map at 512×512."""
    gray_pil    = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    img_resized = gray_pil.resize(UNET_INPUT_SIZE, resample=Image.BILINEAR)
    img_tensor  = TF.to_tensor(img_resized).unsqueeze(0).to(UNET_DEVICE)
    with torch.no_grad():
        inner_pred = torch.argmax(
            torch.softmax(model_unet(img_tensor)["inner"], dim=1), dim=1
        ).squeeze(0).cpu().numpy()
    return inner_pred


def _extract_struct_masks(inner_pred, orig_w, orig_h):
    """Return {struct_name: uint8 mask at orig resolution} for all tracked structures."""
    def _m(cls):
        m = (inner_pred == cls).astype(np.uint8) * 255
        return cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return {name: _m(idx) for name, idx in _UNET_STRUCT_IDX.items()}


# =========================================================================
# PROCESS SINGLE TC IMAGE  (TCD + CM via UNet)
# =========================================================================

def process_single_image_tc(image_path, model_unet):
    """UNet inference on one TC-plane image; measure TCD and CM.

    Handles both plain images and DICOM (.dcm/.dicom) input transparently.

    Returns (tcd_value, cm_value, units), either value may be None on failure.
    units is "mm" for DICOM input (converted using PixelSpacing) or "px"
    for plain images (raw pixel value).
    """
    try:
        img_bgr, pixel_spacing, is_dicom = load_image_generic(image_path)
    except IOError as e:
        print(f"  [WARN] Cannot read: {image_path} ({e})")
        return None, None, "px"

    orig_h, orig_w = img_bgr.shape[:2]
    inner_pred = _unet_run(img_bgr, model_unet)
    masks = _extract_struct_masks(inner_pred, orig_w, orig_h)

    stem = os.path.splitext(os.path.basename(image_path))[0]

    # save_mask_overlay(
    #     img_bgr, masks, "tcd",
    #     os.path.join(out_mask_dir, f"{stem}_mask_overlay.png"),
    # )

    tcd_px = measure_tcd(
        img_bgr,
        masks["Cerebellum"],
        masks["Midline Falx"],
        # os.path.join(tcd_meas_dir, f"{stem}_tcd.png"),
    )

    cm_raw = measure_cm(
        img_bgr,
        masks["cisternae magna"],
        masks["Midline Falx"],
        # os.path.join(cm_meas_dir, f"{stem}_cm.png"),
    )
    cm_px = cm_raw["line_length"] if cm_raw else None

    tcd_result, units = px_to_output_value(tcd_px, pixel_spacing, is_dicom)
    cm_result, _       = px_to_output_value(cm_px,  pixel_spacing, is_dicom)

    return tcd_result, cm_result, units


# =========================================================================
# PROCESS SINGLE TV IMAGE  (LV via UNet)
# =========================================================================

def process_single_image_tv(image_path, model_unet):
    """UNet inference on one TV-plane image; measure LV.

    Handles both plain images and DICOM (.dcm/.dicom) input transparently.

    Returns (lv_value, units), or (None, "px") on failure.
    units is "mm" for DICOM input (converted using PixelSpacing) or "px"
    for plain images (raw pixel value).
    """
    try:
        img_bgr, pixel_spacing, is_dicom = load_image_generic(image_path)
    except IOError as e:
        print(f"  [WARN] Cannot read: {image_path} ({e})")
        return None, "px"

    orig_h, orig_w = img_bgr.shape[:2]
    inner_pred = _unet_run(img_bgr, model_unet)
    masks = _extract_struct_masks(inner_pred, orig_w, orig_h)

    stem = os.path.splitext(os.path.basename(image_path))[0]

    # save_mask_overlay(
    #     img_bgr, masks, "lv",
    #     os.path.join(out_mask_dir, f"{stem}_mask_overlay.png"),
    # )

    lv_px = measure_lv(
        img_bgr,
        masks["Lateral ventricle 2"],
        masks["Choroid Plexus 2"]
    )

    return px_to_output_value(lv_px, pixel_spacing, is_dicom)


# =========================================================================
# CSV HELPERS
# =========================================================================

CSV_COLUMNS = ["image_name", "modality", "units", "tcd", "cm", "lv", "bpd", "ofd", "hc"]


def _empty_row(image_name):
    """Return a dict with all measurement columns set to empty string."""
    return {col: "" for col in CSV_COLUMNS} | {"image_name": image_name}


def save_results_csv(results, csv_path):
    """
    Write *results* (list of dicts keyed by CSV_COLUMNS) to *csv_path*.

    Rows are sorted by image_name for deterministic output.
    Existing file is overwritten.
    """
    results_sorted = sorted(results, key=lambda r: r["image_name"])
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(results_sorted)
    print(f"\n[CSV] Results saved to: {csv_path}")


# =========================================================================
# PROCESS MAIN PATH
# =========================================================================

def process_main_path(main_path):
    """
    Process all images under three fixed sub-folders of main_path:

      tt/  →  BPD, OFD, HC  (UNet)
      tc/  →  TCD + CM       (UNet)
      tv/  →  LV             (UNet)

    Outputs are saved under:
      <main_path>/output/<subfolder>/mask_overlay/
      <main_path>/output/<subfolder>/measurement_overlay/

    For tc/, TCD and CM each get their own measurement_overlay sub-folder:
      <main_path>/output/tc/tcd_measurement_overlay/
      <main_path>/output/tc/cm_measurement_overlay/

    A single CSV with all measurements is saved to:
      <main_path>/output/measurements.csv
    """

    output_base = os.path.join(main_path, "output")

    # Accumulator: image_name → row dict
    # Using a dict lets tc images receive both tcd and cm in the same row.
    results_map = {}   # key: image_name (basename without extension)

    def _get_row(fname):
        """Return (and lazily create) the results row for an image filename."""
        stem = os.path.splitext(os.path.basename(fname))[0]
        if stem not in results_map:
            results_map[stem] = _empty_row(stem)
        return results_map[stem]

    tt_folder = os.path.join(main_path, "tt")
    tc_folder = os.path.join(main_path, "tc")
    tv_folder = os.path.join(main_path, "tv")

    tt_exists = os.path.isdir(tt_folder)
    tc_exists = os.path.isdir(tc_folder)
    tv_exists = os.path.isdir(tv_folder)

    if not UNET_AVAILABLE:
        print("[ERROR] UNet not available — cannot process any folder.")
        return

    if not (tt_exists or tc_exists or tv_exists):
        print(f"[ERROR] No tt/, tc/, or tv/ sub-folder found under: {main_path}")
        return

    print(f"\n[MODEL] Loading UNet head model ...")
    model_unet = setup_unet_model()

    # ─────────────────────────────────────────────────────────────────
    # TT folder  →  BPD / OFD / HC
    # ─────────────────────────────────────────────────────────────────
    tt_mask_dir = os.path.join(output_base, "tt", "mask_overlay")
    tt_meas_dir = os.path.join(output_base, "tt", "measurement_overlay")
    os.makedirs(tt_mask_dir, exist_ok=True)
    os.makedirs(tt_meas_dir, exist_ok=True)

    if not tt_exists:
        print(f"[TT] Folder not found: {tt_folder}")
    else:
        images_tt = [
            f for f in sorted(os.listdir(tt_folder))
            if f.lower().endswith(ALL_SUPPORTED_EXTS)
        ]
        if not images_tt:
            print(f"[TT] No images found in {tt_folder}")
        else:
            print(f"[TT] Processing {len(images_tt)} image(s) from: {tt_folder}")
            for fname in tqdm(images_tt, desc="[TT] BPD/OFD/HC"):
                img_path = os.path.join(tt_folder, fname)
                tt_result = process_single_image_tt(
                    img_path, model_unet, tt_mask_dir, tt_meas_dir,
                )
                row = _get_row(fname)
                row["modality"] = "tt"
                if tt_result is not None:
                    ofd_val, bpd_val, hc_val, units = tt_result
                    row["ofd"]   = ofd_val
                    row["bpd"]   = bpd_val
                    row["hc"]    = hc_val
                    row["units"] = units

    # ─────────────────────────────────────────────────────────────────
    # TC folder  →  TCD + CM
    # ─────────────────────────────────────────────────────────────────
    if not tc_exists:
        print(f"[TC] Folder not found: {tc_folder}")
    else:
        tc_mask_dir = os.path.join(output_base, "tc", "mask_overlay")
        tc_tcd_meas = os.path.join(output_base, "tc", "tcd_measurement_overlay")
        tc_cm_meas  = os.path.join(output_base, "tc", "cm_measurement_overlay")
        os.makedirs(tc_mask_dir, exist_ok=True)
        os.makedirs(tc_tcd_meas, exist_ok=True)
        os.makedirs(tc_cm_meas,  exist_ok=True)

        images_tc = [
            f for f in sorted(os.listdir(tc_folder))
            if f.lower().endswith(ALL_SUPPORTED_EXTS)
        ]
        if not images_tc:
            print(f"[TC] No images found in {tc_folder}")
        else:
            print(f"[TC] Processing {len(images_tc)} image(s) from: {tc_folder}")
            for fname in tqdm(images_tc, desc="[TC] TCD+CM"):
                img_path = os.path.join(tc_folder, fname)
                tcd_result, cm_result, units = process_single_image_tc(
                    img_path, model_unet
                )
                row = _get_row(fname)
                row["modality"] = "tc"
                row["units"]    = units
                if tcd_result is not None:
                    row["tcd"] = tcd_result
                if cm_result is not None:
                    row["cm"] = cm_result

    # ─────────────────────────────────────────────────────────────────
    # TV folder  →  LV
    # ─────────────────────────────────────────────────────────────────
    if not tv_exists:
        print(f"[TV] Folder not found: {tv_folder}")
    else:
        tv_mask_dir = os.path.join(output_base, "tv", "mask_overlay")
        tv_meas_dir = os.path.join(output_base, "tv", "measurement_overlay")
        os.makedirs(tv_mask_dir, exist_ok=True)
        os.makedirs(tv_meas_dir, exist_ok=True)

        images_tv = [
            f for f in sorted(os.listdir(tv_folder))
            if f.lower().endswith(ALL_SUPPORTED_EXTS)
        ]
        if not images_tv:
            print(f"[TV] No images found in {tv_folder}")
        else:
            print(f"[TV] Processing {len(images_tv)} image(s) from: {tv_folder}")
            for fname in tqdm(images_tv, desc="[TV] LV"):
                img_path = os.path.join(tv_folder, fname)
                lv_result, units = process_single_image_tv(
                    img_path, model_unet
                )
                row = _get_row(fname)
                row["modality"] = "tv"
                row["units"]    = units
                if lv_result is not None:
                    row["lv"] = lv_result

    # ─────────────────────────────────────────────────────────────────
    # Write combined CSV
    # ─────────────────────────────────────────────────────────────────
    if results_map:
        csv_path = os.path.join(output_base, "measurements.csv")
        save_results_csv(list(results_map.values()), csv_path)
    else:
        print("\n[CSV] No measurements collected — CSV not written.")

    print(f"\n[DONE] All outputs saved under: {output_base}")


# =========================================================================
# CLI
# =========================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Fetal biometry measurement tool"
        )
    )

    # ── main-path mode: positional or --main_path ─────────────────────
    parser.add_argument(
        "main_path",
        nargs="?",
        default="/home/htic/Videos/auto_brain_png_in", # /home/htic/Documents/biometry_brain_planes/validate_head
        help=(
            "Root folder containing tt/, tc/, tv/ sub-folders. "
            "All planes use UNet: TT→BPD/OFD/HC, TC→TCD/CM, TV→LV. "
            "Accepts plain images (png/jpg/bmp/tiff) and DICOM (.dcm/.dicom) files; "
            "measurements are saved in mm for DICOM input (via PixelSpacing) "
            "and in raw pixels otherwise."
        ),
    )

    return parser.parse_args()


# =========================================================================
# MAIN
# =========================================================================

def main():

    args = parse_args()

    if not args.main_path:
        sys.exit(
            "[ERROR] Provide the root folder containing tt/, tc/, tv/ sub-folders.\n"
            "Usage: python inf_tt_tc_tv.py <main_path>"
        )

    process_main_path(args.main_path)


if __name__ == "__main__":
    main()


###### python3 meas_brain_planes_5.py --measurement tcd --folder /home/htic/Videos/new_validation_data/tcplane/std_plane --output_dir z4