"""
infer_final_face.py
--------------------
Generic-framework inference for the 14-channel sigmoid Face/Nose/Mouth U-Net++
segmentation model.

MODIFICATIONS:
    - Nose prediction is removed if its cleaned area is < 1000 pixels.
    - Orbit components are filtered relative to the largest orbit component.
      Any orbit component smaller than ORBIT_MIN_RELATIVE_AREA times the largest
      orbit component is removed.
    - Orbits and Lenses are Standard only when EXACTLY 2 orbit components AND
      EXACTLY 2 lens components remain after cleaning.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from skimage.measure import find_contours

from config import CLASS_NAMES_FACE, PALETTE_RGB_FACE
from utils.seg_biom_results_json import write_segmentation_result
from utils.extract_panels import extract_panel

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    _DICOM_AVAILABLE = True
except ImportError:
    _DICOM_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

NUM_CLASSES = 14

STRUCTURE_CONFIDENCE_THRESHOLD = 0.99
PREDICTION_THRESHOLD = 0.45
VISUALIZATION_THRESHOLD = 0.45


CLASS_NAMES = CLASS_NAMES_FACE
assert len(CLASS_NAMES) == NUM_CLASSES

# Model input resolution
MODEL_INPUT_SIZE = (512, 512)


SUPPORTED_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tiff",
    ".tif",
    ".dcm",
    ".dicom",
}

_YBR_PYDICOM_CONVERT = {
    "YBR_ICT",
    "YBR_RCT",
}

_YBR_MANUAL_SWAP = {
    "YBR_FULL",
    "YBR_FULL_422",
}


# ═══════════════════════════════════════════════════════════════════════
# NEW STRUCTURE-SPECIFIC CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════

# Nose must have at least this many pixels after component cleaning.
NOSE_MIN_AREA_PX = 1000

# Orbit component filtering:
#
# The largest orbit component is used as the reference.
#
# Example:
#     largest orbit = 10,000 pixels
#     ORBIT_MIN_RELATIVE_AREA = 0.30
#
# Then another orbit component must have at least:
#
#     10,000 * 0.30 = 3,000 pixels
#
# otherwise it is removed.
ORBIT_MIN_RELATIVE_AREA = 0.30

# Maximum number of components allowed after cleaning.
ORBIT_MAX_COMPONENTS = 2
LENS_MAX_COMPONENTS = 2


# Standard plane constraints
STANDARD_MIN_AREA_PX = {
    "Nose": 1000,
    "Upper Lip": 1500,
    "Lower Lip": 1500,
    "Orbits": 5000,
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════════════
# IMAGE I/O
# ═══════════════════════════════════════════════════════════════════════

def _is_ybr_dicom(path: str) -> bool:
    if not _DICOM_AVAILABLE:
        return False

    try:
        ds = pydicom.dcmread(
            path,
            stop_before_pixels=True
        )

        pi = getattr(
            ds,
            "PhotometricInterpretation",
            ""
        ).upper()

        return pi.startswith("YBR")

    except Exception:
        return False


def _load_dicom_as_rgb_uint8(path: str) -> np.ndarray:

    if not _DICOM_AVAILABLE:
        raise ImportError(
            "pydicom is required for DICOM input. "
            "Install with: pip install pydicom"
        )

    ds = pydicom.dcmread(path)

    pi = getattr(
        ds,
        "PhotometricInterpretation",
        ""
    ).upper()

    if pi in _YBR_PYDICOM_CONVERT:

        arr = ds.pixel_array

        if arr.ndim == 3 and arr.shape[-1] in (3, 4):

            return np.clip(
                arr[..., :3],
                0,
                255
            ).astype(np.uint8)

        arr = np.clip(
            arr,
            0,
            255
        ).astype(np.uint8)

        return cv2.cvtColor(
            arr,
            cv2.COLOR_GRAY2RGB
        )

    arr = apply_voi_lut(
        ds.pixel_array.astype(np.float32),
        ds
    )

    if arr.ndim == 3:

        if arr.shape[-1] in (3, 4):

            if (
                hasattr(ds, "PhotometricInterpretation")
                and pi in _YBR_MANUAL_SWAP
            ):

                lo, hi = arr.min(), arr.max()

                arr = (
                    (arr - lo)
                    / (hi - lo + 1e-8)
                    * 255.0
                ).astype(np.uint8)

                arr = arr[..., [0, 2, 1]]

                return cv2.cvtColor(
                    arr[..., :3],
                    cv2.COLOR_YCrCb2RGB
                )

            lo, hi = arr.min(), arr.max()

            arr = (
                (arr - lo)
                / (hi - lo + 1e-8)
                * 255.0
            ).astype(np.uint8)

            return arr[..., :3]

        elif arr.shape[0] == 1:

            arr = arr[0]

        else:

            arr = arr[arr.shape[0] // 2]

    elif arr.ndim != 2:

        raise ValueError(
            f"Unexpected DICOM pixel_array shape: {arr.shape}"
        )

    if (
        hasattr(ds, "PhotometricInterpretation")
        and ds.PhotometricInterpretation == "MONOCHROME1"
    ):
        arr = arr.max() - arr

    lo, hi = arr.min(), arr.max()

    if hi > lo:

        arr = (
            (arr - lo)
            / (hi - lo)
            * 255.0
        ).astype(np.uint8)

    else:

        arr = np.zeros_like(
            arr,
            dtype=np.uint8
        )

    return cv2.cvtColor(
        arr,
        cv2.COLOR_GRAY2RGB
    )


def load_rgb_image(
    image_path: str
) -> np.ndarray | None:

    suffix = Path(image_path).suffix.lower()

    if suffix in (".dcm", ".dicom"):

        try:

            return _load_dicom_as_rgb_uint8(
                image_path
            )

        except Exception as e:

            logging.warning(
                f"Could not load DICOM "
                f"{Path(image_path).name}: {e}"
            )

            return None

    bgr = cv2.imread(
        image_path,
        cv2.IMREAD_COLOR
    )

    if bgr is None:

        logging.warning(
            f"Could not load image: "
            f"{Path(image_path).name}"
        )

        return None

    return cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB
    )


def load_bgr_image(
    image_path: str
) -> np.ndarray | None:

    rgb = load_rgb_image(image_path)

    return (
        None
        if rgb is None
        else cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR
        )
    )


def discover_images(
    image_dir: str
) -> list[Path]:

    return sorted(
        p
        for p in Path(image_dir).rglob("*")
        if (
            p.is_file()
            and p.suffix.lower() in SUPPORTED_EXTS
        )
    )


def preprocess_for_model(
    rgb_uint8: np.ndarray,
    size: tuple[int, int] = MODEL_INPUT_SIZE
) -> torch.Tensor:

    h, w = size

    resized = cv2.resize(
        rgb_uint8,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    img_f = (
        resized.astype(np.float32)
        / 255.0
    )

    return torch.from_numpy(
        img_f
    ).permute(
        2,
        0,
        1
    ).contiguous()


# ═══════════════════════════════════════════════════════════════════════
# OPTIONAL GT LOADING
# ═══════════════════════════════════════════════════════════════════════

def find_gt_mask_path(
    masks_dir: Path | None,
    stem: str
) -> Path | None:

    if masks_dir is None:
        return None

    candidates = [stem]

    if stem.endswith("_image"):

        candidates.append(
            stem[:-len("_image")]
        )

    else:

        candidates.append(
            stem + "_image"
        )

    for cand in candidates:

        p = masks_dir / f"{cand}.npz"

        if p.exists():
            return p

    return None


def load_gt(
    masks_dir: Path,
    stem: str,
    size: tuple[int, int]
) -> np.ndarray | None:

    npz_path = find_gt_mask_path(
        masks_dir,
        stem
    )

    if npz_path is None:
        return None

    try:

        npz = np.load(
            str(npz_path),
            allow_pickle=False
        )

        if "mask" not in npz.files:

            logging.warning(
                f"NPZ {npz_path.name} has no "
                f"'mask' key "
                f"(found {npz.files}) - skipping GT."
            )

            return None

        raw = npz["mask"]

    except Exception as exc:

        logging.warning(
            f"Cannot load NPZ "
            f"{npz_path.name}: {exc}"
        )

        return None

    if raw.shape[0] != NUM_CLASSES:

        logging.warning(
            f"NPZ {npz_path.name} has "
            f"{raw.shape[0]} channels, "
            f"expected {NUM_CLASSES} - skipping GT."
        )

        return None

    h, w = size

    out = np.zeros(
        (NUM_CLASSES, h, w),
        dtype=np.uint8
    )

    for c in range(NUM_CLASSES):

        out[c] = cv2.resize(
            (raw[c] > 0).astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

    return out


# ═══════════════════════════════════════════════════════════════════════
# MODEL — U-NET++
# ═══════════════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),
        )

    def forward(self, x):
        return self.block(x)


class UNetPlusPlus(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 14
    ):

        super().__init__()

        f = [
            32,
            64,
            128,
            256,
            512
        ]

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.conv0_0 = DoubleConv(
            in_channels,
            f[0]
        )

        self.conv1_0 = DoubleConv(
            f[0],
            f[1]
        )

        self.conv2_0 = DoubleConv(
            f[1],
            f[2]
        )

        self.conv3_0 = DoubleConv(
            f[2],
            f[3]
        )

        self.conv4_0 = DoubleConv(
            f[3],
            f[4]
        )

        self.conv0_1 = DoubleConv(
            f[0] + f[1],
            f[0]
        )

        self.conv1_1 = DoubleConv(
            f[1] + f[2],
            f[1]
        )

        self.conv2_1 = DoubleConv(
            f[2] + f[3],
            f[2]
        )

        self.conv3_1 = DoubleConv(
            f[3] + f[4],
            f[3]
        )

        self.conv0_2 = DoubleConv(
            f[0] * 2 + f[1],
            f[0]
        )

        self.conv1_2 = DoubleConv(
            f[1] * 2 + f[2],
            f[1]
        )

        self.conv2_2 = DoubleConv(
            f[2] * 2 + f[3],
            f[2]
        )

        self.conv0_3 = DoubleConv(
            f[0] * 3 + f[1],
            f[0]
        )

        self.conv1_3 = DoubleConv(
            f[1] * 3 + f[2],
            f[1]
        )

        self.conv0_4 = DoubleConv(
            f[0] * 4 + f[1],
            f[0]
        )

        self.final = nn.Conv2d(
            f[0],
            num_classes,
            kernel_size=1
        )

    @staticmethod
    def upsample(
        x,
        target
    ):

        return F.interpolate(
            x,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

    def forward(self, x):

        x0_0 = self.conv0_0(x)

        x1_0 = self.conv1_0(
            self.pool(x0_0)
        )

        x2_0 = self.conv2_0(
            self.pool(x1_0)
        )

        x3_0 = self.conv3_0(
            self.pool(x2_0)
        )

        x4_0 = self.conv4_0(
            self.pool(x3_0)
        )

        x3_1 = self.conv3_1(
            torch.cat(
                [
                    x3_0,
                    self.upsample(
                        x4_0,
                        x3_0
                    )
                ],
                dim=1
            )
        )

        x2_1 = self.conv2_1(
            torch.cat(
                [
                    x2_0,
                    self.upsample(
                        x3_0,
                        x2_0
                    )
                ],
                dim=1
            )
        )

        x1_1 = self.conv1_1(
            torch.cat(
                [
                    x1_0,
                    self.upsample(
                        x2_0,
                        x1_0
                    )
                ],
                dim=1
            )
        )

        x0_1 = self.conv0_1(
            torch.cat(
                [
                    x0_0,
                    self.upsample(
                        x1_0,
                        x0_0
                    )
                ],
                dim=1
            )
        )

        x2_2 = self.conv2_2(
            torch.cat(
                [
                    x2_0,
                    x2_1,
                    self.upsample(
                        x3_1,
                        x2_0
                    )
                ],
                dim=1
            )
        )

        x1_2 = self.conv1_2(
            torch.cat(
                [
                    x1_0,
                    x1_1,
                    self.upsample(
                        x2_1,
                        x1_0
                    )
                ],
                dim=1
            )
        )

        x0_2 = self.conv0_2(
            torch.cat(
                [
                    x0_0,
                    x0_1,
                    self.upsample(
                        x1_1,
                        x0_0
                    )
                ],
                dim=1
            )
        )

        x1_3 = self.conv1_3(
            torch.cat(
                [
                    x1_0,
                    x1_1,
                    x1_2,
                    self.upsample(
                        x2_2,
                        x1_0
                    )
                ],
                dim=1
            )
        )

        x0_3 = self.conv0_3(
            torch.cat(
                [
                    x0_0,
                    x0_1,
                    x0_2,
                    self.upsample(
                        x1_2,
                        x0_0
                    )
                ],
                dim=1
            )
        )

        x0_4 = self.conv0_4(
            torch.cat(
                [
                    x0_0,
                    x0_1,
                    x0_2,
                    x0_3,
                    self.upsample(
                        x1_3,
                        x0_0
                    )
                ],
                dim=1
            )
        )

        return self.final(x0_4)


# ═══════════════════════════════════════════════════════════════════════
# MODEL LOADING + PREDICTION
# ═══════════════════════════════════════════════════════════════════════

def load_model(
    checkpoint: str,
    device: torch.device
) -> UNetPlusPlus:

    ckpt = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False
    )

    model = UNetPlusPlus(
        in_channels=3,
        num_classes=NUM_CLASSES
    ).to(device)

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    model.eval()

    logging.info(
        f"Loaded checkpoint  "
        f"epoch={ckpt.get('epoch', '?')}  "
        f"val_dice={ckpt.get('val_dice', '?')}"
    )

    return model


@torch.no_grad()
def predict(
    model,
    image: torch.Tensor,
    device: torch.device
):

    probabilities = torch.sigmoid(
        model(
            image.unsqueeze(0).to(device)
        )
    )[0].cpu()

    prediction = torch.zeros_like(
        probabilities,
        dtype=torch.bool
    )

    confidences = np.zeros(
        NUM_CLASSES,
        dtype=np.float32
    )

    for c in range(NUM_CLASSES):

        chan = probabilities[c]

        confidence = float(
            chan.max()
        )

        confidences[c] = confidence

        if (
            confidence
            >= STRUCTURE_CONFIDENCE_THRESHOLD
        ):

            prediction[c] = (
                chan >= PREDICTION_THRESHOLD
            )

    return (
        prediction,
        probabilities,
        confidences
    )


# ═══════════════════════════════════════════════════════════════════════
# PIXEL METRICS
# ═══════════════════════════════════════════════════════════════════════

def _pixel_metrics(
    pred: np.ndarray,
    gt: np.ndarray
) -> dict:

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = int(
        (pred & gt).sum()
    )

    fp = int(
        (pred & ~gt).sum()
    )

    fn = int(
        (~pred & gt).sum()
    )

    tn = int(
        (~pred & ~gt).sum()
    )

    if tp + fp + fn == 0:

        dice = 1.0

    else:

        dice = (
            2.0 * tp
        ) / (
            2.0 * tp
            + fp
            + fn
        )

    precision = (
        tp + 1e-6
    ) / (
        tp
        + fp
        + 1e-6
    )

    recall = (
        tp + 1e-6
    ) / (
        tp
        + fn
        + 1e-6
    )

    specificity = (
        tn + 1e-6
    ) / (
        tn
        + fp
        + 1e-6
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }


def assess_structures(
    prediction: torch.Tensor,
    masks_binary: torch.Tensor | None,
    confidences: np.ndarray,
) -> tuple[list[dict], dict | None]:

    results = []

    for c in range(NUM_CLASSES):

        pred_c = prediction[c].numpy()

        confidence = float(
            confidences[c]
        )

        confidence_pass = (
            confidence
            >= STRUCTURE_CONFIDENCE_THRESHOLD
        )

        predicted_present = bool(
            pred_c.any()
        )

        if masks_binary is None:

            results.append({
                "name": CLASS_NAMES[c],
                "predicted_present": predicted_present,
                "confidence": round(
                    confidence,
                    4
                ),
                "confidence_pass": confidence_pass,
            })

            continue

        gt_c = masks_binary[c].numpy()

        pix = _pixel_metrics(
            pred_c,
            gt_c
        )

        gt_present = bool(
            gt_c.any()
        )

        if gt_present and predicted_present:

            status = "PREDICTED"

        elif gt_present and not predicted_present:

            status = (
                "LOW CONF"
                if not confidence_pass
                else "MISSED"
            )

        elif not gt_present and predicted_present:

            status = "UNEXPECTED"

        else:

            status = "ABSENT"

        results.append({
            "name": CLASS_NAMES[c],
            "gt_present": gt_present,
            "predicted_present": predicted_present,
            "confidence": round(
                confidence,
                4
            ),
            "confidence_pass": confidence_pass,
            "dice": round(
                pix["dice"],
                4
            ),
            "precision": round(
                pix["precision"],
                4
            ),
            "recall": round(
                pix["recall"],
                4
            ),
            "specificity": round(
                pix["specificity"],
                4
            ),
            "tp": pix["tp"],
            "fp": pix["fp"],
            "fn": pix["fn"],
            "tn": pix["tn"],
            "status": status,
        })

    if masks_binary is None:

        return results, None

    n_gt_present = sum(
        r["gt_present"]
        for r in results
    )

    n_predicted = sum(
        r["predicted_present"]
        for r in results
    )

    present_dice = [
        r["dice"]
        for r in results
        if r["gt_present"]
    ]

    summary = {
        "n_structures_gt_present":
            n_gt_present,

        "n_structures_predicted":
            n_predicted,

        "mean_dice_gt_present":
            round(
                float(
                    np.mean(present_dice)
                ),
                4
            )
            if present_dice
            else None,

        "mean_dice_all":
            round(
                float(
                    np.mean(
                        [
                            r["dice"]
                            for r in results
                        ]
                    )
                ),
                4
            ),
    }

    return results, summary


def gt_metrics_rows_to_dict(
    struct_results: list[dict]
) -> dict:

    return {
        r["name"]: {
            "dice": r["dice"],
            "precision": r["precision"],
            "recall": r["recall"],
            "specificity": r["specificity"],
            "gt_present": bool(
                r["gt_present"]
            ),
            "predicted_present": bool(
                r["predicted_present"]
            ),
        }
        for r in struct_results
    }


# ═══════════════════════════════════════════════════════════════════════
# CONFIDENCE
# ═══════════════════════════════════════════════════════════════════════

def compute_mean_structure_confidence(
    struct_results: list[dict]
) -> float:

    vals = [
        r["confidence"]
        for r in struct_results
        if r["predicted_present"]
    ]

    return (
        float(np.mean(vals))
        if vals
        else 0.0
    )


# ═══════════════════════════════════════════════════════════════════════
# POLYGONS
# ═══════════════════════════════════════════════════════════════════════

def mask_to_polygons(
    mask: np.ndarray,
    min_pixels: int = 20,
    epsilon_fraction: float = 0.01,
) -> list[list[dict]]:

    if mask.max() == 0:
        return []

    polys = []

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:

        if cv2.contourArea(cnt) < min_pixels:
            continue

        perimeter = cv2.arcLength(
            cnt,
            True
        )

        if perimeter == 0:

            pts = cnt.reshape(
                -1,
                2
            )

        else:

            epsilon = (
                epsilon_fraction
                * perimeter
            )

            approx = cv2.approxPolyDP(
                cnt,
                epsilon,
                True
            )

            pts = approx.reshape(
                -1,
                2
            )

            if len(pts) < 3:

                pts = cnt.reshape(
                    -1,
                    2
                )

        polys.append([
            {
                "x": float(x),
                "y": float(y)
            }
            for x, y in pts
        ])

    return polys


def build_polygons_from_final_mask(
    prediction: torch.Tensor,
    orig_h: int,
    orig_w: int
) -> list[dict]:

    max_occurrences = {
        "Orbits": 2,
        "Lenses": 2
    }

    shapes = []

    for c in range(NUM_CLASSES):

        structure_name = CLASS_NAMES[c]

        mask = prediction[c].numpy().astype(
            np.uint8
        )

        if mask.max() == 0:
            continue

        mask_orig = cv2.resize(
            mask,
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST
        )

        polygons = mask_to_polygons(
            mask_orig
        )

        if not polygons:
            continue

        max_allowed = max_occurrences.get(
            structure_name,
            1
        )

        if len(polygons) > max_allowed:

            def polygon_area(poly):

                pts = np.array(
                    [
                        [p["x"], p["y"]]
                        for p in poly
                    ],
                    dtype=np.float32
                )

                return abs(
                    cv2.contourArea(pts)
                )

            polygons = sorted(
                polygons,
                key=polygon_area,
                reverse=True
            )[:max_allowed]

        for poly_pts in polygons:

            shapes.append({
                "points": poly_pts,
                "structure": structure_name,
                "type": "polygon"
            })

    return shapes


# ═══════════════════════════════════════════════════════════════════════
# PLANE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════

def determine_predicted_plane(
    structures: list[dict]
) -> dict:

    present = {
        s["name"]
        for s in structures
        if s["predicted_present"]
    }

    print(present)

    # Orbits + Lenses
    if (
        "Orbits" in present
        and "Lenses" in present
    ):
        return {
            "predicted_plane": "Orbits and Lenses",
            "plane_info": {
                "predicted_plane": "Orbits and Lenses",
                "plane_standard": "Standard",
            },
        }

    # Nose and Mouth
    mouth_structures = {
        "Nose",
        "Upper Lip",
        "Lower Lip",
    }

    if present.intersection(mouth_structures):

        if (
            "Upper Lip" in present
            or "Lower Lip" in present
        ):
            return {
                "predicted_plane": "Nose and Mouth",
                "plane_info": {
                    "predicted_plane": "Nose and Mouth",
                    "plane_standard": "Standard",
                },
            }

    # Median Facial Profile
    median_structures = {
        "Forehead",
        "Diencephalon",
        "Maxilla",
        "Nasal_Bone",
        "Skinline",
        "Mandible",
        "Chin",
        "Vomer_bone",
        "Lips",
    }

    median_present = present.intersection(
        median_structures
    )

    median_without_chin = (
        median_structures - {"Chin"}
    )

    n_other_median = len(
        present.intersection(
            median_without_chin
        )
    )

    # Explicit condition for Median Facial Profile
    median_plane_condition = (
        (
            "Nose" in present
            and bool(median_present)
        )
        or
        (
            "Chin" in present
            and n_other_median >= 2
        )
    )

    if median_plane_condition:

        return {
            "predicted_plane": "Median Facial Profile",

            "plane_info": {
                "predicted_plane":
                    "Median Facial Profile",

                "plane_standard":
                    (
                        "Standard"
                        if "Diencephalon" in present
                        else "Non-Standard"
                    ),
            },
        }

    # Nose alone
    if "Nose" in present:

        return {
            "predicted_plane": "Nose and Mouth",

            "plane_info": {
                "predicted_plane": "Nose and Mouth",
                "plane_standard": "Standard",
            },
        }

    median_present = present.intersection(
        median_structures
    )

    if median_present:

        return {
            "predicted_plane":
                "Median Facial Profile",

            "plane_info": {
                "predicted_plane":
                    "Median Facial Profile",

                "plane_standard":
                    (
                        "Standard"
                        if "Diencephalon" in present
                        else "Non-Standard"
                    ),
            },
        }

    return {
        "predicted_plane": "Unknown",
        "plane_info": {
            "predicted_plane": "Unknown",
            "plane_standard": "Unknown",
        },
    }

def get_present_structures(
    structures: list[dict]
) -> set[str]:

    return {
        s["name"]
        for s in structures
        if s["predicted_present"]
    }


# ═══════════════════════════════════════════════════════════════════════
# PLANE MANDATORY STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

PLANE_MANDATORY_STRUCTURES = {
    "Nose and Mouth": [
        "Nose",
        "Upper Lip",
        "Lower Lip",
        "Chin"
    ],

    "Orbits and Lenses": [
        "Orbits",
        "Lenses"
    ],

    "Median Facial Profile": [
        "Diencephalon",
        "Forehead",
        "Maxilla",  
        "Nasal_Bone",
        "Skinline",
        "Mandible",
        "Chin",
        "Vomer_bone",
        "Lips"
    ],
}


PLANE_ANCHOR_STRUCTURES = {
    "Nose and Mouth": [
        "Nose"
    ],

    "Orbits and Lenses": [
        "Orbits",
        "Lenses"
    ],

    "Median Facial Profile": [
        "Diencephalon"
    ],
}


PLANE_DETECT_THRESHOLD = 0.50


def build_plane_mandatory_structures(
    present: set[str]
) -> dict:

    out = {}

    for plane, required in (
        PLANE_MANDATORY_STRUCTURES.items()
    ):

        present_structs = [
            s
            for s in required
            if s in present
        ]

        missing_structs = [
            s
            for s in required
            if s not in present
        ]

        anchors = (
            PLANE_ANCHOR_STRUCTURES.get(
                plane,
                []
            )
        )

        out[plane] = {

            "required":
                required,

            "present":
                present_structs,

            "missing":
                missing_structs,

            "detected_fraction":
                round(
                    len(present_structs)
                    / len(required),
                    6
                ),

            "anchor_pass":
                all(
                    a in present
                    for a in anchors
                ),
        }

    return out


def build_plane_candidates(
    mandatory_block: dict,
    winning_plane: str
) -> list[str]:

    candidates = [
        p
        for p, info
        in mandatory_block.items()
        if info["detected_fraction"]
        >= PLANE_DETECT_THRESHOLD
    ]

    if winning_plane not in candidates:
        candidates.append(
            winning_plane
        )

    candidates.sort(
        key=lambda p:
            mandatory_block.get(
                p,
                {}
            ).get(
                "detected_fraction",
                0.0
            ),
        reverse=True
    )

    return candidates


# ═══════════════════════════════════════════════════════════════════════
# COMPONENT CLEANING
# ═══════════════════════════════════════════════════════════════════════

def keep_largest_components(
    mask: np.ndarray,
    max_components: int = 1,
    min_pixels: int = 20,
    min_relative_area: float = 0.05,
) -> np.ndarray:

    mask = (
        mask > 0
    ).astype(np.uint8)

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    components = []

    for label in range(
        1,
        num_labels
    ):

        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA
            ]
        )

        if area >= min_pixels:

            components.append(
                (
                    label,
                    area
                )
            )

    if not components:

        return np.zeros_like(mask)

    components.sort(
        key=lambda x: x[1],
        reverse=True
    )

    largest_area = (
        components[0][1]
    )

    components = [
        (label, area)
        for label, area
        in components
        if area
        >= largest_area
        * min_relative_area
    ]

    components = components[
        :max_components
    ]

    cleaned = np.zeros_like(
        mask
    )

    for label, _ in components:

        cleaned[
            labels == label
        ] = 1

    return cleaned


# ═══════════════════════════════════════════════════════════════════════
# NEW: ORBIT-SPECIFIC COMPONENT CLEANING
# ═══════════════════════════════════════════════════════════════════════

def clean_orbit_components(
    mask: np.ndarray,
    min_pixels: int = 20,
    min_relative_area: float = ORBIT_MIN_RELATIVE_AREA,
    max_components: int = ORBIT_MAX_COMPONENTS,
) -> np.ndarray:
    """
    Clean Orbit connected components.

    The largest orbit component is used as the reference.

    Any other component must satisfy:

        component_area >=
        largest_component_area * min_relative_area

    Otherwise it is dropped.

    At most max_components are retained.
    """

    mask = (
        mask > 0
    ).astype(np.uint8)

    n_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    components = []

    for label in range(
        1,
        n_labels
    ):

        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA
            ]
        )

        if area >= min_pixels:

            components.append(
                (
                    label,
                    area
                )
            )

    if not components:

        return np.zeros_like(mask)

    # Largest first
    components.sort(
        key=lambda x: x[1],
        reverse=True
    )

    largest_label, largest_area = (
        components[0]
    )

    kept = []

    # Largest component is always kept.
    kept.append(
        (
            largest_label,
            largest_area
        )
    )

    # Check remaining components against
    # the largest component.
    for label, area in components[1:]:

        relative_area = (
            area
            / float(largest_area)
        )

        if (
            relative_area
            >= min_relative_area
        ):

            kept.append(
                (
                    label,
                    area
                )
            )

            logging.info(
                f"    [ORBIT AREA CLEAN] "
                f"keeping component: "
                f"{area}px "
                f"({relative_area:.3f} "
                f"of largest)"
            )

        else:

            logging.info(
                f"    [ORBIT AREA CLEAN] "
                f"dropping component: "
                f"{area}px "
                f"({relative_area:.3f} "
                f"of largest; "
                f"required="
                f"{min_relative_area:.3f})"
            )

    # Keep at most two orbit components.
    kept = kept[
        :max_components
    ]

    cleaned = np.zeros_like(
        mask
    )

    for label, _ in kept:

        cleaned[
            labels == label
        ] = 1

    return cleaned


# ═══════════════════════════════════════════════════════════════════════
# CENTROID / SPATIAL CLEANING
# ═══════════════════════════════════════════════════════════════════════

def get_structure_centroid(
    mask: np.ndarray,
) -> tuple[float, float] | None:

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:
        return None

    return (
        float(xs.mean()),
        float(ys.mean())
    )


def remove_spatially_isolated_structures(
    prediction: torch.Tensor,
    min_pixels: int = 20,
    max_distance_fraction: float = 0.25,
    min_cluster_size: int = 2,
) -> torch.Tensor:

    cleaned = prediction.clone()

    h, w = prediction.shape[-2:]

    diagonal = float(
        np.sqrt(
            h * h
            + w * w
        )
    )

    max_distance = (
        diagonal
        * max_distance_fraction
    )

    centroids = {}

    for c in range(NUM_CLASSES):

        mask = (
            prediction[c]
            .numpy()
            .astype(np.uint8)
        )

        if mask.sum() < min_pixels:
            continue

        centroid = get_structure_centroid(
            mask
        )

        if centroid is not None:

            centroids[c] = centroid

    if len(centroids) <= 1:
        return cleaned

    neighbours = {
        c: set()
        for c in centroids
    }

    channels = list(
        centroids.keys()
    )

    for i, c1 in enumerate(
        channels
    ):

        x1, y1 = centroids[c1]

        for c2 in channels[
            i + 1:
        ]:

            x2, y2 = centroids[c2]

            distance = np.sqrt(
                (x1 - x2) ** 2
                + (y1 - y2) ** 2
            )

            if distance <= max_distance:

                neighbours[c1].add(
                    c2
                )

                neighbours[c2].add(
                    c1
                )

    clusters = []
    visited = set()

    for c in channels:

        if c in visited:
            continue

        stack = [c]
        cluster = []

        while stack:

            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            cluster.append(current)

            for neighbour in (
                neighbours[current]
            ):

                if neighbour not in visited:

                    stack.append(
                        neighbour
                    )

        clusters.append(
            cluster
        )

    large_clusters = [
        cluster
        for cluster in clusters
        if len(cluster)
        >= min_cluster_size
    ]

    if not large_clusters:
        return cleaned

    main_cluster = max(
        large_clusters,
        key=len
    )

    main_cluster = set(
        main_cluster
    )

    for cluster in clusters:

        cluster_set = set(
            cluster
        )

        if cluster_set == main_cluster:
            continue

        for c in cluster_set:

            logging.info(
                f"    [SPATIAL CLEAN] "
                f"removing isolated "
                f"{CLASS_NAMES[c]} structure"
            )

            cleaned[c] = False

    return cleaned


# ═══════════════════════════════════════════════════════════════════════
# MODIFIED: COMPLETE PREDICTION MASK CLEANING
# ═══════════════════════════════════════════════════════════════════════

def clean_prediction_mask(
    prediction: torch.Tensor,
    min_pixels: int = 20,
) -> torch.Tensor:

    cleaned = torch.zeros_like(
        prediction,
        dtype=torch.bool
    )

    for c in range(NUM_CLASSES):

        structure_name = (
            CLASS_NAMES[c]
        )

        mask = (
            prediction[c]
            .numpy()
            .astype(np.uint8)
        )

        # =========================================================
        # ORBITS
        # =========================================================

        if structure_name == "Orbits":

            mask_clean = clean_orbit_components(
                mask,
                min_pixels=min_pixels,
                min_relative_area=ORBIT_MIN_RELATIVE_AREA,
                max_components=ORBIT_MAX_COMPONENTS,
            )

        # =========================================================
        # LENSES
        # =========================================================

        elif structure_name == "Lenses":

            mask_clean = keep_largest_components(
                mask,
                max_components=LENS_MAX_COMPONENTS,
                min_pixels=min_pixels,
                min_relative_area=0.05,
            )

        # =========================================================
        # NOSE
        # =========================================================

        elif structure_name == "Nose":

            mask_clean = keep_largest_components(
                mask,
                max_components=1,
                min_pixels=min_pixels,
                min_relative_area=0.05,
            )

            nose_area = int(
                mask_clean.sum()
            )

            if nose_area < NOSE_MIN_AREA_PX:

                logging.info(
                    f"    [NOSE AREA CLEAN] "
                    f"removing Nose: "
                    f"area={nose_area}px "
                    f"< minimum="
                    f"{NOSE_MIN_AREA_PX}px"
                )

                mask_clean = np.zeros_like(
                    mask_clean
                )

        # =========================================================
        # OTHER STRUCTURES
        # =========================================================

        else:

            mask_clean = keep_largest_components(
                mask,
                max_components=1,
                min_pixels=min_pixels,
                min_relative_area=0.05,
            )

        cleaned[c] = torch.from_numpy(
            mask_clean.astype(bool)
        )

    return cleaned


# ═══════════════════════════════════════════════════════════════════════
# PLANE-CONSISTENCY VISUALIZATION FILTER
# ═══════════════════════════════════════════════════════════════════════

def clean_mask_for_predicted_plane(
    prediction: torch.Tensor,
    plane_info: dict,
) -> torch.Tensor:

    plane = plane_info[
        "predicted_plane"
    ]

    allowed_structures = {

        "Nose and Mouth": {
            "Nose",
            "Upper Lip",
            "Lower Lip",
        },

        "Orbits and Lenses": {
            "Orbits",
            "Lenses",
        },

        "Median Facial Profile": {
            "Forehead",
            "Diencephalon",
            "Maxilla",
            "Nasal_Bone",
            "Skinline",
            "Mandible",
            "Chin",
            "Vomer_bone",
        },

        "Unknown": set(),
    }

    allowed = allowed_structures.get(
        plane,
        set()
    )

    cleaned = torch.zeros_like(
        prediction,
        dtype=torch.bool
    )

    for c, name in enumerate(
        CLASS_NAMES
    ):

        if name in allowed:

            cleaned[c] = prediction[c]

    return cleaned


# ═══════════════════════════════════════════════════════════════════════
# COMPONENT COUNTING
# ═══════════════════════════════════════════════════════════════════════

def _count_components(
    mask: np.ndarray,
    min_pixels: int = 1
) -> int:

    mask_u8 = (
        mask > 0
    ).astype(np.uint8)

    n_labels, _, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask_u8,
            connectivity=8
        )
    )

    return sum(
        1
        for lbl in range(
            1,
            n_labels
        )
        if stats[
            lbl,
            cv2.CC_STAT_AREA
        ] >= min_pixels
    )


# ═══════════════════════════════════════════════════════════════════════
# MODIFIED: PLANE QUALITY
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
# PLANE QUALITY
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# COMPONENT COUNTING
# ═══════════════════════════════════════════════════════════════════════

def get_component_masks(
    mask: np.ndarray,
    min_pixels: int = 1,
) -> list[np.ndarray]:

    """
    Return individual connected components as binary masks.
    Components smaller than min_pixels are ignored.
    """

    mask_u8 = (
        mask > 0
    ).astype(np.uint8)

    n_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask_u8,
            connectivity=8
        )
    )

    components = []

    for label in range(
        1,
        n_labels
    ):

        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA
            ]
        )

        if area >= min_pixels:

            component = (
                labels == label
            )

            components.append(
                component
            )

    return components


def _count_components(
    mask: np.ndarray,
    min_pixels: int = 1
) -> int:

    return len(
        get_component_masks(
            mask,
            min_pixels=min_pixels
        )
    )


# ═══════════════════════════════════════════════════════════════════════
# LENS-IN-ORBIT CHECK
# ═══════════════════════════════════════════════════════════════════════

def check_lenses_inside_orbits(
    orbit_mask: np.ndarray,
    lens_mask: np.ndarray,
) -> tuple[bool, int, int]:

    """
    Check whether every lens component is contained inside
    an orbit component.

    Returns:

        all_inside:
            True only if every lens component is inside an orbit.

        n_lenses_inside:
            Number of lens components that are inside an orbit.

        n_lenses:
            Total number of lens components.
    """

    orbit_components = get_component_masks(
        orbit_mask,
        min_pixels=1
    )

    lens_components = get_component_masks(
        lens_mask,
        min_pixels=1
    )

    n_lenses = len(
        lens_components
    )

    if n_lenses == 0:

        return (
            False,
            0,
            0
        )

    if len(orbit_components) == 0:

        return (
            False,
            0,
            n_lenses
        )

    n_inside = 0

    for lens_idx, lens_component in enumerate(
        lens_components,
        start=1
    ):

        # A lens is considered inside an orbit if
        # ALL pixels of that lens component are
        # contained within the union of the orbit
        # components.
        inside_any_orbit = False

        for orbit_idx, orbit_component in enumerate(
            orbit_components,
            start=1
        ):

            lens_pixels = (
                lens_component
            )

            if np.all(
                orbit_component[
                    lens_pixels
                ]
            ):

                inside_any_orbit = True

                logging.info(
                    f"    [LENS/ORBIT CHECK] "
                    f"Lens {lens_idx} is inside "
                    f"Orbit {orbit_idx}"
                )

                break

        if inside_any_orbit:

            n_inside += 1

        else:

            logging.info(
                f"    [LENS/ORBIT CHECK] "
                f"Lens {lens_idx} is NOT inside "
                f"any orbit"
            )

    all_inside = (
        n_inside == n_lenses
    )

    return (
        all_inside,
        n_inside,
        n_lenses
    )


# ═══════════════════════════════════════════════════════════════════════
# PLANE QUALITY
# ═══════════════════════════════════════════════════════════════════════

def refine_plane_quality(
    plane_info: dict,
    cleaned_prediction: torch.Tensor
) -> dict:

    """
    Re-evaluates plane_standard against the CLEANED mask.

    For Orbits and Lenses, the plane is Standard ONLY when:

        1. Exactly 2 orbit components exist.
        2. Exactly 2 lens components exist.
        3. Every lens component is completely inside an orbit.

    The predicted plane itself is NEVER changed.
    """

    plane = plane_info[
        "predicted_plane"
    ]

    was_standard = (
        plane_info[
            "plane_info"
        ][
            "plane_standard"
        ]
        == "Standard"
    )

    updated = {
        "predicted_plane":
            plane,

        "plane_info":
            dict(
                plane_info[
                    "plane_info"
                ]
            ),
    }

    # =========================================================
    # NOSE AND MOUTH
    # =========================================================

    if plane == "Nose and Mouth":

        big_enough = True

        for name, min_area in (
            STANDARD_MIN_AREA_PX.items()
        ):

            c = CLASS_NAMES.index(
                name
            )

            area = int(
                cleaned_prediction[c]
                .sum()
                .item()
            )

            if (
                0 < area
                < min_area
            ):

                big_enough = False

        updated[
            "plane_info"
        ][
            "plane_standard"
        ] = (
            "Standard"
            if (
                was_standard
                and big_enough
            )
            else "Non-Standard"
        )

    # =========================================================
    # ORBITS AND LENSES
    # =========================================================

    elif plane == "Orbits and Lenses":

        orbit_mask = (
            cleaned_prediction[
                CLASS_NAMES.index(
                    "Orbits"
                )
            ]
            .numpy()
        )

        lens_mask = (
            cleaned_prediction[
                CLASS_NAMES.index(
                    "Lenses"
                )
            ]
            .numpy()
        )

        # -----------------------------------------------------
        # Count actual connected components
        # -----------------------------------------------------

        n_orbits = _count_components(
            orbit_mask,
            min_pixels=1
        )

        n_lenses = _count_components(
            lens_mask,
            min_pixels=1
        )

        logging.info(
            f"    [PLANE CHECK] "
            f"Orbit components = {n_orbits}"
        )

        logging.info(
            f"    [PLANE CHECK] "
            f"Lens components  = {n_lenses}"
        )

        # -----------------------------------------------------
        # EXACTLY TWO + EXACTLY TWO
        # -----------------------------------------------------

        exact_orbits = (
            n_orbits == 2
        )

        exact_lenses = (
            n_lenses == 2
        )

        exact_pair = (
            exact_orbits
            and exact_lenses
        )

        # -----------------------------------------------------
        # LENS MUST BE INSIDE ORBIT
        #
        # Only perform containment check when we have
        # exactly 2 + 2. This makes the logic explicit.
        # -----------------------------------------------------

        (
            lenses_inside_orbits,
            n_lenses_inside,
            n_lenses_checked,
        ) = check_lenses_inside_orbits(
            orbit_mask,
            lens_mask
        )

        # -----------------------------------------------------
        # FINAL STANDARD DECISION
        # -----------------------------------------------------

        standard_component_check = (
            exact_pair
            and lenses_inside_orbits
        )

        final_standard = (
            was_standard
            and standard_component_check
        )

        updated[
            "plane_info"
        ][
            "plane_standard"
        ] = (
            "Standard"
            if final_standard
            else "Non-Standard"
        )

        # =====================================================
        # STORE EVERYTHING IN JSON
        # =====================================================

        updated[
            "plane_info"
        ][
            "orbit_count"
        ] = n_orbits

        updated[
            "plane_info"
        ][
            "lens_count"
        ] = n_lenses

        updated[
            "plane_info"
        ][
            "orbit_requirement"
        ] = 2

        updated[
            "plane_info"
        ][
            "lens_requirement"
        ] = 2

        updated[
            "plane_info"
        ][
            "exact_orbit_count"
        ] = exact_orbits

        updated[
            "plane_info"
        ][
            "exact_lens_count"
        ] = exact_lenses

        updated[
            "plane_info"
        ][
            "lenses_inside_orbits"
        ] = lenses_inside_orbits

        updated[
            "plane_info"
        ][
            "n_lenses_inside_orbits"
        ] = n_lenses_inside

        updated[
            "plane_info"
        ][
            "n_lenses_checked"
        ] = n_lenses_checked

        updated[
            "plane_info"
        ][
            "standard_component_check"
        ] = standard_component_check

        logging.info(
            f"    [PLANE CHECK] "
            f"Exact 2 orbits: "
            f"{exact_orbits}"
        )

        logging.info(
            f"    [PLANE CHECK] "
            f"Exact 2 lenses: "
            f"{exact_lenses}"
        )

        logging.info(
            f"    [PLANE CHECK] "
            f"All lenses inside orbits: "
            f"{lenses_inside_orbits}"
        )

        logging.info(
            f"    [PLANE CHECK] "
            f"FINAL: "
            f"{'STANDARD' if final_standard else 'NON-STANDARD'}"
        )

    return updated


# def refine_plane_quality(
#     plane_info: dict,
#     cleaned_prediction: torch.Tensor
# ) -> dict:

#     """
#     Re-evaluates plane_standard using the FINAL CLEANED mask.

#     For Orbits and Lenses:

#         exactly 2 Orbit components
#         AND
#         exactly 2 Lens components

#     are required for Standard.

#     IMPORTANT:
#     Components are NOT hidden when the count is wrong.

#     Example:

#         1 Orbit + 1 Lens
#             -> Non-Standard

#         1 Orbit + 2 Lenses
#             -> Non-Standard

#         2 Orbits + 1 Lens
#             -> Non-Standard

#         2 Orbits + 2 Lenses
#             -> Standard

#     The predicted plane itself is never changed.
#     """

#     plane = plane_info[
#         "predicted_plane"
#     ]

#     updated = {
#         "predicted_plane":
#             plane,

#         "plane_info":
#             dict(
#                 plane_info[
#                     "plane_info"
#                 ]
#             ),
#     }

#     # =========================================================
#     # NOSE AND MOUTH
#     # =========================================================

#     if plane == "Nose and Mouth":

#         big_enough = True

#         for name, min_area in (
#             STANDARD_MIN_AREA_PX.items()
#         ):

#             c = CLASS_NAMES.index(
#                 name
#             )

#             area = int(
#                 cleaned_prediction[c]
#                 .sum()
#                 .item()
#             )

#             if (
#                 0 < area
#                 < min_area
#             ):

#                 big_enough = False

#         updated[
#             "plane_info"
#         ][
#             "plane_standard"
#         ] = (
#             "Standard"
#             if big_enough
#             else "Non-Standard"
#         )

#     # =========================================================
#     # ORBITS AND LENSES
#     # =========================================================

#     elif plane == "Orbits and Lenses":

#         orbit_idx = CLASS_NAMES.index(
#             "Orbits"
#         )

#         lens_idx = CLASS_NAMES.index(
#             "Lenses"
#         )

#         orbit_mask = (
#             cleaned_prediction[
#                 orbit_idx
#             ]
#             .numpy()
#         )

#         lens_mask = (
#             cleaned_prediction[
#                 lens_idx
#             ]
#             .numpy()
#         )

#         # -----------------------------------------------------
#         # Count FINAL cleaned components
#         # -----------------------------------------------------

#         n_orbits = _count_components(
#             orbit_mask,
#             min_pixels=1
#         )

#         n_lenses = _count_components(
#             lens_mask,
#             min_pixels=1
#         )

#         # -----------------------------------------------------
#         # EXACTLY 2 + EXACTLY 2
#         # -----------------------------------------------------

#         exact_orbits = (
#             n_orbits == 2
#         )

#         exact_lenses = (
#             n_lenses == 2
#         )

#         exact_pair = (
#             exact_orbits
#             and exact_lenses
#         )

#         # -----------------------------------------------------
#         # IMPORTANT:
#         # Do NOT use was_standard here.
#         #
#         # The final cleaned component count determines
#         # Standard / Non-Standard directly.
#         # -----------------------------------------------------

#         updated[
#             "plane_info"
#         ][
#             "plane_standard"
#         ] = (
#             "Standard"
#             if exact_pair
#             else "Non-Standard"
#         )

#         # =====================================================
#         # STORE COMPONENT COUNTS
#         # =====================================================

#         updated[
#             "plane_info"
#         ][
#             "orbit_count"
#         ] = n_orbits

#         updated[
#             "plane_info"
#         ][
#             "lens_count"
#         ] = n_lenses

#         updated[
#             "plane_info"
#         ][
#             "orbit_requirement"
#         ] = 2

#         updated[
#             "plane_info"
#         ][
#             "lens_requirement"
#         ] = 2

#         updated[
#             "plane_info"
#         ][
#             "orbit_count_valid"
#         ] = exact_orbits

#         updated[
#             "plane_info"
#         ][
#             "lens_count_valid"
#         ] = exact_lenses

#         updated[
#             "plane_info"
#         ][
#             "standard_component_check"
#         ] = exact_pair

#         # =====================================================
#         # LOG RESULT
#         # =====================================================

#         logging.info(
#             f"    [PLANE COMPONENT CHECK] "
#             f"Orbits={n_orbits}/2, "
#             f"Lenses={n_lenses}/2 -> "
#             f"{'STANDARD' if exact_pair else 'NON-STANDARD'}"
#         )

#     return updated

# ═══════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════

# _CMAP = plt.colormaps.get_cmap(
#     "tab20"
# )

# _CLASS_COLORS = [
#     _CMAP(i / NUM_CLASSES)
#     for i in range(NUM_CLASSES)
# ]
_CLASS_COLORS = [
    tuple(rgb / 255.0)
    for rgb in PALETTE_RGB_FACE
]

# def make_visualization(
#     rgb_orig: np.ndarray,
#     prediction: torch.Tensor,
#     masks_binary: torch.Tensor | None,
#     struct_results: list[dict],
#     plane_info: dict,
#     patient_id: str,
#     view: str,
#     out_path: Path,
#     suspicious_mask: torch.Tensor | None = None,
# ) -> Path:

#     image_np = np.clip(
#         rgb_orig.astype(
#             np.float32
#         ) / 255.0,
#         0,
#         1
#     )

#     h, w = image_np.shape[:2]

#     show_gt = (
#         masks_binary is not None
#     )

#     if show_gt:

#         fig, (
#             gt_ax,
#             pred_ax
#         ) = plt.subplots(
#             1,
#             2,
#             figsize=(18, 9)
#         )

#         axes = [
#             (
#                 gt_ax,
#                 "Ground Truth"
#             ),
#             (
#                 pred_ax,
#                 "Prediction"
#             ),
#         ]

#     else:

#         fig, pred_ax = plt.subplots(
#             1,
#             1,
#             figsize=(12, 10)
#         )

#         axes = [
#             (
#                 pred_ax,
#                 "Prediction"
#             )
#         ]

#     for ax, title in axes:

#         ax.imshow(
#             image_np
#         )

#         ax.set_title(
#             title,
#             fontsize=15,
#             fontweight="bold"
#         )

#         ax.axis("off")

#     legend_handles = []

#     for c in range(NUM_CLASSES):

#         pred_mask_full = cv2.resize(
#             prediction[c]
#             .numpy()
#             .astype(np.uint8),
#             (w, h),
#             interpolation=cv2.INTER_NEAREST
#         ) > 0

#         if suspicious_mask is not None:

#             suspicious_c = cv2.resize(
#                 suspicious_mask[c]
#                 .numpy()
#                 .astype(np.uint8),
#                 (w, h),
#                 interpolation=cv2.INTER_NEAREST
#             ) > 0

#             pred_mask = (
#                 pred_mask_full
#                 & ~suspicious_c
#             )

#         else:

#             pred_mask = pred_mask_full

#         color = _CLASS_COLORS[c]

#         if pred_mask.any():

#             for contour in find_contours(
#                 pred_mask.astype(np.uint8),
#                 level=0.5
#             ):

#                 pred_ax.plot(
#                     contour[:, 1],
#                     contour[:, 0],
#                     color=color,
#                     linewidth=4.0
#                 )

#         if show_gt:

#             gt_mask = cv2.resize(
#                 masks_binary[c]
#                 .numpy()
#                 .astype(np.uint8),
#                 (w, h),
#                 interpolation=cv2.INTER_NEAREST
#             ) > 0

#             if gt_mask.any():

#                 for contour in find_contours(
#                     gt_mask.astype(np.uint8),
#                     level=0.5
#                 ):

#                     gt_ax.plot(
#                         contour[:, 1],
#                         contour[:, 0],
#                         color=color,
#                         linewidth=4.0
#                     )

#         r = struct_results[c]

#         is_suspicious = bool(
#             (
#                 pred_mask_full
#                 & ~pred_mask
#             ).any()
#         ) if suspicious_mask is not None else False

#         suspicious_tag = (
#             " | SUSPICIOUS (hidden)"
#             if is_suspicious
#             else ""
#         )

#         if show_gt:

#             label = (
#                 f"{r['name']} | "
#                 f"Conf: {r['confidence']:.3f} | "
#                 f"Conf "
#                 f"{'PASS' if r['confidence_pass'] else 'REJECT'} | "
#                 f"Dice: {r['dice']:.3f} | "
#                 f"{r['status']}"
#                 f"{suspicious_tag}"
#             )

#         else:

#             label = (
#                 f"{r['name']} | "
#                 f"Conf: {r['confidence']:.3f} | "
#                 f"{'PRESENT' if r['predicted_present'] else 'ABSENT'}"
#                 f"{suspicious_tag}"
#             )

#         if (
#             show_gt
#             or r["predicted_present"]
#         ):

#             legend_handles.append(
#                 Line2D(
#                     [0],
#                     [0],
#                     color=color,
#                     linewidth=5,
#                     label=label
#                 )
#             )

#     # plane_title = (
#     #     f"Predicted Plane: "
#     #     f"{plane_info['predicted_plane']} | "
#     #     f"{plane_info['plane_info']['plane_standard']}"
#     # )

#     plane_name = (
#     plane_info[
#         "predicted_plane"
#     ]
# )

#     plane_quality = (
#         plane_info[
#             "plane_info"
#         ][
#             "plane_standard"
#         ]
#     )

#     if plane_name == "Orbits and Lenses":

#         orbit_count = (
#             plane_info[
#                 "plane_info"
#             ].get(
#                 "orbit_count",
#                 0
#             )
#         )

#         lens_count = (
#             plane_info[
#                 "plane_info"
#             ].get(
#                 "lens_count",
#                 0
#             )
#         )

#         plane_title = (
#             f"Predicted Plane: "
#             f"{plane_name} | "
#             f"{plane_quality} | "
#             f"Orbits: {orbit_count}/2 | "
#             f"Lenses: {lens_count}/2"
#         )

#     else:

#         plane_title = (
#             f"Predicted Plane: "
#             f"{plane_name} | "
#             f"{plane_quality}"
#         )


#     fig.legend(
#         handles=legend_handles,
#         loc="center left",
#         bbox_to_anchor=(0.89, 0.5),
#         fontsize=11,
#         frameon=True,
#         title=(
#             f"Structures\n"
#             f"{plane_title}\n"
#             f"Conf threshold = "
#             f"{STRUCTURE_CONFIDENCE_THRESHOLD:.2f}\n"
#             f"Pixel threshold = "
#             f"{VISUALIZATION_THRESHOLD:.2f}"
#         ),
#         title_fontsize=12,
#     )

#     fig.suptitle(
#         f"Patient: {patient_id} | "
#         f"View: {view}\n"
#         f"{plane_title}",
#         fontsize=16,
#         fontweight="bold"
#     )

#     plt.tight_layout(
#         rect=[
#             0,
#             0,
#             0.87,
#             0.93
#         ]
#     )

#     plt.savefig(
#         out_path,
#         dpi=200,
#         bbox_inches="tight"
#     )

#     plt.close(fig)

#     return out_path



def make_visualization(
    rgb_orig: np.ndarray,
    prediction: torch.Tensor,
    masks_binary: torch.Tensor | None,
    struct_results: list[dict],
    plane_info: dict,
    patient_id: str,
    view: str,
    out_path: Path,
) -> Path:
    """
    Saves a figure with a Prediction panel (always) and a Ground Truth panel (only when
    masks_binary is not None) - unchanged contour/legend/plane-banner content from the
    original script's visualise() / visualise_prediction_only(), merged into one function
    since GT availability is now decided per-image rather than by which script path was
    called.
    """
    image_np = np.clip(rgb_orig.astype(np.float32) / 255.0, 0, 1)
    h, w = image_np.shape[:2]

    show_gt = masks_binary is not None

    if show_gt:
        fig, (gt_ax, pred_ax) = plt.subplots(1, 2, figsize=(18, 9))
        axes = [gt_ax, pred_ax]
    else:
        fig, pred_ax = plt.subplots(1, 1, figsize=(12, 10))
        axes = [pred_ax]

    # --------------------------------------------------
    # IMAGE ONLY — NO TEXT
    # --------------------------------------------------
    for ax in axes:
        ax.imshow(image_np)
        ax.axis("off")

    # --------------------------------------------------
    # SEGMENTATION OVERLAY / CONTOURS
    # --------------------------------------------------
    for c in range(NUM_CLASSES):

        pred_mask = cv2.resize(
            prediction[c].numpy().astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST
        ) > 0

        color = _CLASS_COLORS[c]

        # Prediction overlay
        if pred_mask.any():
            for contour in find_contours(
                pred_mask.astype(np.uint8),
                level=0.5
            ):
                pred_ax.plot(
                    contour[:, 1],
                    contour[:, 0],
                    color=color,
                    linewidth=4.0
                )

        # Ground Truth overlay
        if show_gt:
            gt_mask = cv2.resize(
                masks_binary[c].numpy().astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST
            ) > 0

            if gt_mask.any():
                for contour in find_contours(
                    gt_mask.astype(np.uint8),
                    level=0.5
                ):
                    gt_ax.plot(
                        contour[:, 1],
                        contour[:, 0],
                        color=color,
                        linewidth=4.0
                    )

    # --------------------------------------------------
    # NO LEGEND
    # NO PLANE TITLE
    # NO PATIENT TITLE
    # NO STRUCTURE TEXT
    # --------------------------------------------------

    plt.tight_layout(pad=0)

    plt.savefig(
        out_path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close(fig)

    return out_path


# ═══════════════════════════════════════════════════════════════════════
# JSON EXPORT
# ═══════════════════════════════════════════════════════════════════════

def build_per_image_json_entry(
    image_path: Path,
    view: str,
    plane_info: dict,
    plane_candidates: list[str],
    plane_mandatory_structures: dict,
    mean_structure_confidence: float,
    struct_results: list[dict],
    polygons: list[dict],
    gt_metrics: dict | None,
    struct_summary: dict | None,
    component_report: dict | None = None,
) -> dict:

    return {
        "image_path": str(image_path),
        "input_filename": image_path.name,


        "view": view,

        "plane": plane_info["predicted_plane"],

        "plane_quality":
            plane_info["plane_info"]["plane_standard"],

        "plane_candidates": plane_candidates,

        "plane_mandatory_structures":
            plane_mandatory_structures,

        "mean_structure_confidence":
            round(mean_structure_confidence, 6),

        "structures":
            struct_results,

        "polygons":
            polygons,

        "gt_metrics_available":
            gt_metrics is not None,

        "gt_metrics":
            gt_metrics,

        "gt_summary":
            struct_summary,

        "postprocessing":
            component_report,
    }

# from ui.utils.dcm_png_utils import anonymized_png_for_dicom
# ═══════════════════════════════════════════════════════════════════════
# MAIN INFERENCE LOOP
# ═══════════════════════════════════════════════════════════════════════

def run_inference_face(
    checkpoint: str,
    items,
    masks_dir: str | None = None,
    seed: int = 42,
    device_str: str = "cuda",
):

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s"
    )

    set_seed(seed)
    # num_vis_images = len(image_dir)

    device = torch.device(
        device_str
        if torch.cuda.is_available()
        else "cpu"
    )

    logging.info(
        "\n"
        + "=" * 70
    )

    logging.info(
        "U-NET++ FACE/NOSE/MOUTH INFERENCE"
    )

    logging.info(
        "=" * 70
    )

    logging.info(
        f"Device: {device}"
    )

    if device.type == "cuda":

        logging.info(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # out = Path(
    #     output_dir
    # )

    # out.mkdir(
    #     parents=True,
    #     exist_ok=True
    # )

    # vis_dir = (
    #     out / "visualisations"
    # )

    # vis_dir.mkdir(
    #     parents=True,
    #     exist_ok=True
    # )

    # met_dir = (
    #     out / "metrics"
    # )

    # met_dir.mkdir(
    #     parents=True,
    #     exist_ok=True
    # )

    # gt_dir = (
    #     Path(masks_dir)
    #     if masks_dir
    #     else None
    # )

    # if (
    #     gt_dir is not None
    #     and not gt_dir.is_dir()
    # ):

    #     logging.warning(
    #         f"masks_dir '{masks_dir}' "
    #         f"not found - GT metrics will be "
    #         f"skipped for every image."
    #     )

    #     gt_dir = None

    if not Path(
        checkpoint
    ).exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint}"
        )

    model = load_model(
        checkpoint,
        device
    )

    logging.info(
        f"Parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # img_paths = discover_images(
    #     image_dir
    # )

    # logging.info(
    #     f"\nFound {len(img_paths)} "
    #     f"image(s) under {image_dir}\n"
    #     f"{'-' * 50}"
    # )

    # num_vis_images = min(
    #     num_vis_images,
    #     len(img_paths)
    # )

    # json_entries = []

    n_with_gt = 0

    gt_present_dice_all = []

    results = []

    for idx, item in enumerate(items):

        img_path = Path(item["image_path"])

        logging.info(
            f"[{idx + 1}/{len(items)}] "
            f"{img_path.name}"
        )

        rgb = load_rgb_image(
            str(img_path)
        )

        if rgb is None:
            continue
        # anon_path = anonymized_png_for_dicom(img_path)
        # rgb_from_png = (
        #     load_rgb_image(str(anon_path)) if anon_path is not None else None
        # )

        panel = item["panel"]

        panel_rgb = extract_panel(
            rgb,
            panel
        )
        orig_h, orig_w = (
            panel_rgb.shape[:2]
        )

        # patient_id = stem
        view = "Unknown"

        image_tensor = (
            preprocess_for_model(
                rgb
            )
        )

        (
            prediction,
            probabilities,
            confidences
        ) = predict(
            model,
            image_tensor,
            device
        )

        # =====================================================
        # 1. COMPONENT CLEANING
        #
        # Includes:
        #   - Orbit relative-area filtering
        #   - Nose >= 1000 pixels
        #   - Lens component capping
        #   - Other component capping
        # =====================================================

        prediction = clean_prediction_mask(
            prediction,
            min_pixels=20
        )

        # =====================================================
        # 2. SPATIAL ISOLATION CLEANING
        # =====================================================

        prediction = (
            remove_spatially_isolated_structures(
                prediction,
                min_pixels=20,
                max_distance_fraction=0.25,
                min_cluster_size=2,
            )
        )

        # =====================================================
        # GT
        # =====================================================

        # gt_arr = (
        #     load_gt(
        #         gt_dir,
        #         stem,
        #         MODEL_INPUT_SIZE
        #     )
        #     if gt_dir is not None
        #     else None
        # )

        # masks_binary = (
        #     torch.from_numpy(
        #         gt_arr
        #     ).bool()
        #     if gt_arr is not None
        #     else None
        # )

        # if masks_binary is not None:

        #     n_with_gt += 1

        # else:

        #     logging.info(
        #         f"    [GT] no NPZ found "
        #         f"for '{stem}' - "
        #         f"gt_metrics_available=false"
        #     )

        # =====================================================
        # 3. SCORE CLEANED MASK
        # =====================================================

        (
            struct_results,
            struct_summary
        ) = assess_structures(
            prediction,
            None,
            confidences
        )

        # gt_metrics = (
        #     gt_metrics_rows_to_dict(
        #         struct_results
        #     )
        #     if masks_binary is not None
        #     else None
        # )

        # if (
        #     struct_summary is not None
        #     and struct_summary[
        #         "mean_dice_gt_present"
        #     ] is not None
        # ):

        #     gt_present_dice_all.append(
        #         struct_summary[
        #             "mean_dice_gt_present"
        #         ]
        #     )

        # =====================================================
        # 4. PLANE DECISION
        # =====================================================

        plane_info = (
            determine_predicted_plane(
                struct_results
            )
        )

        # =====================================================
        # 5. VISUALIZATION-ONLY PLANE FILTER
        # =====================================================

        visualization_prediction = (
            clean_mask_for_predicted_plane(
                prediction,
                plane_info
            )
        )

        # =====================================================
        # 6. REFINE PLANE QUALITY
        #
        # Orbits and Lenses:
        # exactly 2 + exactly 2 required.
        # =====================================================

        plane_info = (
            refine_plane_quality(
                plane_info,
                visualization_prediction
            )
        )

        present = (
            get_present_structures(
                struct_results
            )
        )

        plane_mandatory_structures = (
            build_plane_mandatory_structures(
                present
            )
        )

        plane_candidates = (
            build_plane_candidates(
                plane_mandatory_structures,
                plane_info[
                    "predicted_plane"
                ]
            )
        )

        mean_structure_confidence = (
            compute_mean_structure_confidence(
                struct_results
            )
        )

        polygons = (
            build_polygons_from_final_mask(
                visualization_prediction,
                orig_h,
                orig_w
            )
        )

        # =====================================================
        # Ensure JSON structures agree with visualization
        # =====================================================

        visualization_structures = []

        for c, result in enumerate(
            struct_results
        ):

            result_vis = result.copy()

            result_vis[
                "predicted_present"
            ] = bool(
                visualization_prediction[c]
                .any()
            )

            visualization_structures.append(
                result_vis
            )

        # =====================================================
        # VISUALIZATION
        # =====================================================
        is_standard = (
            plane_info["plane_info"]["plane_standard"] == "Standard"
        )

        # if progress_callback:
        #     progress_callback(
        #         str(img_path),
        #         is_standard
        #     )
        # vis_path = None

        # if idx < num_vis_images:
        #     vis_path = vis_dir / f"{stem}_vis.png"
        #     if rgb_from_png is not None:
        #         make_visualization(
        #             rgb_from_png, prediction, masks_binary, struct_results, plane_info,
        #             patient_id, view, vis_path,
        #         )
        #     else:
        #         make_visualization(
        #             rgb, prediction, masks_binary, struct_results, plane_info,
        #             patient_id, view, vis_path,
        #         )
        #     logging.info(f"    [VIS] saved -> {vis_path}")

        # =====================================================
        # JSON ENTRY
        # =====================================================

        entry = build_per_image_json_entry(

            img_path,
            # vis_path,
            view,
            plane_info,
            plane_candidates,
            plane_mandatory_structures,
            mean_structure_confidence,
            visualization_structures,
            polygons,
            None,
            struct_summary,
        )
        if plane_info["plane_info"]["plane_standard"] == "Standard":
            status = "standard"
        elif plane_info["predicted_plane"] == "Unknown":
            status = "unknown"
        else:
            status = "non-standard"

        results.append({
            "image_path": str(img_path),
            "status": status,
            "result": entry,
        })
        write_segmentation_result(item, entry)
        # json_entries.append(
        #     entry
        # )

    # # ═══════════════════════════════════════════════════════════════
    # # AGGREGATE JSON
    # # ═══════════════════════════════════════════════════════════════

    # results_json_path = (
    #     met_dir
    #     / "inference_results.json"
    # )

    # existing_entries = []

    # existing_n_with_gt = 0

    # if results_json_path.exists():

    #     try:

    #         with open(
    #             results_json_path,
    #             "r"
    #         ) as f:

    #             prev = json.load(f)

    #         existing_entries = (
    #             prev.get(
    #                 "images",
    #                 []
    #             )
    #         )

    #         existing_n_with_gt = (
    #             prev.get(
    #                 "run_summary",
    #                 {}
    #             ).get(
    #                 "n_images_with_gt",
    #                 0
    #             )
    #         )

    #         logging.info(
    #             f"[JSON] found existing "
    #             f"{results_json_path} with "
    #             f"{len(existing_entries)} "
    #             f"image(s) - appending "
    #             f"this run's results to it"
    #         )

    #     except Exception as exc:

    #         logging.warning(
    #             f"Could not read existing "
    #             f"{results_json_path} "
    #             f"({exc}) - starting fresh."
    #         )

    # all_entries = (
    #     existing_entries
    #     + json_entries
    # )

    # run_summary = {

    #     "n_images":
    #         len(all_entries),

    #     "n_images_with_gt":
    #         existing_n_with_gt
    #         + n_with_gt,

    #     "mean_dice_gt_present_over_images_with_gt":
    #         (
    #             round(
    #                 float(
    #                     np.mean(
    #                         gt_present_dice_all
    #                     )
    #                 ),
    #                 4
    #             )
    #             if gt_present_dice_all
    #             else None
    #         ),
    # }

    # with open(
    #     results_json_path,
    #     "w"
    # ) as f:

    #     json.dump(
    #         {
    #             "run_summary":
    #                 run_summary,

    #             "images":
    #                 all_entries
    #         },
    #         f,
    #         indent=2
    #     )

    # logging.info(
    #     f"\n[JSON] wrote "
    #     f"{results_json_path} "
    #     f"({len(all_entries)} "
    #     f"images total)"
    # )

    # logging.info(
    #     "\n"
    #     + "=" * 70
    # )

    # logging.info(
    #     "INFERENCE COMPLETE"
    # )

    # logging.info(
    #     "=" * 70
    # )

    # logging.info(
    #     f"Images processed: "
    #     f"{len(json_entries)}"
    # )

    # logging.info(
    #     f"Results JSON:     "
    #     f"{results_json_path}"
    # )

    # logging.info(
    #     f"Visualisations:   "
    #     f"{vis_dir}"
    # )

    # return {

    #     "output_dir":
    #         str(out),

    #     "visualisations_dir":
    #         str(vis_dir),

    #     "results_json_path":
    #         str(results_json_path),

    #     "n_images":
    #         len(json_entries),

    #     "n_images_with_gt":
    #         n_with_gt,
    # }
    return results


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(

#         description=(
#             "Generic-framework inference "
#             "for the 14-channel "
#             "face/nose/mouth U-Net++ "
#             "model. Supports "
#             "PNG/JPEG/BMP/TIFF/DICOM input."
#         )
#     )

#     parser.add_argument(
#         "--checkpoint",
#         default=CHECKPOINT_PATH_DEFAULT,
#         help=(
#             "Path to a "
#             "best_model.pth checkpoint."
#         ),
#     )

#     parser.add_argument(
#         "--image_dir",
#         required=True,
#         help=(
#             "Folder to scan recursively "
#             "for images/DICOM."
#         ),
#     )

#     parser.add_argument(
#         "--output_dir",
#         default=OUTPUT_DIR_DEFAULT,
#         help=(
#             "Where visualisations/ "
#             "and metrics/ for this run go."
#         ),
#     )

#     parser.add_argument(
#         "--masks_dir",
#         default=None,
#         help=(
#             "Optional folder of GT .npz masks."
#         ),
#     )

#     # parser.add_argument(
#     #     "--num_vis_images",
#     #     type=int,
#     #     default=NUM_VIS_IMAGES_DEFAULT
#     # )

#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=42
#     )

#     parser.add_argument(
#         "--device",
#         default="cuda",
#         help="cuda or cpu"
#     )

#     args = parser.parse_args()

#     run_inference_face(

#         checkpoint=args.checkpoint,

#         image_dir=args.image_dir,

#         output_dir=args.output_dir,

#         masks_dir=args.masks_dir,

#         # num_vis_images=args.num_vis_images,


#         seed=args.seed,

#         device_str=args.device,
#     )
