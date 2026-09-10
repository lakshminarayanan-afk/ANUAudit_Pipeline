
# ============================================================
# IMPORTS
# ============================================================

import os
import cv2
import time
import json
import torch
import random
import warnings
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt

from PIL import Image
import shutil

# ------------------------------------------------------------
# Torch
# ------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# ------------------------------------------------------------
# Torchvision
# ------------------------------------------------------------

from torchvision import transforms
from torchvision import models

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ------------------------------------------------------------
# Misc
# ------------------------------------------------------------

warnings.filterwarnings("ignore")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("="*60)
print("DEVICE :", DEVICE)
print("="*60)


# ============================================================
# PATHS
# ============================================================

OLD_MODEL_CKPT = Path(
    "/mnt/data/agilaa/checkpoints-modality-20260909T104119Z-1-001/checkpoints-modality/best_f1.pth"
)

ANATOMY_MODEL_CKPT = Path(
    "/mnt/data/agilaa/checkpoints-anatomy-20260909T104119Z-1-001/checkpoints-anatomy/best_f1.pth"
)

INPUT_FOLDER = Path(
    "/mnt/data/dicom_data_inference/colour_doppler"
)

folder_name = (
    INPUT_FOLDER.name
    .replace(" ", "_")
    .replace("(", "")
    .replace(")", "")
)

# ------------------------------------------------------------
# SAMPLE IMAGE (for the test/demo cells below)
# ------------------------------------------------------------
# INPUT_FOLDER is a master folder of case subfolders, so
# grabbing INPUT_FOLDER.glob("*") no longer returns an image
# directly. This searches recursively for the first actual
# image file to use in the demo cells.
# ------------------------------------------------------------

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SAMPLE_IMAGE_PATH = next(
    p for p in INPUT_FOLDER.rglob("*")
    if p.suffix.lower() in _IMAGE_EXTS
)

# ------------------------------------------------------------
# MASTER FOLDER MODE
# ------------------------------------------------------------
# INPUT_FOLDER is treated as a master folder containing one
# subfolder per case (e.g. K58918_HEMALATHA_AKASH, ...).
# MASTER_OUTPUT_FOLDER mirrors that: one output subfolder per
# case, each with the usual internal structure.
# ------------------------------------------------------------

MASTER_OUTPUT_FOLDER = (
    Path("./integrated_results")
    / folder_name
)

MASTER_OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FOLDER = MASTER_OUTPUT_FOLDER

# ============================================================
# OUTPUT STRUCTURE
# ============================================================
# setup_output_dirs() builds the standard set of output
# folders (visualizations, anatomy_vis, uncertain_cases, roi,
# masks, gradcam, reports) under whatever root it is given,
# and rebinds the module-level dir globals used everywhere
# else in this script (VIS_DIR, GRADCAM_DIR, etc.) so the
# existing per-image saving code works unchanged.
# ============================================================

def setup_output_dirs(output_root):

    global OUTPUT_FOLDER
    global VIS_DIR
    global ANATOMY_VIS_DIR, ANATOMY_3VC_DIR, ANATOMY_3VT_DIR
    global ANATOMY_RENAL_DIR, ANATOMY_INFLOWS_DIR
    global UNCERTAIN_DIR, UNCERTAIN_VIS_DIR
    global ROI_DIR
    global MASK_DIR, VESSEL_MASK_DIR, RED_MASK_DIR, BLUE_MASK_DIR
    global GRADCAM_DIR
    global REPORT_DIR

    OUTPUT_FOLDER = Path(output_root)

    VIS_DIR = OUTPUT_FOLDER / "visualizations"

    ANATOMY_VIS_DIR = OUTPUT_FOLDER / "anatomy_vis"

    ANATOMY_3VC_DIR = ANATOMY_VIS_DIR / "3VC"
    ANATOMY_3VT_DIR = ANATOMY_VIS_DIR / "3VT"
    ANATOMY_RENAL_DIR = ANATOMY_VIS_DIR / "Renal"
    ANATOMY_INFLOWS_DIR = ANATOMY_VIS_DIR / "Inflows"

    UNCERTAIN_DIR = OUTPUT_FOLDER / "uncertain_cases"
    UNCERTAIN_VIS_DIR = UNCERTAIN_DIR / "visualizations"

    ROI_DIR = OUTPUT_FOLDER / "roi"

    MASK_DIR = OUTPUT_FOLDER / "masks"

    VESSEL_MASK_DIR = MASK_DIR / "vessel"
    RED_MASK_DIR = MASK_DIR / "red"
    BLUE_MASK_DIR = MASK_DIR / "blue"

    GRADCAM_DIR = OUTPUT_FOLDER / "gradcam"

    REPORT_DIR = OUTPUT_FOLDER / "reports"

    # Create all output directories
    for p in [

        VIS_DIR,

        ANATOMY_VIS_DIR,
        ANATOMY_3VC_DIR,
        ANATOMY_3VT_DIR,
        ANATOMY_RENAL_DIR,
        ANATOMY_INFLOWS_DIR,

        UNCERTAIN_DIR,
        UNCERTAIN_VIS_DIR,

        ROI_DIR,

        MASK_DIR,
        VESSEL_MASK_DIR,
        RED_MASK_DIR,
        BLUE_MASK_DIR,

        GRADCAM_DIR,

        REPORT_DIR
    ]:
        p.mkdir(parents=True, exist_ok=True)

    return OUTPUT_FOLDER


# Default single-folder setup (kept so earlier/demo cells that
# reference VIS_DIR, GRADCAM_DIR, etc. still work as before).
setup_output_dirs(MASTER_OUTPUT_FOLDER)

print("Output folders ready")


# ============================================================
# OLD MODEL LABELS
# ============================================================

LABEL_NAMES = [

    "b-mode",

    "tinted",

    "colour_doppler",

    "pulse_doppler",

    "split_screen_only",

    "quadrant_images"
]

CLASS_THRESHOLDS = {

    "b-mode":0.95,

    "tinted":0.376,

    "colour_doppler":0.926,

    "pulse_doppler":0.95,

    "split_screen_only":0.95,

    "quadrant_images":0.95
}
# ============================================================
# CONFIDENCE SETTINGS
# ============================================================

UNCERTAINTY_THRESHOLD = 0.80

print(LABEL_NAMES)


# ============================================================
# ANATOMY LABELS
# ============================================================

ANATOMY_CLASSES = {

    0:"3VC",

    1:"3VT",

    2:"Renal",

    3:"Inflows"
}

CLASS_TO_IDX = {

    "3VC":0,

    "3VT":1,

    "Renal":2,

    "Inflows":3
}

print(ANATOMY_CLASSES)


# ============================================================
# OLD MODALITY MODEL
# ============================================================

import timm
import torch.nn as nn


class OldModalityClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=0,
            global_pool="avg"
        )

        feat_dim = self.backbone.num_features

        self.head = nn.Sequential(

            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 6)
        )

    def forward(self, x):

        feat = self.backbone(x)

        return self.head(feat)


ckpt = torch.load(
    "/mnt/data/agilaa/checkpoints-modality-20260909T104119Z-1-001/checkpoints-modality/best_f1.pth",
    map_location="cpu"
)

print(ckpt.keys())

for k,v in ckpt["model"].items():

    if "head" in k:
        print(k, v.shape)


# ============================================================
# CHECKPOINT PATHS
# ============================================================

OLD_CHECKPOINT = Path(
    "/mnt/data/agilaa/checkpoints-modality-20260909T104119Z-1-001/checkpoints-modality/best_f1.pth"
)

ANATOMY_CHECKPOINT = Path(
    "/mnt/data/agilaa/checkpoints-anatomy-20260909T104119Z-1-001/checkpoints-anatomy/best_f1.pth"
)

print("Old :", OLD_CHECKPOINT.exists())
print("New :", ANATOMY_CHECKPOINT.exists())


old_model = OldModalityClassifier().to(DEVICE)

ckpt = torch.load(
    OLD_CHECKPOINT,
    map_location=DEVICE
)

old_model.load_state_dict(
    ckpt["model"]
)

old_model.eval()

print("✅ Old model loaded")
print("Epoch:", ckpt["epoch"])
print("Val F1:", ckpt["val_f1"])

# ============================================================
# INSPECT ANATOMY CHECKPOINT
# ============================================================

ckpt = torch.load(
    ANATOMY_CHECKPOINT,
    map_location="cpu"
)

print(ckpt.keys())

for k,v in ckpt["model"].items():

    if (
        "conv_stem" in k
        or "classifier" in k
        or "head" in k
    ):
        print(k, v.shape)

# ============================================================
# ANATOMY MODEL
# ============================================================

import timm
import torch.nn as nn

class AnatomyEfficientNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            in_chans=6,
            num_classes=0,
            global_pool="avg"
        )

        feat_dim = self.backbone.num_features

        self.classifier = nn.Sequential(

            nn.Linear(feat_dim, 512),

            nn.BatchNorm1d(512),

            nn.SiLU(),

            nn.Dropout(0.30),

            nn.Linear(512, 128),

            nn.BatchNorm1d(128),

            nn.SiLU(),

            nn.Dropout(0.20),

            nn.Linear(128, 4)
        )

    def forward(self, x):

        feat = self.backbone(x)

        return self.classifier(feat)

ckpt = torch.load(
    ANATOMY_CHECKPOINT,
    map_location="cpu"
)

print(ckpt.keys())

for k,v in ckpt.items():

    if k != "model":

        print(k, ":", v)


# ============================================================
# LOAD ANATOMY MODEL
# ============================================================

anatomy_model = AnatomyEfficientNet().to(DEVICE)

ckpt = torch.load(
    ANATOMY_CHECKPOINT,
    map_location=DEVICE
)

anatomy_model.load_state_dict(
    ckpt["model"]
)

anatomy_model.eval()

print("="*60)

print("✅ Anatomy model loaded")

print("Epoch :", ckpt["epoch"])

print("Best F1 :", ckpt["f1"])

print("="*60)


dummy = torch.randn(1,6,224,224).to(DEVICE)

with torch.no_grad():
    out = anatomy_model(dummy)

print(out.shape)


# ============================================================
# VERIFY BOTH MODELS
# ============================================================

print("\nOLD MODEL")

dummy_old = torch.randn(
    1,
    3,
    224,
    224
).to(DEVICE)

with torch.no_grad():

    out_old = old_model(dummy_old)

print(out_old.shape)

print("\nANATOMY MODEL")

dummy_new = torch.randn(
    1,
    6,
    224,
    224
).to(DEVICE)

with torch.no_grad():

    out_new = anatomy_model(dummy_new)

print(out_new.shape)


# ============================================================
# SHARED TRANSFORMS
# ============================================================

import torchvision.transforms as T

MODALITY_TRANSFORM = T.Compose([

    T.ToPILImage(),

    T.Resize(
        (224,224)
    ),

    T.ToTensor(),

    T.Normalize(

        mean=[0.485,0.456,0.406],

        std=[0.229,0.224,0.225]
    )
])

print("✅ Modality transform ready")


# ============================================================
# ANATOMY NORMALIZATION
# ============================================================

ANATOMY_SIZE = 224

print("✅ Anatomy normalization ready")


# ============================================================
# LABEL MAPS
# ============================================================

MODALITY_LABELS = [

    "b-mode",

    "tinted",

    "colour_doppler",

    "pulse_doppler",

    "split_screen_only",

    "quadrant_images"
]

ANATOMY_LABELS = [

    "3VC",

    "3VT",

    "Renal",

    "Inflows"
]

print(MODALITY_LABELS)
print(ANATOMY_LABELS)


# ============================================================
# OLD MODEL PREDICTION
# ============================================================

@torch.no_grad()
def predict_modality(img_bgr):

    img_rgb = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2RGB
    )

    x = MODALITY_TRANSFORM(
        img_rgb
    )

    x = x.unsqueeze(0).to(DEVICE)

    logits = old_model(x)

    probs = torch.sigmoid(
        logits
    )

    probs = probs.squeeze().cpu().numpy()

    pred_labels = []

    for idx, label in enumerate(MODALITY_LABELS):

        threshold = CLASS_THRESHOLDS[label]

        if probs[idx] >= threshold:

            pred_labels.append(label)

    dominant_idx = int(
        np.argmax(probs)
    )

    return {

        "probs": probs,

        "pred_labels": pred_labels,

        "dominant_label":
            MODALITY_LABELS[
                dominant_idx
            ],

        "dominant_prob":
            float(
                probs[
                    dominant_idx
                ]
            )
    }

print("✅ Modality predictor ready")


print(INPUT_FOLDER)
print(INPUT_FOLDER.exists())


class DopplerFlowLocalizer:
    """
    Extracts color Doppler flow regions using HSV analysis.
    Red = flow toward transducer
    Blue = flow away from transducer
    """

    # HSV range for red flow (wraps around 0/180)
    RED_LOWER1 = np.array([0,   80,  80])
    RED_UPPER1 = np.array([10, 255, 255])
    RED_LOWER2 = np.array([165, 80,  80])
    RED_UPPER2 = np.array([180,255, 255])

    # HSV range for blue flow
    BLUE_LOWER = np.array([100, 80,  80])
    BLUE_UPPER = np.array([130,255, 255])

    def __init__(self, min_area=50):
        self.min_area = min_area
        self.kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def extract(self, img_rgb: np.ndarray):
        """
        Returns:
          flow_mask  : binary mask of all flow regions
          red_mask   : red flow
          blue_mask  : blue flow
          overlay    : colored overlay on original
          stats      : dict with flow_pct, bbox, etc.
        """
        if img_rgb.dtype != np.uint8:
            img_rgb = (img_rgb * 255).astype(np.uint8)

        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        # Red mask
        red1 = cv2.inRange(hsv, self.RED_LOWER1, self.RED_UPPER1)
        red2 = cv2.inRange(hsv, self.RED_LOWER2, self.RED_UPPER2)
        red_mask  = cv2.bitwise_or(red1, red2)

        # Blue mask
        blue_mask = cv2.inRange(hsv, self.BLUE_LOWER, self.BLUE_UPPER)

        # Combined
        flow_mask = cv2.bitwise_or(red_mask, blue_mask)

        # Morphological cleanup
        flow_mask = cv2.morphologyEx(flow_mask, cv2.MORPH_OPEN,  self.kernel)
        flow_mask = cv2.morphologyEx(flow_mask, cv2.MORPH_CLOSE, self.kernel)
        flow_mask = cv2.dilate(flow_mask, self.kernel, iterations=1)

        # Remove Doppler color scale and machine overlays

        flow_mask_clean = flow_mask.copy()

        h, w = flow_mask_clean.shape

        flow_mask_clean[:, :100] = 0        # left color scale
        flow_mask_clean[:, w-150:] = 0      # right machine text
        flow_mask_clean[:50, :] = 0         # top overlay
        flow_mask_clean[h-50:, :] = 0       # bottom overlay

        # Stats

        total_px = flow_mask_clean.size
        flow_px = flow_mask_clean.sum() / 255
        flow_pct = 100.0 * flow_px / total_px

        bbox = None
        contours, _ = cv2.findContours(
            flow_mask_clean,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        large_contours = [c for c in contours
                          if cv2.contourArea(c) >= self.min_area]
        if large_contours:
            all_pts = np.vstack(large_contours)
            x, y, w, h = cv2.boundingRect(all_pts)
            bbox = (x, y, x+w, y+h)

        # Colored overlay
        overlay = img_rgb.copy()
        overlay[red_mask  > 0] = [255, 60,  60]
        overlay[blue_mask > 0] = [60,  60, 255]

        stats = {
            'flow_pct': round(flow_pct, 2),
            'flow_px':  int(flow_px),
            'red_px':   int((red_mask > 0 ).sum()),
            'blue_px':  int((blue_mask > 0).sum()),
            'bbox':     bbox
        }
        return flow_mask, red_mask, blue_mask, overlay, stats

doppler_localizer = DopplerFlowLocalizer()
print('✅ Doppler localizer ready')


# ============================================================
# ROI EXTRACTOR
# ============================================================

def extract_roi(img_rgb):

    flow_mask, red_mask, blue_mask, overlay, stats = \
        doppler_localizer.extract(img_rgb)

    bbox = stats["bbox"]

    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox

    w = x2 - x1
    h = y2 - y1

    pad_x = int(w * 0.5)
    pad_y = int(h * 0.5)

    pad_x = max(pad_x, 60)
    pad_y = max(pad_y, 60)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)

    x2 = min(img_rgb.shape[1], x2 + pad_x)
    y2 = min(img_rgb.shape[0], y2 + pad_y)

    roi = img_rgb[y1:y2, x1:x2]

    return {
        "roi": roi,
        "bbox": (x1, y1, x2, y2),

        "roi_x1": x1,
        "roi_y1": y1,
        "roi_x2": x2,
        "roi_y2": y2,

        "flow_mask": flow_mask,
        "red_mask": red_mask,
        "blue_mask": blue_mask,
        "stats": stats
    }


img = cv2.imread(
    str(SAMPLE_IMAGE_PATH)
)

img_rgb = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2RGB
)

result = extract_roi(
    img_rgb
)

if result is None:

    print(
        "No ROI detected"
    )

else:

    print(
        result["bbox"]
    )

    print(
        result["roi"].shape
    )


if result is None:

    print("No ROI detected")

else:

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(img_rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(result["roi"])
    plt.title("ROI")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# MASK GENERATION
# ============================================================

def generate_masks(roi_rgb):

    flow_mask, red_mask, blue_mask, _, stats = \
        doppler_localizer.extract(
            roi_rgb
        )

    vessel_mask = (
        flow_mask > 0
    ).astype(np.uint8) * 255

    red_mask = (
        red_mask > 0
    ).astype(np.uint8) * 255

    blue_mask = (
        blue_mask > 0
    ).astype(np.uint8) * 255

    return {

        "vessel_mask": vessel_mask,

        "red_mask": red_mask,

        "blue_mask": blue_mask,

        "stats": stats
    }

print("✅ Mask generator ready")


# ============================================================
# TEST MASK GENERATION
# ============================================================
if result is None:

    print("No ROI detected")

else:
    masks = generate_masks(
        result["roi"]
    )

    print(
        masks["vessel_mask"].shape
    )

    print(
        masks["red_mask"].shape
    )

    print(
        masks["blue_mask"].shape
    )


# ============================================================
# VISUALIZE MASKS
# ============================================================
if result is None:

    print("No ROI detected")

else:
    fig, ax = plt.subplots(
        1,
        4,
        figsize=(16,4)
    )

    ax[0].imshow(
        result["roi"]
    )
    ax[0].set_title(
        "ROI"
    )

    ax[1].imshow(
        masks["vessel_mask"],
        cmap="gray"
    )
    ax[1].set_title(
        "Vessel"
    )

    ax[2].imshow(
        masks["red_mask"],
        cmap="gray"
    )
    ax[2].set_title(
        "Red"
    )

    ax[3].imshow(
        masks["blue_mask"],
        cmap="gray"
    )
    ax[3].set_title(
        "Blue"
    )

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# BUILD 6-CHANNEL TENSOR
# ============================================================

def build_6channel_tensor(

    roi_rgb,

    vessel_mask,

    red_mask,

    blue_mask
):

    roi_rgb = cv2.resize(

        roi_rgb,

        (224,224),

        interpolation=cv2.INTER_AREA
    )

    vessel_mask = cv2.resize(

        vessel_mask,

        (224,224),

        interpolation=cv2.INTER_NEAREST
    )

    red_mask = cv2.resize(

        red_mask,

        (224,224),

        interpolation=cv2.INTER_NEAREST
    )

    blue_mask = cv2.resize(

        blue_mask,

        (224,224),

        interpolation=cv2.INTER_NEAREST
    )

    roi_rgb = roi_rgb.astype(
        np.float32
    ) / 255.0

    vessel_mask = vessel_mask.astype(
        np.float32
    ) / 255.0

    red_mask = red_mask.astype(
        np.float32
    ) / 255.0

    blue_mask = blue_mask.astype(
        np.float32
    ) / 255.0

    tensor = np.dstack([

        roi_rgb,

        vessel_mask,

        red_mask,

        blue_mask
    ])

    tensor = torch.tensor(
        tensor,
        dtype=torch.float32
    )

    tensor = tensor.permute(
        2,
        0,
        1
    )

    return tensor

print("✅ 6-channel builder ready")


# ============================================================
# VERIFY 6-CHANNEL TENSOR
# ============================================================
if result is None:

    print("No ROI detected")

else:
    tensor6 = build_6channel_tensor(

        result["roi"],

        masks["vessel_mask"],

        masks["red_mask"],

        masks["blue_mask"]
    )

    print(
        tensor6.shape
    )


# ============================================================
# ANATOMY PREDICTION
# ============================================================

@torch.no_grad()
def predict_anatomy(img_bgr):

    img_rgb = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2RGB
    )

    roi_result = extract_roi(
        img_rgb
    )

    if roi_result is None:

        return None

    masks = generate_masks(
        roi_result["roi"]
    )

    tensor6 = build_6channel_tensor(

        roi_result["roi"],

        masks["vessel_mask"],

        masks["red_mask"],

        masks["blue_mask"]
    )

    inp = tensor6.unsqueeze(0).to(DEVICE)

    logits = anatomy_model(inp)

    probs = torch.softmax(
        logits,
        dim=1
    )

    probs = probs.squeeze().cpu().numpy()

    pred_idx = int(
        np.argmax(probs)
    )

    confidence = float(
        probs[pred_idx]
    )

    return {

        "prediction":
            ANATOMY_LABELS[
                pred_idx
            ],

        "pred_idx":
            pred_idx,

        "confidence":
            confidence,

        "probs":
            probs,

        "roi":
            roi_result["roi"],

        "bbox":
            roi_result["bbox"],

        "roi_result":
            roi_result,

        "masks":
            masks,

        "tensor6":
            tensor6
    }

print("✅ Anatomy predictor ready")


# ============================================================
# TEST ANATOMY MODEL - SINGLE IMAGE
# ============================================================

IMAGE_PATH = str(SAMPLE_IMAGE_PATH)

test_image = cv2.imread(
    IMAGE_PATH
)

result = predict_anatomy(
    test_image
)

if result is None:

    print("No anatomy ROI detected")

else:

    print(
        f"Prediction : {result['prediction']}"
    )

    print(
        f"Confidence : {result['confidence']:.4f}"
    )

    print("\nClass Probabilities\n")

    for cls, p in zip(
        ANATOMY_LABELS,
        result["probs"]
    ):

        print(
            f"{cls:10s}: {p:.4f}"
        )


import torch.nn.functional as F


class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = \
            target_layer.register_forward_hook(
                self.save_activation
            )

        self.backward_hook = \
            target_layer.register_full_backward_hook(
                self.save_gradient
            )

    def save_activation(
        self,
        module,
        inp,
        out
    ):
        self.activations = out.detach()

    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        image_tensor,
        class_idx=None
    ):

        self.model.zero_grad()

        logits = self.model(
            image_tensor
        )

        if class_idx is None:

            class_idx = logits.argmax(
                dim=1
            ).item()

        score = logits[
            0,
            class_idx
        ]

        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(
            dim=(2,3),
            keepdim=True
        )

        cam = (
            weights * activations
        ).sum(
            dim=1
        )

        cam = F.relu(cam)

        cam = cam.squeeze()

        cam = cam.cpu().numpy()

        cam -= cam.min()

        cam /= (
            cam.max() + 1e-8
        )

        return cam, class_idx


# ============================================================
# ANATOMY GRADCAM
# ============================================================

target_layer = (
    anatomy_model.backbone.blocks[-1]
)

gradcam_anatomy = GradCAM(

    anatomy_model,

    target_layer
)

print("✅ Anatomy GradCAM ready")


result = predict_anatomy(test_image)


inp = result["tensor6"].unsqueeze(0).to(DEVICE)

cam, pred_idx = gradcam_anatomy.generate(inp)

print("CAM Shape :", cam.shape)

print(
    "Prediction :",
    ANATOMY_LABELS[pred_idx]
)


roi = cv2.resize(
    result["roi"],
    (224,224)
)

cam = cv2.resize(
    cam,
    (224,224)
)

heatmap = cv2.applyColorMap(
    np.uint8(cam * 255),
    cv2.COLORMAP_JET
)

heatmap = cv2.cvtColor(
    heatmap,
    cv2.COLOR_BGR2RGB
)

overlay = (
    0.6 * roi +
    0.4 * heatmap
).astype(np.uint8)

fig, ax = plt.subplots(
    1,
    3,
    figsize=(15,5)
)

ax[0].imshow(roi)
ax[0].set_title("ROI")
ax[0].axis("off")

ax[1].imshow(cam, cmap="jet")
ax[1].set_title("GradCAM")
ax[1].axis("off")

ax[2].imshow(overlay)
ax[2].set_title("Overlay")
ax[2].axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# PANEL PREDICTION
# ============================================================

@torch.no_grad()
def predict_panel(panel_bgr):

    panel_rgb = cv2.cvtColor(
        panel_bgr,
        cv2.COLOR_BGR2RGB
    )

    tensor = MODALITY_TRANSFORM(
        panel_rgb
    )

    tensor = tensor.unsqueeze(0).to(DEVICE)

    logits = old_model(
        tensor
    )

    probs = torch.sigmoid(
        logits
    )[0].cpu().numpy()

    pred_labels = []

    for idx, prob in enumerate(probs):

        label = MODALITY_LABELS[idx]

        threshold = CLASS_THRESHOLDS[label]

        if label == "colour_doppler":
            threshold = 0.85

        if prob >= threshold:

            pred_labels.append(
                label
            )

    return {

        "labels":
            pred_labels,

        "probs":
            probs
    }


print(MODALITY_LABELS)
print(CLASS_THRESHOLDS)


# ============================================================
# SPLIT SCREEN EXTRACTOR
# ============================================================
def extract_split_panels(image_bgr):

    h, w = image_bgr.shape[:2]

    candidates = [
        0.45,
        0.475,
        0.50,
        0.525,
        0.55,
        0.575,
        0.60,
        0.625,
        0.65
    ]

    best_score = -1
    best_ratio = 0.5
    best_split_x = w // 2

    best_left = None
    best_right = None
    best_left_img = None
    best_right_img = None

    for ratio in candidates:

        split_x = int(w * ratio)

        left_panel = image_bgr[:, :split_x]
        right_panel = image_bgr[:, split_x:]

        left_result = predict_panel(left_panel)
        right_result = predict_panel(right_panel)

        score = (
            np.max(left_result["probs"])
            +
            np.max(right_result["probs"])
        )

        if score > best_score:

            best_score = score
            best_ratio = ratio
            best_split_x = split_x

            best_left = left_result
            best_right = right_result
            best_left_img = left_panel
            best_right_img = right_panel

    return {
        "left_result": best_left,
        "right_result": best_right,
        "left_img": best_left_img,
        "right_img": best_right_img,
        "split_ratio": best_ratio,
        "split_x": best_split_x,
        "score": best_score,
    }
# ============================================================
# SPLIT SCREEN ANALYSIS
# ============================================================

def analyze_split_screen(image_path):
    image_bgr = cv2.imread(str(image_path))
    split_result = extract_split_panels(image_bgr)

    left_result = split_result["left_result"]
    right_result = split_result["right_result"]

    best_ratio = split_result["split_ratio"]
    best_split_x = split_result["split_x"]
    best_score = split_result["score"]
    print("\nBest Split Search")
    print("Ratio :", best_ratio)
    print("Split X :", best_split_x)
    print("Score :", best_score)

    # PANEL MODALITY
    left_modality = (
        left_result["labels"][0]
        if len(left_result["labels"]) > 0
        else "unknown"
    )
    right_modality = (
        right_result["labels"][0]
        if len(right_result["labels"]) > 0
        else "unknown"
    )

    # FINAL REPORT
    print("\n" + "=" * 60)
    print("HIERARCHICAL SPLIT SCREEN REPORT")
    print("=" * 60)
    print("\nLayout:")
    print("  Split Screen")
    print("\nPanel Analysis")
    print(f"  Left Panel  : {left_modality}")
    print(f"  Right Panel : {right_modality}")

    # DOPPLER DETECTION
    doppler_locations = []
    if "colour_doppler" in left_result["labels"]:
        doppler_locations.append("LEFT PANEL")
    if "colour_doppler" in right_result["labels"]:
        doppler_locations.append("RIGHT PANEL")

    print("\nColour Doppler:")
    if len(doppler_locations) > 0:
        print("  DETECTED")
        for loc in doppler_locations:
            print(f"  -> {loc}")
    else:
        print("  NOT DETECTED")

    print("=" * 60)

    if len(doppler_locations):
        final_labels = ["split_screen_only", "colour_doppler"]
    else:
        final_labels = ["split_screen_only"]

    # ----------------------------------------------
    # ANATOMY CLASSIFICATION ON DOPPLER PANEL(S)
    # ----------------------------------------------
    # Whichever panel(s) were flagged as colour_doppler
    # get passed to the anatomy classifier individually.
    # ----------------------------------------------

    panel_images = {
        "left": split_result["left_img"],
        "right": split_result["right_img"],
    }

    anatomy_results = {}

    if "colour_doppler" in left_result["labels"]:
        anatomy_results["left"] = predict_anatomy(panel_images["left"])

    if "colour_doppler" in right_result["labels"]:
        anatomy_results["right"] = predict_anatomy(panel_images["right"])

    return {
        "layout": "split_screen",
        "left": left_result,
        "right": right_result,
        "split_ratio": best_ratio,
        "split_x": best_split_x,
        "split_score": best_score,
        "doppler_locations": doppler_locations,
        "final_labels": final_labels,
        "panel_images": panel_images,
        "anatomy_results": anatomy_results,
    }


# ============================================================
# QUADRANT EXTRACTOR
# ============================================================

def extract_quad_panels(image_bgr):
    h, w = image_bgr.shape[:2]
    mid_h = h // 2
    mid_w = w // 2

    return {
        "top_left": image_bgr[:mid_h, :mid_w],
        "top_right": image_bgr[:mid_h, mid_w:],
        "bottom_left": image_bgr[mid_h:, :mid_w],
        "bottom_right": image_bgr[mid_h:, mid_w:],
    }

# ============================================================
# QUADRANT ANALYSIS
# ============================================================

def analyze_quadrant(image_path):
    image_bgr = cv2.imread(str(image_path))
    panels = extract_quad_panels(image_bgr)

    results = {}
    for panel_name, panel_img in panels.items():
        results[panel_name] = predict_panel(panel_img)

    print("\n" + "=" * 60)
    print("HIERARCHICAL QUADRANT REPORT")
    print("=" * 60)
    print("\nLayout:")
    print("  Quadrant")
    print("\nPanel Analysis")

    for panel_name, result in results.items():
        labels = result["labels"]
        modality = labels[0] if len(labels) else "unknown"
        print(f"  {panel_name:15s}: {modality}")

    doppler_locations = []
    for panel_name, result in results.items():
        if "colour_doppler" in result["labels"]:
            doppler_locations.append(panel_name)

    print("\nColour Doppler:")
    if len(doppler_locations):
        print("  DETECTED")
        for loc in doppler_locations:
            print(f"  -> {loc}")
    else:
        print("  NOT DETECTED")

    print("=" * 60)

    if len(doppler_locations):
        final_labels = [
            "quadrant_images",
            "colour_doppler"
        ]
    else:
        final_labels = [
            "quadrant_images"
        ]

    # ----------------------------------------------
    # ANATOMY CLASSIFICATION ON DOPPLER PANEL(S)
    # ----------------------------------------------

    anatomy_results = {}

    for panel_name in doppler_locations:
        anatomy_results[panel_name] = predict_anatomy(
            panels[panel_name]
        )

    return {
        "layout": "quadrant",
        "results": results,
        "doppler_locations": doppler_locations,
        "final_labels": final_labels,
        "panel_images": panels,
        "anatomy_results": anatomy_results,
    }


# ============================================================
# PULSE IMAGE EXTRACTOR
# ============================================================

def extract_pulse_image_region(image_bgr):
    h, w = image_bgr.shape[:2]
    upper_region = image_bgr[:int(h * 0.55), :]
    return upper_region

# ============================================================
# PULSE ANALYSIS
# ============================================================

def analyze_pulse(image_path):
    image_bgr = cv2.imread(str(image_path))
    upper_region = extract_pulse_image_region(image_bgr)
    upper_result = predict_panel(upper_region)

    print("\n" + "=" * 60)
    print("HIERARCHICAL PULSE REPORT")
    print("=" * 60)
    print("\nLayout:")
    print("  Pulse Doppler")
    print("\nUpper Image Analysis")

    if len(upper_result["labels"]):
        for label in upper_result["labels"]:
            print(f"  -> {label}")
    else:
        print("  -> unknown")

    # COLOUR DOPPLER CHECK
    colour_found = "colour_doppler" in upper_result["labels"]

    print("\nColour Doppler:")
    if colour_found:
        print("  DETECTED")
    else:
        print("  NOT DETECTED")

    print("=" * 60)

    if colour_found:
        final_labels = [
            "pulse_doppler",
            "colour_doppler"
        ]
    else:
        final_labels = [
            "pulse_doppler"
        ]

    # ----------------------------------------------
    # ANATOMY CLASSIFICATION ON DOPPLER PANEL
    # ----------------------------------------------

    anatomy_results = {}
    panel_images = {"upper": upper_region}

    if colour_found:
        anatomy_results["upper"] = predict_anatomy(upper_region)

    return {
        "layout": "pulse_doppler",
        "upper_result": upper_result,
        "colour_detected": colour_found,
        "final_labels": final_labels,
        "panel_images": panel_images,
        "anatomy_results": anatomy_results,
    }


# ============================================================
# HIERARCHICAL POSTPROCESS
# ============================================================

def hierarchical_postprocess(
    image_path,
    stage1_result
):

    labels = stage1_result[
        "pred_labels"
    ]

    if "split_screen_only" in labels:

        return analyze_split_screen(
            image_path
        )

    elif "quadrant_images" in labels:

        return analyze_quadrant(
            image_path
        )

    elif "pulse_doppler" in labels:

        return analyze_pulse(
            image_path
        )

    return None

print(
    "✅ Hierarchical postprocess ready"
)


# ============================================================
# MASTER ROUTER
# ============================================================

def master_router(
    image_path
):

    image_bgr = cv2.imread(
        str(image_path)
    )

    # -------------------------
    # STAGE 1
    # -------------------------

    image_bgr = cv2.imread(
        str(image_path)
    )

    modality_result = predict_modality(
        image_bgr
    )

    dominant_label = modality_result[
        "dominant_label"
    ]

    # -------------------------
    # SPLIT
    # -------------------------

    if dominant_label == "split_screen_only":

        hier_result = \
            analyze_split_screen(
                image_path
            )

        return {

            "pipeline":
                "split",

            "modality":
                modality_result,

            "hierarchy":
                hier_result,

            "anatomy":
                hier_result.get("anatomy_results", {})
        }

    # -------------------------
    # QUADRANT
    # -------------------------

    if dominant_label == "quadrant_images":

        hier_result = \
            analyze_quadrant(
                image_path
            )

        return {

            "pipeline":
                "quadrant",

            "modality":
                modality_result,

            "hierarchy":
                hier_result,

            "anatomy":
                hier_result.get("anatomy_results", {})
        }

    # -------------------------
    # PULSE
    # -------------------------

    if dominant_label == "pulse_doppler":

        hier_result = \
            analyze_pulse(
                image_path
            )

        return {

            "pipeline":
                "pulse",

            "modality":
                modality_result,

            "hierarchy":
                hier_result,

            "anatomy":
                hier_result.get("anatomy_results", {})
        }

    # -------------------------
    # COLOUR DOPPLER
    # -------------------------

    if dominant_label == "colour_doppler":

        anatomy_result = \
            predict_anatomy(
                image_bgr
            )

        return {

            "pipeline":
                "anatomy",

            "modality":
                modality_result,

            "hierarchy":
                None,

            "anatomy":
                anatomy_result
        }

    # -------------------------
    # BMODE / TINTED
    # -------------------------

    return {

        "pipeline":
            "modality",

        "modality":
            modality_result,

        "hierarchy":
            None,

        "anatomy":
            None
    }

print(
    "✅ Master router ready"
)


# ============================================================
# TEST MASTER ROUTER
# ============================================================

test_image = SAMPLE_IMAGE_PATH

result = master_router(
    test_image
)

print()

print(
    "Pipeline :",
    result["pipeline"]
)

print()

print(
    "Dominant :",
    result["modality"]["dominant_label"]
)


# ============================================================
# ANATOMY VISUALIZATION
# ============================================================

def create_anatomy_visualization(

    original_bgr,

    anatomy_result,

    save_path,

    cam=None
):

    roi = anatomy_result["roi"]

    masks = anatomy_result["masks"]

    vessel_mask = masks["vessel_mask"]

    red_mask = masks["red_mask"]

    blue_mask = masks["blue_mask"]

    probs = anatomy_result["probs"]

    prediction = anatomy_result["prediction"]

    confidence = anatomy_result["confidence"]

    stats = masks["stats"]

    # --------------------------------------------------
    # GRADCAM
    # --------------------------------------------------

    if cam is None:

        inp = anatomy_result[
            "tensor6"
        ].unsqueeze(0).to(DEVICE)

        cam, _ = gradcam_anatomy.generate(inp)

    cam = cv2.resize(
        cam,
        (224,224)
    )

    roi224 = cv2.resize(
        roi,
        (224,224)
    )

    roi224_bgr = cv2.cvtColor(
        roi224,
        cv2.COLOR_RGB2BGR
    )

    heatmap = cv2.applyColorMap(
        np.uint8(cam * 255),
        cv2.COLORMAP_JET
    )

    gradcam_overlay = (

        0.6 * roi224_bgr +

        0.4 * heatmap

    ).astype(np.uint8)

    # --------------------------------------------------
    # CANVAS LAYOUT (cv2/numpy replacement for the
    # matplotlib figure — same information, no matplotlib
    # figure/savefig cost)
    # --------------------------------------------------

    PANEL = 360
    PAD = 20
    TITLE_H = 30
    HEADER_H = 300
    STATS_H = 130
    COLS = 3
    ROWS = 2

    canvas_w = COLS * PANEL + (COLS + 1) * PAD
    canvas_h = (
        HEADER_H
        + ROWS * (TITLE_H + PANEL)
        + (ROWS + 1) * PAD
        + STATS_H
        + PAD
    )

    canvas = np.ones(
        (canvas_h, canvas_w, 3),
        dtype=np.uint8
    ) * 255

    def place_panel(img, title, x, y):

        panel_img = img

        if panel_img.ndim == 2:
            panel_img = cv2.cvtColor(
                panel_img,
                cv2.COLOR_GRAY2BGR
            )

        h, w = panel_img.shape[:2]

        scale = min(PANEL / w, PANEL / h)

        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        resized = cv2.resize(
            panel_img,
            (new_w, new_h)
        )

        tile = np.zeros(
            (PANEL, PANEL, 3),
            dtype=np.uint8
        )

        ox = (PANEL - new_w) // 2
        oy = (PANEL - new_h) // 2

        tile[oy:oy+new_h, ox:ox+new_w] = resized

        canvas[y:y+PANEL, x:x+PANEL] = tile

        cv2.putText(
            canvas, title, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 0, 0), 1, cv2.LINE_AA
        )

        cv2.rectangle(
            canvas, (x, y), (x+PANEL, y+PANEL),
            (180, 180, 180), 1
        )

    # ==================================================
    # HEADER
    # ==================================================

    cv2.putText(
        canvas, "ULTRASOUND ANATOMY ANALYSIS", (PAD, 45),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA
    )

    cv2.putText(
        canvas, f"Prediction : {prediction}", (PAD, 95),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA
    )

    cv2.putText(
        canvas, f"Confidence : {confidence:.4f}", (PAD, 130),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA
    )

    # ==================================================
    # PROBABILITIES
    # ==================================================

    prob_y = 175

    bar_x = int(canvas_w * 0.45)
    bar_w = int(canvas_w * 0.30)

    for cls, prob in zip(
        ANATOMY_LABELS,
        probs
    ):

        cv2.putText(
            canvas, f"{cls:10s}", (PAD, prob_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA
        )

        cv2.rectangle(
            canvas,
            (bar_x, prob_y - 15),
            (bar_x + bar_w, prob_y + 5),
            (220, 220, 220), -1
        )

        cv2.rectangle(
            canvas,
            (bar_x, prob_y - 15),
            (bar_x + int(bar_w * prob), prob_y + 5),
            (0, 180, 0), -1
        )

        cv2.putText(
            canvas, f"{prob:.3f}",
            (bar_x + bar_w + 15, prob_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA
        )

        prob_y += 32

    # ==================================================
    # PANELS
    # ==================================================

    row1_y = HEADER_H + PAD + TITLE_H
    row2_y = row1_y + PANEL + PAD + TITLE_H

    col_x = [
        PAD,
        PAD + PANEL + PAD,
        PAD + 2 * (PANEL + PAD)
    ]

    place_panel(original_bgr, "Original Image", col_x[0], row1_y)
    place_panel(roi224_bgr, "ROI", col_x[1], row1_y)
    place_panel(vessel_mask, "Vessel Mask", col_x[2], row1_y)

    place_panel(red_mask, "Red Mask", col_x[0], row2_y)
    place_panel(blue_mask, "Blue Mask", col_x[1], row2_y)
    place_panel(gradcam_overlay, "GradCAM", col_x[2], row2_y)

    # ==================================================
    # STATS
    # ==================================================

    stats_y = row2_y + PANEL + PAD + 30

    stats_lines = [
        f"Flow %        : {stats['flow_pct']}",
        f"Flow Pixels   : {stats['flow_px']}",
        f"Red Pixels    : {stats['red_px']}",
        f"Blue Pixels   : {stats['blue_px']}"
    ]

    for i, line in enumerate(stats_lines):

        cv2.putText(
            canvas, line, (PAD, stats_y + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA
        )

    cv2.imwrite(
        str(save_path),
        canvas
    )

    return save_path


# ============================================================
# TEST ANATOMY VISUALIZATION
# ============================================================

test_image_path = SAMPLE_IMAGE_PATH

test_image = cv2.imread(
    str(test_image_path)
)

anatomy_result = predict_anatomy(
    test_image
)

save_path = (

    ANATOMY_VIS_DIR /

    f"{test_image_path.stem}_anatomy_visualization.png"
)

create_anatomy_visualization(

    test_image,

    anatomy_result,

    save_path
)

print(
    f"Saved to: {save_path}"
)


# ============================================================
# VISUALIZATION
# ============================================================

def create_visualization(
    image_bgr,
    image_name,
    probs,
    pred_labels,
    dominant_label,
    dominant_prob,
    modality,
    save_path,
    hier_result=None
):
    original = image_bgr.copy()
    h, w = original.shape[:2]

    canvas_h = max(h, 1600)
    canvas_w = w + 700

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    canvas[:h, :w] = original

    x0 = w + 25
    y = 40

    # HEADER
    cv2.putText(canvas, "ULTRASOUND ANALYSIS", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    y += 50
    cv2.putText(canvas, image_name, (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    # DOMINANT LABEL
    y += 60
    cv2.putText(canvas, f"Dominant : {dominant_label}", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 0), 2)

    y += 35
    cv2.putText(canvas, f"Confidence : {dominant_prob:.4f}", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

    # MODALITY
    y += 60
    cv2.putText(canvas, f"Modality : {modality}", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    # ACTIVE LABELS
    y += 70
    cv2.putText(canvas, "Active Labels", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    y += 40
    for label in pred_labels:
        cv2.putText(canvas, f"✓ {label}", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)
        y += 35

    # PROBABILITIES
    y += 30
    cv2.putText(canvas, "Class Probabilities", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    y += 40
    for idx, label in enumerate(LABEL_NAMES):
        prob = float(probs[idx])
        threshold = CLASS_THRESHOLDS[label]

        cv2.putText(canvas, label, (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        bar_x = x0
        bar_y = y + 10

        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + 300, bar_y + 20), (220, 220, 220), -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + int(prob * 300), bar_y + 20), (0, 180, 0), -1)

        cv2.putText(canvas, f"{prob:.3f}", (bar_x + 320, bar_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        cv2.putText(canvas, f"T={threshold:.3f}", (bar_x + 420, bar_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)

        y += 55

    # LAYOUT
    y += 20
    cv2.putText(canvas, "Layout Interpretation", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    y += 40
    if "split_screen_only" in pred_labels:
        cv2.putText(canvas, "Detected Split Screen", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        y += 35

    if "quadrant_images" in pred_labels:
        cv2.putText(canvas, "Detected Quadrant Layout", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        y += 35

    # HIERARCHICAL ANALYSIS
    if hier_result is not None:
        y += 20
        cv2.putText(canvas, "Hierarchical Analysis", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        y += 40
        cv2.putText(canvas, f"Layout : {hier_result['layout']}", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

        y += 35
        if hier_result.get("layout") == "split_screen":
            left_mod  = hier_result["left"]["labels"][0]  if len(hier_result["left"]["labels"])  else "unknown"
            right_mod = hier_result["right"]["labels"][0] if len(hier_result["right"]["labels"]) else "unknown"
            cv2.putText(canvas, f"Left Panel  : {left_mod}", (x0, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
            y += 30
            cv2.putText(canvas, f"Right Panel : {right_mod}", (x0, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
            y += 30
            doppler_loc = ", ".join(hier_result["doppler_locations"]) if hier_result["doppler_locations"] else "None"
            cv2.putText(canvas, f"Doppler Location : {doppler_loc}", (x0, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
            y += 30
            ####################################################
            # SPLIT OVERLAY
            ####################################################

            mid_x = hier_result["split_x"]

            cv2.line(
                canvas,
                (mid_x,0),
                (mid_x,h),
                (0,255,255),
                3
            )
            ####################################################
            # PANEL LABELS ON IMAGE
            ####################################################

            left_label = ", ".join(
                hier_result["left"]["labels"]
            )

            right_label = ", ".join(
                hier_result["right"]["labels"]
            )

            cv2.putText(
                canvas,
                f"LEFT: {left_label}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

            cv2.putText(
                canvas,
                f"RIGHT: {right_label}",
                (mid_x + 20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

            if "LEFT PANEL" in hier_result["doppler_locations"]:

                cv2.rectangle(
                    canvas,
                    (10,10),
                    (mid_x-10,h-10),
                    (0,0,255),
                    5
                )

            if "RIGHT PANEL" in hier_result["doppler_locations"]:

                cv2.rectangle(
                    canvas,
                    (mid_x+10,10),
                    (w-10,h-10),
                    (0,0,255),
                    5
                )

        elif hier_result.get("layout") == "quadrant":
            for panel_name, res in hier_result["results"].items():
                mod = res["labels"][0] if len(res["labels"]) else "unknown"
                cv2.putText(canvas, f"{panel_name:15s}: {mod}", (x0, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
                y += 30
            doppler_loc = ", ".join(hier_result["doppler_locations"]) if hier_result["doppler_locations"] else "None"
            cv2.putText(canvas, f"Doppler Location : {doppler_loc}", (x0, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
            y += 30
            ####################################################
            # QUADRANT GRID OVERLAY
            ####################################################

            mid_h = h // 2
            mid_w = w // 2

            cv2.line(
                canvas,
                (mid_w, 0),
                (mid_w, h),
                (0,255,255),
                3
            )

            cv2.line(
                canvas,
                (0, mid_h),
                (w, mid_h),
                (0,255,255),
                3
            )
            
            ####################################################
            # PANEL LABELS ON IMAGE
            ####################################################

            panel_positions = {

                "top_left": (20,40),

                "top_right": (mid_w + 20,40),

                "bottom_left": (20,mid_h + 40),

                "bottom_right": (mid_w + 20,mid_h + 40)
            }

            for panel_name, res in hier_result["results"].items():

                label = ", ".join(
                    res["labels"]
                )

                cv2.putText(

                    canvas,

                    f"{panel_name}: {label}",

                    panel_positions[panel_name],

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    (0,255,0),

                    2
                )
            ####################################################
            # DOPPLER HIGHLIGHT BOXES
            ####################################################

            for loc in hier_result["doppler_locations"]:

                if loc == "top_left":

                    cv2.rectangle(
                        canvas,
                        (10,10),
                        (mid_w-10, mid_h-10),
                        (0,0,255),
                        5
                    )

                elif loc == "top_right":

                    cv2.rectangle(
                        canvas,
                        (mid_w+10,10),
                        (w-10, mid_h-10),
                        (0,0,255),
                        5
                    )

                elif loc == "bottom_left":

                    cv2.rectangle(
                        canvas,
                        (10, mid_h+10),
                        (mid_w-10, h-10),
                        (0,0,255),
                        5
                    )

                elif loc == "bottom_right":

                    cv2.rectangle(
                        canvas,
                        (mid_w+10, mid_h+10),
                        (w-10, h-10),
                        (0,0,255),
                        5
                    )
        elif hier_result.get("layout") == "pulse_doppler":
            upper_labels = ", ".join(hier_result["upper_result"]["labels"]) if hier_result["upper_result"]["labels"] else "unknown"
            cv2.putText(canvas, f"Upper Region : {upper_labels}", (x0, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
            y += 30
            colour_str = "DETECTED" if hier_result["colour_detected"] else "NOT DETECTED"
            cv2.putText(canvas, f"Colour Doppler : {colour_str}", (x0, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
            y += 30
            ####################################################
            # PULSE REGION OVERLAY
            ####################################################

            cutoff = int(h * 0.55)

            cv2.line(
                canvas,
                (0, cutoff),
                (w, cutoff),
                (0,255,255),
                3
            )
            ####################################################
            # UPPER LABEL ON IMAGE
            ####################################################

            upper_label = ", ".join(
                hier_result["upper_result"]["labels"]
            )

            upper_color = (
                (0,0,255)
                if "colour_doppler" in hier_result["upper_result"]["labels"]
                else (0,255,0)
            )

            cv2.putText(

                canvas,

                f"UPPER: {upper_label}",

                (20,40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                upper_color,

                2
            )

            if hier_result["colour_detected"]:

                cv2.rectangle(
                    canvas,
                    (10,10),
                    (w-10, cutoff-10),
                    (0,0,255),
                    5
                )
    print("Saving visualization:",save_path)

    if hier_result is not None:
        print(
        "Hierarchical visualization added"
        )
    cv2.imwrite(str(save_path), canvas)


print(create_visualization)


# ============================================================
# UNIFIED VISUALIZATION DISPATCHER
# ============================================================

def save_prediction_visualization(

    image_path,

    router_result,

    save_path,

    cam=None
):

    image_bgr = cv2.imread(
        str(image_path)
    )

    pipeline = router_result[
        "pipeline"
    ]

    # --------------------------------------------------
    # ANATOMY PIPELINE
    # --------------------------------------------------

    if pipeline == "anatomy":

        create_anatomy_visualization(

            image_bgr,

            router_result["anatomy"],

            save_path,

            cam=cam
        )

        return

    # --------------------------------------------------
    # MODALITY PIPELINE
    # --------------------------------------------------

    create_visualization(

        image_bgr=image_bgr,

        image_name=image_path.name,

        probs=router_result["modality"]["probs"],

        pred_labels=router_result["modality"]["pred_labels"],

        dominant_label=
            router_result["modality"]["dominant_label"],

        dominant_prob=
            router_result["modality"]["dominant_prob"],

        modality=
            router_result["pipeline"],

        save_path=save_path,

        hier_result=
            router_result["hierarchy"]
    )

print(
    "✅ Visualization dispatcher ready"
)


# ============================================================
# TEST VISUALIZATION DISPATCHER
# ============================================================

test_image = SAMPLE_IMAGE_PATH

router_result = master_router(
    test_image
)

save_path = (

    VIS_DIR /

    f"{test_image.stem}_final.png"
)

save_prediction_visualization(

    test_image,

    router_result,

    save_path
)

print(
    f"Saved: {save_path}"
)


# ============================================================
# SHARED ANATOMY CASE SAVER
# ============================================================
# Used both for full-frame colour_doppler images and for
# doppler panels cropped out of split/quadrant/pulse layouts.
# Saves gradcam/masks/roi, copies (or writes) the classified
# image into its predicted anatomy folder, saves the
# visualization, and handles the uncertain-case copy.
# ============================================================

ANATOMY_DIR_MAP_KEYS = {
    "3VC": lambda: ANATOMY_3VC_DIR,
    "3VT": lambda: ANATOMY_3VT_DIR,
    "Renal": lambda: ANATOMY_RENAL_DIR,
    "Inflows": lambda: ANATOMY_INFLOWS_DIR,
}

def save_anatomy_case(
    anatomy,
    source_bgr,
    stem_tag,
    copy_original_path=None
):

    inp = anatomy["tensor6"].unsqueeze(0).to(DEVICE)

    cam_raw, _ = gradcam_anatomy.generate(inp)

    cam = cv2.resize(cam_raw, (224, 224))

    cv2.imwrite(
        str(GRADCAM_DIR / f"{stem_tag}.png"),
        np.uint8(cam * 255)
    )

    cv2.imwrite(
        str(VESSEL_MASK_DIR / f"{stem_tag}.png"),
        anatomy["masks"]["vessel_mask"]
    )

    cv2.imwrite(
        str(RED_MASK_DIR / f"{stem_tag}.png"),
        anatomy["masks"]["red_mask"]
    )

    cv2.imwrite(
        str(ROI_DIR / f"{stem_tag}.png"),
        cv2.cvtColor(anatomy["roi"], cv2.COLOR_RGB2BGR)
    )

    cv2.imwrite(
        str(BLUE_MASK_DIR / f"{stem_tag}.png"),
        anatomy["masks"]["blue_mask"]
    )

    pred_class = anatomy["prediction"]

    anatomy_dir = ANATOMY_DIR_MAP_KEYS.get(
        pred_class,
        ANATOMY_DIR_MAP_KEYS["Inflows"]
    )()

    # ----------------------------------
    # PUT THE CLASSIFIED IMAGE INTO THE
    # PREDICTED ANATOMY FOLDER
    # ----------------------------------
    # Full-frame colour_doppler: copy the original file as-is.
    # Doppler panel crop: no original file on disk, so write
    # the panel pixels directly.
    # ----------------------------------

    if copy_original_path is not None:

        shutil.copy(
            copy_original_path,
            anatomy_dir / f"{stem_tag}{copy_original_path.suffix}"
        )

    else:

        cv2.imwrite(
            str(anatomy_dir / f"{stem_tag}.png"),
            source_bgr
        )

    # ----------------------------------
    # VISUALIZATION
    # ----------------------------------

    vis_save_path = VIS_DIR / f"{stem_tag}.png"

    create_anatomy_visualization(
        source_bgr,
        anatomy,
        vis_save_path,
        cam=cam_raw
    )

    # ----------------------------------
    # UNCERTAIN CASES
    # ----------------------------------

    if anatomy["confidence"] < UNCERTAINTY_THRESHOLD:

        shutil.copy(
            vis_save_path,
            UNCERTAIN_VIS_DIR / f"{stem_tag}.png"
        )

    stats = anatomy["masks"]["stats"]

    return {
        "anatomy_prediction": anatomy["prediction"],
        "anatomy_confidence": anatomy["confidence"],
        "flow_pct": stats["flow_pct"],
        "flow_px": stats["flow_px"],
        "red_px": stats["red_px"],
        "blue_px": stats["blue_px"],
    }

print("✅ Anatomy case saver ready")


# ============================================================
# FOLDER INFERENCE ENGINE
# ============================================================

def run_folder_inference(
    input_folder=INPUT_FOLDER
):

    input_folder = Path(input_folder)

    records = []

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    }

    image_paths = [

        p for p in input_folder.rglob("*")

        if p.suffix.lower() in image_extensions
    ]

    print(f"\nFound {len(image_paths)} images")

    # ----------------------------------
    # PATIENT ID PREFIX (from the case
    # subfolder name, e.g.
    # "K58918_HEMALATHA_AKASH" -> "K58918")
    # ----------------------------------

    patient_id = input_folder.name.split("_")[0]

    for image_path in tqdm(image_paths):

        try:

            router_result = master_router(
                image_path
            )

            # ----------------------------------
            # SAVE LOCATION
            # ----------------------------------

            anatomy_fields = None

            if router_result["pipeline"] == "anatomy":

                anatomy = router_result["anatomy"]

                anatomy_fields = save_anatomy_case(
                    anatomy=anatomy,
                    source_bgr=cv2.imread(str(image_path)),
                    stem_tag=f"{patient_id}_{image_path.stem}",
                    copy_original_path=image_path
                )

            else:

                # ----------------------------------
                # SAVE VISUALIZATION (MAIN VIS FOLDER)
                # split / quadrant / pulse / plain modality
                # ----------------------------------

                save_path = (
                    VIS_DIR /
                    f"{patient_id}_{image_path.stem}.png"
                )

                save_prediction_visualization(
                    image_path,
                    router_result,
                    save_path,
                    cam=None
                )

            # ----------------------------------
            # RECORD (WHOLE-IMAGE ROW)
            # ----------------------------------

            record = {

                "filename":
                    image_path.name,

                "pipeline":
                    router_result["pipeline"],

                "dominant_label":
                    router_result["modality"]["dominant_label"],

                "dominant_prob":
                    router_result["modality"]["dominant_prob"]
            }

            if anatomy_fields is not None:

                record.update(anatomy_fields)

            else:

                record.update({
                    "anatomy_prediction": None,
                    "anatomy_confidence": None,
                    "flow_pct": None,
                    "flow_px": None,
                    "red_px": None,
                    "blue_px": None
                })

            records.append(record)

            # ----------------------------------
            # DOPPLER PANEL(S) FROM SPLIT / QUADRANT /
            # PULSE LAYOUTS — each detected doppler panel
            # is classified and saved the same way as a
            # full-frame colour_doppler image.
            # ----------------------------------

            if router_result["pipeline"] in ("split", "quadrant", "pulse"):

                panel_anatomy = router_result["anatomy"]  # dict: panel_name -> anatomy result / None
                panel_images = router_result["hierarchy"]["panel_images"]

                for panel_name, panel_anatomy_result in panel_anatomy.items():

                    if panel_anatomy_result is None:
                        continue

                    panel_stem_tag = f"{patient_id}_{image_path.stem}_{panel_name}"

                    panel_fields = save_anatomy_case(
                        anatomy=panel_anatomy_result,
                        source_bgr=panel_images[panel_name],
                        stem_tag=panel_stem_tag,
                        copy_original_path=None
                    )

                    panel_record = {
                        "filename": f"{image_path.name} [{panel_name}]",
                        "pipeline": f"{router_result['pipeline']}_panel",
                        "dominant_label": "colour_doppler",
                        "dominant_prob":
                            router_result["modality"]["dominant_prob"]
                    }

                    panel_record.update(panel_fields)

                    records.append(panel_record)

        except Exception as e:

            print(
                f"Failed: {image_path}"
            )

            print(e)

    df = pd.DataFrame(
        records
    )

    return df

print("\nOUTPUT")
print("="*60)
print("Visualizations :", VIS_DIR)
print("Anatomy :", ANATOMY_VIS_DIR)
print("GradCAM :", GRADCAM_DIR)
print("ROI :", ROI_DIR)
print("Reports :", REPORT_DIR)
print("="*60)



# ============================================================
# RUN INFERENCE — MASTER FOLDER (ALL CASE SUBFOLDERS)
# ============================================================
# INPUT_FOLDER is the master folder. Every immediate
# subfolder inside it (one per case) is processed in turn.
# For each case subfolder "X", results go to:
#   MASTER_OUTPUT_FOLDER / X / (visualizations, anatomy_vis,
#                                uncertain_cases, roi, masks,
#                                gradcam, reports)
# i.e. the same structure as before, just nested per case.
# ============================================================

def process_case_folder(case_input_folder, case_output_folder):
    """
    Runs the full existing pipeline (run_folder_inference +
    predictions.csv + summary.txt + uncertain_cases.csv) for
    a single case folder, writing into case_output_folder.
    Identical logic to the original single-folder driver,
    just parameterized so it can run per-subfolder.
    """

    setup_output_dirs(case_output_folder)

    print("\n" + "=" * 60)
    print("CASE :", case_input_folder.name)
    print("=" * 60)
    print("Input  :", case_input_folder)
    print("Output :", case_output_folder)
    print("Visualizations :", VIS_DIR)
    print("Anatomy :", ANATOMY_VIS_DIR)
    print("GradCAM :", GRADCAM_DIR)
    print("ROI :", ROI_DIR)
    print("Reports :", REPORT_DIR)
    print("=" * 60)

    # --------------------------------------------------
    # RUN INFERENCE
    # --------------------------------------------------

    results_df = run_folder_inference(
        input_folder=case_input_folder
    )

    print(results_df.head())

    print(f"\nTotal Images: {len(results_df)}")

    # --------------------------------------------------
    # SAVE PREDICTIONS CSVmmas
    # --------------------------------------------------

    csv_path = REPORT_DIR / "predictions.csv"

    results_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 60)
    print("PREDICTIONS CSV SAVED")
    print("=" * 60)
    print(f"Total Records : {len(results_df)}")
    print(f"Location      : {csv_path}")

    # --------------------------------------------------
    # SUMMARY REPORT
    # --------------------------------------------------

    summary_path = REPORT_DIR / "summary.txt"

    with open(summary_path, "w") as f:

        f.write("INTEGRATED ULTRASOUND ANALYSIS REPORT\n")
        f.write(f"Case : {case_input_folder.name}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total Images : {len(results_df)}\n\n")

        f.write("PIPELINE DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        f.write(str(results_df["pipeline"].value_counts()))
        f.write("\n\n")

        f.write("MODALITY DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        f.write(str(results_df["dominant_label"].value_counts()))
        f.write("\n\n")

        anatomy_df = results_df[
            results_df["anatomy_prediction"].notna()
        ]

        if len(anatomy_df):

            f.write("ANATOMY DISTRIBUTION\n")
            f.write("-" * 40 + "\n")
            f.write(str(anatomy_df["anatomy_prediction"].value_counts()))
            f.write("\n\n")

            f.write("ANATOMY CONFIDENCE\n")
            f.write("-" * 40 + "\n")
            f.write(f"Mean : {anatomy_df['anatomy_confidence'].mean():.4f}\n")
            f.write(f"Min  : {anatomy_df['anatomy_confidence'].min():.4f}\n")
            f.write(f"Max  : {anatomy_df['anatomy_confidence'].max():.4f}\n")
            f.write("\n")

            f.write("FLOW STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Mean Flow % : {anatomy_df['flow_pct'].mean():.2f}\n")
            f.write(f"Mean Red Px : {anatomy_df['red_px'].mean():.0f}\n")
            f.write(f"Mean Blue Px : {anatomy_df['blue_px'].mean():.0f}\n")
            f.write("\n")

        uncertain_count = len(
            results_df[
                (results_df["dominant_prob"] < UNCERTAINTY_THRESHOLD)
                |
                (results_df["anatomy_confidence"] < UNCERTAINTY_THRESHOLD)
            ]
        )

        f.write("UNCERTAIN CASES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Count : {uncertain_count}\n\n")

        f.write("=" * 70 + "\n")

    print("\n" + "=" * 60)
    print("SUMMARY REPORT SAVED")
    print("=" * 60)
    print(f"Location : {summary_path}")

    # --------------------------------------------------
    # UNCERTAIN CASES REPORT
    # --------------------------------------------------

    uncertain_df = results_df[
        (results_df["anatomy_confidence"] < UNCERTAINTY_THRESHOLD)
        |
        (results_df["dominant_prob"] < UNCERTAINTY_THRESHOLD)
    ]

    uncertain_path = REPORT_DIR / "uncertain_cases.csv"

    uncertain_df.to_csv(uncertain_path, index=False)

    print(f"Uncertain Cases: {len(uncertain_df)}")
    print(f"Saved: {uncertain_path}")

    return results_df



# ============================================================
# DISCOVER CASE SUBFOLDERS AND RUN
# ============================================================

case_folders = sorted(
    p for p in INPUT_FOLDER.iterdir()
    if p.is_dir()
)

print(f"Found {len(case_folders)} case folders under {INPUT_FOLDER}")

all_case_results = {}

for case_folder in case_folders:

    case_output_folder = (
        MASTER_OUTPUT_FOLDER
        / case_folder.name
    )

    try:
        all_case_results[case_folder.name] = process_case_folder(
            case_input_folder=case_folder,
            case_output_folder=case_output_folder
        )

    except Exception as e:

        print(f"Failed case folder: {case_folder}")
        print(e)

print("\n" + "=" * 60)
print("ALL CASES DONE")
print("=" * 60)
print(f"Master output folder : {MASTER_OUTPUT_FOLDER}")
print(f"Cases processed       : {len(all_case_results)} / {len(case_folders)}")