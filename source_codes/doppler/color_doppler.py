from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm


# ============================================================
# CONSTANTS
# ============================================================

ANATOMY_LABELS = [
    "3 Vessel Cord",
    "3 Vessel Trachea",
    "Renal Arteries",
    "Inflows",
]

ANATOMY_CHECKPOINT = Path(
    "/mnt/data/agilaa/checkpoints-anatomy-20260909T104119Z-1-001/checkpoints-anatomy/best_f1.pth"
)


# ============================================================
# DOPPLER FLOW LOCALIZER 
# ============================================================

class DopplerFlowLocalizer:
    """
    Extracts color Doppler flow regions using HSV analysis.
    Red  = flow toward transducer
    Blue = flow away from transducer
    """

    # HSV range for red flow (wraps around 0/180)
    RED_LOWER1 = np.array([0,   80,  80])
    RED_UPPER1 = np.array([10, 255, 255])
    RED_LOWER2 = np.array([165, 80,  80])
    RED_UPPER2 = np.array([180, 255, 255])

    # HSV range for blue flow
    BLUE_LOWER = np.array([100, 80,  80])
    BLUE_UPPER = np.array([130, 255, 255])

    def __init__(self, min_area=50):
        self.min_area = min_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

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
        red_mask = cv2.bitwise_or(red1, red2)

        # Blue mask
        blue_mask = cv2.inRange(hsv, self.BLUE_LOWER, self.BLUE_UPPER)

        # Combined
        flow_mask = cv2.bitwise_or(red_mask, blue_mask)

        # Morphological cleanup
        flow_mask = cv2.morphologyEx(flow_mask, cv2.MORPH_OPEN, self.kernel)
        flow_mask = cv2.morphologyEx(flow_mask, cv2.MORPH_CLOSE, self.kernel)
        flow_mask = cv2.dilate(flow_mask, self.kernel, iterations=1)

        # Remove Doppler color scale and machine overlays
        flow_mask_clean = flow_mask.copy()
        h, w = flow_mask_clean.shape

        flow_mask_clean[:, :100] = 0        # left color scale
        flow_mask_clean[:, w - 150:] = 0    # right machine text
        flow_mask_clean[:50, :] = 0         # top overlay
        flow_mask_clean[h - 50:, :] = 0     # bottom overlay

        # Stats
        total_px = flow_mask_clean.size
        flow_px = flow_mask_clean.sum() / 255
        flow_pct = 100.0 * flow_px / total_px

        bbox = None
        contours, _ = cv2.findContours(
            flow_mask_clean,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        large_contours = [c for c in contours if cv2.contourArea(c) >= self.min_area]
        if large_contours:
            all_pts = np.vstack(large_contours)
            x, y, w, h = cv2.boundingRect(all_pts)
            bbox = (x, y, x + w, y + h)

        # Colored overlay
        overlay = img_rgb.copy()
        overlay[red_mask > 0] = [255, 60, 60]
        overlay[blue_mask > 0] = [60, 60, 255]

        stats = {
            "flow_pct": round(flow_pct, 2),
            "flow_px": int(flow_px),
            "red_px": int((red_mask > 0).sum()),
            "blue_px": int((blue_mask > 0).sum()),
            "bbox": bbox,
        }
        return flow_mask, red_mask, blue_mask, overlay, stats


# ============================================================
# ROI EXTRACTOR  (reused as-is)
# ============================================================

def extract_roi(img_rgb: np.ndarray, localizer: DopplerFlowLocalizer) -> Optional[dict]:
    flow_mask, red_mask, blue_mask, overlay, stats = localizer.extract(img_rgb)

    bbox = stats["bbox"]
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    pad_x = max(int(w * 0.5), 60)
    pad_y = max(int(h * 0.5), 60)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_rgb.shape[1], x2 + pad_x)
    y2 = min(img_rgb.shape[0], y2 + pad_y)

    roi = img_rgb[y1:y2, x1:x2]

    return {
        "roi": roi,
        "bbox": (x1, y1, x2, y2),
        "flow_mask": flow_mask,
        "red_mask": red_mask,
        "blue_mask": blue_mask,
        "stats": stats,
    }


# ============================================================
# MASK GENERATION  (reused as-is)
# ============================================================

def generate_masks(roi_rgb: np.ndarray, localizer: DopplerFlowLocalizer) -> dict:
    flow_mask, red_mask, blue_mask, _, stats = localizer.extract(roi_rgb)

    vessel_mask = (flow_mask > 0).astype(np.uint8) * 255
    red_mask = (red_mask > 0).astype(np.uint8) * 255
    blue_mask = (blue_mask > 0).astype(np.uint8) * 255

    return {
        "vessel_mask": vessel_mask,
        "red_mask": red_mask,
        "blue_mask": blue_mask,
        "stats": stats,
    }


# ============================================================
# BUILD 6-CHANNEL TENSOR  (reused as-is)
# ============================================================

def build_6channel_tensor(roi_rgb, vessel_mask, red_mask, blue_mask) -> torch.Tensor:
    roi_rgb = cv2.resize(roi_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    vessel_mask = cv2.resize(vessel_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
    red_mask = cv2.resize(red_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
    blue_mask = cv2.resize(blue_mask, (224, 224), interpolation=cv2.INTER_NEAREST)

    roi_rgb = roi_rgb.astype(np.float32) / 255.0
    vessel_mask = vessel_mask.astype(np.float32) / 255.0
    red_mask = red_mask.astype(np.float32) / 255.0
    blue_mask = blue_mask.astype(np.float32) / 255.0

    tensor = np.dstack([roi_rgb, vessel_mask, red_mask, blue_mask])
    tensor = torch.tensor(tensor, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1)  # [C, H, W]

    return tensor


# ============================================================
# ANATOMY MODEL  (reused as-is)
# ============================================================

class AnatomyEfficientNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            in_chans=6,
            num_classes=0,
            global_pool="avg",
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
            nn.Linear(128, 4),
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)


# ============================================================
# MODEL LOADING (startup-time only — call once)
# ============================================================

def load_color_doppler_model(
    checkpoint_path: Path = ANATOMY_CHECKPOINT,
    device: Optional[torch.device] = None,
) -> "tuple[AnatomyEfficientNet, torch.device]":
    """
    Loads the Anatomy checkpoint once. Call this during application startup,
    never per-image / per-request.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AnatomyEfficientNet().to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    return model, device


# ============================================================
# PRODUCTION CLASSIFIER
# ============================================================

class ColourDopplerAnatomyClassifier:
    """
    Wraps an already-loaded Anatomy model for per-panel inference.

    Usage (startup):
        anatomy_model, device = load_anatomy_model()
        anatomy_classifier = ColourDopplerAnatomyClassifier(anatomy_model, device)

    Usage (per panel, inside the existing panel loop):
        result = anatomy_classifier.classify_colour_doppler_panel(
            panel_bgr=panel,
            panel_modality_result=modality_result,
        )
    """

    def __init__(
        self,
        model: AnatomyEfficientNet,
        device: torch.device,
        localizer: Optional[DopplerFlowLocalizer] = None,
    ):
        self.model = model
        self.device = device
        self.localizer = localizer or DopplerFlowLocalizer()

    @staticmethod
    def _is_colour_doppler(panel_modality_result: Dict[str, Any]) -> bool:
        """
        Adapts to either representation of the modality result:
          - {"labels": [...]}       (multi-label list)
          - {"pred_labels": [...]}  (as produced by the existing predict_modality())
          - {"dominant_label": "..."}
        """
        if panel_modality_result is None:
            return False

        labels = (
            panel_modality_result.get("labels")
            or panel_modality_result.get("pred_labels")
            or []
        )
        if "colour_doppler" in labels:
            return True

        dominant = panel_modality_result.get("dominant_label")
        if dominant == "colour_doppler":
            return True

        return False

    @torch.no_grad()
    def classify_colour_doppler_panel(
        self,
        panel_bgr: np.ndarray,
        panel_modality_result: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Second-stage classifier for an already-extracted Colour Doppler panel.

        Does NOT run any modality/panel/layout inference itself — it trusts
        panel_modality_result, which the caller has already computed.

        Returns None if the panel is not Colour Doppler (caller should route
        B-mode/Tinted panels to the existing 27-class plane classifier instead).
        """
        if not self._is_colour_doppler(panel_modality_result):
            return None

        img_rgb = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)

        roi_result = extract_roi(img_rgb, self.localizer)
        if roi_result is None:
            return {
                "is_colour_doppler": True,
                "prediction": None,
                "reason": "no_doppler_roi",
            }

        masks = generate_masks(roi_result["roi"], self.localizer)

        tensor6 = build_6channel_tensor(
            roi_result["roi"],
            masks["vessel_mask"],
            masks["red_mask"],
            masks["blue_mask"],
        )

        inp = tensor6.unsqueeze(0).to(self.device)

        logits = self.model(inp)
        probs = torch.softmax(logits, dim=1)
        probs = probs.squeeze().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        return {
            "is_colour_doppler": True,
            "prediction": ANATOMY_LABELS[pred_idx],
            "pred_idx": pred_idx,
            "confidence": confidence,
            "probs": {
                label: float(probs[i]) for i, label in enumerate(ANATOMY_LABELS)
            },
        }


# ============================================================
# EXAMPLE INTEGRATION (not executed on import)
# ============================================================

def _example_integration():
    """
    Illustrates how the existing production panel loop calls this module.
    This function is not part of the production API — it is documentation.
    """
    # --- startup, once ---
    anatomy_model, device = load_anatomy_model()
    anatomy_classifier = ColourDopplerAnatomyClassifier(anatomy_model, device)

    # --- existing per-image production logic (NOT reimplemented here) ---
    # extracted_panels = existing_split_quadrant_pulse_extractor(full_image)

    extracted_panels = []  # placeholder — supplied by existing pipeline

    for panel in extracted_panels:
        # Already implemented elsewhere in production:
        modality_result = existing_panel_modality_classifier(panel)  # noqa: F821

        if panel_is_bmode_or_tinted(modality_result):  # noqa: F821
            plane_result = existing_27_class_plane_classifier(panel)  # noqa: F821
            _ = plane_result

        elif panel_is_colour_doppler(modality_result):  # noqa: F821
            anatomy_result = anatomy_classifier.classify_colour_doppler_panel(
                panel_bgr=panel,
                panel_modality_result=modality_result,
            )
            _ = anatomy_result