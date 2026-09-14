# --- UNIFIED INFERENCE (Replaces Old Inference) ---
import os
import cv2
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import shutil
import torch 
import torchvision.transforms as T
import json
from source_codes.modality.models import HierarchicalUltrasoundModel
from utils.image_utils import load_image
# from models import HierarchicalUltrasoundModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── CONFIG: set your image folder here ───────────────────────────────────────
# folder = r"C:\Users\laksh\ANU\12_Full_Image_Datasets\10"
INPUT_MODE   = 'folder'

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

plane2idx = {
    "3 Vessel ViewOrTrachea or PAS": 0,
    "4 Chamber View of Heart": 1,
    "Abdominal Circumference": 2,
    "Amniotic Fluid or Liquor": 3,
    "Both Feet": 4,
    "Cervix": 5,
    "Cord Insertion": 6,
    "Coronal Kidneys": 7,
    "Coronal Spine": 8,
    "Femur": 9,
    "Full Body Coronal View": 10,
    "Humerus": 11,
    "LVOT": 12,
    "Median Facial Profile": 13,
    "Nose and Mouth": 14,
    "Open Hands": 15,
    "Orbits and Lenses": 16,
    "Placenta": 17,
    "Premaxillary Triangle": 18,
    "RVOT": 19,
    "Radius and Ulna": 20,
    "Sagittal Spine": 21,
    "Tibia and Fibula": 22,
    "Transcerebellar plane": 23,
    "Transthalamic plane": 24,
    "Transventricular plane": 25,
    "Transverse Kidneys": 26
  }

idx2plane = {idx: plane for plane, idx in plane2idx.items()}


anatomy2idx = {
    "Abdomen": 0,
    "Face": 1,
    "Fetal Environment": 2,
    "Head": 3,
    "Lower Limbs": 4,
    "Spine": 5,
    "Thorax": 6,
    "Upper Limbs": 7
}

idx2anatomy = {
    idx: anatomy
    for anatomy, idx in anatomy2idx.items()
}

valid_planes_for_anatomy = {
    "0": [2, 6, 7, 26],
    "3": [25, 24, 23],
    "6": [1, 12, 19, 0],
    "7": [11, 20, 15],
    "4": [9, 22, 4],
    "1": [13, 14, 16, 18],
    "5": [21, 8, 10],
    "2": [3, 17, 5]
  }

valid_planes_for_anatomy = {
    int(k): [int(x) for x in v]
    for k, v in valid_planes_for_anatomy.items()
}

NUM_ANATOMIES = len(idx2anatomy)
NUM_PLANES = len(idx2plane)

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.dcm', '.dicom'}
MODEL_PATH= "/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/weights/CLASS/hierarchicalmodel27 (1).pt"
model = HierarchicalUltrasoundModel(
num_anatomies=NUM_ANATOMIES,
num_planes=NUM_PLANES,
backbone_name='convnext_small',
pretrained=False,
dropout=0.3,
).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)

if isinstance(checkpoint, dict) and "model" in checkpoint:
    state_dict = checkpoint["model"]
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict, strict= True)
model = model.to(device)
model.eval()



def get_image_paths(mode, folder=None, df=None):
    """Return a list of absolute image path strings from a folder or a dataframe."""
    if mode == 'folder':
        folder = Path(folder)
        paths  = sorted([str(p) for p in folder.rglob('*') if p.suffix.lower() in SUPPORTED_EXTS])
        print(f'Found {len(paths)} images in {folder}')
        return paths
    else:
        paths = [p for p in df['image_path'].values if isinstance(p, str) and os.path.exists(p)]
        print(f'Found {len(paths)} valid image paths from CSV')
        return paths
# ─────────────────────────────────────────────────────────────────────────────

LABEL_NAMES = ['b-mode', 'tinted', 'colour_doppler', 'pulse_doppler', 'split_screen_only', 'quadrant_images']
CLASS_THRESHOLDS = {'b-mode': 0.950, 'tinted': 0.376, 'colour_doppler': 0.926, 'pulse_doppler': 0.950, 'split_screen_only': 0.950, 'quadrant_images': 0.950}


class UltrasoundClassifier(nn.Module):
    def __init__(self, num_classes, dense1, dense2, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0, global_pool='avg')
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, dense1), nn.BatchNorm1d(dense1), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(dense1, dense2), nn.BatchNorm1d(dense2), nn.SiLU(), nn.Dropout(dropout*0.7),
            nn.Linear(dense2, num_classes)
        )
    def forward(self, x):
        return self.head(self.backbone(x))

def load_modality_model(model_path):
    print("Loading Stage 1 Modality Classifier...")

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    dense1 = state_dict["head.0.weight"].shape[0]
    dense2 = state_dict["head.4.weight"].shape[0]
    num_classes = state_dict["head.8.weight"].shape[0]

    modality_model = UltrasoundClassifier(
        num_classes,
        dense1,
        dense2
    ).to(device)

    modality_model.load_state_dict(state_dict)
    modality_model.eval()

    print("Stage 1 Modality Classifier loaded successfully.")

    return modality_model

def add_prediction_text(
    image_bgr,
    anatomy_name,
    anatomy_conf,
    plane_name,
    plane_conf,
    position=(10, 30),
    font_scale=0.7,
    thickness=2
):
    """
    Draw Top-1 anatomy and plane predictions on an image.
    """

    annotated = image_bgr.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX

    anatomy_text = f"Anatomy: {anatomy_name} ({anatomy_conf:.3f})"
    plane_text = f"Plane: {plane_name} ({plane_conf:.3f})"

    x, y = position

    # Text colors
    anatomy_color = (0, 255, 0)   # Green
    plane_color = (0, 255, 255)   # Yellow

    # Black background rectangles for readability
    (w1, h1), _ = cv2.getTextSize(
        anatomy_text, font, font_scale, thickness
    )
    (w2, h2), _ = cv2.getTextSize(
        plane_text, font, font_scale, thickness
    )

    max_w = max(w1, w2)

    cv2.rectangle(
        annotated,
        (x - 5, y - h1 - 10),
        (x + max_w + 10, y + h2 + 45),
        (0, 0, 0),
        -1
    )

    # Anatomy
    cv2.putText(
        annotated, anatomy_text, (x, y),
        font,
        font_scale,
        anatomy_color,
        thickness,
        cv2.LINE_AA
    )

    # Plane
    cv2.putText(
        annotated,
        plane_text,
        (x, y + 35),
        font,
        font_scale,
        plane_color,
        thickness,
        cv2.LINE_AA
    )

    return annotated


@torch.no_grad()
def predict_panel(image_bgr):
    tensor = modality_tf(image=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))['image'].unsqueeze(0).to(device)
    probs = torch.sigmoid(modality_model(tensor))[0].cpu().numpy()
    labels = [LABEL_NAMES[i] for i, p in enumerate(probs) if p >= (0.85 if LABEL_NAMES[i]=='colour_doppler' else CLASS_THRESHOLDS[LABEL_NAMES[i]])]
    return labels, probs

# def extract_candidate_split(image_bgr, ratio):
#     split_x = int(image_bgr.shape[1] * ratio)
#     return image_bgr[:, :split_x], image_bgr[:, split_x:]

# def find_best_split_ratio(image_bgr):
#     best_ratio, max_score = 0.5, -1
#     for ratio in [0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65]:
#         lp, rp = extract_candidate_split(image_bgr, ratio)
#         score = np.max(predict_panel(lp)[1]) + np.max(predict_panel(rp)[1])
#         if score > max_score:
#             max_score, best_ratio = score, ratio
#     return best_ratio

# def find_best_split_ratio(image_bgr):
#     return 0.5

def generate_masked_panels(image_bgr, ratio):
    split_x = int(image_bgr.shape[1] * ratio)
    left_masked, right_masked = image_bgr.copy(), image_bgr.copy()
    left_masked[:, split_x:] = 0
    right_masked[:, :split_x] = 0
    return left_masked, right_masked

# ── NEW: Quadrant panel extractor (reused from test.py) ──────────────────────
def extract_quad_panels(image_bgr):
    """Split image_bgr into four equal quadrants at exact midpoints."""
    h, w  = image_bgr.shape[:2]
    mid_h = h // 2
    mid_w = w // 2
    return {
        'top_left':     image_bgr[:mid_h, :mid_w],
        'top_right':    image_bgr[:mid_h,  mid_w:],
        'bottom_left':  image_bgr[mid_h:, :mid_w],
        'bottom_right': image_bgr[mid_h:,  mid_w:],
    }

# ── run_edl at module scope so all branches (split, quadrant, single) can call it ──
def hierarchical_predict(
    anatomy_logits,
    plane_logits,
    valid_planes_for_anatomy
):
    pred_anatomy = anatomy_logits.argmax(dim=1)

    pred_plane_raw = plane_logits.argmax(dim=1)

    masked = plane_logits.clone()
    num_planes = masked.shape[1]

    for b in range(masked.size(0)):

        aidx = pred_anatomy[b].item()

        valid = valid_planes_for_anatomy.get(
            aidx,
            list(range(num_planes))
        )

        invalid_mask = torch.ones(
            num_planes,
            dtype=torch.bool,
            device=masked.device
        )

        invalid_mask[valid] = False

        masked[b, invalid_mask] = -1e9

    pred_plane_masked = masked.argmax(dim=1)

    return (
        pred_anatomy,
        pred_plane_masked,
        pred_plane_raw,
        masked
    )

def run_model(img_bgr, model, transform, idx2anatomy, idx2plane, device):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    img = img.convert("L")

    image = Image.merge(
        "RGB",
        [img, img, img]
    )

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(input_tensor)

        anatomy_logits = output["anatomy"]
        plane_logits = output["plane"]

        # =================================================
        # 1. ANATOMY TOP-3
        # =================================================

        anatomy_probs = torch.softmax(anatomy_logits, dim=1)

        anatomy_top3_scores, anatomy_top3_indices = torch.topk(anatomy_probs, k=3, dim=1)

        pred_anatomy_idx = anatomy_top3_indices[0, 0].item()
        pred_anatomy_name = idx2anatomy[pred_anatomy_idx]
        anatomy_score = anatomy_top3_scores[0, 0].item()

        pred_anatomy2_idx = anatomy_top3_indices[0, 1].item()
        pred_anatomy2_name = idx2anatomy[pred_anatomy2_idx]
        anatomy2_score = anatomy_top3_scores[0, 1].item()

        pred_anatomy3_idx = anatomy_top3_indices[0, 2].item()
        pred_anatomy3_name = idx2anatomy[pred_anatomy3_idx]
        anatomy3_score = anatomy_top3_scores[0, 2].item()

        # =================================================
        # 2. HIERARCHICAL PLANE
        # =================================================

        pred_anatomy, pred_plane_masked, pred_plane_raw, masked_plane_logits = hierarchical_predict(anatomy_logits, plane_logits, valid_planes_for_anatomy)
        

        # =================================================
        # 3. HIERARCHICAL PLANE TOP-3
        # =================================================

        plane_probs = torch.softmax(masked_plane_logits, dim=1)

        plane_top3_scores, plane_top3_indices = torch.topk(plane_probs, k=3, dim=1)

        pred_plane_idx = plane_top3_indices[0, 0].item()
        pred_plane_name = idx2plane[pred_plane_idx]
        plane_score = plane_top3_scores[0, 0].item()

        pred_plane2_idx = plane_top3_indices[0, 1].item()
        pred_plane2_name = idx2plane[pred_plane2_idx]
        plane2_score = plane_top3_scores[0, 1].item()

        pred_plane3_idx = plane_top3_indices[0, 2].item()
        pred_plane3_name = idx2plane[pred_plane3_idx]
        plane3_score = plane_top3_scores[0, 2].item()

        return (
            pred_anatomy_name,
            anatomy_score,
            pred_anatomy2_name,
            anatomy2_score,
            pred_anatomy3_name,
            anatomy3_score,

            pred_plane_name,
            plane_score,
            pred_plane2_name,
            plane2_score,
            pred_plane3_name,
            plane3_score
        )



def get_classification_result(panel_bgr):
    (
        pred_anat1,
        anat1_conf,
        pred_anat2,
        anat2_conf,
        pred_anat3,
        anat3_conf,
        pred_plane1,
        plane1_conf,
        pred_plane2,
        plane2_conf,
        pred_plane3,
        plane3_conf
    ) = run_model(
        panel_bgr,
        model,
        transform,
        idx2anatomy,
        idx2plane,
        device
    )

    return {
        "1st": {
            "Anatomy": pred_anat1,
            "Standard plane": pred_plane1,
            "Standard_plane_confidence":plane1_conf,
        },

        "2nd": {
            "Anatomy": pred_anat2,
            "Standard plane": pred_plane2,
            "Standard_plane_confidence":plane2_conf,
        },

        "3rd": {
            "Anatomy": pred_anat3,
            "Standard plane": pred_plane3,
            "Standard_plane_confidence":plane3_conf,
        }
    }

def get_doppler_placeholder(panel_type):
    return {
        "result": None
    }

# ── Resolve image list from folder or CSV ─────────────────────────────────────

# Create image_path -> ground truth lookup
# gt_lookup = df.set_index("image_path")[["anatomy", "plane"]].to_dict("index")

def modality_split(image_folder = None):
    image_paths = get_image_paths(INPUT_MODE, folder=image_folder, df=df if INPUT_MODE == 'csv' else None)
    print('Starting Unified Inference...')
    for img_path in tqdm(image_paths, desc='Unified Inference'):
        image_rgb = load_image(img_path)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        image_name = Path(img_path).name
        if not os.path.exists(img_path):
            continue
        # image_bgr = cv2.imread(img_path)
        H, W, _ = image_bgr.shape
        if image_bgr is None:
            print(f'Warning: could not read {img_path}, skipping.')
            continue
        labels, probs = predict_panel(image_bgr)
        if image_name == "3899671.jpg":
            print("\n--- Modality probabilities ---")
            for name, prob in zip(LABEL_NAMES, probs):
                print(f"{name:20s}: {prob:.4f}")

            print("Selected labels:", labels)
            print(f"Labels for image {image_name} : {labels}")

        is_bmode = "b-mode" in labels
        is_tinted = "tinted" in labels
        is_colour_doppler = "colour_doppler" in labels
        is_pulse_doppler = "pulse_doppler" in labels

        is_doppler = (
            is_colour_doppler or
            is_pulse_doppler
        )

        is_bisplit = "split_screen_only" in labels
        is_quadsplit = "quadrant_images" in labels

        result = {
            "image_path": str(img_path),
            "modality": {
                "split": None
            },
            "classification": {}
        }

        # ============================================================
        # CASE 1.1
        # B-MODE BI-SPLIT / QUAD-SPLIT
        # ============================================================

        if is_bisplit and not is_doppler:

            result["modality"] = {
                "split": "bi-split",
                "type": {
                    "left": "b-mode",
                    "right": "b-mode"
                }
            }

            split_x = W // 2

            panels = {
                "left": image_bgr[:, :split_x],
                "right": image_bgr[:, split_x:]
            }


            for panel_name, panel_bgr in panels.items():

                result["classification"][panel_name] = (
                    get_classification_result(panel_bgr)
                )


        elif is_quadsplit and not is_doppler:

            result["modality"] = {
                "split": "quad-split",
                "type": {
                    "top_left": "b-mode",
                    "top_right": "b-mode",
                    "bottom_left": "b-mode",
                    "bottom_right": "b-mode"
                }
            }

            panels = extract_quad_panels(image_bgr)

            for panel_name, panel_bgr in panels.items():

                result["classification"][panel_name] = (
                    get_classification_result(panel_bgr)
                )


        # ============================================================
        # CASE 1.2
        # SINGLE B-MODE / TINTED
        # ============================================================

        elif is_bmode or is_tinted:

            result["modality"] = {
                "split": "single",
                "type": "tinted" if is_tinted else "b-mode"
            }

            result["classification"] = (
                get_classification_result(image_bgr)
            )

        # ============================================================
        # CASE 2
        # DOPPLER + BI/QUAD SPLIT
        # ============================================================

        elif is_doppler and is_bisplit:

            result["modality"] = {
                "split": "bi-split",
                "type": {}
            }

            H, W = image_bgr.shape[:2]
            split_x = W // 2

            panels = {
                "left": image_bgr[:, :split_x],
                "right": image_bgr[:, split_x:]
            }

            for panel_name, panel_bgr in panels.items():

                panel_labels, _ = predict_panel(panel_bgr)

                # ----------------------------
                # Determine panel modality
                # ----------------------------

                if "colour_doppler" in panel_labels:
                    panel_type = "colour_doppler"

                elif "pulse_doppler" in panel_labels:
                    panel_type = "pulse_doppler"

                elif "tinted" in panel_labels:
                    panel_type = "tinted"

                elif "b-mode" in panel_labels:
                    panel_type = "b-mode"

                else:
                    panel_type = "unknown"

                result["modality"]["type"][panel_name] = panel_type

                if panel_type in ["b-mode", "tinted"]:

                    result["classification"][panel_name] = (
                        get_classification_result(panel_bgr)
                    )

                elif panel_type in [
                    "colour_doppler",
                    "pulse_doppler"
                ]:

                    result["classification"][panel_name] = (
                        get_doppler_placeholder(panel_type)
                    )

        elif is_doppler and is_quadsplit:

            result["modality"] = {
                "split": "quad-split",
                "type": {}
            }

            panels = extract_quad_panels(image_bgr)

            for panel_name, panel_bgr in panels.items():

                # --------------------------------
                # Run modality again on each panel
                # --------------------------------

                panel_labels, _ = predict_panel(panel_bgr)

                if "colour_doppler" in panel_labels:
                    panel_type = "colour_doppler"

                elif "pulse_doppler" in panel_labels:
                    panel_type = "pulse_doppler"

                elif "tinted" in panel_labels:
                    panel_type = "tinted"

                elif "b-mode" in panel_labels:
                    panel_type = "b-mode"

                else:
                    panel_type = "unknown"

                result["modality"]["type"][panel_name] = panel_type

                if panel_type in ["b-mode", "tinted"]:

                    result["classification"][panel_name] = (
                        get_classification_result(panel_bgr)
                    )

                elif panel_type in [
                    "colour_doppler",
                    "pulse_doppler"
                ]:

                    result["classification"][panel_name] = (
                        get_doppler_placeholder(panel_type)
                    )

        # ============================================================
        # CASE 3
        # DOPPLER WITHOUT BI/QUAD
        # ============================================================

        elif is_doppler:

            # result["modality"]["split"] = "single"

            if is_colour_doppler:
                classification = {
                    "result": None
                }

                result = {
                    "modality": {
                        "split" : "single",
                        "type": "colour_doppler"
                    },
                    "classification": classification
                }

                # colour_result = run_colour_doppler_model(image_bgr)
                # result["classification"] = colour_result

            elif is_pulse_doppler:
                classification = {
                    "result": None
                }

                result = {
                    "modality": {
                        "split" : "single",
                        "type": "pulse_doppler"
                    },
                    "classification": classification
                }

                # whatever pulse Doppler information you want
                # result["classification"] = {...}

        json_path = Path(output_dir) / f"{Path(img_path).stem}.json"

        with open(json_path, "w") as f:
            json.dump(result, f, indent=4)

        # print(f"Saved: {json_path}")

def modality_inference(IMAGE_DIRECTORY, MODEL_PATH, OUTPUT_FOLDER_PATH):

    global modality_model
    global modality_tf
    global folder
    global model_path
    global output_dir

    if os.path.exists(OUTPUT_FOLDER_PATH):
        shutil.rmtree(OUTPUT_FOLDER_PATH)

    os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)

    folder = IMAGE_DIRECTORY
    model_path = MODEL_PATH
    output_dir = OUTPUT_FOLDER_PATH

    # Load Stage 1 model AFTER MODEL_PATH is available
    modality_model = load_modality_model(MODEL_PATH)

    modality_tf = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])

    modality_split(image_folder=IMAGE_DIRECTORY)



if __name__ == "__main__":
    modality_inference(
       IMAGE_DIRECTORY = r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/12_Full_Image_Datasets/8",
        MODEL_PATH = "/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/weights/MODALITY/modality_model.pth",
       OUTPUT_FOLDER_PATH = "/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/TEMP_OUTPUT/MODALITY"
    )

    