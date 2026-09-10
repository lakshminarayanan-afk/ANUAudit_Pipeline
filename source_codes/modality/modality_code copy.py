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

from source_codes.models import HierarchicalUltrasoundModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── CONFIG: set your image folder here ───────────────────────────────────────
# folder = r"C:\Users\laksh\ANU\12_Full_Image_Datasets\10"
INPUT_MODE   = 'folder'


# output_dir.mkdir(parents=True, exist_ok=True)

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

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

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

print('Loading Stage 1 Modality Classifier...')
model_path = model_path
checkpoint = torch.load(model_path, map_location=d)
state_dict = checkpoint['model']
dense1 = state_dict['head.0.weight'].shape[0]
dense2 = state_dict['head.4.weight'].shape[0]
num_classes = state_dict['head.8.weight'].shape[0]
modality_model = UltrasoundClassifier(num_classes, dense1, dense2).to(device)
modality_model.load_state_dict(state_dict)
modality_model.eval()
modality_tf = A.Compose([A.Resize(224, 224), A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()])

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
def run_model(img_bgr, model, transform, idx2anatomy, idx2plane, device):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    img = img.convert("L")

    image = Image.merge(
        "RGB",
        [img, img, img]
    )

    input_tensor = (
        transform(image)
        .unsqueeze(0)
        .to(device)
    )
    with torch.no_grad():
        output = model(input_tensor)
        # anatomy_logits = output["anatomy"]

        # anatomy_probs = torch.softmax(
        #     anatomy_logits,
        #     dim=1
        # )

        # top3_scores, top3_indices = torch.topk(
        #     anatomy_probs,
        #     k=3,
        #     dim=1
        # )

        # # Top-1
        # pred_anatomy_idx = top3_indices[0, 0].item()
        # pred_anatomy_name = idx2anatomy[pred_anatomy_idx]
        # anatomy_score = top3_scores[0, 0].item()
        # # =================================================
        # # 2. RAW PLANE
        # # =================================================

        # plane_logits = output["plane"]

        # plane_probs = torch.softmax(
        #     plane_logits,
        #     dim=1
        # )

        # pred_raw_plane_idx = plane_probs.argmax(
        #     dim=1
        # ).item()

        # pred_raw_plane_name = idx2plane[
        #     pred_raw_plane_idx
        # ]

        # raw_plane_score = plane_probs[
        #     0,
        #     pred_raw_plane_idx
        # ].item()
        anatomy_logits = output["anatomy"]
        
        anatomy_probs = torch.softmax(
            anatomy_logits,
            dim=1
        )

        top3_scores, top3_indices = torch.topk(
            anatomy_probs,
            k=3,
            dim=1
        )

        # Top-1
        pred_anatomy_idx = top3_indices[0, 0].item()
        pred_anatomy_name = idx2anatomy[pred_anatomy_idx]
        anatomy_score = top3_scores[0, 0].item()

        # Top-2
        pred_anatomy2_idx = top3_indices[0, 1].item()
        pred_anatomy2_name = idx2anatomy[pred_anatomy2_idx]
        anatomy2_score = top3_scores[0, 1].item()

        # Top-3
        pred_anatomy3_idx = top3_indices[0, 2].item()
        pred_anatomy3_name = idx2anatomy[pred_anatomy3_idx]
        anatomy3_score = top3_scores[0, 2].item()

        # =================================================
        # 2. RAW PLANE
        # =================================================

        plane_logits = output["plane"]

        plane_probs = torch.softmax(
            plane_logits,
            dim=1
        )

        # pred_raw_plane_idx = plane_probs.argmax(
        #     dim=1
        # ).item()

        # pred_raw_plane_name = idx2plane[
        #     pred_raw_plane_idx
        # ]

        # raw_plane_score = plane_probs[
        #     0,
        #     pred_raw_plane_idx
        # ].item()

        top3_scores, top3_indices = torch.topk(
            plane_probs,
            k=3,
            dim=1
        )

        # Top-1
        pred_plane_idx = top3_indices[0, 0].item()
        pred_plane_name = idx2plane[pred_plane_idx]
        plane_score = top3_scores[0, 0].item()

        # Top-2
        pred_plane2_idx = top3_indices[0, 1].item()
        pred_plane2_name = idx2plane[pred_plane2_idx]
        plane2_score = top3_scores[0, 1].item()

        # Top-3
        pred_plane3_idx = top3_indices[0, 2].item()
        pred_plane3_name = idx2plane[pred_plane3_idx]
        plane3_score = top3_scores[0, 2].item()
        

        return pred_anatomy_name, anatomy_score, pred_anatomy2_name, anatomy2_score, pred_anatomy3_name, anatomy3_score, pred_plane_name, plane_score, pred_plane2_name, plane2_score, pred_plane3_name, plane3_score

# ── Resolve image list from folder or CSV ─────────────────────────────────────

# Create image_path -> ground truth lookup
# gt_lookup = df.set_index("image_path")[["anatomy", "plane"]].to_dict("index")

def modality_split(image_folder = None):
    image_paths = get_image_paths(INPUT_MODE, folder=image_folder, df=df if INPUT_MODE == 'csv' else None)
    results = []
    print('Starting Unified Inference...')
    for img_path in tqdm(image_paths, desc='Unified Inference'):
        if not os.path.exists(img_path):
            continue
        image_bgr = cv2.imread(img_path)
        H, W, _ = image_bgr.shape
        if image_bgr is None:
            print(f'Warning: could not read {img_path}, skipping.')
            continue
        labels, _ = predict_panel(image_bgr)

        if 'colour_doppler' in labels:
            output_dir = Path("colour_doppler")
            output_dir.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(
                str(output_dir / Path(img_path).name),
                image_bgr
            )

        elif 'pulse_doppler' in labels:
            output_dir = Path("pulse_doppler")
            output_dir.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(
                str(output_dir / Path(img_path).name),
                image_bgr
            )


        elif 'split_screen_only' in labels:
            # ── Existing split-screen branch (unchanged) ──────────────────────────
            # ratio = find_best_split_ratio(image_bgr)
            split_x = W // 2
            lbgr = image_bgr[:, :split_x]
            rbgr = image_bgr[:, split_x:]
            # lbgr, rbgr = generate_masked_panels(image_bgr, ratio)
            # pred_anatomy_name, anatomy_score, pred_raw_plane_name, raw_plane_score
            l_pred_anat1, l_anat1_conf, l_pred_anat2, l_anat2_conf, l_pred_anat3, l_anat3_conf, l_pred_plane1, l_plane1_conf, l_pred_plane2, l_plane2_conf, l_pred_plane3, l_plane3_conf = run_model(lbgr, model, transform, idx2anatomy, idx2plane, device)
            r_pred_anat1, r_anat1_conf, r_pred_anat2, r_anat2_conf, r_pred_anat3, r_anat3_conf, r_pred_plane1, r_plane1_conf, r_pred_plane2, r_plane2_conf, r_pred_plane3, r_plane3_conf = run_model(rbgr, model, transform, idx2anatomy, idx2plane, device)

            # Add predictions to each panel
            left_annotated = add_prediction_text(
                lbgr,
                l_pred_anat1,
                l_anat1_conf,
                l_pred_plane1,
                l_plane1_conf
            )

            right_annotated = add_prediction_text(
                rbgr,
                r_pred_anat1,
                r_anat1_conf,
                r_pred_plane1,
                r_plane1_conf
            )

            # Recombine panels
            annotated_image = np.hstack([left_annotated, right_annotated])

            # Save annotated split-screen image
            output_path = output_dir / Path(img_path).name
            cv2.imwrite(str(output_path), annotated_image)


            # results.append({'image_path': img_path, 'panel': 'left',  'anatomy_prediction': l_pred_anat, 'anatomy_confidence': l_anat_conf, 'plane_prediction': l_pred_plane, 'plane_confidence': l_plane_conf})
            # results.append({'image_path': img_path, 'panel': 'right', 'anatomy_prediction': r_pred_anat, 'anatomy_confidence': r_anat_conf, 'plane_prediction': r_pred_plane, 'plane_confidence': r_plane_conf})
            results.append({
                'image_path': img_path,
                # 'anatomy': gt_anatomy,
                # 'plane': gt_plane,
                'panel': 'split_screen',
                # Left panel
                'left_anatomy_prediction': l_pred_anat1,
                'left_anatomy_confidence': l_anat1_conf,
                'left_anatomy2_prediction': l_pred_anat2,
                'left_anatomy2_confidence': l_anat2_conf,
                'left_anatomy3_prediction': l_pred_anat3,
                'left_anatomy3_confidence': l_anat3_conf,
                'left_plane_prediction': l_pred_plane1,
                'left_plane_confidence': l_plane1_conf,
                'left_plane2_prediction': l_pred_plane2,
                'left_plane2_confidence': l_plane2_conf,
                'left_plane3_prediction': l_pred_plane3,
                'left_plane3_confidence': l_plane3_conf,

                # Right panel
                'right_anatomy_prediction': r_pred_anat1,
                'right_anatomy_confidence': r_anat1_conf,
                'right_anatomy2_prediction': r_pred_anat2,
                'right_anatomy2_confidence': r_anat2_conf,
                'right_anatomy3_prediction': r_pred_anat3,
                'right_anatomy3_confidence': r_anat3_conf,
                'right_plane_prediction': r_pred_plane1,
                'right_plane_confidence': r_plane1_conf,
                'right_plane2_prediction': r_pred_plane2,
                'right_plane2_confidence': r_plane2_conf,
                'right_plane3_prediction': r_pred_plane3,
                'right_plane3_confidence': r_plane3_conf,
            })


        elif 'quadrant_images' in labels:
            # ── NEW: Quadrant branch ──────────────────────────────────────────────
            panels = extract_quad_panels(image_bgr)
            for panel_name, panel_bgr in panels.items():
                pred_anat, anat_conf, pred_anat2, anat2_conf, pred_anat3, anat3_conf, pred_plane, plane_conf, pred_plane2, plane2_conf, pred_plane3, plane3_conf = run_model(panel_bgr, model, transform, idx2anatomy, idx2plane, device)   # same run_edl, no shortcuts

                annotated_panel = add_prediction_text(
                    panel_bgr,
                    pred_anat,
                    anat_conf,
                    pred_plane,
                    plane_conf
                )

                # Save each quadrant separately
                output_name = (
                    f"{Path(img_path).stem}_{panel_name}"
                    f"{Path(img_path).suffix}"
                )

                output_path = output_dir / output_name
                cv2.imwrite(str(output_path), annotated_panel)


                results.append({
                    'image_path': img_path,
                    # 'anatomy': gt_anatomy,
                    # 'plane': gt_plane,
                    'panel':      panel_name,
                    'anatomy_prediction': pred_anat,
                    'anatomy_confidence': anat_conf,
                    'plane_prediction': pred_plane,
                    'plane_confidence': plane_conf,

                    'anatomy2_prediction': pred_anat2,
                    'anatomy2_confidence': anat2_conf,
                    'plane2_prediction': pred_plane2,
                    'plane2_confidence': plane2_conf,

                    'anatomy3_prediction': pred_anat3,
                    'anatomy3_confidence': anat3_conf,
                    'plane3_prediction': pred_plane3,
                    'plane3_confidence': plane3_conf
                })

        else:
            # ── Existing single-image branch (unchanged) ──────────────────────────
            pred_anatomy_name, anatomy_score, pred_anatomy2_name, anatomy2_score, pred_anatomy3_name, anatomy3_score, pred_plane_name, plane_score, pred_plane2_name, plane2_score, pred_plane3_name, plane3_score  = run_model(image_bgr, model, transform, idx2anatomy, idx2plane, device)

            annotated_image = add_prediction_text(
                image_bgr,
                pred_anatomy_name,
                anatomy_score,
                pred_plane_name,
                plane_score
            )

            output_path = output_dir / Path(img_path).name
            cv2.imwrite(str(output_path), annotated_image)


            results.append({
                'image_path': img_path, 
                # 'anatomy': gt_anatomy, 
                # 'plane': gt_plane, 
                'panel': 'full', 
                'anatomy_prediction': pred_anatomy_name, 'anatomy_confidence': anatomy_score, 
                'anatomy2_prediction': pred_anatomy2_name, 'anatomy2_confidence': anatomy2_score,
                'anatomy3_prediction': pred_anatomy3_name, 'anatomy3_confidence': anatomy3_score,
                'plane_prediction': pred_plane_name, 'plane_confidence': plane_score,
                'plane2_prediction': pred_plane2_name, 'plane2_confidence': plane2_score,
                'plane3_prediction': pred_plane3_name, 'plane3_confidence': plane3_score,
                })

    unified_df = pd.DataFrame(results)
    unified_df.to_csv('unified_inference_results_march2026.csv', index=False)
    print(f'Saved {len(unified_df)} predictions to unified_inference_results.csv')
    unified_df.head(10)

def modality_inference(IMAGE_DIRECTORY, MODEL_PATH, OUTPUT_FOLDER_PATH):
    if os.path.exists(OUTPUT_FOLDER_PATH):
        shutil.rmtree(OUTPUT_FOLDER_PATH)
    os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)
    global folder
    folder = IMAGE_DIRECTORY
    global model_path
    model_path = MODEL_PATH
    global output_dir
    output_dir = OUTPUT_FOLDER_PATH
    modality_split(image_folder = IMAGE_DIRECTORY)



if __name__ == "__main__":
    modality_inference(
       IMAGE_DIRECTORY = r"/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/12_Full_Image_Datasets/8",
        MODEL_PATH = "/home/htic/MLN/PIPELINE/ANUAudit_Pipeline/weights/MODALITY/modality_model.pth",
       OUTPUT_FOLDER_PATH = Path(r"C:\Users\laksh\MLN\ANUAudit_Pipeline\TEMP_OUTPUT\MODALITY"),
    )

    