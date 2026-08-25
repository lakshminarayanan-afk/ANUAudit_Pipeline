import os
import random
import torch
import torch.nn.functional as F
import timm
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.swa_utils import AveragedModel
import shutil
from pathlib import Path
import pydicom




from pathlib import Path
import pydicom
import cv2
import numpy as np

import torch.nn as nn


class HierarchicalUltrasoundModel(nn.Module):
    def __init__(self, num_anatomies, num_planes, backbone_name="convnext_base", pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, drop_path_rate=dropout, num_classes=0,
        )
        feat_dim = self.backbone.num_features

        self.anatomy_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_anatomies)
        )
        # plane head now takes backbone features + anatomy logits concatenated
        self.plane_head = nn.Sequential(
            nn.LayerNorm(feat_dim + num_anatomies),
            nn.Dropout(dropout),
            nn.Linear(feat_dim + num_anatomies, num_planes)
        )

    def forward(self, x):
        features = self.backbone(x)
        # feature_map = self.backbone.forward_features(x)
        # features = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        anatomy_logits = self.anatomy_head(features)
        # detach so plane-head gradient doesn't corrupt anatomy head's own training signal
        plane_input = torch.cat([features, anatomy_logits.detach()], dim=-1)
        plane_logits = self.plane_head(plane_input)
        return {"anatomy": anatomy_logits, "plane": plane_logits} #, "feature_maps": feature_map, "features": features}

# -----------------------------
# Config
# -----------------------------
# ROOT_DIR = "/mnt/data/mln/20_class/Abdomen-Abdominal_Circumference"      # Replace with your dataset root
# MODEL_PATH = "/home/htic/mln/pyqt/ml/swa_final_27-classes-DACL+ContrLoss+DeeperClsHead#1.pth"
# CSV_OUTPUT = "/mnt/data/ConvNeXt_HybridPooling/DACL_ContrLoss_DeeperClsHead/predictions.csv"
# OUTPUT_FOLDER_PATH = "/home/htic/mln/pyqt/ml/outputs"
# 28 classes


plane2idx = {
    "3 Vessel View or PAS": 0,
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
    "Transcerebellar Plane": 23,
    "Transthalamic Plane": 24,
    "Transventricular Plane": 25,
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
    "0": [      2,      7,      26    ],
    "3": [      25,     24,     23    ],
    "6": [      1,      12,      19    ],
    "7": [      11,      20,     15    ],
    "4": [      9,       22    ],
    "1": [       13,       14,       16,       18     ],
    "5": [       21,       8,       10     ],
    "2": [       3,       17,       5     ]
  }


valid_planes_for_anatomy = {
    int(k): [int(x) for x in v]
    for k, v in valid_planes_for_anatomy.items()
}

def convert_dicoms_to_png(input_root, output_root):

    input_root = Path(input_root)
    output_root = Path(output_root)

    for dcm_file in input_root.rglob("*.dcm"):

        ds = pydicom.dcmread(dcm_file)
        image = ds.pixel_array

        # --------------------------
        # Grayscale images
        # --------------------------
        if image.ndim == 2:

            image = image.astype(np.float32)


            if hasattr(ds, "RescaleSlope"):
                image *= ds.RescaleSlope


            if hasattr(ds, "RescaleIntercept"):
                image += ds.RescaleIntercept


            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                image = image.max() - image


            image -= image.min()
            if image.max() > 0:
                image /= image.max()


            image = (image * 255).astype(np.uint8)


        # --------------------------
        # Color images
        # --------------------------
        elif image.ndim == 3:


            # pydicom returns RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        relative_path = dcm_file.relative_to(input_root)
        png_path = output_root / relative_path.with_suffix(".png")
        png_path.parent.mkdir(parents=True, exist_ok=True)


        cv2.imwrite(str(png_path), image)


        print(f"Saved {png_path}")


# -----------------------------
# Load model
# -----------------------------
# def load_convnext(model_path, num_classes, device):
#     model = timm.create_model(
#         "convnext_tiny",
#         pretrained=False,
#         num_classes=num_classes
#     )
#     state_dict = torch.load(model_path, map_location=device)
#     model.load_state_dict(state_dict)
#     model.to(device)
#     model.eval()
#     return model


# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess_image(img_path, device, IMG_SIZE):
    if img_path.lower().endswith('.dcm'):
        # print('DICOM image detected:', img_path)
        ds = pydicom.dcmread(img_path)
        image = ds.pixel_array.astype(np.float32)


        # Apply rescale slope/intercept if present
        if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
            image = image * ds.RescaleSlope + ds.RescaleIntercept


        # Handle MONOCHROME1 (invert)
        if ds.PhotometricInterpretation == "MONOCHROME1":
            image = np.max(image) - image


        # Normalize to [0,255]
        image -= image.min()
        image /= image.max() + 1e-6
        image *= 255.0
        image = image.astype(np.uint8)


        # If grayscale → convert to RGB
        if image.ndim == 2:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            # DICOM RGB is already RGB
            img_rgb = image


    else:
        # Regular image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


    # Resize
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))


    # Normalize to [0,1]
    img_float = img_resized.astype(np.float32) / 255.0


    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_float - mean) / std


    # To tensor
    # tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)
    tensor = torch.from_numpy(img_norm).permute(2, 0, 1)
    tensor = tensor.to(device=device, dtype=torch.float32)

    return tensor, img_float


# -----------------------------
# Run inference and Grad-CAM
# -----------------------------
def hierarchical_predict(
    anatomy_logits,
    plane_logits,
    valid_planes_for_anatomy,
    top_k=3
):

    pred_anatomy = anatomy_logits.argmax(dim=1)
    pred_plane_raw = plane_logits.argmax(dim=1)


    masked = plane_logits.clone()
    num_planes = masked.shape[1]


    for b in range(masked.size(0)):

        aidx = int(pred_anatomy[b])


        valid = valid_planes_for_anatomy.get(
            aidx,
            list(range(num_planes))
        )


        invalid_mask = torch.ones( ## all planes r true
            num_planes,
            dtype=torch.bool,
            device=masked.device
        )


        invalid_mask[valid] = False

        masked[b][invalid_mask] = -1e9

    # Convert masked logits to probabilities
    plane_probs = F.softmax(masked, dim=1)

    # Get top 3 valid planes
    top_k = min(top_k, num_planes)

    top_probs, top_indices = torch.topk(
        plane_probs,
        k=top_k,
        dim=1
    )

    # Top 1 is the final hierarchical prediction
    pred_plane_masked = top_indices[:, 0]

    return (
        pred_anatomy,
        pred_plane_masked,
        pred_plane_raw,
        top_indices,
        top_probs
    )

import time 

def infer(model, img_paths, root_dir, output_root, num_classes, device, num_gradcam=5, IMG_SIZE=None, BATCH_SIZE = 32):
    results = []
    inference_times = []

    # -----------------------------------------
    # GPU WARMUP
    # -----------------------------------------
    if device.type == "cuda":
        print("Warming up GPU...")

        warmup_tensors = []

        for img_path in img_paths[:BATCH_SIZE]:
            tensor, _ = preprocess_image(
                img_path,
                device,
                IMG_SIZE
            )
            warmup_tensors.append(tensor)

        warmup_batch = torch.stack(warmup_tensors)

        with torch.no_grad():
            for _ in range(20):
                out = model(warmup_batch)

                _ = hierarchical_predict(
                    out["anatomy"],
                    out["plane"],
                    valid_planes_for_anatomy,
                    top_k=3
                )

        torch.cuda.synchronize()

        print("GPU warmup complete.")
        
    

    inference_times = []
    total_images = 0

    for start in range(0, len(img_paths), BATCH_SIZE):

        batch_paths = img_paths[start:start + BATCH_SIZE]

        batch_tensors = []
        for img_path in batch_paths:
            tensor, _ = preprocess_image(
                img_path,
                device,
                IMG_SIZE
            )
            batch_tensors.append(tensor)

        batch = torch.stack(batch_tensors)

        # -----------------------------------------
        # START TIMING
        # -----------------------------------------
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():
            out = model(batch)

            pred_anat, pred_plane, pred_plane_raw, \
            top_indices, top_probs = hierarchical_predict(
                out["anatomy"],
                out["plane"],
                valid_planes_for_anatomy,
                top_k=3
            )

        # -----------------------------------------
        # STOP TIMING
        # -----------------------------------------
        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        batch_size = len(batch_paths)

        inference_times.append({
            "batch_size": batch_size,
            "batch_time_ms": elapsed_ms,
            "per_image_ms": elapsed_ms / batch_size
        })

        total_images += batch_size

        print(
            f"Batch {start // BATCH_SIZE + 1}: "
            f"{batch_size} images | "
            f"Batch: {elapsed_ms:.2f} ms | "
            f"Per image: {elapsed_ms / batch_size:.2f} ms"
        )

        # -----------------------------------------
        # Process every image in the batch
        # -----------------------------------------
        for i, img_path in enumerate(batch_paths):

            # Top 3 predictions for this image
            top1_idx = top_indices[i, 0].item()
            top2_idx = top_indices[i, 1].item()
            top3_idx = top_indices[i, 2].item()

            top1_class = idx2plane[top1_idx]
            top2_class = idx2plane[top2_idx]
            top3_class = idx2plane[top3_idx]

            top1_prob = top_probs[i, 0].item()
            top2_prob = top_probs[i, 1].item()
            top3_prob = top_probs[i, 2].item()

            # Anatomy prediction for this image
            anatomy_idx = pred_anat[i].item()
            anatomy_class = idx2anatomy[anatomy_idx]

            # Final hierarchical plane prediction
            pred_plane_idx = pred_plane[i].item()
            pred_plane_class = idx2plane[pred_plane_idx]

            # Save JSON
            json_path = save_image_classification_json(
                img_path=img_path,
                root_dir=root_dir,
                output_root=output_root,
                top_indices=top_indices[i:i+1],
                top_probs=top_probs[i:i+1],
                idx2plane=idx2plane
            )

            # print(f"Saved JSON → {json_path}")

            results.append({
                "image_path": img_path,
                "predicted_model_class": pred_plane_class,
                "confidence": top1_prob,
                "top1_class": top1_class,
                "top1_prob": top1_prob,
                "top2_class": top2_class,
                "top2_prob": top2_prob,
                "top3_class": top3_class,
                "top3_prob": top3_prob,
            })

    if inference_times:
        total_batch_time_ms = sum(
            x["batch_time_ms"] for x in inference_times
        )

        average_per_image_ms = total_batch_time_ms / total_images

        throughput = total_images / (total_batch_time_ms / 1000)

        print("\n========================================")
        print(f"Total images       : {total_images}")
        print(f"Total inference    : {total_batch_time_ms:.2f} ms")
        print(f"Average per image  : {average_per_image_ms:.2f} ms")
        print(f"Throughput         : {throughput:.2f} images/sec")
        print("========================================")
                
    return results


# -----------------------------
# Collect all images recursively
# -----------------------------
def get_all_images(root_dir):
    exts = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".dcm")
    img_paths = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(exts):
                img_paths.append(os.path.join(root, f))
    return img_paths

import json


def save_image_classification_json(
    img_path,
    root_dir,
    output_root,
    top_indices,
    top_probs,
    idx2plane
):
    img_path = Path(img_path)
    root_dir = Path(root_dir)
    output_root = Path(output_root)


    relative_path = img_path.relative_to(root_dir)

    # Get patient folder
    patient_name = root_dir.name

    # Create:
    # OUTPUT_FOLDER_PATH/patient_001/
    patient_output_dir = output_root / patient_name
    patient_output_dir.mkdir(parents=True, exist_ok=True)

    # JSON filename
    json_path = patient_output_dir / f"{img_path.stem}.json"

    data = {
        "image": str(relative_path),
        "classification": {
            "top1": {
                "class": idx2plane[top_indices[0, 0].item()],
                "probability": top_probs[0, 0].item()
            },
            "top2": {
                "class": idx2plane[top_indices[0, 1].item()],
                "probability": top_probs[0, 1].item()
            },
            "top3": {
                "class": idx2plane[top_indices[0, 2].item()],
                "probability": top_probs[0, 2].item()
            }
        }
    }

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    return json_path
# -----------------------------
# Main
# -----------------------------
def inference_fn(ROOT_DIR, MODEL_PATH, OUTPUT_FOLDER_PATH):

    if os.path.exists(OUTPUT_FOLDER_PATH):
        shutil.rmtree(OUTPUT_FOLDER_PATH)
    os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE = 224
    NUM_ANATOMIES = len(idx2anatomy)
    NUM_PLANES = len(idx2plane)


    model = HierarchicalUltrasoundModel(
    num_anatomies=NUM_ANATOMIES,
    num_planes=NUM_PLANES,
    backbone_name='convnext_small',
    pretrained=True,
    dropout=0.3,
    ).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict= True)
    model = model.to(DEVICE)
    model.eval()


    img_paths = get_all_images(ROOT_DIR)
    print(f"Found {len(img_paths)} images.")


    results = infer(
        model=model,
        img_paths=img_paths,
        root_dir=ROOT_DIR,
        output_root=OUTPUT_FOLDER_PATH,
        num_classes=NUM_PLANES,
        device=DEVICE,
        num_gradcam=5,
        IMG_SIZE=IMG_SIZE,
        BATCH_SIZE = 32
    )

    print("Folders saved to:", {OUTPUT_FOLDER_PATH} )


