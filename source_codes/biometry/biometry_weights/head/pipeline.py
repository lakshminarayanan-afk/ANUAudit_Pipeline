'''
Fetal BPD Inference Script: Single Image Implementation
Customizable Input/Output paths with clinical symmetry logic.
'''
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import numpy as np
import pandas as pd
import math
import sys
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF
from segment_anything import sam_model_registry

# Import custom modules (ensure these are in your working directory)
from config import Config
from model import MedSAMFineTune


# USER CONFIGURATION
# 1. Path to the specific image you want to process
INPUT_IMAGE_PATH = "in/head.jpeg"

# 2. Path to the folder where you want to save the output (Image + CSV)
CUSTOM_OUTPUT_FOLDER = "out"


# UTILITY FUNCTIONS
def make_image_square_with_zero_padding(image):
    """Pads the image to be square to maintain aspect ratio for the model."""
    width, height = image.size
    max_side = max(width, height)
    image = image.convert('RGB')
    new_image = Image.new('RGB', (max_side, max_side), (0, 0, 0))
    padding_left = (max_side - width) // 2
    padding_top = (max_side - height) // 2
    new_image.paste(image, (padding_left, padding_top))
    return new_image, (padding_left, padding_top)

def setup_medsam_model(checkpoint_path):
    """Initializes MedSAM and loads fine-tuned weights."""
    sam_model = sam_model_registry["vit_b"](checkpoint=Config.MEDSAM_CHECKPOINT) 
    model = MedSAMFineTune(sam_model, num_classes=Config.NUM_CLASSES)
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(Config.DEVICE).eval()
    print(f" Model Loaded.")
    return model

def get_medsam_masks(image_tensor, model):
    """Generates binary masks using the MedSAM model."""
    with torch.no_grad():
        logits = model(image_tensor.to(Config.DEVICE)) 
        masks = (torch.sigmoid(logits) > Config.THRESHOLD).cpu().numpy()[0] 
    return {
        "inner_cal": masks[Config.INNER_CALVARIUM],
        "midline": masks[Config.MIDLINE_FALX],
        "outer_cal": masks[Config.OUTER_CALVARIUM]
    }

def resize_mask_to_original(pred_mask_1024, orig_size, padded_size):
    """Crops and resizes model mask back to original image dimensions."""
    mask_res = Image.fromarray((pred_mask_1024 * 255).astype(np.uint8))
    mask_padded = mask_res.resize(padded_size, resample=Image.NEAREST)
    left = (padded_size[0] - orig_size[0]) // 2
    top = (padded_size[1] - orig_size[1]) // 2
    return np.array(mask_padded.crop((left, top, left + orig_size[0], top + orig_size[1])))

def extract_refined_contour(mask_np):
    """Extracts the largest contour from a binary mask."""
    if mask_np is None or mask_np.sum() == 0: return None
    kernel = np.ones((5,5), np.uint8)
    mask_closed = cv2.morphologyEx(mask_np.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None

def compute_pca_line(mask_np):
    """Calculates the orientation (angle) of the midline mask."""
    kernel = np.ones((5,5), np.uint8)
    mask_np = cv2.morphologyEx(mask_np.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: raise ValueError("Falx not found")
    pts = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    mean = np.mean(pts, axis=0)
    _, eig = np.linalg.eigh(np.cov(pts - mean, rowvar=False))
    axis = eig[:, 1]
    return mean, axis, math.degrees(math.atan2(axis[1], axis[0]))

def line_contour_intersection(p1, p2, contour):
    """Finds pixel intersections between a line and a contour."""
    if contour is None: return []
    ints = []
    for i in range(len(contour)):
        seg_p3, seg_p4 = tuple(contour[i][0]), tuple(contour[(i+1)%len(contour)][0])
        x1,y1=p1; x2,y2=p2; x3,y3=seg_p3; x4,y4=seg_p4
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) > 1e-10:
            t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4))/denom
            u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3))/denom
            if 0<=t<=1 and 0<=u<=1:
                ints.append((x1+t*(x2-x1), y1+t*(y2-y1)))
    unique = []
    for pt in ints:
        if not any(np.linalg.norm(np.array(pt) - np.array(u)) < 2.0 for u in unique):
            unique.append(pt)
    return unique

def get_clinical_symmetry_bpd(image_np, ofd_pts, principal_axis, outer_cont, inner_cont):
    """BPD calculation with symmetry and intensity refinement."""
    if len(ofd_pts) < 2:
        return 0, {'top': None, 'bottom': None}
    
    mid_x = (ofd_pts[0][0] + ofd_pts[-1][0]) / 2
    mid_y = (ofd_pts[0][1] + ofd_pts[-1][1]) / 2
    perp_vec = np.array([-principal_axis[1], principal_axis[0]])
    perp_vec = perp_vec / np.linalg.norm(perp_vec)
    
    p1 = (int(mid_x - perp_vec[0] * 1024), int(mid_y - perp_vec[1] * 1024))
    p2 = (int(mid_x + perp_vec[0] * 1024), int(mid_y + perp_vec[1] * 1024))
    
    o_ints = line_contour_intersection(p1, p2, outer_cont)
    i_ints = line_contour_intersection(p1, p2, inner_cont)
    
    if not (o_ints and i_ints): return 0, {'top': None, 'bottom': None}

    top_pt = min(o_ints, key=lambda p: p[1])
    bot_pt_initial = max(i_ints, key=lambda p: p[1])

    # Intensity snapping logic
    refined_bot_y = bot_pt_initial[1]
    search_range = 15 
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    best_gradient = -1
    for offset in range(-5, search_range):
        y = int(bot_pt_initial[1] + offset)
        x = int(bot_pt_initial[0])
        if 0 <= y + 1 < gray.shape[0]:
            gradient = int(gray[y+1, x]) - int(gray[y, x])
            if gradient > best_gradient:
                best_gradient = gradient
                refined_bot_y = y

    bot_pt = (bot_pt_initial[0], refined_bot_y)
    dist = np.linalg.norm(np.array(top_pt) - np.array(bot_pt))
    return dist, {'top': top_pt, 'bottom': bot_pt}

def save_accurate_visualization(orig_img, ofd_pts, bpd_pts, midline_mask, save_path):
    """Overlays measurements on the original image and saves."""
    img = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
    overlay = img.copy()
    if midline_mask is not None:
        overlay[midline_mask > 0] = [128, 0, 128] 
        img = cv2.addWeighted(overlay, 0.2, img, 0.8, 0)
    
    if len(ofd_pts) >= 2:
        p_front, p_back = tuple(map(int, ofd_pts[0])), tuple(map(int, ofd_pts[-1]))
        cv2.line(img, p_front, p_back, (255, 255, 0), 2)
        mid_x, mid_y = (p_front[0] + p_back[0]) // 2, (p_front[1] + p_back[1]) // 2
        cv2.circle(img, (mid_x, mid_y), 5, (0, 0, 255), -1)

    if bpd_pts['top'] and bpd_pts['bottom']:
        p1, p2 = tuple(map(int, bpd_pts['top'])), tuple(map(int, bpd_pts['bottom']))
        cv2.line(img, p1, p2, (0, 255, 0), 2)
        cv2.circle(img, p1, 4, (0, 0, 255), -1) 
        cv2.circle(img, p2, 4, (0, 0, 255), -1) 
        
    cv2.imwrite(save_path, img)

# MODIFIED CORE PIPELINE
def run_custom_inference(img_path, output_dir):
    """Processes a single image and saves all outputs to a user-defined folder."""
    if not os.path.exists(img_path):
        print(f" Error: Input image {img_path} not found.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(img_path)

    # 1. Setup Model
    model = setup_medsam_model(Config.BEST_CHECKPOINT)
    
    # 2. Load and Prepare Image
    orig_img = Image.open(img_path).convert("RGB")
    img_padded, _ = make_image_square_with_zero_padding(orig_img)
    img_tensor = TF.to_tensor(img_padded.resize((1024, 1024))).unsqueeze(0).to(Config.DEVICE)
    
    # 3. Get Masks and Metrics
    masks = get_medsam_masks(img_tensor, model)
    masks_orig = {k: resize_mask_to_original(v, orig_img.size, img_padded.size) for k, v in masks.items()}
    
    try:
        mean_pt, principal_axis, angle = compute_pca_line(masks_orig['midline'])
    except:
        print(f"Falx detection failed for {filename}")
        return

    outer_cont = extract_refined_contour(masks_orig['outer_cal'])
    inner_cont = extract_refined_contour(masks_orig['inner_cal'])

    # OFD calculation
    ofd_line_p1 = (int(mean_pt[0] - principal_axis[0]*1024), int(mean_pt[1] - principal_axis[1]*1024))
    ofd_line_p2 = (int(mean_pt[0] + principal_axis[0]*1024), int(mean_pt[1] + principal_axis[1]*1024))
    ofd_ints = sorted(line_contour_intersection(ofd_line_p1, ofd_line_p2, outer_cont), key=lambda p: p[0])

    ofd_px_full = np.linalg.norm(np.array(ofd_ints[0]) - np.array(ofd_ints[-1])) if len(ofd_ints) >= 2 else 0
    ofd_px = round(ofd_px_full, 2)

    # BPD calculation
    bpd_px_full, bpd_pts = get_clinical_symmetry_bpd(np.array(orig_img), ofd_ints, principal_axis, outer_cont, inner_cont) 
    bpd_px = round(bpd_px_full, 2)
    
    # 4. Save Visualization to Custom Folder
    vis_save_path = os.path.join(output_dir, f"measured_{filename}")
    save_accurate_visualization(orig_img, ofd_ints, bpd_pts, masks_orig['midline'], vis_save_path)
    
    # 5. Save Results CSV to Custom Folder
        # 5. Save Results CSV to Custom Folder
    csv_data = {
        'filename': filename,
        'ofd_pixel_dist': ofd_px,
        'bpd_pixel_dist': bpd_px,
    }

    csv_save_path = os.path.join(output_dir, f"data_{os.path.splitext(filename)[0]}.csv")
    pd.DataFrame([csv_data]).to_csv(csv_save_path, index=False)
    
    print(f"\n--- SUCCESS ---")
    print(f" Visualization: {vis_save_path}")
    print(f" CSV Data: {csv_save_path}")
    print(f"----------------\n")

    return ofd_px, bpd_px

if __name__ == "__main__":
    # Runs using the variables defined in the USER CONFIGURATION section
    ofd_px, bpd_px = run_custom_inference(INPUT_IMAGE_PATH, CUSTOM_OUTPUT_FOLDER)

    print(ofd_px, bpd_px)