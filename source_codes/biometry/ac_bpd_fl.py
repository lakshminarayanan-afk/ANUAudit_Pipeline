####################################################################################################################3
### DICOM -> MODE1


import torch
import cv2
import numpy as np
import pydicom
import os
from skimage.morphology import skeletonize
import math
import csv
import json
from glob import glob
from scipy.optimize import brentq
from tqdm import main

from biometry.weights.head.head_unet.model import UNet
from biometry.weights.abd.model import YNet
from biometry.weights.limbs.model import YNet as YNet_limbs
# from weights.abd.model import YNet
# from weights.limbs.model import YNet as YNet_limbs

import pandas as pd
from PIL import Image
import torchvision.transforms.functional as TF
from pathlib import Path

# from ui.utils.dcm_png_utils import anonymized_png_for_dicom

from utils.seg_biom_results_json import write_biometry_result
from utils.patient_values_json import write_patient_results

# from segment_anything import sam_model_registry
# from weights.head.head_unet.model import UNet
# from weights.head.config_biom import Config_biom
# from weights.head.model import MedSAMFineTune


# from config_chumma_1line_audit import Config
from config import Config
#########################################################################
# CONFIGURATION
#########################################################################

# OUTPUT_BASE_DIR = '/home/htic/MLN/ANU-Audit/biometry/mode1'

# OVERLAY_DIR                   = os.path.join(OUTPUT_BASE_DIR, 'overlays')
# OVERLAY_DIR_HEAD              = os.path.join(OVERLAY_DIR, 'head')
# OVERLAY_DIR_ABD               = os.path.join(OVERLAY_DIR, 'abdomen')
# OVERLAY_DIR_FEMUR             = os.path.join(OVERLAY_DIR, 'femur')
# OVERLAY_DIR_FEMUR_SPLITSCREEN = os.path.join(OVERLAY_DIR, 'femur_splitscreen')

# MASK_OVERLAY_DIR                   = os.path.join(OUTPUT_BASE_DIR, 'mask_overlays')
# MASK_OVERLAY_DIR_HEAD              = os.path.join(MASK_OVERLAY_DIR, 'head')
# MASK_OVERLAY_DIR_ABD               = os.path.join(MASK_OVERLAY_DIR, 'abdomen')
# MASK_OVERLAY_DIR_FEMUR             = os.path.join(MASK_OVERLAY_DIR, 'femur')
# MASK_OVERLAY_DIR_FEMUR_SPLITSCREEN = os.path.join(MASK_OVERLAY_DIR, 'femur_splitscreen')


# WEIGHTS_PATH = Config.ABD_CHECKPOINT
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SIZE = (224, 224)

# ── UNet (BPD) configuration ──────────────────────────────────────────
UNET_CHECKPOINT_PATH = Config.UNET_CHECKPOINT_PATH
UNET_DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNET_INPUT_SIZE      = (512, 512)   # (width, height) for cv2 / PIL resize
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'} # '.dcm'}
CSV_COLUMNS = [
    "image_path",
    "visualization_path",
    "modality",
    "BPD_mm", "OFD_mm",
    "AC_mm",
    "FL_mm",
    "EFW_g",
    "GA_weeks", "GA_days",
    "CI_index",           # Cephalic Index = (BPD/OFD)*100
    "confidence_score",
]
JSON_COLUMNS = [
    "image_path",
    "visualization_path",
    "modality",
    "BPD_mm", "OFD_mm",
    "AC_mm", "FL_mm",
    "EFW_g",
    "GA_weeks", "GA_days",
    "CI_index",
    "confidence_score",
    "points",
]

MASK_COLORS_HEAD = {
    'outer_calvarium': (255, 200,   0),
    'inner_calvarium': (  0, 255, 180),
    'midline_falx'   : (200,   0, 255),
}
MASK_COLOR_ABD   = (  0, 255,   0)
MASK_COLOR_FEMUR = (  0, 200, 255)


#########################################################################
# SHARED HELPERS
#########################################################################

def is_image_file(path):
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def is_dicom_file(path):
    ext = os.path.splitext(path)[1].lower()
    return ext in {'.dcm', '.dicom', ''}


# def load_image_generic(path):
#     if is_image_file(path):
#         img_bgr = cv2.imread(path)
#         if img_bgr is None:
#             raise IOError(f"Could not read image: {path}")
#         pil_rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
#         pixel_spacing = (1.0, 1.0)
#         print(f"Loaded plain image : {os.path.basename(path)}  "
#               f"(no pixel spacing — measurements will be in pixels)")
#         return img_bgr, pil_rgb, pixel_spacing, False
#     else:
#         ds = pydicom.dcmread(path)
#         pixels = ds.pixel_array.astype(float)
#         pixels = (pixels - np.min(pixels)) / (np.max(pixels) - np.min(pixels) + 1e-8) * 255.0
#         pixels = pixels.astype(np.uint8)

#         if pixels.ndim == 2:
#             img_bgr = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
#         else:
#             img_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

#         pil_rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

#         spacing = ds.get("PixelSpacing", None) or ds.get("ImagerPixelSpacing", [1.0, 1.0])
#         pixel_spacing = (float(spacing[0]), float(spacing[1]))
#         print(f"Loaded DICOM       : {os.path.basename(path)}  "
#               f"| PixelSpacing: {pixel_spacing} mm/px")
#         return img_bgr, pil_rgb, pixel_spacing, True

def load_image_generic(path):
    if is_image_file(path):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise IOError(f"Could not read image: {path}")
        pil_rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        pixel_spacing = (1.0, 1.0)
        print(f"Loaded plain image : {os.path.basename(path)}  "
              f"(no pixel spacing — measurements will be in pixels)")
        return img_bgr, pil_rgb, pixel_spacing, False
    else:
        ds = pydicom.dcmread(path)
        pixels = ds.pixel_array.astype(float)
        pixels = (pixels - np.min(pixels)) / (np.max(pixels) - np.min(pixels) + 1e-8) * 255.0
        pixels = pixels.astype(np.uint8)

        if pixels.ndim == 2:
            img_bgr = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

        pil_rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        # Standard tags (Voluson), else fall back to US Regions sequence (Sonoscape)
        spacing = ds.get("PixelSpacing") or ds.get("ImagerPixelSpacing")
        if not spacing:
            us_regions = ds.get((0x0018, 0x6011))
            if us_regions:
                r = us_regions[0]
                dx = r.get((0x0018, 0x602C))
                dy = r.get((0x0018, 0x602E))
                spacing = [abs(float(dy.value)) * 10.0, abs(float(dx.value)) * 10.0] if dx and dy else [1.0, 1.0]
            else:
                spacing = [1.0, 1.0]

        pixel_spacing = (float(spacing[0]), float(spacing[1]))
        print(f"Loaded DICOM       : {os.path.basename(path)}  "
              f"| PixelSpacing: {pixel_spacing} mm/px")
        return img_bgr, pil_rgb, pixel_spacing, True


#########################################################################
# MASK OVERLAY HELPER
#########################################################################

def save_mask_overlay(img_bgr, masks_dict, save_path, alpha=0.45):
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
    print(f"Mask overlay saved : {save_path}")
    return result


#########################################################################
# GEOMETRY FUNCTIONS
#########################################################################

def compute_pca_line(mask_np):
    kernel = np.ones((5, 5), np.uint8)
    mask_np = cv2.morphologyEx(mask_np.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Midline not found")
    pts = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    mean = np.mean(pts, axis=0)
    _, eig = np.linalg.eigh(np.cov(pts - mean, rowvar=False))
    axis = eig[:, 1]
    return mean, axis, math.degrees(math.atan2(axis[1], axis[0]))


def line_contour_intersection(p1, p2, contour):
    if contour is None:
        return []
    ints = []
    for i in range(len(contour)):
        seg_p3 = tuple(contour[i][0])
        seg_p4 = tuple(contour[(i + 1) % len(contour)][0])
        x1, y1 = p1; x2, y2 = p2
        x3, y3 = seg_p3; x4, y4 = seg_p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) > 1e-10:
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
            if 0 <= t <= 1 and 0 <= u <= 1:
                pt = (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
                ints.append(pt)
    return ints


#########################################################################
# VISUALIZATION
#########################################################################

def save_visualization(orig_img, ofd_pts, bpd_pts, midline_mask, save_path):
    # img = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
    img = orig_img.copy()
    overlay = img.copy()
    if midline_mask is not None:
        overlay[midline_mask > 0] = [128, 0, 128]
        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
    if len(ofd_pts) >= 2:
        ofd_p1 = tuple(map(int, ofd_pts[0]))
        ofd_p2 = tuple(map(int, ofd_pts[-1]))

        cv2.line(img, ofd_p1, ofd_p2, (0, 255, 255), 2)

        # Red dots at OFD endpoints
        cv2.circle(img, ofd_p1, 5, (0, 0, 255), -1)
        cv2.circle(img, ofd_p2, 5, (0, 0, 255), -1)
        # cv2.line(img, tuple(map(int, ofd_pts[0])), tuple(map(int, ofd_pts[-1])), (0, 255, 255), 2)
    if bpd_pts['top'] and bpd_pts['bottom']:
        p1 = tuple(map(int, bpd_pts['top']))
        p2 = tuple(map(int, bpd_pts['bottom']))
        cv2.line(img, p1, p2, (0, 255, 0), 2)
        cv2.circle(img, p1, 5, (0, 0, 255), -1)
        cv2.circle(img, p2, 5, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)
    return img


#########################################################################
# HEAD — BPD / OFD
#########################################################################

# def setup_model():
#     sam_model = sam_model_registry["vit_b"](checkpoint=Config.MEDSAM_CHECKPOINT)
#     model = MedSAMFineTune(sam_model, num_classes=Config_biom.NUM_CLASSES)
#     checkpoint = torch.load(Config.BEST_CHECKPOINT, map_location=Config_biom.DEVICE)
#     model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
#     return model.to(Config_biom.DEVICE).eval()

def setup_unet_model():
    """Load the Dual-Head UNet from best_model.pth."""
    model      = UNet()
    checkpoint = torch.load(UNET_CHECKPOINT_PATH, map_location=UNET_DEVICE)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    return model.to(UNET_DEVICE).eval()


def get_clinical_symmetry_bpd(image_np, ofd_pts, principal_axis, outer_cont, inner_cont):
    if len(ofd_pts) < 2:
        return 0, {'top': None, 'bottom': None}

    mid_x = (ofd_pts[0][0] + ofd_pts[-1][0]) / 2
    mid_y = (ofd_pts[0][1] + ofd_pts[-1][1]) / 2
    perp_vec = np.array([-principal_axis[1], principal_axis[0]])
    perp_vec /= np.linalg.norm(perp_vec)

    p1 = (int(mid_x - perp_vec[0] * 1024), int(mid_y - perp_vec[1] * 1024))
    p2 = (int(mid_x + perp_vec[0] * 1024), int(mid_y + perp_vec[1] * 1024))

    o_ints = line_contour_intersection(p1, p2, outer_cont)
    i_ints = line_contour_intersection(p1, p2, inner_cont)

    if not (o_ints and i_ints):
        return 0, {'top': None, 'bottom': None}

    top_pt = min(o_ints, key=lambda p: p[1])
    bot_pt_initial = max(i_ints, key=lambda p: p[1])

    refined_bot_y = bot_pt_initial[1]
    search_range  = 15
    gray          = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    best_gradient = -1
    for offset in range(-5, search_range):
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


# ── MODIFIED: now computes and returns confidence_score ───────────────
def line_intersection(p1, p2, p3, p4):
    """
    Find intersection of two lines (p1-p2) and (p3-p4)
    Returns (x, y) or None if parallel
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if denom == 0:
        return None

    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom

    return (px, py)

def process_single_file_bpd(dicom_path=None, image_path=None):
    DEVIATION_THRESHOLD = 20 
    file_path = dicom_path if dicom_path is not None else image_path
    if file_path is None:
        raise ValueError("Provide either dicom_path or image_path.")

    # model = setup_model()
    model = setup_unet_model()
    img_bgr, orig_img, spacing, is_dicom = load_image_generic(file_path)
    # Image used only for visualization
    vis_img_bgr = orig_img.copy()

    # if is_dicom:
    #         anonymized_path = anonymized_png_for_dicom(file_path)

    #         if anonymized_path is not None:
    #             vis_img_bgr = cv2.imread(str(anonymized_path))

    #             if vis_img_bgr is None:
    #                 vis_img_bgr = img_bgr.copy()

    #             elif vis_img_bgr.shape[:2] != img_bgr.shape[:2]:
    #                 vis_img_bgr = cv2.resize(
    #                     vis_img_bgr,
    #                     (img_bgr.shape[1], img_bgr.shape[0]),
    #                     interpolation=cv2.INTER_LINEAR
    #                 )

    # ── UNet preprocessing: grayscale 512×512, single-channel tensor ──
    orig_w, orig_h = orig_img.size                       # PIL: (width, height)
    gray_pil       = orig_img.convert("L")
    img_resized    = gray_pil.resize(UNET_INPUT_SIZE, resample=Image.BILINEAR)
    img_tensor     = TF.to_tensor(img_resized).unsqueeze(0).to(UNET_DEVICE)

    with torch.no_grad():
        outputs     = model(img_tensor)
        # Head 1 — softmax: midline falx is class index 3
        inner_probs = torch.softmax(outputs["inner"], dim=1)
        inner_pred  = torch.argmax(inner_probs, dim=1).squeeze(0).cpu().numpy()
        # Head 2 — sigmoid: ch0 = inner calvarium, ch1 = outer calvarium
        calv_probs  = torch.sigmoid(outputs["calv"]).squeeze(0).cpu().numpy()

    # Confidence score: mean sigmoid probability over foreground calv pixels
    foreground = calv_probs[calv_probs > 0.5]
    confidence_score = round(float(foreground.mean()) if len(foreground) > 0 else 0.0, 4)
    print(f"Head confidence score: {confidence_score}")

    # ── Binary masks at 512×512 → resize to original image dimensions ─
    mid_512   = (inner_pred == 3).astype(np.uint8)
    inner_512 = (calv_probs[0] > 0.5).astype(np.uint8)
    outer_512 = (calv_probs[1] > 0.5).astype(np.uint8)

    def resize_mask(mask_512):
        """Nearest-neighbour resize from 512×512 to original image size."""
        return cv2.resize(mask_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    outer = resize_mask(outer_512)
    inner = resize_mask(inner_512)
    mid   = resize_mask(mid_512)

    mean_pt, axis, angle = compute_pca_line(mid)

    def get_c(m):
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(cnts, key=cv2.contourArea) if cnts else None

    outer_c = get_c(outer)
    inner_c = get_c(inner)

    p1 = (int(mean_pt[0] - axis[0] * 1024), int(mean_pt[1] - axis[1] * 1024))
    p2 = (int(mean_pt[0] + axis[0] * 1024), int(mean_pt[1] + axis[1] * 1024))

    ofd_pts = sorted(line_contour_intersection(p1, p2, outer_c), key=lambda p: p[0])

    ofd_px = 0
    if len(ofd_pts) >= 2:
        ofd_px = np.linalg.norm(np.array(ofd_pts[0]) - np.array(ofd_pts[-1]))
    ofd_px = round(ofd_px, 2)

    bpd_px, bpd_pts = get_clinical_symmetry_bpd(
        np.array(orig_img), ofd_pts, axis, outer_c, inner_c
    )
    bpd_px = round(bpd_px, 2)

    # ─────────────────────────────────────────────
    # OFD–BPD deviation
    # ─────────────────────────────────────────────
    deviation_px = 0

    if len(ofd_pts) >= 2 and bpd_pts['top'] is not None and bpd_pts['bottom'] is not None:

        bpd_top = bpd_pts['top']
        bpd_bot = bpd_pts['bottom']

        mid_bpd = (
            (bpd_top[0] + bpd_bot[0]) / 2,
            (bpd_top[1] + bpd_bot[1]) / 2
        )

        ofd_p1 = ofd_pts[0]
        ofd_p2 = ofd_pts[-1]

        intersect_pt = line_intersection(ofd_p1, ofd_p2, bpd_top, bpd_bot)

        if intersect_pt is not None:
            deviation_px = np.linalg.norm(
                np.array(intersect_pt) - np.array(mid_bpd)
            )
            deviation_px = round(deviation_px, 2)
        else:
            print("OFD and BPD lines are parallel — skipping case.")
            return None

    else:
        print("Insufficient points — skipping case.")
        return None

    # Convert to mm if needed
    if is_dicom:
        deviation_val = round(deviation_px * spacing[0], 2)
        ofd_val = round(ofd_px * spacing[0], 2)
        bpd_val = round(bpd_px * spacing[0], 2)
        unit = "mm"
    else:
        deviation_val = deviation_px
        ofd_val = ofd_px
        bpd_val = bpd_px
        unit = "px"

    print(f"OFD–BPD deviation ({unit}): {deviation_val}")

    # ─────────────────────────────────────────────
    # 🚫 QUALITY FILTER
    # ─────────────────────────────────────────────
    if deviation_val > DEVIATION_THRESHOLD:
        print(f"❌ Rejected (deviation {deviation_val} > {DEVIATION_THRESHOLD})")
        return None

    print(f"✅ Accepted (deviation {deviation_val} <= {DEVIATION_THRESHOLD})")

    # # ─────────────────────────────────────────────
    # # SAVE ONLY IF PASSED
    # # ─────────────────────────────────────────────
    # os.makedirs(MASK_OVERLAY_DIR_HEAD, exist_ok=True)
    # mask_overlay_path = os.path.join(
    #     MASK_OVERLAY_DIR_HEAD, os.path.basename(file_path) + "_mask_overlay.png"
    # )

    # save_mask_overlay(
    #     img_bgr,
    #     {
    #         'outer_calvarium': (outer, MASK_COLORS_HEAD['outer_calvarium']),
    #         'inner_calvarium': (inner, MASK_COLORS_HEAD['inner_calvarium']),
    #         'midline_falx'   : (mid,   MASK_COLORS_HEAD['midline_falx']),
    #     },
    #     mask_overlay_path,
    # )

    # os.makedirs(OVERLAY_DIR_HEAD, exist_ok=True)
    # vis_path  = os.path.join(OVERLAY_DIR_HEAD, os.path.basename(file_path) + "_overlay.png")

    # img_inst  = save_visualization(vis_img_bgr, ofd_pts, bpd_pts, mid, vis_path)

    # print(f"BPD ({unit}): {bpd_val}  |  OFD ({unit}): {ofd_val}  |  Angle: {round(angle, 2)}°")
    # print(f"Overlay saved to: {vis_path}")

    return (
        ofd_val,
        bpd_val,
        is_dicom,
        confidence_score,
        deviation_val,
        {
            "ofd_start": (
                list(map(float, ofd_pts[0]))
                if len(ofd_pts) >= 2 else None
            ),
            "ofd_end": (
                list(map(float, ofd_pts[-1]))
                if len(ofd_pts) >= 2 else None
            ),
            "bpd_top": (
                list(map(float, bpd_pts["top"]))
                if bpd_pts["top"] is not None else None
            ),
            "bpd_bottom": (
                list(map(float, bpd_pts["bottom"]))
                if bpd_pts["bottom"] is not None else None
            ),
        }
    )


#########################################################################
# ABDOMEN — AC
#########################################################################

def calculate_perfected_ac(mask, original_shape):
    h_orig, w_orig = original_shape[:2]
    mask_res = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None, None
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    perimeter_hull = cv2.arcLength(hull, True)
    epsilon = 0.002 * perimeter_hull
    smooth_skin_contour = cv2.approxPolyDP(hull, epsilon, True)
    ac_px = cv2.arcLength(smooth_skin_contour, True)
    return ac_px, smooth_skin_contour, mask_res


# ── MODIFIED: now computes and returns confidence_score ───────────────
def run_inference_abd(dicom_path=None, image_path=None):
    WEIGHTS_PATH = Config.ABD_CHECKPOINT
    file_path = dicom_path if dicom_path is not None else image_path
    if file_path is None:
        print("Error: provide either dicom_path or image_path.")
        return None, None, None, None

    img_orig, _, pixel_spacing, is_dicom = load_image_generic(file_path)
    # Image used only for visualization
    vis_img_bgr = img_orig.copy()

    # if is_dicom:
    #     anonymized_path_ac = anonymized_png_for_dicom(file_path)

    #     if anonymized_path_ac is not None:
    #         vis_img_bgr = cv2.imread(str(anonymized_path_ac))
    #         print(f"ANONYM PATH AC: {anonymized_path_ac}")

    #         if vis_img_bgr is None:
    #             print(f"NO vis_img_bgr")
    #             vis_img_bgr = img_orig.copy()

    #         elif vis_img_bgr.shape[:2] != img_orig.shape[:2]:
    #             print(f"COMING HEREEEE")
    #             vis_img_bgr = cv2.resize(
    #                 vis_img_bgr,
    #                 (img_orig.shape[1], img_orig.shape[0]),
    #                 interpolation=cv2.INTER_LINEAR
    #             )

    #         else:

    #             vis_img_bgr = vis_img_bgr 
    fname = os.path.basename(file_path)

    model = YNet(input_channels=1, output_channels=64, n_class=1).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()

    gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, TARGET_SIZE)
    normalized = (resized / 255.0 - 0.5) / 0.5
    input_tensor = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        _, pred = model(input_tensor)

    pred_np = pred.squeeze().cpu().numpy()
    foreground = pred_np[pred_np > 0.5]
    confidence_score = round(float(foreground.mean()) if len(foreground) > 0 else 0.0, 4)
    print(f"Abdomen confidence score: {confidence_score}")
    # ──────────────────────────────────────────────────────────────────

    mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

    ac_px_full, smooth_contour, mask_resized = calculate_perfected_ac(mask, img_orig.shape)
    if ac_px_full is None:
        print("No valid contour detected.")
        return None, None, is_dicom, None

    ac_px = round(ac_px_full, 2)

    # os.makedirs(MASK_OVERLAY_DIR_ABD, exist_ok=True)
    # mask_overlay_path = os.path.join(
    #     MASK_OVERLAY_DIR_ABD, fname + "_mask_overlay.png"
    # )
    # save_mask_overlay(
    #     vis_img_bgr,
    #     {'abdomen': (mask_resized, MASK_COLOR_ABD)},
    #     mask_overlay_path,
    # )

    if is_dicom:
        mean_spacing = (pixel_spacing[0] + pixel_spacing[1]) / 2.0
        ac_val = round(ac_px_full * mean_spacing, 2)
        unit = "mm"
    else:
        ac_val = ac_px
        unit = "px"

    print(f"AC ({unit}): {ac_val}")

    # os.makedirs(OVERLAY_DIR_ABD, exist_ok=True)
    # ac_overlay_path = os.path.join(OVERLAY_DIR_ABD, fname + "_overlay.png")
    # vis_img = vis_img_bgr.copy()
    # cv2.drawContours(vis_img, [smooth_contour], -1, (0, 255, 0), 3)
    # cv2.imwrite(ac_overlay_path, vis_img)
    # print(f"Overlay saved to: {ac_overlay_path}")
    # Store ALL contour points
    ac_points = smooth_contour.reshape(-1, 2).astype(float).tolist()

    # ── MODIFIED: returns confidence_score as extra value ─────────────
    return ac_val, is_dicom, confidence_score, ac_points


#########################################################################
# FEMUR — FL
#########################################################################

input_channels  = 1
output_channels = 64
n_class         = 1
target_size     = (224, 224)
ANGLE_THRESHOLD_DEG = 20.0
SEG_THRESHOLD       = 0.85
PADDING_PERCENT     = 0.05
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_fl = YNet_limbs(input_channels=input_channels,
                      output_channels=output_channels,
                      n_class=n_class)
model_fl.load_state_dict(torch.load(Config.FEMUR_CHECKPOINT, map_location=device))
model_fl = model_fl.to(device)
model_fl.eval()


def detectCenterLines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mid_x = w // 2
    PADDING_MARGIN = int(w * PADDING_PERCENT)
    cx_min, cx_max = mid_x - PADDING_MARGIN, mid_x + PADDING_MARGIN
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=h // 3, maxLineGap=20)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if 88 <= angle <= 92 and cx_min <= (x1 + x2) / 2 <= cx_max:
                return True
    return False


@torch.no_grad()
def run_inference_fl(frame_np):
    gray = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY) if len(frame_np.shape) == 3 else frame_np
    img_res = cv2.resize(gray, target_size) / 255.0
    input_tensor = torch.from_numpy(img_res).float().to(device).view(1, 1, 1, 224, 224)
    _, seg_output = model_fl(input_tensor)
    return seg_output.cpu().numpy().squeeze()


def get_angle_from_horizontal(p1, p2):
    if p1 is None or p2 is None:
        return True, 0.0, None, None
    if p1[0] <= p2[0]:
        origin, other_end = tuple(p1), tuple(p2)
    else:
        origin, other_end = tuple(p2), tuple(p1)
    dx = other_end[0] - origin[0]
    dy = other_end[1] - origin[1]
    angle_deg = 90.0 if abs(dx) < 1e-6 else np.degrees(np.arctan2(dy, dx))
    angle_deg_abs = abs(angle_deg)
    is_rejected = angle_deg_abs > ANGLE_THRESHOLD_DEG
    return is_rejected, angle_deg_abs, origin, other_end


def get_clinical_best_fl(seg_map, original_img):
    img_h, img_w = original_img.shape[:2]
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    best_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    full_mask = (seg_map > SEG_THRESHOLD).astype(np.uint8) * 255
    full_mask = cv2.resize(full_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, full_mask, best_mask, None, 0.0, None, None

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 600:
            continue
        c_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.drawContours(c_mask, [c], -1, 255, -1)
        brightness = cv2.mean(gray, mask=c_mask)[0]
        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), rect_angle = rect
        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
        if aspect_ratio < 2.0:
            continue
        score = (brightness * 2.0) + (area / 100.0) + (img_h - cy)
        candidates.append((score, c, rect))

    if not candidates:
        return None, None, full_mask, best_mask, None, 0.0, None, None

    _, best_contour, rect = max(candidates, key=lambda x: x[0])
    (cx, cy), (w, h), rect_angle = rect
    if w < h:
        rect_angle += 90
    axis = np.array([np.cos(np.deg2rad(rect_angle)), np.sin(np.deg2rad(rect_angle))])

    cv2.drawContours(best_mask, [best_contour], -1, 255, -1)
    ys, xs  = np.where(best_mask > 0)
    pts     = np.stack([xs, ys], axis=1)
    projs   = pts @ axis
    edges   = cv2.Canny(best_mask, 50, 150)
    ey, ex  = np.where(edges > 0)
    e_pts   = np.stack([ex, ey], axis=1)
    e_projs = e_pts @ axis

    if len(e_pts) == 0:
        return None, None, full_mask, best_mask, None, 0.0, None, None

    proj_range    = projs.max() - projs.min()
    p1_candidates = e_pts[e_projs < projs.min() + proj_range * 0.05]
    p2_candidates = e_pts[e_projs > projs.max() - proj_range * 0.05]

    if len(p1_candidates) == 0 or len(p2_candidates) == 0:
        return None, None, full_mask, best_mask, None, 0.0, None, None

    p1 = p1_candidates.mean(axis=0).astype(int)
    p2 = p2_candidates.mean(axis=0).astype(int)
    is_rejected, angle_deg, origin, other_end = get_angle_from_horizontal(
        tuple(p1), tuple(p2)
    )
    box = np.int64(cv2.boxPoints(rect))
    return tuple(p1), tuple(p2), full_mask, best_mask, box, angle_deg, origin, other_end


# ── MODIFIED: now computes and returns confidence_score ───────────────
def process_single_file_fl(dicom_path=None, image_path=None):

    file_path = dicom_path if dicom_path is not None else image_path
    if file_path is None:
        print("Error: provide either dicom_path or image_path.")
        return None, None, None, None

    img_bgr, _, pixel_spacing, is_dicom = load_image_generic(file_path)

    has_split = detectCenterLines(img_bgr)

    # ---------------------------------------------------------
    # Visualization image:
    # Use anonymized PNG for DICOMs, but keep original image
    # for inference/measurement.
    # ---------------------------------------------------------
    vis_img_bgr = img_bgr.copy()

    # if is_dicom:
    #     anonymized_path = anonymized_png_for_dicom(file_path)

    #     if anonymized_path is not None:
    #         vis_img_bgr = cv2.imread(str(anonymized_path))

    #         if vis_img_bgr is None:
    #             print(f"⚠️ Could not read anonymized PNG: {anonymized_path}")
    #             vis_img_bgr = img_bgr.copy()
    #         else:
    #             # Make sure visualization image has exactly the same
    #             # dimensions as the image used for inference.
    #             if vis_img_bgr.shape[:2] != img_bgr.shape[:2]:
    #                 vis_img_bgr = cv2.resize(
    #                     vis_img_bgr,
    #                     (img_bgr.shape[1], img_bgr.shape[0]),
    #                     interpolation=cv2.INTER_LINEAR
    #                 )

    #             print(f"Visualization image: {anonymized_path}")
    #     else:
    #         print(f"⚠️ No anonymized PNG found for: {file_path}")
    #         print("   Falling back to original image for visualization.")
            
    if has_split:
        frames = [
            (img_bgr[:, :img_bgr.shape[1] // 2], "left",  0),
            (img_bgr[:, img_bgr.shape[1] // 2:], "right", img_bgr.shape[1] // 2),
        ]
    else:
        frames = [(img_bgr, "full", 0)]

    # master_vis = img_bgr.copy()
    
    master_vis = vis_img_bgr.copy()
    best_result = None   # (distance_val, drawing, confidence_score)

    for frame_np, side, offset in frames:
        seg_map = run_inference_fl(frame_np)
        p1, p2, full_mask, best_mask, box, angle_deg, origin, other_end = \
            get_clinical_best_fl(seg_map, frame_np)

        # combined_frame = frame_np.copy()
        if has_split:
            vis_frame_np = vis_img_bgr[
                :,
                offset:offset + frame_np.shape[1]
            ]
        else:
            vis_frame_np = vis_img_bgr

        combined_frame = vis_frame_np.copy()

        if p1 is None or p2 is None or origin is None or other_end is None:
            master_vis[:, offset:offset + frame_np.shape[1]] = combined_frame
            continue

        status = "Rejected" if angle_deg > ANGLE_THRESHOLD_DEG else "Accepted"
        color  = (0, 0, 255) if status == "Rejected" else (0, 255, 0)

        cv2.line(combined_frame, origin, other_end, color, 2)
        cv2.circle(combined_frame, origin,    5, (0, 0, 255), -1)
        cv2.circle(combined_frame, other_end, 5, (0, 0, 255), -1)
        # cv2.line(combined_frame, origin, (origin[0] + 100, origin[1]), (255, 255, 0), 2)
        # cv2.putText(
        #     combined_frame,
        #     f"{status}: {angle_deg:.1f}deg",
        #     (max(10, origin[0]), max(20, origin[1] - 10)),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        # )

        master_vis[:, offset:offset + frame_np.shape[1]] = combined_frame

        if status == "Rejected" or has_split:
            continue

        # ── Confidence score: mean seg_map probability within best_mask ─
        # Using the masked region gives a score specific to the detected
        # bone region rather than the whole image (which is mostly background).
        if best_mask is not None and best_mask.any():
            # seg_map is at model resolution; resize to match best_mask
            seg_map_resized = cv2.resize(
                seg_map.astype(np.float32),
                (best_mask.shape[1], best_mask.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            mask_bool = best_mask > 0
            confidence_score = round(float(seg_map_resized[mask_bool].mean()), 4)
        else:
            confidence_score = round(float(seg_map.mean()), 4)
        print(f"Femur confidence score: {confidence_score}")
        # ──────────────────────────────────────────────────────────────

        # if best_mask is not None:
        #     os.makedirs(MASK_OVERLAY_DIR_FEMUR, exist_ok=True)
        #     mask_overlay_path = os.path.join(
        #         MASK_OVERLAY_DIR_FEMUR,
        #         os.path.basename(file_path) + "_mask_overlay.png",
        #     )
        #     save_mask_overlay(
        #         img_bgr,
        #         {'femur': (best_mask, MASK_COLOR_FEMUR)},
        #         mask_overlay_path,
        #     )

        dist_px = float(np.linalg.norm(np.array(origin) - np.array(other_end)))

        if is_dicom:
            dx_mm        = (other_end[0] - origin[0]) * pixel_spacing[1]
            dy_mm        = (other_end[1] - origin[1]) * pixel_spacing[0]
            distance_val = round(float(np.sqrt(dx_mm ** 2 + dy_mm ** 2)), 2)
            unit = "mm"
        else:
            distance_val = round(dist_px, 2)
            unit = "px"

        print(f"FL ({unit}): {distance_val}  |  angle: {angle_deg:.1f}°  |  {status}")

        # os.makedirs(OVERLAY_DIR_FEMUR, exist_ok=True)
        # femur_overlay_path = os.path.join(
        #     OVERLAY_DIR_FEMUR,
        #     os.path.basename(file_path) + "_overlay.png",
        # )
        # cv2.imwrite(femur_overlay_path, master_vis)
        # print(f"Overlay saved to: {femur_overlay_path}")

        best_result = (distance_val, master_vis, confidence_score)

        return (
            best_result[0],
            best_result[1],
            is_dicom,
            best_result[2],
            {
                "start": [
                    float(origin[0]),
                    float(origin[1])
                ],
                "end": [
                    float(other_end[0]),
                    float(other_end[1])
                ],
            }
        )

    # ── Split-screen overlays (unchanged logic) ───────────────────────
    if has_split:
        # os.makedirs(OVERLAY_DIR_FEMUR_SPLITSCREEN, exist_ok=True)
        # splitscreen_overlay_path = os.path.join(
        #     OVERLAY_DIR_FEMUR_SPLITSCREEN,
        #     os.path.basename(file_path) + "_overlay.png",
        # )
        # cv2.imwrite(splitscreen_overlay_path, master_vis)
        # print(f"Split-screen overlay saved to: {splitscreen_overlay_path}")

        combined_best_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        for frame_np, side, offset in frames:
            seg_map_ss = run_inference_fl(frame_np)
            _, _, _, best_mask_ss, _, _, _, _ = get_clinical_best_fl(seg_map_ss, frame_np)
            if best_mask_ss is not None:
                combined_best_mask[
                    :, offset:offset + frame_np.shape[1]
                ] = best_mask_ss

        if combined_best_mask.any():
            # os.makedirs(MASK_OVERLAY_DIR_FEMUR_SPLITSCREEN, exist_ok=True)
            # splitscreen_mask_overlay_path = os.path.join(
            #     MASK_OVERLAY_DIR_FEMUR_SPLITSCREEN,
            #     os.path.basename(file_path) + "_mask_overlay.png",
            # )
            # save_mask_overlay(
            #     img_bgr,
            #     {'femur': (combined_best_mask, MASK_COLOR_FEMUR)},
            #     splitscreen_mask_overlay_path,
            # )
            print(f"SPLIT")


    print("Could not detect femur or no accepted measurement found.")
    return None, img_bgr, is_dicom, None


#########################################################################
# EFW / GA CALCULATIONS  (unchanged)
#########################################################################

def calculate_efw_hadlock(bpd_cm, ac_cm, fl_cm):
    log10_efw = (
        1.335
        - 0.0034 * (ac_cm * fl_cm)
        + 0.0316 * bpd_cm
        + 0.0457 * ac_cm
        + 0.1623 * fl_cm
    )
    return round(10 ** log10_efw, 2)


def calculate_efw_hadlock_ind(bpd_cm, ofd_cm, ac_cm, fl_cm):
    hc_cm = 1.62 * (bpd_cm + ofd_cm)
    log10_efw_ind = (
        1.3596
        + 0.0064 * hc_cm
        + 0.0424 * ac_cm
        + 0.174  * fl_cm
        + 0.00061 * bpd_cm * ac_cm
        - 0.00386 * ac_cm * fl_cm
    )
    return round(10 ** log10_efw_ind, 2)


CSV_PATH = Config.BI0MET_CSV_PATH
GA_MIN, GA_MAX = 14.0, 42.0


def load_efw_coefficients(csv_path=CSV_PATH):
    coefficients = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["fetalDimension"] == "EFW":
                q = float(row["quantile"])
                coefficients[q] = (
                    float(row["b0"]), float(row["b1"]),
                    float(row["b2"]), float(row["b3"]),
                    float(row["b4"]),
                )
    return coefficients


def _poly(ga, b0, b1, b2, b3, b4):
    return b0 + b1 * ga + b2 * ga ** 2 + b3 * ga ** 3 + b4 * ga ** 4


def efw_to_ga(efw_grams, quantile=0.5, coefficients=None):
    if coefficients is None:
        coefficients = load_efw_coefficients()
    if quantile not in coefficients:
        raise ValueError(f"Quantile {quantile} not found.")
    coefs   = coefficients[quantile]
    log_efw = math.log(efw_grams)

    def equation(ga):
        return _poly(ga, *coefs) - log_efw

    try:
        return brentq(equation, GA_MIN, GA_MAX)
    except ValueError:
        return None


#########################################################################
# MAIN PIPELINE
#########################################################################

def bio_ga(model_inputs):

    # ---------------------------------------------------------
    # Get patient name
    # ---------------------------------------------------------

    all_inputs = (
        model_inputs.get("bpd", [])
        + model_inputs.get("abdomen", [])
        + model_inputs.get("limbs", [])
    )

    patient_name = Path(
        all_inputs[0]["image_path"]
    ).parent.name

    patient_json_path = all_inputs[0]["json_path"]
    print(f"Patient_json_path:{patient_json_path}")

    if not all_inputs:
        raise ValueError("No biometry inputs found.")

    patient_name = Path(
        all_inputs[0]["image_path"]
    ).parent.name

    print(f"Patient: {patient_name}")

    # ---------------------------------------------------------
    # Best measurement storage
    # ---------------------------------------------------------

    best = {
        "bpd": {
            "value": None,
            "ofd": None,
            "confidence": -1.0,
            "is_dcm": True,
            "image_path": None,
        },

        "ac": {
            "value": None,
            "confidence": -1.0,
            "is_dcm": True,
            "image_path": None,
        },

        "fl": {
            "value": None,
            "confidence": -1.0,
            "is_dcm": True,
            "image_path": None,
        },
    }

    # =========================================================
    # BPD
    # =========================================================

    for item in model_inputs.get("bpd", []):

        (
            ofd_val,
            bpd_val,
            is_dicom,
            confidence_score,
            deviation_val,
            points,
        ) = process_single_file_bpd(
            item["image_path"]
        )

        if bpd_val is None or ofd_val is None:
            continue

        # Write per-image JSON
        json_entry = {
            "BPD": float(bpd_val),
            "OFD": float(ofd_val),
            "units": "mm",
            "confidence": float(confidence_score),
            "points": points,
        }

        write_biometry_result(item, json_entry)

        # Select best BPD
        if confidence_score > best["bpd"]["confidence"]:

            best["bpd"] = {
                "value": float(bpd_val),
                "ofd": float(ofd_val),
                "confidence": float(confidence_score),
                "is_dcm": is_dicom,
                "image_path": item["image_path"],
            }

    # =========================================================
    # ABDOMEN / AC
    # =========================================================

    for item in model_inputs.get("abdomen", []):

        (
            ac_val,
            is_dicom,
            confidence_score,
            ac_points,
        ) = run_inference_abd(
            item["image_path"]
        )

        if ac_val is None:
            continue

        # Write per-image JSON
        json_entry = {
            "AC": float(ac_val),
            "units": "mm",
            "confidence": float(confidence_score),
            "points": ac_points,
        }

        write_biometry_result(item, json_entry)

        # Select best AC
        if confidence_score > best["ac"]["confidence"]:

            best["ac"] = {
                "value": float(ac_val),
                "confidence": float(confidence_score),
                "is_dcm": is_dicom,
                "image_path": item["image_path"],
            }

    # =========================================================
    # FEMUR / FL
    # =========================================================

    for item in model_inputs.get("limbs", []):

        (
            fl_val,
            is_dicom,
            confidence_score,
            fl_points,
        ) = process_single_file_fl(
            item["image_path"]
        )

        if fl_val is None:
            continue

        # Write per-image JSON
        json_entry = {
            "FL": float(fl_val),
            "units": "mm",
            "confidence": float(confidence_score),
            "points": fl_points,
        }

        write_biometry_result(item, json_entry)

        # Select best FL
        if confidence_score > best["fl"]["confidence"]:

            best["fl"] = {
                "value": float(fl_val),
                "confidence": float(confidence_score),
                "is_dcm": is_dicom,
                "image_path": item["image_path"],
            }

    # =========================================================
    # Get final best measurements
    # =========================================================

    bpd_val = best["bpd"]["value"]
    ofd_val = best["bpd"]["ofd"]
    ac_val = best["ac"]["value"]
    fl_val = best["fl"]["value"]

    # ── Cephalic Index ───────────────────────────────────────────────────────
    ci_val = None
    if bpd_val is not None and ofd_val is not None and ofd_val != 0:
        ci_val = round((bpd_val / ofd_val) * 100, 2)

    efw = None
    w = None
    d = None

    all_dicom = (
        best["bpd"]["is_dcm"]
        and best["ac"]["is_dcm"]
        and best["fl"]["is_dcm"]
    )

    if (
        bpd_val is not None
        and ofd_val is not None
        and ac_val is not None
        and fl_val is not None
        and all_dicom
    ):
        efw = calculate_efw_hadlock_ind(
            bpd_val / 10.0,
            ofd_val / 10.0,
            ac_val / 10.0,
            fl_val / 10.0,
        )

        if efw is not None:
            ga = efw_to_ga(efw)

            if ga is not None:
                w = int(ga)
                d = round((ga - w) * 7)

    # =========================================================
    # WRITE PATIENT RESULTS
    # =========================================================

    write_patient_results(
        json_path=patient_json_path,
        patient_name=patient_name,
        best=best,
        bpd_val=bpd_val,
        ofd_val=ofd_val,
        ac_val=ac_val,
        fl_val=fl_val,
        ci_val=ci_val,
        efw=efw,
        ga_weeks=w,
        ga_days=d,
        units="mm",
    )

    # =========================================================
    # Return values
    # =========================================================

    return (
        bpd_val,
        ac_val,
        fl_val,
        w,
        d,
        ci_val,
    )
    
def biomet_inf(model_inputs):


    return bio_ga(
        model_inputs
    )

#########################################################################
# ENTRY POINT
#########################################################################

if __name__ == "__main__":
    biomet_inf(
        input_dir="/home/htic/MLN/ANU-Audit/OUTPUT/CLASS/ClinicalPatient_1_dcm/Biometry",
        output_dir="/home/htic/MLN/ANU-Audit/OUTPUT/BIOM_TEST",
    )