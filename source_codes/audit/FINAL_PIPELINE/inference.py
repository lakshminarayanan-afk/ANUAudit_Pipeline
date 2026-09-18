import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import argparse
import math
import shutil
import gc
import pydicom
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF
from ultralytics import YOLO
import json
from pathlib import Path

# Import model architectures from their respective tracking files
from source_codes.audit.FINAL_PIPELINE.model_ac import YNet as YNetAC
from source_codes.audit.FINAL_PIPELINE.model_bpd import UNet as UNetBPD
from source_codes.audit.FINAL_PIPELINE.model_fl import YNet as YNetFL

from utils.seg_biom_results_json import write_biometry_audit_result

from config import Config

# from model_ac import YNet as YNetAC
# from model_bpd import UNet as UNetBPD
# from model_fl import YNet as YNetFL

def get_audit_term(audit_type, point_name):
    return Config.AUDIT_TERMINOLOGY.get(
        audit_type, {}
    ).get(point_name, point_name)

# =================================================================
# UTILITY: UNIVERSAL IMAGE & IO LOADER
# =================================================================
def load_image_bgr(img_path):
    """Loads standard images or DICOM files into BGR format natively."""
    if img_path.lower().endswith('.dcm'):
        try:
            ds = pydicom.dcmread(img_path)
            img = ds.pixel_array
            if img.dtype != np.uint8:
                if img.max() > img.min():
                    img = ((img - img.min()) / (img.max() - img.min()) * 255.0).astype(np.uint8)
                else:
                    img = np.zeros(img.shape, dtype=np.uint8)
            
            if len(img.shape) == 2:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img
            return img_bgr
        except Exception as e:
            print(f"Error reading DICOM {img_path}: {e}")
            return None
    else:
        return cv2.imread(img_path)

def create_visualization_overlay(
    image_path,
    visualization_dir,
    output_path
):

    # ------------------------------------------------------------
    # Original filename information
    # ------------------------------------------------------------
    original_name = os.path.basename(image_path)
    original_stem = os.path.splitext(original_name)[0]

    # ------------------------------------------------------------
    # Search recursively for corresponding visualization
    # ------------------------------------------------------------
    matched_visualization = None

    for root, _, files in os.walk(visualization_dir):

        for fname in files:

            # Only image files
            if not fname.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):
                continue

            vis_stem = os.path.splitext(fname)[0]        # IMG_20260220_1_8.dcm_overlay

            # Remove known visualization suffix
            vis_stem = vis_stem.replace("_overlay", "")   # IMG_20260220_1_8.dcm
            vis_stem = vis_stem.replace("_vis", "")

            # Strip any leftover original-file extension (e.g. embedded .dcm)
            vis_stem = os.path.splitext(vis_stem)[0]      # IMG_20260220_1_8

            if vis_stem == original_stem:
                matched_visualization = os.path.join(
                    root,
                    fname
                )
                break

        if matched_visualization is not None:
            break

    # ------------------------------------------------------------
    # Visualization not found
    # ------------------------------------------------------------
    if matched_visualization is None:
        print(
            f"No visualization found for: {original_name}"
        )
        return None

    print(
        f"Visualization found:\n"
        f"  Input : {image_path}\n"
        f"  Visual: {matched_visualization}"
    )

    # return output_path
    overlay_base = cv2.imread(matched_visualization)
    return overlay_base

def compute_iou(box1, box2):
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    xi1, yi1 = max(x1, x1b), max(y1, y1b)
    xi2, yi2 = min(x2, x2b), min(y2, y2b)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
    box2_area = max(0, (x2b - x1b)) * max(0, (y2b - y1b))
    union = box1_area + box2_area - inter_area
    return inter_area / union if union > 0 else 0

def point_inside_box(point, box):
    """
    Check whether a point (x, y) lies inside a bounding box
    (x1, y1, x2, y2).
    """
    x, y = point
    x1, y1, x2, y2 = box

    return (
        x1 <= x <= x2 and
        y1 <= y <= y2
    )

def make_box_from_point(point, half_size=13):
    """
    Build a bounding box (x1, y1, x2, y2) centered on a single (x, y)
    point, using `half_size` pixels in each direction. Used to convert
    a JSON ground-truth point into a comparable box.
    """
    x, y = point
    return (x - half_size, y - half_size, x + half_size, y + half_size)


def boxes_overlap(box1, box2):
    """
    Check whether two (x1, y1, x2, y2) boxes overlap at all
    (non-zero intersection area).
    """
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    xi1, yi1 = max(x1, x1b), max(y1, y1b)
    xi2, yi2 = min(x2, x2b), min(y2, y2b)
    return (xi2 > xi1) and (yi2 > yi1)

def segment_intersects_box(p1, p2, box):
    """
    Check whether the line segment p1->p2 intersects/enters
    the given box (x1, y1, x2, y2), using the Liang-Barsky
    line clipping algorithm.
    """
    x1b, y1b, x2b, y2b = box
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    t0, t1 = 0.0, 1.0
    p = [-dx, dx, -dy, dy]
    q = [p1[0] - x1b, x2b - p1[0], p1[1] - y1b, y2b - p1[1]]

    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return False  # parallel and outside
        else:
            t = qi / pi
            if pi < 0:
                if t > t1:
                    return False
                if t > t0:
                    t0 = t
            else:
                if t < t0:
                    return False
                if t < t1:
                    t1 = t

    return t0 <= t1


def any_contour_point_inside_box(contour, box):
    """
    Check whether ANY point of a contour lies inside a bounding box,
    OR whether any segment between consecutive contour points
    passes through the box (handles sparse polygons).
    """
    if contour is None or len(contour) == 0:
        return False

    contour = np.asarray(contour)
    if contour.ndim == 3:
        contour = contour.reshape(-1, 2)

    x1, y1, x2, y2 = box

    # Original point-in-box check
    inside = (
        (contour[:, 0] >= x1) &
        (contour[:, 0] <= x2) &
        (contour[:, 1] >= y1) &
        (contour[:, 1] <= y2)
    )
    if bool(np.any(inside)):
        return True

    # Segment-based check for sparse polygons
    n = len(contour)
    for i in range(n):
        p1 = contour[i]
        p2 = contour[(i + 1) % n]  # wrap around, since it's a closed contour
        if segment_intersects_box(p1, p2, box):
            return True

    return False

def get_biometry_points(json_path, panel, plane):

    if not json_path or not os.path.exists(json_path):
        return None

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        biometry = data.get("biometry", {})

        panel_data = biometry.get(panel)
        if panel_data is None:
            return None

        plane_data = panel_data.get(plane)
        if plane_data is None:
            return None

        points = plane_data.get("points")

        if not isinstance(points, (dict, list)):
            return None

        return points

    except Exception as e:
        print(f"Error reading biometry JSON {json_path}: {e}")
        return None

def serialize_yolo_boxes(calipers):
    """
    Convert YOLO detections into JSON-serializable dictionaries.
    """

    boxes = []

    for caliper in calipers:

        box = caliper.get("box", caliper.get("coords"))
        center = caliper.get("center")
        conf = caliper.get("conf")

        boxes.append({
            "box": [float(x) for x in box],
            "center": [float(x) for x in center],
            "confidence": float(conf)
        })

    return boxes
# =================================================================
# CLASS 1: ABDOMINAL CIRCUMFERENCE (AC)
# =================================================================
class AC:
    def __init__(self, yolo_path, ynet_path, device, conf_thresh=0.4, iou_thresh=0.5):
        self.device = device
        self.yolo_model = YOLO(yolo_path)
        self.ynet_model = self.load_ynet_ac(ynet_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.CALIPER_TOLERANCES = [10.0, 10.0, 10.0, 10.0]  # [Top, Bottom, Left, Right]
        self.ANGLE_TOLERANCE = 15.0  # Perpendicular tolerance window around 90 degrees
        self.HORIZ_TOLERANCE = 30.0  # Max allowed deviation from true horizontal baseline
        self.ELLIPSE_SIZE_TOLERANCE = 20.0  # Pixel tolerance window for vertical axis vs major diameter
        self.dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    def load_ynet_ac(self, weights_path):
        # Instantiate model structure with 4 target classes to match the weight checkpoint dimensions
        model = YNetAC(input_channels=1, output_channels=64, n_class=4).to(self.device)
        
        # Ensure out_layer matches the exact checkpoint parameters [4, 4, 3, 3]
        model.out_layer = nn.Conv2d(
            in_channels=4, 
            out_channels=4, 
            kernel_size=3, 
            padding=1
        ).to(self.device)
        
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    def preprocess_ynet_ac(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (224, 224)) / 255.0
        tensor = torch.from_numpy(gray).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def calculate_perfected_ac_with_dims(self, mask, original_shape):
        """
        Finds the accurate model skin-line and applies an ellipse-guided 
        Distance Transform correction to mends flat shadow lines.
        """
        h_orig, w_orig = original_shape[:2]
        mask_res = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        
        # Break minor pixel bridges using an elliptical opening kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_res = cv2.morphologyEx(mask_res, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return None
        
        real_contour = max(contours, key=cv2.contourArea)
        if len(real_contour) < 5:
            return None

        # Fit hidden auxiliary ellipse to act as our anatomical reference guide
        ellipse = cv2.fitEllipse(real_contour)
        
        # Generate an ellipse border profile
        ellipse_border = np.zeros_like(mask_res)
        cv2.ellipse(ellipse_border, ellipse, 255, 1)
        
        # Create L2 Distance Transform map relative to reference ellipse line
        dist_map = cv2.distanceTransform(cv2.bitwise_not(ellipse_border), cv2.DIST_L2, 3)
        
        # Get continuous coordinates along the guide ellipse
        ellipse_contours, _ = cv2.findContours(ellipse_border, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ellipse_pts = ellipse_contours[0].reshape(-1, 2)
        
        # Selective snap-to-guide point optimization
        corrected_points = []
        MAX_ALLOWED_DEVIATION = 4
        
        for pt in real_contour:
            x, y = pt[0]
            if y >= h_orig or x >= w_orig or y < 0 or x < 0:
                continue
                
            distance = dist_map[y, x]
            if distance > MAX_ALLOWED_DEVIATION:
                sq_distances = np.sum((ellipse_pts - np.array([x, y])) ** 2, axis=1)
                nearest_ellipse_pt = ellipse_pts[np.argmin(sq_distances)]
                corrected_points.append([nearest_ellipse_pt])
            else:
                corrected_points.append([[x, y]])
                
        if len(corrected_points) < 15:
            return None
            
        corrected_contour = np.array(corrected_points, dtype=np.int32).squeeze()
        
        # Extract independent 1D spatial tracks to clean jagged staircase pixelation artifacts
        x_coords = corrected_contour[:, 0]
        y_coords = corrected_contour[:, 1]
        
        # Pad trajectories periodically to remove sharp notches at sequence wrap-around points
        pad_size = 15
        x_padded = np.concatenate([x_coords[-pad_size:], x_coords, x_coords[:pad_size]])
        y_padded = np.concatenate([y_coords[-pad_size:], y_coords, y_coords[:pad_size]])
        
        # Define a 1D Gaussian kernel window window to smooth curves biologically
        window_size = 15 
        sigma = 4
        kernel_1d = cv2.getGaussianKernel(window_size, sigma).squeeze()
        
        x_smooth = np.convolve(x_padded, kernel_1d, mode='same')
        y_smooth = np.convolve(y_padded, kernel_1d, mode='same')
        
        x_final = x_smooth[pad_size:-pad_size]
        y_final = y_smooth[pad_size:-pad_size]
        
        smooth_skin_contour = np.stack([x_final, y_final], axis=1).astype(np.int32).reshape(-1, 1, 2)
        ac_px = cv2.arcLength(smooth_skin_contour, True)

        # Extract orientation-based anatomical extrema coordinates
        leftmost = tuple(smooth_skin_contour[smooth_skin_contour[:, :, 0].argmin()][0])
        rightmost = tuple(smooth_skin_contour[smooth_skin_contour[:, :, 0].argmax()][0])
        topmost = tuple(smooth_skin_contour[smooth_skin_contour[:, :, 1].argmin()][0])
        bottommost = tuple(smooth_skin_contour[smooth_skin_contour[:, :, 1].argmax()][0])

        # Calculate precise orthogonal dimensions
        apad_px = abs(bottommost[1] - topmost[1])  # Anatomical Vertical Dimension
        tad_px = abs(rightmost[0] - leftmost[0])    # Anatomical Horizontal Dimension

        return {
            "ac_px": ac_px,
            "contour": smooth_skin_contour,
            "apad_px": apad_px,
            "tad_px": tad_px,
            "points": (topmost, bottommost, leftmost, rightmost)
        }

    def detect_calipers(self, img_bgr, conf_thresh, iou_thresh):
        results = self.yolo_model.predict(source=img_bgr, conf=0.01, verbose=False)
        raw_dets = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf < conf_thresh: 
                    continue
                coords = box.xyxy[0].cpu().numpy()
                raw_dets.append({
                    'box': coords, 
                    'conf': conf, 
                    'center': (int((coords[0]+coords[2])/2), int((coords[1]+coords[3])/2))
                })

        raw_dets = sorted(raw_dets, key=lambda x: x['conf'], reverse=True)
        keep = []
        while raw_dets:
            best = raw_dets.pop(0)
            keep.append(best)
            raw_dets = [d for d in raw_dets if compute_iou(best['box'], d['box']) < iou_thresh]

        calipers = keep[:4]
        if len(calipers) != 4: 
            return calipers  

        sorted_by_y = sorted(calipers, key=lambda c: c['center'][1])
        top_cal, bottom_cal = sorted_by_y[0], sorted_by_y[-1]
        remaining = sorted(sorted_by_y[1:3], key=lambda c: c['center'][0])
        left_cal, right_cal = remaining[0], remaining[-1]
        return [top_cal, bottom_cal, left_cal, right_cal]

    def validate_caliper_placement(self, calipers, smooth_contour):
        mismatches = []
        distances = []
        failed_indices = []
        misplaced_caliper_names = []
        names = ["Top", "Bottom", "Left", "Right"]

        for i, cal in enumerate(calipers):
            tolerance = self.CALIPER_TOLERANCES[i]
            dist = abs(cv2.pointPolygonTest(smooth_contour, cal['center'], True))
            distances.append(dist)
            if dist > tolerance:
                mismatches.append(f"{names[i]}({dist:.1f}px)")
                failed_indices.append(i)
                misplaced_caliper_names.append(names[i])

        is_valid = len(mismatches) == 0
        reason = "Correct" if is_valid else f"Misplaced: {', '.join(mismatches)}"

        return is_valid, reason, distances, failed_indices, misplaced_caliper_names

    def check_line_intersection(self, pt1, pt2, mask, steps=200):
        x_vals = np.linspace(pt1[0], pt2[0], steps).astype(np.int32)
        y_vals = np.linspace(pt1[1], pt2[1], steps).astype(np.int32)
        h, w = mask.shape[:2]
        x_vals = np.clip(x_vals, 0, w - 1)
        y_vals = np.clip(y_vals, 0, h - 1)
        return np.any(mask[y_vals, x_vals] == 1)

    def find_segments_intersection(self, p1, p2, p3, p4):
        x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:
            return int((x1 + x2 + x3 + x4) / 4), int((y1 + y2 + y3 + y4) / 4)
        num_x = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        num_y = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        return int(num_x / denominator), int(num_y / denominator)

    def calculate_intersection_angle(self, p1_v1, p2_v1, p1_v2, p2_v2):
        vec1 = np.array([p2_v1[0] - p1_v1[0], p2_v1[1] - p1_v1[1]])
        vec2 = np.array([p2_v2[0] - p1_v2[0], p2_v2[1] - p1_v2[1]])
        dot_product = np.dot(vec1, vec2)
        norm_v1 = np.linalg.norm(vec1)
        norm_v2 = np.linalg.norm(vec2)
        if norm_v1 == 0.0 or norm_v2 == 0.0:
            return 0.0
        cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    def calculate_horizontal_deviation(self, left_pt, right_pt):
        vec = np.array([right_pt[0] - left_pt[0], right_pt[1] - left_pt[1]])
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            return 0.0
        cos_theta = np.clip(abs(vec[0]) / norm, -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    def process_and_connect_spine(self, spine_mask, distance_threshold=70):
        output_mask = spine_mask.copy()
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(spine_mask)
        if num_labels <= 2:
            return output_mask, []
        connected_pairs = []
        for i in range(1, num_labels):
            x1, y1 = centroids[i]
            min_dist = float('inf')
            closest_idx = -1
            for j in range(1, num_labels):
                if i == j: continue
                x2, y2 = centroids[j]
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = j
            if min_dist <= distance_threshold and closest_idx != -1:
                pt1 = (int(centroids[i][0]), int(centroids[i][1]))
                pt2 = (int(centroids[closest_idx][0]), int(centroids[closest_idx][1]))
                cv2.line(output_mask, pt1, pt2, 1, thickness=12)
                connected_pairs.append((pt1, pt2))
        return output_mask, connected_pairs

    def validate(self, model_input):
        img_path = model_input["image_path"]
        json_path = model_input["json_path"]
        panel = model_input["panel"]
        plane = model_input["plane"]
        img_name = os.path.basename(img_path)
        img = load_image_bgr(img_path)
        # overlay_base = create_visualization_overlay(img_path, VIS_DIR, None)
        # if overlay_base is None:
        #     print("NO OVERLAY")
        #     overlay_base = img.copy()
        if img is None:
            return (
                {
                    "name": img_name,
                    "type": "AC",
                    "yolo detection": "Read Error"
                },
                [],
                np.zeros((512, 512, 3), dtype=np.uint8)
            )


        h, w = img.shape[:2]
        reject_reasons_list = []
        placement_ok = False
        cal_distances = []
        failed_caliper_indices = []
        misplaced_caliper_names = []

        log_entry = {
            'name': img_name,
            'no of points': 0,
            'points': "N/A",
            'type': "AC",
            'yolo detection': "wrong",
            'point misplaced': "1",
            'intersection mismatch': "1",
            'APAD range': "1",
            'TAD length': "1",
            'misplaced caliper': "",
            'mismatched APAD angle': "",
            'mismatched intersection angle': "",
            'TAD difference': "",
            'RESULTS': ""
        }

        calipers = self.detect_calipers(img, self.conf_thresh, self.iou_thresh)
        log_entry['no of points'] = len(calipers)

        # ============================================================
        # CALIPER PLACEMENT AUDIT
        # ============================================================

        # 1. No calipers detected
        if len(calipers) == 0:

            log_entry['RESULTS'] = "No calipers placed"
            log_entry['yolo detection'] = "wrong"

        else:

            # --------------------------------------------------------
            # Read JSON points
            # --------------------------------------------------------
            json_points = get_biometry_points(
                json_path=json_path,
                panel=panel,
                plane=plane
            )

            if json_points is None:
                log_entry['RESULTS'] = "BAD AUDIT - JSON not found"

            else:
                
                if json_points is None or len(json_points) == 0:
                    log_entry['RESULTS'] = "BAD AUDIT - JSON points missing"

                else:
                    # Convert JSON points to Nx2 array
                    json_points = np.asarray(
                        json_points,
                        dtype=np.float32
                    ).reshape(-1, 2)

                    # ------------------------------------------------
                    # Check every detected caliper
                    # ------------------------------------------------
                    caliper_names = [
                        "Top",
                        "Bottom",
                        "Left",
                        "Right"
                    ]

                    failed_calipers = []

                    for i, caliper in enumerate(calipers):

                        box = caliper["box"]

                        passed = any_contour_point_inside_box(
                            json_points,
                            box
                        )

                        if not passed:

                            if i < len(caliper_names):
                                failed_calipers.append(
                                    get_audit_term(
                                        "AC",
                                        caliper_names[i]
                                    )
                                )
                            else:
                                failed_calipers.append(
                                    f"Caliper {i + 1}"
                                )

                    # ------------------------------------------------
                    # Final result
                    # ------------------------------------------------
                    if failed_calipers:

                        log_entry['RESULTS'] = (
                            "BAD AUDIT - " +
                            ", ".join(failed_calipers)
                        )

                    else:

                        log_entry['RESULTS'] = "GOOD"
            

        gray_input = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray_input, (224, 224)) / 255.0
        tensor = torch.from_numpy(gray_resized).float().unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, preds = self.ynet_model(tensor)
        masks = (preds.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

        skin_mask_raw = masks[0]
        abdomen_bubble = cv2.resize(masks[1], (w, h), interpolation=cv2.INTER_NEAREST)
        portal_vein = cv2.resize(masks[2], (w, h), interpolation=cv2.INTER_NEAREST)
        spine = cv2.resize(masks[3], (w, h), interpolation=cv2.INTER_NEAREST)

        perfected_data = self.make_perfected_ac_with_dims_call(skin_mask_raw, img.shape)

        if perfected_data is None:
            reject_reasons_list.append("Morphology Processing Error")
            return log_entry, img.copy()

        smooth_contour = perfected_data["contour"]
        
        spine_connected, spine_bridges = self.process_and_connect_spine(spine, distance_threshold=70)
        spine_final_mask = cv2.dilate(spine_connected, self.dilation_kernel, iterations=1)
        portal_vein_dilated = cv2.dilate(portal_vein, self.dilation_kernel, iterations=1)

        if len(calipers) != 4:
            reject_reasons_list.append("Count Error")
            failed_caliper_indices = list(range(len(calipers)))
            log_entry['point misplaced'] = "1"
            log_entry['misplaced caliper'] = "All (Count Error)"
        else:
            log_entry['yolo detection'] = "correct"
            log_entry['points'] = f"{calipers[0]['center']} {calipers[1]['center']} {calipers[2]['center']} {calipers[3]['center']}"

            placement_ok, placement_reason, cal_distances, failed_caliper_indices, misplaced_caliper_names = \
                self.validate_caliper_placement(calipers, smooth_contour)
            
            if placement_ok:
                log_entry['point misplaced'] = "0"
            else:
                reject_reasons_list.append(placement_reason)
                log_entry['point misplaced'] = "1"
                log_entry['misplaced caliper'] = ", ".join(misplaced_caliper_names)

        # overlay = overlay_base.copy()
        # overlay[abdomen_bubble == 1] = [255, 0, 0]  # Blue
        # overlay[portal_vein == 1] = [0, 255, 0]     # Green
        # overlay[spine == 1] = [0, 0, 255]           # Red
        # vis_img = cv2.addWeighted(overlay_base, 0.7, overlay, 0.3, 0)

        # cv2.drawContours(vis_img, [smooth_contour], -1, (255, 0, 255), 1, cv2.LINE_AA)

        intersection_msg = "Anatomical Axis Check: Failed"
        intersection_color = (0, 0, 255)
        perp_msg = "Perpendicular Check: N/A"; perp_color = (255, 255, 255)
        horiz_range_msg = "APAD range Check: N/A"; horiz_range_color = (255, 255, 255)
        vert_longest_msg = "Vertical Longest Check: N/A"; vert_longest_color = (255, 255, 255)

        if len(calipers) == 4 and placement_ok:
            top_pt = calipers[0]['center']; bottom_pt = calipers[1]['center']
            left_pt = calipers[2]['center']; right_pt = calipers[3]['center']

            inter_x, inter_y = self.find_segments_intersection(top_pt, bottom_pt, left_pt, right_pt)
            # cv2.line(vis_img, (0, inter_y), (w, inter_y), (128, 128, 128), 1, cv2.LINE_AA)

            horiz_deviation = self.calculate_horizontal_deviation(left_pt, right_pt)
            if horiz_deviation <= self.HORIZ_TOLERANCE:
                horiz_range_msg = f"Horiz Range: Pass ({horiz_deviation:.1f} deg)"
                horiz_range_color = (0, 255, 0)
                log_entry['APAD range'] = "0"
            else:
                horiz_range_msg = f"Horiz Range: Fail ({horiz_deviation:.1f} deg)"
                horiz_range_color = (0, 0, 255)
                reject_reasons_list.append("APAD range Error")
                log_entry['APAD range'] = "1"
                log_entry['mismatched APAD angle'] = f"{horiz_deviation:.1f} deg"

            measured_angle = self.calculate_intersection_angle(top_pt, bottom_pt, left_pt, right_pt)
            angle_deviation = abs(measured_angle - 90.0)
            if angle_deviation <= self.ANGLE_TOLERANCE:
                perp_msg = f"Perpendicular: Yes ({measured_angle:.1f} deg)"
                perp_color = (0, 255, 0)
                log_entry['intersection mismatch'] = "0"
            else:
                perp_msg = f"Perpendicular: No ({measured_angle:.1f} deg)"
                perp_color = (0, 0, 255)
                reject_reasons_list.append("Perpendicular Angle Error")
                log_entry['intersection mismatch'] = "1"
                log_entry['mismatched intersection angle'] = f"{measured_angle:.1f} deg"

            horiz_hits_vein = self.check_line_intersection(left_pt, right_pt, portal_vein_dilated)
            horiz_hits_spine = self.check_line_intersection(left_pt, right_pt, spine_final_mask)
            h_spine_txt = "Yes" if horiz_hits_spine else "No"
            h_vein_txt = "Yes" if horiz_hits_vein else "No"

            if horiz_hits_vein and horiz_hits_spine:
                # cv2.line(vis_img, left_pt, right_pt, (255, 128, 0), 1)
                intersection_msg = "Horizontal line passes: Spine [Yes], Vein [Yes]"
                intersection_color = (0, 255, 0)
            else:
                vert_hits_vein = self.check_line_intersection(top_pt, bottom_pt, portal_vein_dilated)
                vert_hits_spine = self.check_line_intersection(top_pt, bottom_pt, spine_final_mask)
                if vert_hits_vein and vert_hits_spine:
                    # cv2.line(vis_img, top_pt, bottom_pt, (255, 128, 0), 1, cv2.LINE_AA)
                    intersection_msg = f"Vertical line passes: Spine [Yes], Vein [Yes] (Horiz: Spine [{h_spine_txt}], Vein [{h_vein_txt}])"
                    intersection_color = (0, 255, 0)
                else:
                    intersection_msg = f"Axis Fail -> Horiz: Spine [{h_spine_txt}], Vein [{h_vein_txt}]"
                    intersection_color = (0, 0, 255)
                    reject_reasons_list.append("Intersection Error")

            ellipse_params = cv2.fitEllipse(smooth_contour)
            center, axes, angle = ellipse_params
            ellipse_w, ellipse_h = axes
            major_axis_len = max(ellipse_w, ellipse_h)
            vert_len = np.linalg.norm(np.array(top_pt) - np.array(bottom_pt))
            
            # cv2.ellipse(vis_img, ellipse_params, (255, 255, 255), 1, cv2.LINE_AA)
            angle_rad = np.radians(angle if ellipse_h > ellipse_w else angle + 90.0)
            dir_x = np.sin(angle_rad); dir_y = -np.cos(angle_rad)
            pt_major1 = (int(center[0] + dir_x * (major_axis_len / 2)), int(center[1] + dir_y * (major_axis_len / 2)))
            pt_major2 = (int(center[0] - dir_x * (major_axis_len / 2)), int(center[1] - dir_y * (major_axis_len / 2)))
            # cv2.line(vis_img, pt_major1, pt_major2, (200, 200, 200), 1, cv2.LINE_AA)

            if abs(vert_len - major_axis_len) <= self.ELLIPSE_SIZE_TOLERANCE:
                vert_longest_msg = f"Vert Longest: Pass (Vert: {vert_len:.1f}px, Oval Max: {major_axis_len:.1f}px)"
                vert_longest_color = (0, 255, 0)
                log_entry['TAD length'] = "0"
            else:
                vert_longest_msg = f"Vert Longest: Fail (Vert: {vert_len:.1f}px, Oval Max: {major_axis_len:.1f}px)"
                vert_longest_color = (0, 0, 255)
                reject_reasons_list.append("Vertical Line Not Longest Axis")
                log_entry['TAD length'] = "1"
                log_entry['TAD difference'] = f"{abs(vert_len - major_axis_len):.1f}px"

        is_good = (len(calipers) == 4) and (len(reject_reasons_list) == 0)

        caliper_names = ["Top", "Bottom", "Left", "Right"]
        for i, det in enumerate(calipers):
            x1, y1, x2, y2 = map(int, det['box'])
            box_color = (0, 255, 0) if i in failed_caliper_indices else (0, 255, 0)
            # cv2.rectangle(vis_img, (x1, y1), (x2, y2), box_color, 1)  
            # cv2.circle(vis_img, det['center'], 1, box_color, -1)      
            dist_info = f"({cal_distances[i]:.1f}px)" if (cal_distances and i < len(cal_distances)) else ""
            label = f"{caliper_names[i]}: {det['conf']:.2f} {dist_info}" if len(calipers) == 4 else f"Caliper: {det['conf']:.2f}"
            # cv2.putText(vis_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1, cv2.LINE_AA)

        status_txt = "VALID" if is_good else f"REJECTED - {', '.join(reject_reasons_list)}"
        # cv2.putText(vis_img, status_txt, (10s, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0) if is_good else (0, 0, 255), 1, cv2.LINE_AA)
        
        # if len(calipers) == 4 and (len(failed_caliper_indices) == 0 or any(r in log_entry['TAD difference'] or "Error" in status_txt or "REJECTED" in status_txt for r in ["px"])):
        #     cv2.putText(vis_img, intersection_msg, (10, h - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.42, intersection_color, 1, cv2.LINE_AA)
        #     cv2.putText(vis_img, perp_msg, (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.42, perp_color, 1, cv2.LINE_AA)
        #     cv2.putText(vis_img, horiz_range_msg, (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.42, horiz_range_color, 1, cv2.LINE_AA)
        #     cv2.putText(vis_img, vert_longest_msg, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, vert_longest_color, 1, cv2.LINE_AA)

        # yolo_boxes = serialize_yolo_boxes(calipers)

        return log_entry

    def make_perfected_ac_with_dims_call(self, mask, original_shape):
        """Helper call method forwarding parameters to main morphological implementation engine."""
        return self.calculate_perfected_ac_with_dims(mask, original_shape)


# =================================================================
# CLASS 2: BIPARIETAL DIAMETER (BPD)
# =================================================================
class BPD:
    def __init__(self, yolo_path, unet_path, device):
        self.device = device
        self.yolo_model = YOLO(yolo_path)
        self.unet_model = UNetBPD().to(device)
        checkpoint = torch.load(unet_path, map_location=device, weights_only=False)
        self.unet_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        self.unet_model.eval()
        
        self.POINT_THRESHOLDS_PX = {'top': 10, 'right': 10, 'left': 10, 'bottom': 10}
        self.MIN_FALX_PERCENT = 0.01 
        self.ANGLE_TARGET = 90.0 
        self.ANGLE_TOLERANCE = 5.0 
        self.MIDPOINT_TOLERANCE_PX = 7.5

    def apply_custom_nms(self, detections, iou_threshold=0.45):
        detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            detections = [d for d in detections if compute_iou(best['coords'], d['coords']) < iou_threshold]
        return keep

    def validate_anatomy_by_distance(self, detections, inner_mask, outer_mask):
        sorted_x = sorted(detections, key=lambda d: d['center'][0])
        l_pt, r_pt = sorted_x[0], sorted_x[-1]
        rem = sorted(sorted_x[1:3], key=lambda d: d['center'][1])
        t_pt, b_pt = rem[0], rem[-1]

        mismatches = []
        def get_min_dist(mask, pt):
            mask_pts = np.argwhere(mask == 1)
            if mask_pts.size == 0: return 999.0
            target = np.array([pt['center'][1], pt['center'][0]]) 
            return np.min(np.linalg.norm(mask_pts - target, axis=1))

        dists = {
            'Top': get_min_dist(outer_mask, t_pt), 'Right': get_min_dist(outer_mask, r_pt),
            'Left': get_min_dist(outer_mask, l_pt), 'Bottom': get_min_dist(inner_mask, b_pt)
        }

        point_status = {
            'Top': dists['Top'] <= self.POINT_THRESHOLDS_PX['top'],
            'Right': dists['Right'] <= self.POINT_THRESHOLDS_PX['right'],
            'Left': dists['Left'] <= self.POINT_THRESHOLDS_PX['left'],
            'Bottom': dists['Bottom'] <= self.POINT_THRESHOLDS_PX['bottom']
        }

        if not point_status['Top']: mismatches.append(f"Top({dists['Top']:.1f}px)")
        if not point_status['Right']: mismatches.append(f"Right({dists['Right']:.1f}px)")
        if not point_status['Left']: mismatches.append(f"Left({dists['Left']:.1f}px)")
        if not point_status['Bottom']: mismatches.append(f"Bottom({dists['Bottom']:.1f}px)")

        return mismatches, (t_pt, b_pt, l_pt, r_pt), point_status

    def validate(self, model_input):
        img_path = model_input["image_path"]
        json_path = model_input["json_path"]
        panel = model_input["panel"]
        plane = model_input["plane"]

        filename = os.path.basename(img_path)
        filename = os.path.basename(img_path)
        reject_reason = ""
        is_good = False
        point_validation_map = {}
        
        log_entry = {
            'name': filename, 'no of points': 0, 'points': "", 'type': "BPD", 'yolo detection': "wrong",
            'point misplaced': "0", 'angle mismatch': "0", 'midpoint mismatch': "0", 'falx mismatch': "0",
            'misplaced caliper': "", 'mismatched angle': ""
        }
        json_boxes = {}

        try:
            img_bgr = load_image_bgr(img_path)
            yolo_res = self.yolo_model.predict(source=img_bgr, conf=0.4, verbose=False)
            raw_dets = []
            for box in yolo_res[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                raw_dets.append({'coords': (x1, y1, x2, y2), 'center': (int((x1+x2)/2), int((y1+y2)/2)), 'conf': float(box.conf[0])})
            
            detections = self.apply_custom_nms(raw_dets)
            if len(detections) > 4:
                detections = sorted(detections, key=lambda x: x['conf'], reverse=True)[:4]
                
            log_entry['no of points'] = len(detections)

            # ============================================================
            # BPD CALIPER PLACEMENT AUDIT
            # ============================================================

            # 1. No calipers detected
            if len(detections) == 0:

                log_entry["RESULTS"] = "No calipers placed"

            else:

                # --------------------------------------------------------
                # Read JSON
                # --------------------------------------------------------
                json_points = get_biometry_points(
                    json_path=json_path,
                    panel=panel,
                    plane=plane
                )

                if json_points is None:

                    log_entry["RESULTS"] = "BAD AUDIT - JSON not found"

                else:

                    

                    if not isinstance(json_points, dict):

                        log_entry["RESULTS"] = (
                            "BAD AUDIT - JSON points missing"
                        )

                    else:

                        failed_points = []

                        point_names = [
                            "ofd_start",
                            "ofd_end",
                            "bpd_top",
                            "bpd_bottom"
                        ]

                        # ------------------------------------------------
                        # Sort detections anatomically (same logic as
                        # validate_anatomy_by_distance)
                        # ------------------------------------------------
                        if len(detections) == 4:
                            sorted_x_audit = sorted(detections, key=lambda d: d['center'][0])
                            l_det, r_det = sorted_x_audit[0], sorted_x_audit[-1]
                            rem_audit = sorted(sorted_x_audit[1:3], key=lambda d: d['center'][1])
                            t_det, b_det = rem_audit[0], rem_audit[-1]
                        else:
                            l_det = r_det = t_det = b_det = None

                        # ------------------------------------------------
                        # Explicit anatomical mapping:
                        # ofd_start/ofd_end -> Left/Right detections
                        # bpd_top/bpd_bottom -> Top/Bottom detections
                        # ------------------------------------------------
                        name_to_det = {
                            "ofd_start": l_det,
                            "ofd_end": r_det,
                            "bpd_top": t_det,
                            "bpd_bottom": b_det,
                        }

                        # ------------------------------------------------
                        # Check every JSON point against its box
                        # ------------------------------------------------
                        for point_name in point_names:

                            point = json_points.get(point_name) # biom
                            det = name_to_det.get(point_name) # audit

                            if point is not None:
                                json_boxes[point_name] = make_box_from_point(point, half_size=13)

                            # if point is None or det is None:
                            #     failed_points.append(point_name)
                            #     continue
                            if point is None or det is None:
                                failed_points.append(
                                    get_audit_term("BPD", point_name)
                                )
                                continue

                            box = det.get("box", det.get("coords")) # audit 

                            passed = boxes_overlap(json_boxes[point_name], box) 

                            # if not passed:
                            #     failed_points.append(point_name)
                            if not passed:
                                failed_points.append(
                                    get_audit_term("BPD", point_name)
                                )


                        # ------------------------------------------------
                        # Wrong number of detections
                        # ------------------------------------------------
                        if len(detections) != 4:

                            log_entry["RESULTS"] = (
                                "BAD AUDIT - Expected 4 calipers, "
                                f"found {len(detections)}"
                            )

                        # ------------------------------------------------
                        # Some points outside boxes
                        # ------------------------------------------------
                        elif failed_points:

                            log_entry["RESULTS"] = (
                                "BAD AUDIT - " +
                                ", ".join(failed_points)
                            )

                        # ------------------------------------------------
                        # Everything correct
                        # ------------------------------------------------
                        else:

                            log_entry["RESULTS"] = "GOOD"

            if len(detections) != 4:
                reject_reason = f"Detection Error: Found {len(detections)} points"
                log_entry['yolo detection'] = "wrong"
            else:
                log_entry['yolo detection'] = "correct"
                
                if img_path.lower().endswith('.dcm'):
                    orig_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                else:
                    orig_pil = Image.open(img_path)
                    
                orig_size = orig_pil.size
                img_tensor = TF.to_tensor(orig_pil.convert("L").resize((512, 512))).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.unet_model(img_tensor)
                    inner_pred = torch.argmax(torch.softmax(outputs["inner"], dim=1), dim=1).squeeze().cpu().numpy()
                    calv_probs = torch.sigmoid(outputs["calv"]).squeeze().cpu().numpy()
                
                mid_m = cv2.resize((inner_pred == 3).astype(np.uint8), orig_size, interpolation=cv2.INTER_NEAREST)
                inn_m = cv2.resize((calv_probs[0] > 0.5).astype(np.uint8), orig_size, interpolation=cv2.INTER_NEAREST)
                out_m = cv2.resize((calv_probs[1] > 0.5).astype(np.uint8), orig_size, interpolation=cv2.INTER_NEAREST)

                mismatches, sorted_pts, point_status = self.validate_anatomy_by_distance(detections, inn_m, out_m)
                
                labels = ['Top', 'Bottom', 'Left', 'Right']
                for label, pt in zip(labels, sorted_pts):
                    point_validation_map[pt['center']] = point_status[label]

                pts_centers = [p['center'] for p in sorted_pts]
                log_entry['points'] = str(pts_centers)

                if mismatches:
                    reject_reason = f"Misplaced Points: {', '.join(mismatches)}"
                    log_entry['point misplaced'] = "1"
                    log_entry['misplaced caliper'] = ", ".join(mismatches)
                else:
                    log_entry['point misplaced'] = "0"
                    log_entry['misplaced caliper'] = ""
                    
                    t_pt, b_pt, l_pt, r_pt = pts_centers
                    ofd_len = np.linalg.norm(np.array(t_pt) - np.array(b_pt))
                    bpd_len = np.linalg.norm(np.array(l_pt) - np.array(r_pt))
                    mid_ofd = (np.array(t_pt) + np.array(b_pt)) / 2
                    p1, p2 = np.array(l_pt), np.array(r_pt)
                    
                    numerator = np.abs((p2[0] - p1[0]) * (p1[1] - mid_ofd[1]) - (p1[0] - mid_ofd[0]) * (p2[1] - p1[1]))
                    denominator = np.linalg.norm(p2 - p1)
                    midpoint_diff = numerator / denominator if denominator != 0 else 0.0
                    
                    v_ofd = np.array(b_pt) - np.array(t_pt)
                    v_bpd = np.array(r_pt) - np.array(l_pt)
                    cos_theta = np.dot(v_ofd, v_bpd) / (np.linalg.norm(v_ofd) * np.linalg.norm(v_bpd))
                    angle = math.degrees(math.acos(np.clip(cos_theta, -1.0, 1.0)))
                    if angle > 90: angle = 180 - angle
                    
                    h_only_mask = np.zeros_like(mid_m)
                    cv2.line(h_only_mask, l_pt, r_pt, 1, thickness=2)
                    pass_perc = (np.sum(cv2.bitwise_and(h_only_mask, mid_m)) / np.sum(h_only_mask)) * 100

                    is_midpoint_invalid = midpoint_diff > self.MIDPOINT_TOLERANCE_PX
                    is_angle_invalid = not (self.ANGLE_TARGET - self.ANGLE_TOLERANCE <= angle <= self.ANGLE_TARGET + self.ANGLE_TOLERANCE)
                    is_falx_invalid = pass_perc < self.MIN_FALX_PERCENT

                    log_entry['midpoint mismatch'] = "1" if is_midpoint_invalid else "0"
                    log_entry['angle mismatch'] = "1" if is_angle_invalid else "0"
                    log_entry['falx mismatch'] = "1" if is_falx_invalid else "0"
                    
                    log_entry['mismatched angle'] = f"{angle:.2f} deg" if is_angle_invalid else ""

                    if is_midpoint_invalid: reject_reason = f"Symmetry Error: BPD offset {midpoint_diff:.1f}px from OFD mid"
                    elif is_angle_invalid: reject_reason = f"Geometry Error: Angle is {angle:.1f}deg"
                    elif is_falx_invalid: reject_reason = f"Falx Error: Only {pass_perc:.1f}% overlap"
                    else: is_good = True

            # final_img = cv2.cvtColor(np.array(orig_pil.convert("L")), cv2.COLOR_GRAY2BGR) if 'orig_pil' in locals() else img_bgr.copy()
            # overlay_base = create_visualization_overlay(img_path, VIS_DIR, None)
            # final_img = overlay_base if overlay_base is not None else (cv2.cvtColor(np.array(orig_pil.convert("L")), cv2.COLOR_GRAY2BGR) if 'orig_pil' in locals() else img_bgr.copy())

            # if 'mid_m' in locals():
            #     contours_m, _ = cv2.findContours(mid_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #     cv2.drawContours(final_img, contours_m, -1, (128, 0, 128), 2) 
            #     contours_i, _ = cv2.findContours(inn_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #     cv2.drawContours(final_img, contours_i, -1, (255, 0, 0), 2)   
            #     contours_o, _ = cv2.findContours(out_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #     cv2.drawContours(final_img, contours_o, -1, (0, 255, 0), 2)  

        #     if (is_good or "Symmetry Error" in reject_reason) and len(detections) == 4:
        #         hud_metrics = [
        #             f"OFD Len: {ofd_len:.1f}px", f"BPD Len: {bpd_len:.1f}px", 
        #             f"Midpoint Diff: {midpoint_diff:.1f}px", f"Intersect Angle: {angle:.1f}deg", 
        #             f"Falx Overlap: {pass_perc:.1f}%"
        #         ]

        #     for d in detections:
        #         x1, y1, x2, y2 = map(int, d['coords'])
        #         center_pt = d['center']
        #         box_color = (0, 255, 255) if point_validation_map.get(center_pt, False) else (0, 255, 255)
        #         cv2.rectangle(final_img, (x1, y1), (x2, y2), box_color, 1)
        #         # cv2.circle(final_img, center_pt, 4, (26, 26, 255), -1)

        #     for pname, jbox in json_boxes.items():
        #         jx1, jy1, jx2, jy2 = map(int, jbox)
        #         cv2.rectangle(final_img, (jx1, jy1), (jx2, jy2), (0, 255, 0), 1)
            
        #     status_color = (0, 255, 0) if is_good else (0, 0, 255)
        #     # cv2.putText(final_img, "VALID" if is_good else f"REJECTED: {str(reject_reason)}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        # except Exception as e:
        #     print(f"Error processing {filename}: {e}")
        #     final_img = img_bgr.copy() if 'img_bgr' in locals() else np.zeros((512, 512, 3), dtype=np.uint8)

            # yolo_boxes = serialize_yolo_boxes(detections)
            return log_entry

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            # yolo_boxes = serialize_yolo_boxes(
            #     detections if 'detections' in locals() else []
            # )
            return log_entry


# =================================================================
# CLASS 3: FEMUR LENGTH (FL)
# =================================================================
class FL:
    def __init__(self, yolo_path, ynet_path, device, yolo_thresh=0.4, seg_thresh=0.98, distance_tol=15):
        self.device = device
        self.yolo = YOLO(yolo_path)
        self.ynet = YNetFL(input_channels=1, output_channels=64, n_class=1).to(device)
        state_dict = torch.load(ynet_path, map_location=device, weights_only=False)
        self.ynet.load_state_dict(state_dict, strict=False)
        self.ynet.eval()
        
        self.distance_tol = distance_tol
        self.yolo_thresh = yolo_thresh
        self.seg_thresh = seg_thresh
        self.search_width = 40  

    def find_informational_gradient_shift(self, gray_img, pt, side="Left"):
        x, y = int(pt[0]), int(pt[1])
        h, w = gray_img.shape
        x_start = max(0, x - self.search_width // 2)
        x_end = min(w - 1, x + self.search_width // 2)
        y_start, y_end = max(0, y - 2), min(h, y + 3)
        roi_rows = gray_img[y_start:y_end, x_start:x_end].copy()
        
        if roi_rows.size == 0: return x, 0.0, 999.0
        local_max_val = np.max(roi_rows)
        if local_max_val > 150: roi_rows[roi_rows >= int(local_max_val * 0.90)] = 0

        line_profile = roi_rows.mean(axis=0).astype(np.float32)
        line_profile = cv2.GaussianBlur(line_profile.reshape(1, -1), (3, 1), 0).flatten()
        gradient_shift = np.diff(line_profile)
        
        local_caliper_x_idx = x - x_start
        expanded_radius = 6  
        excl_start = max(0, local_caliper_x_idx - expanded_radius)
        excl_end = min(len(gradient_shift), local_caliper_x_idx + expanded_radius)
        gradient_shift[excl_start:excl_end] = 0.0

        if side == "Left":
            peak_idx = np.argmax(gradient_shift)
            peak_grad_val = gradient_shift[peak_idx]
        else:
            peak_idx = np.argmin(gradient_shift)
            peak_grad_val = abs(gradient_shift[peak_idx])

        change_point_x = x if peak_grad_val == 0 else x_start + peak_idx
        return change_point_x, peak_grad_val, abs(change_point_x - x)

    def process_calipers(self, img_bgr):
        res = self.yolo.predict(source=img_bgr, conf=0.1, verbose=False)
        raw_dets = []
        if res and res[0].boxes:
            for box in res[0].boxes:
                conf = float(box.conf[0])
                if conf < self.yolo_thresh: continue
                coords = box.xyxy[0].cpu().numpy()
                center_x = (coords[0] + coords[2]) / 2
                center_y = (coords[1] + coords[3]) / 2
                raw_dets.append({'box': list(map(int, coords)), 'conf': conf, 'center': (float(center_x), float(center_y))})

        raw_dets = sorted(raw_dets, key=lambda x: x['conf'], reverse=True)
        keep = []
        while raw_dets:
            best = raw_dets.pop(0)
            keep.append(best)
            raw_dets = [d for d in raw_dets if compute_iou(best['box'], d['box']) < 0.5]
        return sorted(keep[:2], key=lambda x: x['center'][0])

    def validate(self, model_input):
        img_path = model_input["image_path"]
        json_path = model_input["json_path"]
        panel = model_input["panel"]
        plane = model_input["plane"]
        fname = os.path.basename(img_path)
        img_bgr = load_image_bgr(img_path)
        if img_bgr is None:
            return (
                {
                    "name": fname,
                    "type": "FL",
                    "yolo detection": "Read Error"
                },
                [],
                np.zeros((512, 512, 3), dtype=np.uint8)
            )

        vis_img = img_bgr.copy()
        # overlay_base = create_visualization_overlay(img_path, VIS_DIR, None)
        # vis_img = overlay_base if overlay_base is not None else img_bgr.copy()
        h_orig, w_orig = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        audit = {"name": fname, "no of points": 0, "points": [], "type": "FL", "yolo detection": "correct", "point misplaced": 0, "misplaced caliper": ""}

        calipers = self.process_calipers(img_bgr)
        audit["no of points"] = len(calipers)
        audit["points"] = [(int(c['center'][0]), int(c['center'][1])) for c in calipers]

        # ============================================================
        # FL CALIPER PLACEMENT AUDIT
        # ============================================================

        # 1. No calipers detected
        if len(calipers) == 0:

            audit["RESULTS"] = "No calipers placed"

        else:

            # --------------------------------------------------------
            # Read JSON
            # --------------------------------------------------------
            json_points = get_biometry_points(
                json_path=json_path,
                panel=panel,
                plane=plane
            )

            if json_points is None:

                audit["RESULTS"] = "BAD AUDIT - JSON not found"

            else:

                
                if not isinstance(json_points, dict):
                    audit["RESULTS"] = "BAD AUDIT - JSON points missing"

                else:

                    failed_points = []

                    # ------------------------------------------------
                    # Check START point
                    # ------------------------------------------------
                    if len(calipers) >= 1:

                        start_point = json_points.get("start")

                        if start_point is None:
                            failed_points.append(
                                get_audit_term("FL", "Start")
                            )

                        else:

                            start_box = make_box_from_point(start_point, half_size=13)
                            # cv2.rectangle(
                            #     vis_img,
                            #     (int(start_box[0]), int(start_box[1])),
                            #     (int(start_box[2]), int(start_box[3])),
                            #     (0, 255, 0), 1
                            # )
                            start_pass = boxes_overlap(start_box, calipers[0]["box"])

                            if not start_pass:
                                failed_points.append(
                                    get_audit_term("FL", "Start")
                                )


                    # ------------------------------------------------
                    # Check END point
                    # ------------------------------------------------
                    if len(calipers) >= 2:

                        end_point = json_points.get("end")

                        if end_point is None:
                            failed_points.append(
                                get_audit_term("FL", "End")
                            )

                        else:

                            end_box = make_box_from_point(end_point, half_size=13)
                            # cv2.rectangle(
                            #     vis_img,
                            #     (int(end_box[0]), int(end_box[1])),
                            #     (int(end_box[2]), int(end_box[3])),
                            #     (0, 255, 0), 1
                            # )
                            end_pass = boxes_overlap(end_box, calipers[1]["box"])

                            if not end_pass:
                                failed_points.append(
                                    get_audit_term("FL", "End")
                                )


                    # ------------------------------------------------
                    # Wrong number of detections
                    # ------------------------------------------------
                    if len(calipers) != 2:

                        audit["RESULTS"] = (
                            "BAD AUDIT - Expected 2 calipers, "
                            f"found {len(calipers)}"
                        )

                    # ------------------------------------------------
                    # Final result
                    # ------------------------------------------------
                    elif failed_points:

                        audit["RESULTS"] = (
                            "BAD AUDIT - " +
                            ", ".join(failed_points)
                        )

                    else:

                        audit["RESULTS"] = "GOOD"

        if len(calipers) != 2:
            audit["yolo detection"] = "wrong"
            # cv2.putText(vis_img, f"BAD: Found {len(calipers)} calipers", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return (audit, None)

        x1, y1 = calipers[0]['center']
        x2, y2 = calipers[1]['center']
        left_x, right_x = int(min(x1, x2)), int(max(x1, x2))
        top_y, bottom_y = max(0, int(min(y1, y2)) - 15), min(h_orig, int(max(y1, y2)) + 15) 

        img_gray = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (224, 224))
        tensor_input = torch.from_numpy(img_gray / 255.0).float().to(self.device).view(1, 1, 224, 224)
        with torch.no_grad():
            _, seg_output = self.ynet(tensor_input)
        
        mask_bin = (seg_output.cpu().numpy().squeeze() > self.seg_thresh).astype(np.uint8) * 255
        mask_res = cv2.resize(mask_bin, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        constrained_mask = np.zeros_like(mask_res)
        constrained_mask[top_y:bottom_y, left_x:right_x] = mask_res[top_y:bottom_y, left_x:right_x]

        kernel = np.ones((5, 5), np.uint8)
        clean_mask = cv2.morphologyEx(constrained_mask, cv2.MORPH_CLOSE, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
        clean_mask = cv2.GaussianBlur(clean_mask, (5, 5), 0)
        raw_contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        smooth_contours = []
        for cnt in raw_contours:
            if cv2.contourArea(cnt) > 20:   
                smooth_contours.append(cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True))

        # if smooth_contours: cv2.drawContours(vis_img, smooth_contours, -1, (255, 0, 255), 1, cv2.LINE_AA)

        misplaced_sides, contour_metrics = [], []
        for i, c in enumerate(calipers):
            pt, side = c['center'], "Left" if i == 0 else "Right"
            min_contour_dist = float('inf')
            for contour in smooth_contours:
                abs_dist = abs(cv2.pointPolygonTest(contour, pt, True))
                if abs_dist < min_contour_dist: min_contour_dist = abs_dist

            min_contour_dist = 999.0 if min_contour_dist == float('inf') else min_contour_dist
            is_contour_valid = min_contour_dist <= self.distance_tol
            contour_metrics.append((min_contour_dist, is_contour_valid))
            if not is_contour_valid: misplaced_sides.append(side)

        if misplaced_sides:
            audit["point misplaced"] = 1
            audit["misplaced caliper"] = ", ".join(misplaced_sides)

        l_shift_x, _, l_grad_dist = self.find_informational_gradient_shift(gray, calipers[0]['center'], "Left")
        r_shift_x, _, r_grad_dist = self.find_informational_gradient_shift(gray, calipers[1]['center'], "Right")
        l_g_v, r_g_v = l_grad_dist <= self.distance_tol, r_grad_dist <= self.distance_tol

        for i, c in enumerate(calipers):
            x, y = int(c['center'][0]), int(c['center'][1])
            box, _, is_contour_valid = c['box'], *contour_metrics[i]
            shift_x = l_shift_x if i == 0 else r_shift_x
            color = (0, 255, 255) if is_contour_valid else (0, 255, 255)
            # cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), color, 1)
            # cv2.circle(vis_img, (x, y), 2, color, -1)

        l_c_dist, l_c_v = contour_metrics[0]
        r_c_dist, r_c_v = contour_metrics[1]
        l_contour_str = "L-Contour: OK" if l_c_v else f"L-Contour: FAIL ({l_c_dist:.1f}px > {self.distance_tol})"
        l_grad_str = "L-GradShift: OK" if l_g_v else f"L-GradShift: FAIL ({l_grad_dist:.1f}px > {self.distance_tol})"
        r_contour_str = "R-Contour: OK" if r_c_v else f"R-Contour: FAIL ({r_c_dist:.1f}px > {self.distance_tol})"
        r_grad_str = "R-GradShift: OK" if r_g_v else f"R-GradShift: FAIL ({r_grad_dist:.1f}px > {self.distance_tol})"
        
        status_all = l_c_v and r_c_v
        # cv2.putText(vis_img, f"FL: {'VALID' if status_all else 'REJECTED'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if status_all else (0, 0, 255), 2)
        del mask_bin, mask_res, constrained_mask, raw_contours, smooth_contours, img_gray, tensor_input, gray
        
        # yolo_boxes = serialize_yolo_boxes(calipers)

        return audit #, yolo_boxes, None


# =================================================================
# MAIN INTEGRATED PIPELINE RUNNER
# =================================================================
def main(dataset: str, output: str, json_path: str, vis_dir: str, yolo_ac_bpd: str, yolo_fl: str, bpd_mask: str, ac_mask: str, fl_mask: str):
    # parser = argparse.ArgumentParser(description="Integrated Fetal Ultrasound Biometry Quality Evaluation Pipeline")
    # parser.add_argument("--dataset", required=True, help="Root folder (Dataset)")
    # parser.add_argument("--output", required=True, help="Root folder (Output)")
    # # Model Weights Matrix References
    # parser.add_argument("--yolo_ac_bpd", required=True, help="YOLO for BPD/AC")
    # parser.add_argument("--yolo_fl", required=True, help="YOLO for FL")
    # parser.add_argument("--bpd_mask", required=True, help="UNet for BPD")
    # parser.add_argument("--ac_mask", required=True, help="YNet for AC")
    # parser.add_argument("--fl_mask", required=True, help="YNet for FL")
    # args = parser.parse_args()

    global JSON_PATH 
    JSON_PATH = json_path
    global VIS_DIR
    vis_dir = Path(vis_dir) / "overlays"
    print(f"Visualization directory set to: {vis_dir}")
    VIS_DIR = vis_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize Modality Validators Map
    validators = {
        "AC": AC(yolo_ac_bpd, ac_mask, device),
        "BPD": BPD(yolo_ac_bpd, bpd_mask, device),
        "FL": FL(yolo_fl, fl_mask, device)
    }

    # Setup Workspace Output Folder Hierarchies
    audit_dir = os.path.join(output, "audit")
    vis_dir = os.path.join(output, "visualization")
    
    if os.path.exists(output):
        shutil.rmtree(output)
    os.makedirs(audit_dir, exist_ok=True)
    
    for category in ["AC", "BPD", "FL"]:
        os.makedirs(os.path.join(vis_dir, category), exist_ok=True)

    # Sequentially Evaluate Sourced Frame Category Directories
    for category in ["AC", "BPD", "FL"]:
        input_dir = os.path.join(dataset, category)
        if not os.path.exists(input_dir):
            continue

        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm'))]
        results = []
        
        print(f"Processing {category} Modality...")
        for fname in tqdm(image_files):
            img_path = os.path.join(input_dir, fname)
            
            # Singular instance sequential pass routine execution
            audit, vis_img = validators[category].validate(img_path)
            results.append(audit)

            # Save annotated output files
            # cv2.imwrite(os.path.join(vis_dir, category, fname), vis_img)
            cv2.imwrite(os.path.join(vis_dir, category, os.path.splitext(fname)[0] + "_vis.png"), vis_img)
            
            # Intensive stacked variable cache flush
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Generate metrics CSV file outputs directly mapping diagnostic keys
        if results:
            csv_file_path = os.path.join(audit_dir, f"{category}.csv")
            with open(csv_file_path, "w", newline="") as f:
                if category == "AC":
                    fieldnames = [
                        'name', 'no of points', 'points', 'type', 'yolo detection', 
                        'point misplaced', 'intersection mismatch', 'APAD range', 
                        'TAD length', 'misplaced caliper', 'mismatched APAD angle', 
                        'mismatched intersection angle', 'TAD difference', 'RESULTS'
                    ]
                elif category == "BPD":
                    fieldnames = [
                        'name', 'no of points', 'points', 'type', 'yolo detection',
                        'point misplaced', 'angle mismatch', 'midpoint mismatch',
                        'falx mismatch', 'misplaced caliper', 'mismatched angle', 'RESULTS'
                    ]
                elif category == "FL":
                    fieldnames = [
                        'name', 'no of points', 'points', 'type', 
                        'yolo detection', 'point misplaced', 'misplaced caliper', 'RESULTS'
                    ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

    return output
    # print(f"\nProcessing Complete. Metrics logged cleanly to directory: {output}")


def setup_biometry_validators(device):

    validators = {
        "AC": AC(
            Config.YOLO_AC_BPD,
            Config.AC_MASK,
            device
        ),

        "BPD": BPD(
            Config.YOLO_AC_BPD,
            Config.BPD_MASK,
            device
        ),

        "FL": FL(
            Config.YOLO_FL,
            Config.FL_MASK,
            device
        )
    }

    return validators

def run_biometry_audit(model_inputs, validators):

    audit_results = {
        "BPD": [],
        "AC": [],
        "FL": []
    }

    model_to_validator = {
        "bpd": ("BPD", validators["BPD"]),
        "abdomen": ("AC", validators["AC"]),
        "limbs": ("FL", validators["FL"])
    }

    for model_name, items in model_inputs.items():

        if model_name not in model_to_validator:
            continue

        audit_name, validator = model_to_validator[model_name]

        for item in items:

            # ----------------------------------------------------
            # Run validator
            # ----------------------------------------------------
            audit_result = validator.validate(item)

            # ----------------------------------------------------
            # Write audit + YOLO boxes into same JSON
            # ----------------------------------------------------
            write_biometry_audit_result(
                item=item,
                audit_result=audit_result
            )

            # ----------------------------------------------------
            # Keep result in memory if you need CSV/summary
            # ----------------------------------------------------
            audit_results[audit_name].append(audit_result)

    return audit_results

if __name__ == "__main__":
    main(
        dataset="/home/htic/MLN/ANU-Audit/OUTPUT/CLASS/ClinicalPatient_1_dcm/Biometry",
        output="/home/htic/MLN/ANU-Audit/OUTPUT/BIOM_AUDIT_TEST",
        vis_dir= "/home/htic/MLN/ANU-Audit/OUTPUT/BIOM_TEST/visualizations/overlays",
        json_path="/home/htic/MLN/ANU-Audit/OUTPUT/BIOM_TEST/combined_measurements.json",
        yolo_ac_bpd= "/home/htic/MLN/weights/AUDIT/yolo_ac_bpd.pt",
        yolo_fl= "/home/htic/MLN/weights/AUDIT/yolo_fl.pt",
        bpd_mask= "/home/htic/MLN/weights/AUDIT/mask_bpd.pth",
        ac_mask= "/home/htic/MLN/weights/AUDIT/mask_ac.pt",
        fl_mask= "/home/htic/MLN/weights/AUDIT/mask_fl.pt",
    )