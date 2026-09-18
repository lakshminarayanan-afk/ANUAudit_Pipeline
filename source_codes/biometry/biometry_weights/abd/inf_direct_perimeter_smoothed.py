"""
Single Image Inference Script
Performs inference using trained YNet model
Computes clinically smoothed AC using:
    Convex Hull + Polygon Approximation
Saves:
    1. Single output visualization image
    2. CSV file with AC measurement
"""

import torch
import cv2
import numpy as np
import os
import csv
from model import YNet 

# ---------------- CONFIGURATION ----------------
WEIGHTS_PATH = "abd/best_ac_skin_line.pt"
INPUT_IMAGE_PATH = "in/abd.jpeg"
OUTPUT_IMAGE_PATH = "out/ac.jpg"
CSV_OUTPUT_PATH = "out/ac.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SIZE = (224, 224)

# ---------------- SMOOTHING & MEASUREMENT ----------------
def calculate_perfected_ac(mask, original_shape):
    h_orig, w_orig = original_shape[:2]

    mask_res = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(mask_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None

    cnt = max(contours, key=cv2.contourArea)

    # Step 1: Convex Hull
    hull = cv2.convexHull(cnt)

    # Step 2: Smooth using approximation
    perimeter_hull = cv2.arcLength(hull, True)
    epsilon = 0.002 * perimeter_hull
    smooth_skin_contour = cv2.approxPolyDP(hull, epsilon, True)

    # Step 3: Final perimeter
    ac_px = cv2.arcLength(smooth_skin_contour, True)

    return ac_px, smooth_skin_contour


# ---------------- INFERENCE ----------------
def run_inference_abd():

    # Ensure output folder exists
    os.makedirs(os.path.dirname(OUTPUT_IMAGE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CSV_OUTPUT_PATH), exist_ok=True)

    # Load Model
    model = YNet(input_channels=1, output_channels=64, n_class=1).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()

    results = [["filename", "AC_pixels"]]

    # Read single image
    img_orig = cv2.imread(INPUT_IMAGE_PATH)
    if img_orig is None:
        print("Error: Could not read input image.")
        return

    fname = os.path.basename(INPUT_IMAGE_PATH)

    # Preprocess
    gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, TARGET_SIZE)
    normalized = (resized / 255.0 - 0.5) / 0.5

    input_tensor = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    # Forward pass
    with torch.no_grad():
        _, pred = model(input_tensor)

    # Binary mask
    mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

    # Smooth perimeter calculation
    ac_px_full, smooth_contour = calculate_perfected_ac(mask, img_orig.shape)
    ac_px = round(ac_px_full, 2)

    if ac_px_full is None:
        print("No valid contour detected.")
        return

    results.append([fname, round(ac_px_full, 2)])

    print("------ Perfected AC Result ------")
    print(f"AC (pixels): {round(ac_px_full, 2)}")

    # Visualization
    vis_img = img_orig.copy()

    cv2.drawContours(vis_img, [smooth_contour], -1, (0, 255, 0), 3)

    cv2.imwrite(OUTPUT_IMAGE_PATH, vis_img)

    # Save CSV
    with open(CSV_OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(results)

    print(f"\nOutput image saved at: {OUTPUT_IMAGE_PATH}")
    print(f"CSV saved at: {CSV_OUTPUT_PATH}")

    return ac_px, vis_img


if __name__ == "__main__":
    ac_px, vis_img = run_inference_abd()
