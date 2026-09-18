import torch
import cv2
import numpy as np
from model import YNet
import os
from tqdm import tqdm
from skimage.morphology import skeletonize
import math


# ---------------- CONFIG ----------------
input_channels = 1
output_channels = 64
n_class = 1
target_size = (224, 224)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------
model = YNet(input_channels=input_channels, output_channels=output_channels, n_class=n_class)
model.load_state_dict(torch.load("limbs/fuvai_weights.pt", map_location=device))
model = model.to(device)
model.eval()


# ---------------- ENABLE DROPOUT ----------------
def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
            m.train()


# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image_for_model(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, target_size)
    image_norm = image / 255.0
    tensor = torch.from_numpy(image_norm).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)
    return tensor.to(device), image


def preprocess_image_rgb(img_path):
    image_rgb = cv2.imread(img_path)
    image_rgb_resized = cv2.resize(image_rgb, target_size)
    return image_rgb, image_rgb_resized


# ---------------- INFERENCE ----------------
@torch.no_grad()
def run_inference(image_path, monte_carlo=False, n_samples=10):
    input_tensor, resized_gray = preprocess_image_for_model(image_path)

    if monte_carlo:
        model.train()
        enable_dropout(model)

        class_outputs = []
        seg_outputs = []

        for _ in range(n_samples):
            class_output, seg_output = model(input_tensor)
            class_outputs.append(torch.softmax(class_output, dim=1).cpu().numpy())
            seg_outputs.append(seg_output.cpu().numpy())

        class_outputs = np.stack(class_outputs, axis=0)
        seg_outputs = np.stack(seg_outputs, axis=0)

        mean_class_probs = np.mean(class_outputs, axis=0).squeeze()
        predicted_class = int(np.argmax(mean_class_probs))
        mean_seg_map = np.mean(seg_outputs, axis=0).squeeze()

        model.eval()
        return predicted_class, mean_class_probs, mean_seg_map, resized_gray, None, None

    model.eval()
    class_output, seg_output = model(input_tensor)
    class_probs = torch.softmax(class_output, dim=1).cpu().numpy().squeeze()
    predicted_class = int(np.argmax(class_probs))
    seg_map = seg_output.cpu().numpy().squeeze()
    return predicted_class, class_probs, seg_map, resized_gray, None, None


# ---------------- POSTPROCESS ----------------
def scaling_contour(rgb_image, seg_map):
    img_h, img_w = rgb_image.shape[:2]
    mask = (seg_map * 255).astype(np.uint8)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask_h, mask_w = mask.shape[:2]
    scale_x, scale_y = img_w / mask_w, img_h / mask_h

    scaled_contours = []
    for c in contours:
        c = c.astype(np.float32)
        c[:, 0, 0] *= scale_x
        c[:, 0, 1] *= scale_y
        scaled_contours.append(c.astype(np.int32))

    return scaled_contours, rgb_image.copy()


def scaled_overlay(rgb_image, scaled_contours):
    binary_mask = np.zeros(rgb_image.shape[:2], np.uint8)
    cv2.fillPoly(binary_mask, scaled_contours, 255)
    return None, None, binary_mask


# ---------------- EXTREMA + DISTANCE ----------------
def extremafit_2pts(rgb_image, contours, binary_mask):
    drawing = rgb_image.copy()

    if len(contours) == 0:
        return drawing, None

    largest_contour = max(contours, key=cv2.contourArea)
    largest_binary_mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(largest_binary_mask, [largest_contour], 255)

    skeleton = skeletonize(largest_binary_mask)
    skeleton_uint8 = (skeleton.astype(np.uint8) * 255)

    coords = cv2.findNonZero(skeleton_uint8)
    if coords is None:
        return drawing, None

    leftmost = tuple(coords[coords[:,:,0].argmin()][0])
    rightmost = tuple(coords[coords[:,:,0].argmax()][0])

    # Draw markers
    cv2.drawMarker(drawing, leftmost, (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
    cv2.drawMarker(drawing, rightmost, (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
    cv2.line(drawing, leftmost, rightmost, (0, 255, 0), 2)

    # -------- Euclidean Distance --------
    distance = math.sqrt(
        (rightmost[0] - leftmost[0])**2 +
        (rightmost[1] - leftmost[1])**2
    )

    return drawing, distance


# ---------------- PROCESS SINGLE IMAGE ----------------
def process_single_image(image_path, output_path):

    predicted_class, class_probs, seg_map, resized_gray, _, _ = run_inference(
        image_path, monte_carlo=True, n_samples=10
    )

    image_rgb, _ = preprocess_image_rgb(image_path)

    scaled_contours, _ = scaling_contour(image_rgb, seg_map)
    _, _, binary_mask = scaled_overlay(image_rgb, scaled_contours)

    drawing, distance = extremafit_2pts(image_rgb, scaled_contours, binary_mask)

    if distance is not None:
        print(f"Euclidean pixel distance: {distance:.2f} px")
    else:
        print("Could not compute distance.")

    cv2.imwrite(output_path, drawing)
    print(f"Successfully saved output to: {output_path}")

    return distance, drawing


# ---------------- MAIN ----------------
if __name__ == "__main__":

    input_file = "in/femur.jpeg"
    output_file = "out/f.png"

    distance, drawing = process_single_image(input_file, output_file)


        