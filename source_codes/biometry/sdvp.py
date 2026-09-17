'''Full Improvised Code for Fetal Liquor (Amniotic Fluid) Inference.
IMPROVISED: Added Gradient-based Edge Search for Caliper Precision and Mask Dilation.'''
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import json
import numpy as np
import shutil
from tqdm import tqdm
from PIL import Image
import albumentations as A
import open_clip
from pathlib import Path

_LIQUOR_MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "model", "liquor")
)
sys.path.insert(0, _LIQUOR_MODEL_DIR)

from source_codes.biometry.model.liquor.utils_get_embeddings import EncoderWrapper

# --- 1. CONFIGURATION ---
_CONFIG_PATH = os.path.join(_LIQUOR_MODEL_DIR, "config.json")
print(f"_CONFIG_PATH:{_CONFIG_PATH}")
with open(_CONFIG_PATH, 'r') as file:
    config = json.load(file)

# Output directories
OVERLAY_DIR = 'out_liq/overlay'
MASK_DIR = 'out_liq/mask_overlays'
BINARY_DIR = 'out_liq/binary_masks'
LOW_CONF_DIR = 'out_liq/Low_Confidence_Cases'

# Path to your newly trained checkpoint
# CKPT_PATH = "/home/htic/Documents/deploy_biometry_3/automate/weights/liquor/sdvp_liquor-epoch=25-val_loss=0.20.ckpt"
DIR_DATA = "/home/htic/vasanth/fetalnet/liq_in"

SAVE_IMAGE_SIZE = 512 
LIST_CLASSES = ['liquor'] 
NUM_CLASSES  = len(LIST_CLASSES)
INIT_FILTERS = 32
CONFIDENCE_THRESHOLD = 0.95 

PATH_FETALCLIP_WEIGHT = config['paths']['path_fetalclip_weight']
PATH_FETALCLIP_CONFIG = config['paths']['path_fetalclip_config']
ARCH_NAME = 'FetalCLIP'

# Create all folders
for d in [OVERLAY_DIR, MASK_DIR, BINARY_DIR, LOW_CONF_DIR]:
    os.makedirs(d, exist_ok=True)

# for d in [OVERLAY_DIR, MASK_DIR]:
#     os.makedirs(d, exist_ok=True)

preprocessing = A.Compose([
    A.Resize(SAVE_IMAGE_SIZE, SAVE_IMAGE_SIZE, interpolation=cv2.INTER_CUBIC, mask_interpolation=0),
])

# --- 2. MODEL ARCHITECTURE ---

class SingleDeconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1):
        super().__init__()
        self.block = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0, groups=groups)
    def forward(self, x): return self.block(x)

class SingleConv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1):
        super().__init__()
        self.block = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size - 1) // 2), groups=groups)
    def forward(self, x): return self.block(x)

class Conv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(SingleConv2DBlock(in_planes, in_planes, kernel_size, groups=in_planes), nn.BatchNorm2d(in_planes), nn.ReLU(True),
                                   SingleConv2DBlock(in_planes, out_planes, 1), nn.BatchNorm2d(out_planes), nn.ReLU(True))
    def forward(self, x): return self.block(x)

class Deconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(SingleDeconv2DBlock(in_planes, in_planes, groups=in_planes), SingleConv2DBlock(in_planes, in_planes, kernel_size, groups=in_planes),
                                   nn.BatchNorm2d(in_planes), nn.ReLU(True), SingleConv2DBlock(in_planes, out_planes, 1), nn.BatchNorm2d(out_planes), nn.ReLU(True))
    def forward(self, x): return self.block(x)

class SingleDWConv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.Sequential(SingleDeconv2DBlock(in_planes, in_planes, groups=in_planes), SingleConv2DBlock(in_planes, out_planes, 1))
    def forward(self, x): return self.block(x)

class UNETR(nn.Module):
    def __init__(self, transformer_width, output_dim, input_dim, init_filters):
        super().__init__()
        self.decoder0 = nn.Sequential(Conv2DBlock(input_dim, init_filters, 3), Conv2DBlock(init_filters, init_filters, 3))
        self.decoder3 = nn.Sequential(Deconv2DBlock(transformer_width, 8*init_filters), Deconv2DBlock(8*init_filters, 4*init_filters), Deconv2DBlock(4*init_filters, 2*init_filters))
        self.decoder6 = nn.Sequential(Deconv2DBlock(transformer_width, 8*init_filters), Deconv2DBlock(8*init_filters, 4*init_filters))
        self.decoder9 = Deconv2DBlock(transformer_width, 8*init_filters)
        self.decoder12_upsampler = SingleDWConv2DBlock(transformer_width, 8*init_filters)
        self.decoder9_upsampler = nn.Sequential(Conv2DBlock(16*init_filters, 8*init_filters), Conv2DBlock(8*init_filters, 8*init_filters), Conv2DBlock(8*init_filters, 8*init_filters), SingleDWConv2DBlock(8*init_filters, 4*init_filters))
        self.decoder6_upsampler = nn.Sequential(Conv2DBlock(8*init_filters, 4*init_filters), Conv2DBlock(4*init_filters, 4*init_filters), SingleDWConv2DBlock(4*init_filters, 2*init_filters))
        self.decoder3_upsampler = nn.Sequential(Conv2DBlock(4*init_filters, 2*init_filters), Conv2DBlock(2*init_filters, 2*init_filters), SingleDWConv2DBlock(2*init_filters, init_filters))
        self.decoder0_header = nn.Sequential(Conv2DBlock(2*init_filters, init_filters), Conv2DBlock(init_filters, init_filters), SingleConv2DBlock(init_filters, output_dim, 1))
    
    def forward(self, x):
        z0, z3, z6, z9, z12 = x
        z12 = self.decoder12_upsampler(z12); z9 = self.decoder9(z9); z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6); z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3); z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0); z3 = F.interpolate(z3, size=z0.shape[2:], mode='bilinear', align_corners=False)
        return self.decoder0_header(torch.cat([z0, z3], dim=1))

class LitModel(nn.Module):
    def __init__(self, transformer_width, num_classes, input_dim, init_filters):
        super().__init__()
        self.model = UNETR(transformer_width, num_classes, input_dim, init_filters)
    def forward(self, x): return self.model(x)

# --- 3. LOGIC & UTILITIES ---

def find_clinical_deepest_pocket(contour, mask_shape, gray_img):
    if len(contour) == 0: return None
    x_min, y_min, w, h = cv2.boundingRect(contour)
    FLUID_THRESHOLD = 75 # Standard acoustic cutoff
    
    column_data = []
    heights = []
    
    # --- PHASE 1: Internal Search ---
    for x in range(x_min, x_min + w):
        max_h, best_y1, best_y2 = 0, 0, 0
        curr_y1 = None
        for y in range(y_min, y_min + h):
            in_mask = cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0
            is_clear = gray_img[y, x] < FLUID_THRESHOLD if in_mask else False
            if is_clear:
                if curr_y1 is None: curr_y1 = y
            else:
                if curr_y1 is not None:
                    curr_h = y - curr_y1
                    if curr_h > max_h: max_h, best_y1, best_y2 = curr_h, curr_y1, y
                    curr_y1 = None
        if curr_y1 is not None:
            curr_h = (y_min + h) - curr_y1
            if curr_h > max_h: max_h, best_y1, best_y2 = curr_h, curr_y1, (y_min+h)
        heights.append(max_h)
        column_data.append((x, best_y1, best_y2))
        
    if not heights: return None
    smoothed = np.convolve(heights, np.ones(7)/7, mode='same')
    best_idx = np.argmax(smoothed)
    x, y1, y2 = column_data[best_idx]

    # --- PHASE 2: SMART EDGE SEARCH ---
    # Scans outside the mask to snap to tissue boundaries
    SEARCH_MARGIN = 20 
    
    # Snap Upper Caliper (y1) to tissue spike
    for test_y in range(y1, max(0, y1 - SEARCH_MARGIN), -1):
        if gray_img[test_y, x] > FLUID_THRESHOLD + 15:
            y1 = test_y
            break

    # Snap Lower Caliper (y2) to tissue spike
    for test_y in range(y2, min(gray_img.shape[0]-1, y2 + SEARCH_MARGIN)):
        if gray_img[test_y, x] > FLUID_THRESHOLD + 15:
            y2 = test_y
            break

    return x, y1, y2

def make_image_square_with_zero_padding(image):
    w, h = image.size
    side = max(w, h)
    new_img = Image.new(image.mode, (side, side), 0)
    new_img.paste(image, ((side - w) // 2, (side - h) // 2))
    return new_img, image

def extract_original_from_padded(padded_image, orig_size):
    pw, ph = padded_image.size
    oh, ow = orig_size
    l, t = (pw - ow) // 2, (ph - oh) // 2
    return padded_image.crop((l, t, l + ow, t + oh))

def setup_model(ckpt_path):
    with open(PATH_FETALCLIP_CONFIG, "r") as f:
        open_clip.factory._MODEL_CONFIGS[ARCH_NAME] = json.load(f)
    m_clip, _, prep_img = open_clip.create_model_and_transforms(ARCH_NAME, pretrained=PATH_FETALCLIP_WEIGHT)
    enc = EncoderWrapper(m_clip.visual).eval().cuda()
    model = LitModel(enc.transformer.width, NUM_CLASSES, 3, INIT_FILTERS)
    
    print(f"Loading weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cuda')
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    
    new_model_dict = {}
    new_enc_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_model_dict[k.replace("model.", "")] = v
        elif k.startswith("encoder."):
            key = k.replace("encoder.", "")
            if "transformer.visual_" in key and "transformer.visual_encoder" not in key:
                key = key.replace("transformer.visual_", "transformer.visual_encoder.")
            new_enc_dict[key] = v
            
    if new_model_dict:
        model.model.load_state_dict(new_model_dict, strict=True)
    if new_enc_dict:
        enc.load_state_dict(new_enc_dict, strict=True)
        
    print("Model and Encoder weights loaded successfully.")
    return enc.eval(), model.eval().cuda(), prep_img

def run_infer_img(img, encoder, model, preprocess_img):
    img_np = np.array(img)
    img_resized = Image.fromarray(preprocessing(image=img_np)['image'])
    img_t = preprocess_img(img_resized).unsqueeze(0).cuda()
    with torch.no_grad():
        z = encoder(img_t)
        pred = model([img_t, *[z[k] for k in ['z3', 'z6', 'z9', 'z12']]])
        probs = torch.sigmoid(pred)
    mask = (probs > 0.5).float().squeeze().cpu().numpy()
    avg_conf = probs[probs > 0.5].mean().item() if (probs > 0.5).any() else 0
    return mask, img_resized, avg_conf

def save_prediction(img_name, pred, orig_img, pad_size, confidence):
    mask_raw = Image.fromarray((pred * 255).astype(np.uint8)).resize(pad_size, Image.NEAREST)
    mask_cv = np.array(extract_original_from_padded(mask_raw, orig_img.size[::-1])).astype(np.uint8)
    cv2.imwrite(os.path.join(BINARY_DIR, img_name.replace(".jpg", ".png")), mask_cv)

    # LOOSENING: Use Dilation to expand the mask coverage
    kernel_dilate = np.ones((9,9), np.uint8) 
    mask_cv = cv2.dilate(mask_cv, kernel_dilate, iterations=1) 
    mask_cv = cv2.morphologyEx(mask_cv, cv2.MORPH_CLOSE, kernel_dilate)

    img_cv = np.array(orig_img.convert("RGB"))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    
    mask_overlay = img_cv.copy()
    mask_overlay[mask_cv > 0] = [0, 255, 0]
    cv2.addWeighted(mask_overlay, 0.4, img_cv, 0.6, 0, mask_overlay)
    cv2.imwrite(os.path.join(MASK_DIR, img_name), cv2.cvtColor(mask_overlay, cv2.COLOR_RGB2BGR))

    cnts, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        lrg = max(cnts, key=cv2.contourArea)
        smoothed = cv2.approxPolyDP(lrg, 0.002 * cv2.arcLength(lrg, True), True)
        cv2.drawContours(img_cv, [smoothed], -1, (0, 255, 0), 2)
        line = find_clinical_deepest_pocket(smoothed, mask_cv.shape, gray)
        if line:
            x, y1, y2 = line
            cv2.line(img_cv, (x, y1), (x, y2), (255, 0, 0), 1)
            cv2.circle(img_cv, (x, y1), 3, (0, 255, 255), -1)
            cv2.circle(img_cv, (x, y2), 3, (0, 255, 255), -1)

    final_out = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(OVERLAY_DIR, img_name), final_out)
    
    if confidence < CONFIDENCE_THRESHOLD:
        shutil.copy(os.path.join(OVERLAY_DIR, img_name), os.path.join(LOW_CONF_DIR, img_name))

    if line:
        x, y1, y2 = line

        return {
            "SDVP": round(float(y2 - y1), 2),
            "units": "px",
            "confidence": round(float(confidence), 4)
        }

    return {
        "SDVP": None,
        "units": "px",
        "confidence": round(float(confidence), 4)
    }

# def run_infer_folder(path):
#     exts = ('.jpg', '.jpeg', '.png', '.JPG')
#     imgs = [f for f in os.listdir(path) if f.endswith(exts)]
#     enc, model, prep = setup_model(CKPT_PATH)
    
#     for f in tqdm(imgs):
#         orig = Image.open(os.path.join(path, f)).convert("RGB")
#         pad, _ = make_image_square_with_zero_padding(orig)
#         mask, _, conf = run_infer_img(pad, enc, model, prep)
#         save_prediction(f, mask, orig, pad.size, conf)

def process_single_image_sdvp(
    image_path,
    encoder,
    model,
    preprocess_img
):

    orig = Image.open(image_path).convert("RGB")
    pad, _ = make_image_square_with_zero_padding(orig)
    mask, _, confidence = run_infer_img(
        pad,
        encoder,
        model,
        preprocess_img
    )
    result = save_prediction(
        Path(image_path).name,
        mask,
        orig,
        pad.size,
        confidence
    )

    return result

# if __name__ == "__main__":
#     run_infer_folder(DIR_DATA)