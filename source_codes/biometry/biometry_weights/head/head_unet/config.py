import torch
import os

class Config:
    # ==============================
    # 1. ESSENTIAL PATHS (Update these for your system)
    # ==============================
    # Path to your fine-tuned UNet model weights (.pth)
    INFER_CHECKPOINT = "/home/intern/segmentation/Model_Improve_bpd/UNET/best_model.pth" 
    
    # Path to the folder containing the images you want to process
    INFER_IMAGE_DIR  = "/home/intern/DATA/Transthalamic_plane/Proper_data"
    
    # Path where you want the output (measured images and CSV) to be saved
    INFER_OUTPUT_DIR = "/home/intern/segmentation/Model_Improve_bpd/UNET/output"
    
    DROPOUT  = 0.2    # Required by model.py line 106
    BILINEAR = True   # Required by model.py line 105

    # ==============================
    # 2. MODEL PARAMETERS (Must match training)
    # ==============================
    IN_CHANNELS      = 1       # Grayscale input
    NUM_CLASSES      = 23      # 22 inner structures + 1 background (Softmax Head)
    NUM_CALV_CLASSES = 2       # Inner/Outer Calvarium (Sigmoid Head)
    IMAGE_SIZE       = (512, 512)
    INFER_DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
    BASE_FEATURES    = 64
    BILINEAR         = True

    # ==============================
    # 3. STRUCTURE INDICES (Crucial for BPD Algorithm)
    # ==============================
    # These indices are based on your provided STRUCTURES list (0-based list)
    # In the argmax output: Index 0 = Background
    # Therefore, Index = ListPosition + 1
    
    # Needed for PCA orientation (Midline Falx is at index 2 in list -> Index 3 in model)
    MIDLINE_FALX_IDX = 3 
    
    # Needed for BPD/OFD points (Handled by the Sigmoid Head)
    CALV_INNER_CH = 0   # Channel 0 of Sigmoid head
    CALV_OUTER_CH = 1   # Channel 1 of Sigmoid head

    # ==============================
    # 4. STRUCTURE LIST (Inner structures only)
    # ==============================
    STRUCTURES = [
        "hemisphere 1",                         # 1
        "hemisphere 2",                         # 2
        "midline falx",                         # 3 <-- Critical for PCA
        "thalamus 2",                           # 4
        "hippocampal gyrus 2",                  # 5
        "thalamus 1",                           # 6
        "hippocampal gyrus 1",                  # 7
        "anterior horn of lateral ventricles 2",# 8
        "anterior midline falx",                # 9
        "csp (cavum septum pellucidum)",        # 10
        "arrow sign",                           # 11
        "choroid plexus 2",                     # 12
        "lateral ventricle 2",                  # 13
        "posterior horn of lateral ventricle 2",# 14
        "lateral sulcus 2",                     # 15
        "cerebral peduncle 1",                  # 16
        "cerebral peduncle 2",                  # 17
        "cerebellum",                           # 18
        "cerebellar vermis",                    # 19
        "cisternae magna",                      # 20
        "pillars of fornix",                    # 21
        "anterior horn of lateral ventricle 1", # 22
    ]


"""OLD ORIGINAL CONFIG# config.py

import torch
import numpy as np


class Config:
    # ==============================
    # PATHS — 3 image folders, 3 mask folders (paired by index)
    # ==============================
    IMAGE_DIRS = [
        "/mnt/data/Anusha/UNET_SoftMax/Data_seg/TC/images",
        "/mnt/data/Anusha/UNET_SoftMax/Data_seg/TT/images",
        "/mnt/data/Anusha/UNET_SoftMax/Data_seg/TV/images",
    ]
    MASK_DIRS = [
        "/mnt/data/Anusha/UNET_SoftMax/Data_seg/TC/masks",
        "/mnt/data/Anusha/UNET_SoftMax/Data_seg/TT/masks",
        "/mnt/data/Anusha/UNET_SoftMax/Data_seg/TV/masks",
    ]

    IMAGE_EXT = ".jpeg"
    MASK_EXT  = ".npz"

    # ==============================
    # INFERENCE
    # ==============================
    INFER_CHECKPOINT = "/mnt/data/Anusha/UNET_SoftMax/checkpoints_multihead/best_model.pth"
    INFER_IMAGE_DIR  = "/mnt/data/Anusha/UNET_SoftMax/Data_seg/VAL/TV/images"
    INFER_MASK_DIR   = "/mnt/data/Anusha/UNET_SoftMax/Data_seg/VAL/TV/masks"
    INFER_OUTPUT_DIR = "/mnt/data/Anusha/UNET_SoftMax/results_pp/TV"
    INFER_THRESHOLD  = 0.5    # sigmoid threshold for calvarium head
    INFER_DEVICE     = "cuda"

    # ==============================
    # SPLIT
    # ==============================
    VAL_SPLIT  = 0.15
    TEST_SPLIT = 0.05
    SEED       = 42

    # ==============================
    # MODEL
    # ==============================
    IN_CHANNELS   = 1
    # ── HEAD 1 (softmax): 22 inner foreground structures + 1 background = 23
    # ── HEAD 2 (sigmoid): inner calvarium + outer calvarium = 2 binary channels
    #
    # NUM_CLASSES     = total classes for the softmax head (bg + 22 inner structs)
    # NUM_CALV_CLASSES = number of binary channels for the calvarium head (always 2)
    #
    # STRUCTURES list below has 22 entries (calvarium removed).
    # STRUCTURE_TO_IDX maps those 22 names → class indices 1–22 for the softmax head.
    # Calvarium GT is produced as a separate (2, H, W) float mask.
    # ───────────────────────────────────────────────────────────────────────
    NUM_CLASSES      = 23   # 22 inner structures + 1 background (softmax head)
    NUM_CALV_CLASSES = 2    # inner calvarium (ch 0) + outer calvarium (ch 1)

    # Convenience: calvarium class indices inside CALV mask channels
    CALV_INNER_CH = 0   # channel 0 of mask_calv → inner calvarium
    CALV_OUTER_CH = 1   # channel 1 of mask_calv → outer calvarium

    BASE_FEATURES = 64
    BILINEAR      = True
    DROPOUT       = 0.2

    # ==============================
    # TRAINING
    # ==============================
    EPOCHS               = 150
    BATCH_SIZE           = 4
    LR                   = 5e-5
    WEIGHT_DECAY         = 1e-5
    LR_SCHEDULER         = "cosine"
    WARMUP_EPOCHS        = 10
    EARLY_STOP_PATIENCE  = 20
    GRAD_CLIP            = 0.5
    MIXED_PRECISION      = True
    NUM_WORKERS          = 0
    PIN_MEMORY           = True

    # ==============================
    # LOSS
    # ── Inner head (softmax): CE + Tversky
    # ── Calvarium head (sigmoid): BCE + Dice
    # ── Total = INNER_LOSS_WEIGHT * inner + CALV_LOSS_WEIGHT * calv
    # ==============================
    CE_WEIGHT     = 0.5
    TV_WEIGHT     = 0.5
    LOSS_SMOOTH   = 1e-6
    TVERSKY_ALPHA = 0.3   # weight on FP
    TVERSKY_BETA  = 0.7   # weight on FN

    # Calvarium head loss weights
    CALV_BCE_WEIGHT  = 0.5   # weight of BCE inside calvarium loss
    CALV_DICE_WEIGHT = 0.5   # weight of Dice inside calvarium loss
    CALV_LOSS_WEIGHT = 0.5   # how much calvarium loss contributes to total loss
    INNER_LOSS_WEIGHT = 1.0  # how much inner-structure loss contributes to total

    USE_CLASS_WEIGHTS = True

    # ── Class weighting strategy ──────────────────────────────────────────
    CLASS_WEIGHT_STRATEGY = "inv_smooth"
    CLASS_WEIGHT_ALPHA    = 0.7
    CLASS_WEIGHT_MIN      = 0.5
    CLASS_WEIGHT_MAX      = 6.0

    # ==============================
    # AUGMENTATION
    # ==============================
    IMAGE_SIZE = (512, 512)

    AUG_HFLIP         = 0.5
    AUG_VFLIP         = 0.2
    AUG_ROTATE        = 0.5
    AUG_ROTATE_LIMIT  = 15
    AUG_ELASTIC       = 0.4
    AUG_ELASTIC_ALPHA = 80
    AUG_ELASTIC_SIGMA = 8
    AUG_GRID_DIST     = 0.3
    AUG_RANDOM_CROP   = 0.3
    AUG_CROP_SCALE    = (0.8, 1.0)
    AUG_BRIGHTNESS    = 0.4
    AUG_CONTRAST      = 0.4
    AUG_GAMMA         = 0.4
    AUG_SPECKLE       = 0.5
    AUG_SHADOW        = 0.3
    AUG_REVERBERATION = 0.2
    AUG_COMET         = 0.2

    # ==============================
    # PATHS
    # ==============================
    CHECKPOINT_DIR = "./checkpoints_multihead1"
    LOG_DIR        = "./logs"
    BEST_MODEL     = "./checkpoints_multihead1/best_model.pth"

    # ==============================
    # STRUCTURES  (inner structures only — calvarium excluded)
    # ── Index 0 is BACKGROUND (implicit).
    # ── Indices 1–22 correspond to STRUCTURES[0]–STRUCTURES[21].
    # ── Calvarium is handled by the separate sigmoid head (mask_calv).
    # ==============================
    STRUCTURES = [
        "hemisphere 1",                             # 1
        "hemisphere 2",                             # 2
        "midline falx",                             # 3
        "thalamus 2",                               # 4
        "hippocampal gyrus 2",                      # 5
        "thalamus 1",                               # 6
        "hippocampal gyrus 1",                      # 7
        "anterior horn of lateral ventricles 2",    # 8
        "anterior midline falx",                    # 9
        "csp (cavum septum pellucidum)",            # 10
        "arrow sign",                               # 11
        "choroid plexus 2",                         # 12
        "lateral ventricle 2",                      # 13
        "posterior horn of lateral ventricle 2",    # 14
        "lateral sulcus 2",                         # 15
        "cerebral peduncle 1",                      # 16
        "cerebral peduncle 2",                      # 17
        "cerebellum",                               # 18
        "cerebellar vermis",                        # 19
        "cisternae magna",                          # 20
        "pillars of fornix",                        # 21
        "anterior horn of lateral ventricle 1",     # 22
    ]

    # Calvarium structure names (used in dataset to identify which channels
    # go to the calvarium head instead of the softmax head)
    CALVARIUM_STRUCTURES = {
        "inner calvarium": CALV_INNER_CH,   # → mask_calv channel 0
        "outer calvarium": CALV_OUTER_CH,   # → mask_calv channel 1
    }

    # Maps inner-structure name → class index (1-based; 0 = background)
    STRUCTURE_TO_IDX = {name: i + 1 for i, name in enumerate(STRUCTURES)}

    IGNORE_KEYWORDS = ["shadow", "artefact", "artifact", "reverberation"]

    @staticmethod
    def normalize_name(name: str) -> str:
        name = " ".join(name.strip().lower().split())
        if "dural fold" in name:
            return "dural fold(s)"
        if "pillars of fornix" in name or "pillars of fonix" in name:
            return "pillars of fornix"
        if "petrous" in name:
            if "1" in name:
                return "petrous part of temporal bone 1"
            if "2" in name:
                return "petrous part of temporal bone 2"
        return name

    @staticmethod
    def is_valid_structure(name: str) -> bool:
        name = name.lower()
        return not any(k in name for k in Config.IGNORE_KEYWORDS)"""



