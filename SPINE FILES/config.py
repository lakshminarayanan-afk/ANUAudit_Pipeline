# SPINE 
# config.py - Unified Configuration for 3-Plane Spine Segmentation
#
# Merges coronal / sagittal / full_body_coronal into ONE config used by a
# single shared-encoder, 3-head UNet. Does NOT touch or replace
# config_coronal.py / config_spine_3structs.py / config_spine_5structs.py /
# config_spine.py — those stay as-is for their original single-plane runs.
#
# Structure lists + counts below are CONFIRMED from mask_dataset_report.py
# output (masks_27, run on /mnt/data4tb/Aravind/spine_segmentation/data):
#   coronal            : 343 images, 9 structures
#   sagittal           : 396 images, 7 structures
#   full_body_coronal  : 352 images, 11 structures
# `masks` and `masks_27` report IDENTICAL structure sets per plane —
# masks_27 is just the same data padded to a fixed 27-channel container
# (with unused_channel_N placeholders), so masks_27 is used throughout.

import torch
from pathlib import Path


class Config:
    # ==============================
    # PLANES
    # ==============================
    PLANES = ["coronal", "sagittal", "full_body_coronal"]

    # ==============================
    # PATHS  (confirmed from folder listing + mask_dataset_report.py)
    # ==============================
    DATA_ROOT = "/mnt/data4tb/anusha/Prathicksha_Spine/data"

    IMAGE_DIRS = {
        "coronal":            f"{DATA_ROOT}/coronal/images",
        "sagittal":            f"{DATA_ROOT}/sagittal/images",
        "full_body_coronal":  f"{DATA_ROOT}/full_body_coronal/images",
    }
    MASK_DIRS = {
        "coronal":            f"{DATA_ROOT}/coronal/masks_27",
        "sagittal":            f"{DATA_ROOT}/sagittal/masks_27",
        "full_body_coronal":  f"{DATA_ROOT}/full_body_coronal/masks_27",
    }

    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    MASK_EXT   = ".npz"

    # Output paths
    CHECKPOINT_DIR = "./checkpoints"
    LOG_DIR        = "./logs_multiplane_new"
    BEST_MODEL     = "./checkpoints/best_model.pth"
    # Inference
    INFER_CHECKPOINT = "./checkpoints/best_model.pth"
    INFER_OUTPUT_DIR = "./results_final"
    INFER_IMAGE_DIR  = ""
    INFER_DEVICE      = "cuda"
    INFER_CONF_THRESH = 0.30

    # ==============================
    # DATASET SPLIT
    # Applied PER PLANE, then concatenated, so every split is
    # plane-stratified (every train/val/test set contains all 3 planes).
    # ==============================
    VAL_SPLIT  = 0.15
    TEST_SPLIT = 0.10
    SEED       = 42

    # ==============================S
    # MODEL ARCHITECTURE
    # (unchanged from model_spine.py — shared encoder, three 1x1-conv
    # output heads instead of one)
    # ==============================
    IN_CHANNELS   = 1
    BASE_FEATURES = 64
    BILINEAR      = True
    DROPOUT       = 0.3

    NUM_CLASSES_PER_PLANE = {
        "coronal":              10,  # 9 structures + background
        "sagittal_exclusive":    4,  # skin line + 2 oss. centers + background
        "full_body_coronal":    11,  # 10 structures + background (Lungs merged)
    }

    # Sigmoid multi-label region head (no background channel — regions
    # are independent, can overlap ossification-center classes above).
    NUM_REGIONS_PER_PLANE = {"sagittal": 4}
    REGION_LOSS_WEIGHT    = 3.0

    # ==============================
    # TRAINING HYPERPARAMETERS
    # Reused from config_spine.py (closest match among the 4 old spine
    # configs — smaller batch since combined dataset spans more variety).
    # ==============================
    EPOCHS              = 150
    BATCH_SIZE           = 8
    LR =  1e-3  # was 1e-4 — fine-tuning, not fresh training
    WEIGHT_DECAY          = 1e-5
    LR_SCHEDULER          = "cosine"
    WARMUP_EPOCHS         = 5
    EARLY_STOP_PATIENCE   = 15
    GRAD_CLIP             = 1.0
    MIXED_PRECISION       = True
    NUM_WORKERS           = 2
    PIN_MEMORY            = True

    # ==============================
    # LOSS  (CE + Tversky per plane, same formula as loss.py)
    # ==============================
    CE_WEIGHT     = 0.5
    TV_WEIGHT     = 0.5
    LOSS_SMOOTH   = 1e-6
    TVERSKY_ALPHA = 0.3
    TVERSKY_BETA  = 0.7

    USE_CLASS_WEIGHTS      = True
    CLASS_WEIGHT_STRATEGY  = "inv_smooth"
    CLASS_WEIGHT_ALPHA     = 0.7
    CLASS_WEIGHT_MIN       = 0.5
    CLASS_WEIGHT_MAX = 10.0

    # NOTE: class weights are computed dynamically per-plane from training
    # data frequencies (see loss.py's compute_class_weights pattern) — not
    # hardcoded. Flagging the structures the mask report showed as very
    # rare, since these will land near CLASS_WEIGHT_MAX automatically and
    # may be worth a manual boost if that's not enough:
    #   coronal            -> "Coronal spine- Cervical Region"  (1.7% presence)
    #   full_body_coronal  -> "Gall Bladder"                    (3.1% presence)
    #   full_body_coronal  -> "Diaphragm"                       (3.7% presence)

    # ── Cross-plane batch loss combination: a batch may mix samples from
    # more than one plane. Per-plane CE+Tversky is computed only over that
    # plane's samples in the batch; the per-plane losses PRESENT in the
    # batch are then averaged (not summed) so batch composition doesn't
    # change gradient magnitude. See train_multiplane.py.

    # ==============================
    # AUGMENTATION  (reused verbatim from config_spine.py / sagittal defaults)
    # ==============================
    IMAGE_SIZE = (512, 512)   # forced resize target — native sizes vary
                               # (5 distinct shapes in coronal, 4 in sagittal
                               # incl. a 1080x1920 outlier, 5 in full_body_coronal)

    AUG_HFLIP         = 0.3
    AUG_VFLIP         = 0.0
    AUG_ROTATE        = 0.5
    AUG_ROTATE_LIMIT  = 10
    AUG_ELASTIC       = 0.3
    AUG_ELASTIC_ALPHA = 50
    AUG_ELASTIC_SIGMA = 5
    AUG_GRID_DIST     = 0.2
    AUG_RANDOM_CROP   = 0.3
    AUG_CROP_SCALE    = (0.85, 1.0)
    AUG_BRIGHTNESS    = 0.4
    AUG_CONTRAST      = 0.4
    AUG_GAMMA         = 0.4
    AUG_SPECKLE       = 0.5
    AUG_SHADOW        = 0.2
    AUG_REVERBERATION = 0.15
    AUG_COMET         = 0.1

    # ==============================
    # STRUCTURES — confirmed real per-plane lists (mask_dataset_report.py),
    # NOT the old stale per-plane config lists.
    # Index 0 is background (implicit) for every plane's head.
    # ==============================
    STRUCTURES = {
        "coronal": [
            "Coronal spine- Cervical Region",              # 1
            "Coronal spine- Sacral Region",                # 2
            "Coronal spine- Thoracic Region",               # 3
            "Coronal spine- Lumbar Region",                 # 4
            "Iliac Crest 1",                                # 5
            "Iliac Crest 2",                                # 6
            "Anterior Vertebral Body",                      # 7
            "Ossification Center - Posterior arch 1",       # 8
            "Ossification Center - Posterior arch 2",       # 9
        ],
        "sagittal_exclusive": [
            "Skin line",                                     # 1
            "Ossification Center - Arch of vertebra",        # 2
            "Ossification Center- Vertebral Body",           # 3
        ],
        "sagittal_regions": [
            "Sagittal spine - Sacral Region",                # 0 (sigmoid, no bg)
            "Sagittal spine - Cervical Region",              # 1
            "Sagittal spine - Lumbar Region",                # 2
            "Sagittal spine - Thoracic Region",              # 3
        ],
        "full_body_coronal": [
            "Gall Bladder",                                  # 1
            "Diaphragm",                                     # 2
            "Lungs",                                          # 3
            "Heart",                                          # 4
            "Ribs 1",                                         # 5
            "Stomach",                                        # 6
            "Bladder",                                        # 7
            "Liver",                                          # 8
            "Ribs 2",                                         # 9
            "Bowel",                                          # 10
        ],
    }

    # Raw npz channel names that must map onto one merged class.
    STRUCTURE_ALIASES = {
        "full_body_coronal": {
            "Left Lung":  "Lungs",
            "Right Lung": "Lungs",
        }
    }

    # Maps: plane -> {structure_name: class_idx (1-based; 0=background)}
    STRUCTURE_TO_IDX = {
        plane: {name: i + 1 for i, name in enumerate(names)}
        for plane, names in STRUCTURES.items()
    }

    # Names in the npz `structures` array that must be skipped when
    # building the index mask, regardless of plane (masks_27 pads to 27
    # channels with these placeholders beyond each plane's real count).
    SKIP_STRUCTURE_PREFIX = "unused_channel_"

    IGNORE_KEYWORDS = ["artifact", "artefact", "shadow", "motion", "quality"]

    # ==============================
    # UTILITY METHODS
    # ==============================
    @staticmethod
    def is_valid_structure(name: str) -> bool:
        name_l = name.lower()
        if name.startswith(Config.SKIP_STRUCTURE_PREFIX) or "unused_channel" in name_l:
            return False
        return not any(k in name_l for k in Config.IGNORE_KEYWORDS)

    @staticmethod
    def get_class_index(plane: str, name: str) -> int:
        """Exact-match lookup against the real structure names for `plane`.
        Raw npz channel names listed in STRUCTURE_ALIASES[plane] are
        remapped to their merged target name first (e.g. Left/Right Lung
        -> Lungs), so both channels write into the same class index.
        Returns 0 (background) if not found.
        """
        name = Config.STRUCTURE_ALIASES.get(plane, {}).get(name, name)
        return Config.STRUCTURE_TO_IDX.get(plane, {}).get(name, 0)

    @staticmethod
    def get_active_structures(plane: str) -> list:
        """Convenience: the real (non-placeholder) structure list for a plane."""
        return Config.STRUCTURES.get(plane, [])