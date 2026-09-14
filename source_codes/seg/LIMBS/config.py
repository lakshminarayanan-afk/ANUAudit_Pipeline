"""
config.py - Central configuration for the Limbs segmentation project (Swin-UNet + SupCon pipeline)
=====================================================================================================
Dataset: fetal limb ultrasound, split by plane folder: HU (Humerus) | RU (Radius_Ulna) |
         FE (Femur) | TF (Tibia_Fibula)

[... unchanged docstring - see your existing config.py ...]

CCDC ADDITION (see bottom of class body, clearly marked)
-------------------------------------------------------------------------------------------------
Adds one new config block for the CCDC-style contrastive branch (high-confidence positive mining +
hard-negative class mining + confidence/confusion-weighted contrastive loss, see model.py's
CCDCProjectionHead and loss.py's CCDCStyleLoss). Nothing else in this file was touched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from torch import split


class Config:

    # ---------------------------------------------------------------------
    # Paths - EDIT these two lines for your machine, everything else derives from them.
    # ---------------------------------------------------------------------
    DATA_ROOT = Path("/mnt/data4tb/anusha/Limbs_Model_copy")
    DATASET_DIR = DATA_ROOT / "Limbs_Data"

    CHECKPOINT_DIR = DATA_ROOT / "checkpoints"
    LOG_DIR = DATA_ROOT / "logs"

    # ---------------------------------------------------------------------
    # Planes (each plane folder holds images that predominantly show one bone group,
    # but all 4 are pooled together at train time into a single dataset/model).
    # ---------------------------------------------------------------------
    PLANES: List[str] = ["HU", "RU", "FE", "TF"]
    PLANE_NAMES: Dict[str, str] = {
        "HU": "Humerus",
        "RU": "Radius_Ulna",
        "FE": "Femur",
        "TF": "Tibia_Fibula",
    }

    from pathlib import Path

    @staticmethod
    def images_dir(split, plane):
        if split.lower() == "train":
            return Config.DATASET_DIR / plane / "images"
        # Unseen Femur inference
        if plane == "FE":
            return Config.DATA_ROOT / "Online_data" / "FE"

        return Config.DATASET_DIR / split.upper() / plane / "images"

    @staticmethod
    def masks_dir(split, plane):
        if split.lower() == "train":
            return Config.DATASET_DIR / plane / "masks"
        else:
            return Config.DATASET_DIR / split.upper() / plane / "masks"

    @classmethod
    def train_images_dir(cls, plane: str) -> Path:
        return cls.images_dir("TRAIN", plane)

    @classmethod
    def train_masks_dir(cls, plane: str) -> Path:
        return cls.masks_dir("TRAIN", plane)

    @classmethod
    def val_images_dir(cls, plane: str) -> Path:
        return cls.images_dir("VAL", plane)

    @classmethod
    def val_masks_dir(cls, plane: str) -> Path:
        return cls.masks_dir("VAL", plane)

    @classmethod
    def test_images_dir(cls, plane: str) -> Path:
        return cls.images_dir("TEST", plane)

    @classmethod
    def test_masks_dir(cls, plane: str) -> Path:
        return cls.masks_dir("TEST", plane)

    NATIVE_H = 1080
    NATIVE_W = 1920
    PAD_MULTIPLE = 32

    STRUCTURES: List[str] = [
        "Humerus", "Placenta", "Radius", "Ulna", "Hand", "Femur", "Thigh",
        "Femur Length", "Tibia", "Fibula", "Foot", "Lower Leg",
    ]

    ARTIFACTS: List[str] = [
        "Humerus Shadowing", "Limb Shadowing", "Rib Shadowing", "Other Shadowing",
        "Radius Ulna Shadowing", "Femur Shadowing", "Tibia Fibula Shadowing", "Foot Shadowing",
    ] 

    MASK_FOREGROUND_VALUE = 255
    MASK_THRESHOLD = 127

    BONE_STRUCTURES: List[str] = ["Humerus", "Radius", "Ulna", "Femur", "Tibia", "Fibula"]

    LABEL_MAP: Dict[int, str] = {
        0: "background",
        1: "femur",
        2: "humerus",
        3: "radius_ulna",
        4: "tibia_fibula",
    }
    NUM_SEG_CLASSES: int = len(LABEL_MAP)
    NUM_BONE_CLASSES: int = len(LABEL_MAP) - 1
    IGNORE_INDEX: int = 255

    BONE_NAME_TO_CLASS: Dict[str, int] = {
        "Femur": 1, "Humerus": 2, "Radius": 3, "Ulna": 3, "Tibia": 4, "Fibula": 4,
    }

    PLANE_TO_BONE_NAMES: Dict[str, List[str]] = {
        "HU": ["Humerus"], "RU": ["Radius", "Ulna"], "FE": ["Femur"], "TF": ["Tibia", "Fibula"],
    }

    STRUCTURE_LABEL_PRIORITY: List[str] = ["Radius", "Ulna", "Humerus", "Femur", "Tibia", "Fibula"]

    BONE_CLASS_TO_SHADOW_NAMES: Dict[int, List[str]] = {
        1: ["Femur Shadowing"], 2: ["Humerus Shadowing"],
        3: ["Radius Ulna Shadowing"], 4: ["Tibia Fibula Shadowing"],
    }

    BACKGROUND_WEIGHT: float = 0.1
    CLASS_WEIGHTS: List[float] = [BACKGROUND_WEIGHT, 0.276, 0.369, 0.3815, 0.3765]

    # ---------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------
    IN_CHANNELS: int = 1
    EMBED_DIM: int = 96
    DEPTHS: List[int] = [2, 2, 6, 2]
    NUM_HEADS: List[int] = [3, 6, 12, 24]
    WINDOW_SIZE: int = 7
    FPN_CHANNELS: int = 128
    PROJ_DIM: int = 128            # existing pooled, image-level SupCon projection head output dim

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------
    BATCH_SIZE: int = 1
    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True
    NUM_EPOCHS: int = 150
    LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    LR_PATIENCE: int = 10
    LR_FACTOR: float = 0.5
    EARLY_STOP: int = 20

    # ---------------------------------------------------------------------
    # Loss weights (segmentation / classification / supervised contrastive)
    # ---------------------------------------------------------------------
    W_SEG: float = 1.0
    W_CLS: float = 0.3
    W_CON: float = 0.5
    DICE_WEIGHT: float = 1.0
    CE_WEIGHT: float = 1.0
    SUPCON_TEMPERATURE: float = 0.07

    RANDOM_SEED: int = 42

    AUG_HFLIP: float = 0.5
    AUG_BRIGHTNESS_CONTRAST: float = 0.5
    AUG_BRIGHTNESS_LIMIT: float = 0.15
    AUG_CONTRAST_LIMIT: float = 0.15
    AUG_GAUSS_NOISE: float = 0.3
    AUG_CLAHE: float = 0.2
    AUG_AFFINE: float = 0.5
    AUG_AFFINE_TRANSLATE: float = 0.05
    AUG_AFFINE_ROTATE_LIMIT: int = 10
    AUG_AFFINE_SCALE: tuple = (0.95, 1.05)
    AUG_AFFINE_SHEAR: int = 5

    LOG_INTERVAL: int = 20

    # =======================================================================
    # STANDALONE CONTOUR-ONLY INFERENCE (infer_contours.py) - unchanged
    # =======================================================================
    INFER_IMAGES_DIR: Dict[str, Path] = {
    "HU": Path("/mnt/data4tb/anusha/Limbs_Model_copy/Limbs_Data/HU/images"),
    "RU": Path("/mnt/data4tb/anusha/Limbs_Model_copy/Limbs_Data/RU/images"),
    "FE": Path("/mnt/data4tb/anusha/Limbs_Model_copy/Limbs_Data/FE/images"),
    "TF": Path("/mnt/data4tb/anusha/Limbs_Model_copy/Limbs_Data/TF/images"),
}

    INFER_MASKS_DIR: Dict[str, Path] = {
    "HU": Path("/mnt/data4tb/anusha/Limbs_Model_copy/Limbs_Data/HU/masks"),
    "RU": Path("/mnt/data4tb/anusha/Limbs_Model_copy/Limbs_Data/RU/masks"),
    "FE": Path("/mnt/data4tb/anusha/Limbs_Model_copy/Limbs_Data/FE/masks"),
    "TF": Path("/mnt/data4tb/anusha/Limbs_Model_copy/Limbs_Data/TF/masks"),
    }
    INFER_OUTPUT_DIR: Path = DATA_ROOT / "inference_contours_agreement"

    POSTPROCESS_ENABLED: bool = True
    POSTPROCESS_MORPH_KERNEL: int = 3
    POSTPROCESS_MIN_AREA: int = 40
    POSTPROCESS_KEEP_LARGEST_ONLY: bool = True
    POSTPROCESS_FILL_HOLES: bool = True

    CONTOUR_THICKNESS: int = 2
    CONTOUR_CLASS_COLORS_BGR: Dict[int, tuple] = {
        1: (60, 60, 230), 2: (60, 200, 60), 3: (230, 160, 60), 4: (60, 220, 220),
    }

    # =======================================================================
    # NEW: CCDC-STYLE CONTRASTIVE LEARNING
    # -----------------------------------------------------------------------
    # High-confidence positive mining + hard-negative class mining + confidence/confusion
    # weighted InfoNCE contrastive loss (model.py: CCDCProjectionHead, loss.py: CCDCStyleLoss).
    # This is the ONLY new config block added for the CCDC experiment. Every value below has
    # exactly one source of truth - nothing is hard-coded elsewhere.
    # =======================================================================
    CCDC_PROJ_DIM: int = 32                           # pixel-level embedding channels
    CCDC_POSITIVE_CONFIDENCE_THRESHOLD: float = 0.50    # P(GT class) required to count as a positive
    CCDC_MAX_SAMPLES_PER_CLASS: int = 32               # cap on mined pixels per class, per batch
    CCDC_TEMPERATURE: float = 0.07                      # InfoNCE temperature (cosine similarities)
    CCDC_LAMBDA: float = 0.05                           # weight of L_CCDC in L_total

    @classmethod
    def make_dirs(cls) -> None:
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        cls.INFER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)