"""
config.py  –  Central configuration for the Abdomen segmentation project
=========================================================================
Channel changes vs. original:
  • Kidney Cortex 1 + 2  →  merged into single "Kidney Cortex" channel (ch 9)
  • Renal Pelvis  1 + 2  →  merged into single "Renal Pelvis"  channel (ch 10)
  • Total output channels: 13 structures + 1 skin line = 14
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict


class Config_abd:

    # ── Data ──────────────────────────────────────────────────────────────────
    DATA_ROOT = Path("/mnt/data/Anusha/UNET_Abdomen/DATA_abdomen")

    PLANE_ROOTS = {
        "AC": DATA_ROOT / "AC",
        "TK": DATA_ROOT / "TK",
        "CI": DATA_ROOT / "CI",
    }

    @classmethod
    def images_dir(cls, plane: str) -> Path:
        return cls.PLANE_ROOTS[plane] / "images"

    @classmethod
    def masks_dir(cls, plane: str) -> Path:
        return cls.PLANE_ROOTS[plane] / "masks"

    # ── Checkpoint / logs ─────────────────────────────────────────────────────
    CHECKPOINT_DIR = Path("/mnt/data/Anusha/UNET_Abdomen/checkpoints_20_06(S)")
    LOG_DIR        = Path("/mnt/data/Anusha/UNET_Abdomen/logs")

    # ── Channel definitions ───────────────────────────────────────────────────
    #
    #   Kidney Cortex 1 + 2  are MERGED  →  single "Kidney Cortex" channel
    #   Renal Pelvis  1 + 2  are MERGED  →  single "Renal Pelvis"  channel
    #
    #   Raw label values in the mask files (unchanged):
    #       label 10  →  Kidney Cortex 1  ┐
    #       label 11  →  Kidney Cortex 2  ┘  → merged output ch 9
    #       label 12  →  Renal Pelvis 1   ┐
    #       label 13  →  Renal Pelvis 2   ┘  → merged output ch 10
    #
    # STRUCTURES = [
    #     "Stomach",        # ch 0  ← label 1
    #     "Umbilical Vein", # ch 1  ← label 2
    #     "Vertebrae",      # ch 2  ← label 3
    #     "Aorta",          # ch 3  ← label 4
    #     "IVC",            # ch 4  ← label 5
    #     "Adrenal 1",      # ch 5  ← label 6
    #     "Adrenal 2",      # ch 6  ← label 7
    #     "Placenta",       # ch 7  ← label 8
    #     "Gall Bladder",   # ch 8  ← label 9
    #     "Kidney Cortex",  # ch 9  ← labels 10 + 11  (merged)
    #     "Renal Pelvis",   # ch 10 ← labels 12 + 13  (merged)
    #     "Bladder",        # ch 11 ← label 14
    #     "Cord Insertion", # ch 12 ← label 15
    # ]


    STRUCTURES = [
    "Stomach",        # ch 0  ← label 1
    "Umbilical Vein", # ch 1  ← label 2
    "Vertebrae",      # ch 2  ← label 3
    "Aorta",          # ch 3  ← label 4
    "IVC",            # ch 4  ← label 5
    "Adrenal",        # ch 5  ← labels 6 + 7  (merged)
    "Placenta",       # ch 6  ← label 8        ← shifted down
    "Gall Bladder",   # ch 7  ← label 9        ← shifted down
    "Kidney Cortex",  # ch 8  ← labels 10 + 11 ← shifted down
    "Renal Pelvis",   # ch 9  ← labels 12 + 13 ← shifted down
    "Bladder",        # ch 10 ← label 14       ← shifted down
    "Cord Insertion", # ch 11 ← label 15       ← shifted down
     ]

    SKIN_LINE_NAME = "Skin Line"  # ch 13

    NUM_STRUCTURES = len(STRUCTURES)      # 13
    NUM_CHANNELS   = NUM_STRUCTURES + 1  # 14  (13 structures + 1 skin line)

    # For loss / present vector:
    #   index 0             = background (always excluded from loss)
    #   indices 1–13        = structures
    #   index 14            = skin line
    PRESENT_SIZE = NUM_CHANNELS + 1       # 15

    # Backward compat
    NUM_CLASSES      = NUM_STRUCTURES + 1  # 14  (includes background label 0)
    NUM_CALV_CLASSES = 1
    STRUCTURE_TO_IDX = {s: i + 1 for i, s in enumerate(STRUCTURES)}  # 1-based

    # Skin line present-index (in the present vector)
    SKIN_PRESENT_IDX = NUM_CHANNELS        # 14

    # ── Raw label → output channel mapping ────────────────────────────────────
    #   Labels 1–9, 14, 15 map 1-to-1 (shifted to 0-based channel index).
    #   Labels 10 & 11  →  channel 9   (Kidney Cortex, union-merged)
    #   Labels 12 & 13  →  channel 10  (Renal Pelvis,  union-merged)
    #   Label  16       →  channel 13  (Skin Line)
    # RAW_LABEL_TO_CHANNEL: Dict[int, int] = {
    #     1:  0,   # Stomach
    #     2:  1,   # Umbilical Vein
    #     3:  2,   # Vertebrae
    #     4:  3,   # Aorta
    #     5:  4,   # IVC
    #     6:  5,   # Adrenal 1
    #     7:  6,   # Adrenal 2
    #     8:  7,   # Placenta
    #     9:  8,   # Gall Bladder
    #     10: 9,   # Kidney Cortex 1  ┐ merged → ch 9
    #     11: 9,   # Kidney Cortex 2  ┘
    #     12: 10,  # Renal Pelvis 1   ┐ merged → ch 10
    #     13: 10,  # Renal Pelvis 2   ┘
    #     14: 11,  # Bladder
    #     15: 12,  # Cord Insertion
    #     16: 13,  # Skin Line
    # }
    RAW_LABEL_TO_CHANNEL: Dict[int, int] = {
    1:  0,   # Stomach
    2:  1,   # Umbilical Vein
    3:  2,   # Vertebrae
    4:  3,   # Aorta
    5:  4,   # IVC
    6:  5,   # Adrenal 1  ┐ merged → ch 5
    7:  5,   # Adrenal 2  ┘
    8:  6,   # Placenta        (was ch 7, now ch 6)
    9:  7,   # Gall Bladder    (was ch 8, now ch 7)
    10: 8,   # Kidney Cortex 1 ┐ merged → ch 8  (was ch 9)
    11: 8,   # Kidney Cortex 2 ┘
    12: 9,   # Renal Pelvis 1  ┐ merged → ch 9  (was ch 10)
    13: 9,   # Renal Pelvis 2  ┘
    14: 10,  # Bladder         (was ch 11, now ch 10)
    15: 11,  # Cord Insertion  (was ch 12, now ch 11)
    16: 12,  # Skin Line       (was ch 13, now ch 12)
    }
    # ── Model ─────────────────────────────────────────────────────────────────
    IN_CHANNELS   = 1
    BASE_FEATURES = 64
    BILINEAR      = True
    DROPOUT       = 0.15

    # ── Training ──────────────────────────────────────────────────────────────
    IMAGE_SIZE   = (512, 512)
    BATCH_SIZE   = 8
    NUM_WORKERS  = 4
    PIN_MEMORY   = True
    NUM_EPOCHS   = 150
    LR           = 1e-4
    WEIGHT_DECAY = 1e-5
    LR_PATIENCE  = 10
    LR_FACTOR    = 0.5
    EARLY_STOP   = 20

    # ── Loss ──────────────────────────────────────────────────────────────────
    BCE_WEIGHT  = 0.5
    DICE_WEIGHT = 0.5
    LOSS_SMOOTH = 1e-6

    # ── Class weight computation ───────────────────────────────────────────────
    CLASS_WEIGHT_ALPHA = 0.7
    CLASS_WEIGHT_MIN   = 0.5
    CLASS_WEIGHT_MAX   = 10.0

    # ── Data split ────────────────────────────────────────────────────────────
    VAL_FRACTION  = 0.15
    TEST_FRACTION = 0.10
    RANDOM_SEED   = 42

    # ── Augmentation probabilities ────────────────────────────────────────────
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
    AUG_CONTRAST      = 0.4   # used via RandomBrightnessContrast contrast_limit
    AUG_GAMMA         = 0.4
    AUG_SPECKLE       = 0.5
    AUG_SHADOW        = 0.3
    AUG_REVERBERATION = 0.2
    AUG_COMET         = 0.2

    LOG_INTERVAL = 10

    @classmethod
    def make_dirs(cls) -> None:
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)