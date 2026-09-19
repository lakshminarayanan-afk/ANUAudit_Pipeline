# """
# config.py - Central configuration for the Limbs segmentation project (Swin-UNet + SupCon pipeline)
# =====================================================================================================
# Dataset: fetal limb ultrasound, split by plane folder: HU (Humerus) | RU (Radius_Ulna) |
#          FE (Femur) | TF (Tibia_Fibula)

# [... unchanged docstring - see your existing config.py ...]

# CCDC ADDITION (see bottom of class body, clearly marked)
# -------------------------------------------------------------------------------------------------
# Adds one new config block for the CCDC-style contrastive branch (high-confidence positive mining +
# hard-negative class mining + confidence/confusion-weighted contrastive loss, see model.py's
# CCDCProjectionHead and loss.py's CCDCStyleLoss). Nothing else in this file was touched.
# """

# from __future__ import annotations

# from pathlib import Path
# from typing import Dict, List

# from torch import split


# class Config:
