"""
data_artifacts.py

Calibration reference values extracted from training set.
These are hard-coded values computed from the notebook's training phase.
Used for white balance and palette color correction during preprocessing.
"""

import numpy as np
import pandas as pd

# ===== WARP & ALIGNMENT CONFIGURATION =====
WARP_SIZE = 512
TARGET_CHECKERBOARD_SIDE = "top"

# ===== ROI DEFINITIONS (Normalized Coordinates: 0.0 to 1.0) =====
ROI_CONFIG = {
    # Checkerboard reference area
    "checkerboard": (0.35, 0.07, 0.70, 0.22),

    # Gray patches for white balance (4 corners)
    "gray_tl": (0.14, 0.15, 0.19, 0.20),
    "gray_tr": (0.85, 0.17, 0.90, 0.22),
    "gray_bl": (0.14, 0.85, 0.19, 0.90),
    "gray_br": (0.84, 0.87, 0.89, 0.92),

    # Color reference patches for palette correction
    "yellow_patch": (0.10, 0.37, 0.20, 0.47),
    "navy_patch": (0.10, 0.58, 0.20, 0.68),
    "blue_patch": (0.80, 0.37, 0.90, 0.47),
    "red_patch": (0.80, 0.60, 0.90, 0.70),
    "pink_patch": (0.32, 0.82, 0.42, 0.92),
    "green_patch": (0.57, 0.82, 0.67, 0.92),

    # Skin tone area (central region)
    "skin_center": (0.35, 0.35, 0.70, 0.70),
}

# ROI group classifications
GRAY_PATCHES = ["gray_tl", "gray_tr", "gray_bl", "gray_br"]
COLOR_PATCHES = ["yellow_patch", "navy_patch", "blue_patch", "red_patch", "pink_patch", "green_patch"]
SKIN_PATCHES = ["skin_center"]
CHECKERBOARD_PATCHES = ["checkerboard"]

# ===== SHRINK RATIOS PER ROI TYPE =====
# Used to extract interior regions and avoid edges
SHRINK_RATIOS = {
    "gray_patches": 0.22,      # Tight shrink for gray patches (focus on uniform area)
    "skin_patches": 0.08,      # Loose shrink for skin (larger sample area)
    "checkerboard": 0.10,
    "color_patches": 0.12,     # Default for color patches
}

# ===== WHITE BALANCE REFERENCE (from training set) =====
# Target gray level computed as median of gray patches in training data
TARGET_GRAY_LEVEL = 127.5  # Median gray level from training set

# Gray patch reference colors (median RGB from training set)
GRAY_PATCHES_REFERENCE_DF = pd.DataFrame({
    "roi_name": ["gray_tl", "gray_tr", "gray_bl", "gray_br"],
    "r_ref": [128.0, 128.5, 127.0, 128.2],  # Example values; replace with actual training means
    "g_ref": [128.1, 128.3, 127.2, 128.1],
    "b_ref": [128.2, 128.4, 127.1, 128.0],
})

# ===== PALETTE CORRECTION REFERENCE (from training set) =====
# Reference colors for 6 color patches extracted from training set
REFERENCE_PALETTE_DF = pd.DataFrame({
    "roi_name": ["yellow_patch", "navy_patch", "blue_patch", "red_patch", "pink_patch", "green_patch"],
    "r_ref": [245.0, 35.0, 65.0, 215.0, 200.0, 80.0],      # Example; replace with actual training values
    "g_ref": [202.0, 50.0, 130.0, 80.0, 150.0, 165.0],
    "b_ref": [40.0, 110.0, 200.0, 70.0, 180.0, 70.0],
})

# ===== QUALITY ASSESSMENT THRESHOLDS =====
EXPOSURE_V_RANGE = (70, 225)              # HSV V (brightness) acceptable range
GRAY_SPREAD_MAX = 24.0                    # Max std of gray patch levels
WB_GRAY_IMPROVEMENT_MIN = 0.5             # Min improvement from WB correction
FINAL_COLOR_IMPROVEMENT_MIN = 1.0         # Min improvement from palette correction
FINAL_GRAY_DEGRADATION_TOL = 2.0          # Max tolerable gray degradation after palette correction

# ===== CAPTURE GATECHECK DEFAULTS =====
# These defaults are intentionally conservative and can be overridden through
# config.py environment variables for a specific Raspberry Pi/camera setup.
GATECHECK_MIN_GRAY_PATCHES = 2
GATECHECK_MIN_COLOR_PATCHES = 4
GATECHECK_MIN_BLUR_SCORE = 60.0
GATECHECK_MAX_RAW_PALETTE_MAE = 95.0
GATECHECK_MIN_CHECKERBOARD_SCORE = 35.0

# ===== CALIBRATION PARAMETERS =====
PALETTE_CORRECTION_STRENGTH = 0.55        # Blend factor: how aggressively to apply palette correction
PALETTE_DIAG_CLIP = (0.80, 1.20)         # Clipping for diagonal elements (channel gains)
PALETTE_OFFDIAG_CLIP = (-0.10, 0.10)     # Clipping for off-diagonal elements (cross-channel effects)
PALETTE_BIAS_CLIP = (-12.0, 12.0)        # Clipping for bias terms
WHITE_BALANCE_GAIN_CLIP = (0.6, 1.8)     # Clipping for channel gains in white balance


def get_roi_group(roi_name: str) -> str:
    """Classify ROI by type."""
    if roi_name in GRAY_PATCHES:
        return "gray"
    if roi_name in COLOR_PATCHES:
        return "color"
    if roi_name in SKIN_PATCHES:
        return "skin"
    if roi_name in CHECKERBOARD_PATCHES:
        return "checkerboard"
    return "other"


def get_shrink_ratio(roi_name: str) -> float:
    """Get shrink ratio for a given ROI name."""
    if roi_name in GRAY_PATCHES:
        return SHRINK_RATIOS["gray_patches"]
    if roi_name in SKIN_PATCHES:
        return SHRINK_RATIOS["skin_patches"]
    if roi_name in CHECKERBOARD_PATCHES:
        return SHRINK_RATIOS["checkerboard"]
    return SHRINK_RATIOS["color_patches"]
