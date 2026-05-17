"""
preprocessing.py

Core image preprocessing pipeline for bilirubin prediction.
Handles:
  1. Card detection and perspective alignment
  2. ROI extraction
  3. White balance correction
  4. Palette color correction
  5. Quality assessment and mode selection
"""

import cv2
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from pathlib import Path

from data_artifacts import (
    WARP_SIZE, TARGET_CHECKERBOARD_SIDE,
    ROI_CONFIG, GRAY_PATCHES, COLOR_PATCHES, SKIN_PATCHES,
    SHRINK_RATIOS,
    GRAY_PATCHES_REFERENCE_DF, REFERENCE_PALETTE_DF,
    EXPOSURE_V_RANGE, GRAY_SPREAD_MAX,
    WB_GRAY_IMPROVEMENT_MIN, FINAL_COLOR_IMPROVEMENT_MIN, FINAL_GRAY_DEGRADATION_TOL,
    PALETTE_CORRECTION_STRENGTH, PALETTE_DIAG_CLIP, PALETTE_OFFDIAG_CLIP, PALETTE_BIAS_CLIP,
    WHITE_BALANCE_GAIN_CLIP,
    get_roi_group, get_shrink_ratio
)

try:
    from config import (
        GATECHECK_REQUIRE_PALETTE,
        GATECHECK_MIN_GRAY_PATCHES,
        GATECHECK_MIN_COLOR_PATCHES,
        GATECHECK_MIN_BLUR_SCORE,
        GATECHECK_MAX_RAW_PALETTE_MAE,
        GATECHECK_MIN_CHECKERBOARD_SCORE,
    )
except Exception:
    from data_artifacts import (
        GATECHECK_MIN_GRAY_PATCHES,
        GATECHECK_MIN_COLOR_PATCHES,
        GATECHECK_MIN_BLUR_SCORE,
        GATECHECK_MAX_RAW_PALETTE_MAE,
        GATECHECK_MIN_CHECKERBOARD_SCORE,
    )

    GATECHECK_REQUIRE_PALETTE = True


# ===== CARD DETECTION & PERSPECTIVE ALIGNMENT =====

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points in standard order: top_left, top_right, bottom_right, bottom_left.
    Uses sum and diff of coordinates to identify corners.
    """
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


def detect_card_corners(image_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Detect card corners in image using edge detection and contour analysis.
    
    Returns:
        (corners: 4x2 array or None, edges: edge map)
    """
    h, w = image_bgr.shape[:2]
    image_area = h * w

    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # Morphological operations to close gaps
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    best_score = -1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Filter by minimum area (at least 8% of image)
        if area < 0.08 * image_area:
            continue

        rect = cv2.minAreaRect(cnt)
        (_, _), (rw, rh), _ = rect

        # Filter by minimum dimensions
        if rw < 80 or rh < 80:
            continue

        box = cv2.boxPoints(rect).astype("float32")
        box_area = rw * rh
        if box_area <= 0:
            continue

        # Score: prefer large contours, square-ish, and rectangular
        squareness = min(rw, rh) / max(rw, rh)
        rectangularity = min(area / box_area, 1.0)
        score = area * (0.7 * squareness + 0.3 * rectangularity)

        if score > best_score:
            best_score = score
            best_box = box

    return best_box, edges


def warp_card(image_bgr: np.ndarray, corners: np.ndarray, output_size: int = WARP_SIZE) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply perspective transform to warp card to fixed output_size x output_size square.
    
    Returns:
        (warped_image, ordered_corners, transform_matrix)
    """
    src = order_points(corners)
    dst = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image_bgr, M, (output_size, output_size))
    return warped, src, M


def checkerboard_score(roi_bgr: np.ndarray) -> float:
    """
    Score ROI for checkerboard-ness using Laplacian variance and edge transitions.
    Checkerboard has high frequency content and many black-white transitions.
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # High-frequency score
    lap_var = cv2.Laplacian(gray, cv2.CV_32F).var()

    # Transition score
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = bw.astype(np.float32) / 255.0

    trans_x = np.mean(np.abs(np.diff(bw, axis=1)))
    trans_y = np.mean(np.abs(np.diff(bw, axis=0)))

    return float(lap_var + 200.0 * (trans_x + trans_y))


def get_side_rois(warped_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """Extract ROIs from each side of the warped card."""
    h, w = warped_bgr.shape[:2]
    margin = int(0.18 * w)
    band = int(0.20 * h)

    return {
        "top": warped_bgr[0:band, margin:w - margin],
        "right": warped_bgr[margin:h - margin, w - band:w],
        "bottom": warped_bgr[h - band:h, margin:w - margin],
        "left": warped_bgr[margin:h - margin, 0:band],
    }


def orient_card_by_checkerboard(warped_bgr: np.ndarray, target_side: str = "top") -> Tuple[np.ndarray, str, Dict[str, float]]:
    """
    Detect checkerboard side and rotate image so checkerboard is at target_side.
    
    Returns:
        (oriented_image, detected_side, side_scores_dict)
    """
    side_rois = get_side_rois(warped_bgr)
    side_scores = {side: checkerboard_score(roi) for side, roi in side_rois.items()}

    detected_side = max(side_scores, key=side_scores.get)

    side_order = ["top", "right", "bottom", "left"]
    detected_idx = side_order.index(detected_side)
    target_idx = side_order.index(target_side)

    # np.rot90: rotates counter-clockwise
    k = (detected_idx - target_idx) % 4
    oriented = np.rot90(warped_bgr, k=k)
    oriented = np.ascontiguousarray(oriented)

    return oriented, detected_side, side_scores


# ===== ROI EXTRACTION & STATISTICS =====

def denormalize_roi(roi: Tuple[float, float, float, float], image_shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
    """Convert normalized ROI (0-1) to pixel coordinates."""
    h, w = image_shape[:2]
    x1 = int(roi[0] * w)
    y1 = int(roi[1] * h)
    x2 = int(roi[2] * w)
    y2 = int(roi[3] * h)
    return x1, y1, x2, y2


def shrink_roi(roi: Tuple[float, float, float, float], shrink_ratio: float = 0.12) -> Tuple[float, float, float, float]:
    """Shrink ROI inward by shrink_ratio factor to avoid edges."""
    x1, y1, x2, y2 = roi
    w = x2 - x1
    h = y2 - y1

    new_x1 = x1 + w * shrink_ratio
    new_y1 = y1 + h * shrink_ratio
    new_x2 = x2 - w * shrink_ratio
    new_y2 = y2 - h * shrink_ratio

    return (new_x1, new_y1, new_x2, new_y2)


def crop_roi(image_rgb: np.ndarray, roi: Tuple[float, float, float, float]) -> np.ndarray:
    """Crop normalized ROI from image."""
    x1, y1, x2, y2 = denormalize_roi(roi, image_rgb.shape)
    return image_rgb[y1:y2, x1:x2]


def blur_score_laplacian(image_rgb: np.ndarray) -> float:
    """Estimate focus sharpness using Laplacian variance."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ===== WHITE BALANCE =====

def extract_gray_patch_summary(image_rgb: np.ndarray, roi_config: Dict, gray_patch_names: List[str]) -> pd.DataFrame:
    """Extract median RGB values from gray patches."""
    rows = []

    for roi_name in gray_patch_names:
        shrink_ratio = get_shrink_ratio(roi_name)
        crop = crop_roi(image_rgb, shrink_roi(roi_config[roi_name], shrink_ratio))

        if crop.size == 0:
            continue

        pixels = crop.reshape(-1, 3).astype(np.float32)
        rows.append({
            "roi_name": roi_name,
            "r": float(np.median(pixels[:, 0])),
            "g": float(np.median(pixels[:, 1])),
            "b": float(np.median(pixels[:, 2])),
        })

    return pd.DataFrame(rows)


def fit_gray_white_balance(
    image_rgb: np.ndarray,
    roi_config: Dict,
    gray_patch_names: List[str],
    gray_reference_df: pd.DataFrame,
    gain_clip: Tuple[float, float] = WHITE_BALANCE_GAIN_CLIP
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute per-channel white balance gains from gray patches.

    Each observed gray patch is compared to its reference RGB value.
    Per-patch per-channel gains are computed; the median across patches is taken
    for robustness against individual patch extraction errors.

    Returns:
        (gains: 3-element array [r, g, b], gray_obs_df: observed gray values)
    """
    gray_obs_df = extract_gray_patch_summary(image_rgb, roi_config, gray_patch_names)

    if len(gray_obs_df) < 2:
        raise ValueError("Less than 2 valid gray patches for white balance.")

    merged = gray_obs_df.merge(gray_reference_df, on="roi_name", how="inner")

    if len(merged) >= 2:
        r_obs = merged["r"].to_numpy(dtype=np.float32)
        g_obs = merged["g"].to_numpy(dtype=np.float32)
        b_obs = merged["b"].to_numpy(dtype=np.float32)
        r_ref = merged["r_ref"].to_numpy(dtype=np.float32)
        g_ref = merged["g_ref"].to_numpy(dtype=np.float32)
        b_ref = merged["b_ref"].to_numpy(dtype=np.float32)
        gains = np.array([
            float(np.median(r_ref / np.maximum(r_obs, 1e-6))),
            float(np.median(g_ref / np.maximum(g_obs, 1e-6))),
            float(np.median(b_ref / np.maximum(b_obs, 1e-6))),
        ], dtype=np.float32)
    else:
        # Fallback: scalar target from reference median
        r_ref = float(gray_reference_df["r_ref"].median())
        g_ref = float(gray_reference_df["g_ref"].median())
        b_ref = float(gray_reference_df["b_ref"].median())
        gains = np.array([
            r_ref / max(float(gray_obs_df["r"].median()), 1e-6),
            g_ref / max(float(gray_obs_df["g"].median()), 1e-6),
            b_ref / max(float(gray_obs_df["b"].median()), 1e-6),
        ], dtype=np.float32)

    gains = np.clip(gains, gain_clip[0], gain_clip[1])
    return gains, gray_obs_df


def apply_channel_gains(image_rgb: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """Apply per-channel gain correction."""
    corrected = image_rgb.astype(np.float32).copy()
    corrected[..., 0] *= gains[0]
    corrected[..., 1] *= gains[1]
    corrected[..., 2] *= gains[2]
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected


def gray_neutrality_score(gray_obs_df: pd.DataFrame) -> float:
    """Compute gray neutrality as std of channel medians."""
    channel_medians = np.array([
        gray_obs_df["r"].median(),
        gray_obs_df["g"].median(),
        gray_obs_df["b"].median(),
    ], dtype=np.float32)
    return float(np.std(channel_medians))


# ===== PALETTE COLOR CORRECTION =====

def extract_patch_medians(image_rgb: np.ndarray, roi_config: Dict, patch_names: List[str]) -> pd.DataFrame:
    """Extract median RGB values from color patches."""
    rows = []

    for roi_name in patch_names:
        shrink_ratio = get_shrink_ratio(roi_name)
        crop = crop_roi(image_rgb, shrink_roi(roi_config[roi_name], shrink_ratio))

        if crop.size == 0:
            continue

        pixels = crop.reshape(-1, 3).astype(np.float32)
        rows.append({
            "roi_name": roi_name,
            "r_obs": float(np.median(pixels[:, 0])),
            "g_obs": float(np.median(pixels[:, 1])),
            "b_obs": float(np.median(pixels[:, 2])),
        })

    return pd.DataFrame(rows)


def evaluate_patch_error(observed_patch_df: pd.DataFrame, reference_patch_df: pd.DataFrame) -> pd.DataFrame:
    """Compute color mismatch between observed and reference patches."""
    merged = observed_patch_df.merge(reference_patch_df, on="roi_name", how="inner").copy()

    merged["abs_err_r"] = np.abs(merged["r_obs"] - merged["r_ref"])
    merged["abs_err_g"] = np.abs(merged["g_obs"] - merged["g_ref"])
    merged["abs_err_b"] = np.abs(merged["b_obs"] - merged["b_ref"])
    merged["mean_abs_err_rgb"] = merged[["abs_err_r", "abs_err_g", "abs_err_b"]].mean(axis=1)

    return merged


def fit_palette_transform_regularized(
    observed_patch_df: pd.DataFrame,
    reference_patch_df: pd.DataFrame,
    correction_strength: float = PALETTE_CORRECTION_STRENGTH,
    diag_clip: Tuple[float, float] = PALETTE_DIAG_CLIP,
    offdiag_clip: Tuple[float, float] = PALETTE_OFFDIAG_CLIP,
    bias_clip: Tuple[float, float] = PALETTE_BIAS_CLIP,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Fit affine color transform (3x3 + bias) from observed to reference colors.
    Regularized to avoid over-correction.
    
    Returns:
        (transform_matrix: 4x3, merged_data)
    """
    merged = observed_patch_df.merge(reference_patch_df, on="roi_name", how="inner")

    if len(merged) < 4:
        raise ValueError("Need at least 4 reference patches for stable color transform.")

    X = merged[["r_obs", "g_obs", "b_obs"]].to_numpy(dtype=np.float32)
    Y = merged[["r_ref", "g_ref", "b_ref"]].to_numpy(dtype=np.float32)

    # Augment X with bias term
    X_aug = np.concatenate([X, np.ones((len(X), 1), dtype=np.float32)], axis=1)

    # Least squares fit
    T_est, _, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)

    # Identity transform (no correction)
    T_identity = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float32)

    # Blend toward identity to avoid over-correction
    T = T_identity + correction_strength * (T_est - T_identity)

    # Clip for stability
    for i in range(3):
        T[i, i] = np.clip(T[i, i], diag_clip[0], diag_clip[1])

    for r in range(3):
        for c in range(3):
            if r != c:
                T[r, c] = np.clip(T[r, c], offdiag_clip[0], offdiag_clip[1])

    T[3, :] = np.clip(T[3, :], bias_clip[0], bias_clip[1])

    return T, merged


def apply_palette_transform(image_rgb: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """Apply affine color transform to image."""
    h, w = image_rgb.shape[:2]

    pixels = image_rgb.reshape(-1, 3).astype(np.float32)
    pixels_aug = np.concatenate([pixels, np.ones((len(pixels), 1), dtype=np.float32)], axis=1)

    corrected = pixels_aug @ transform_matrix
    corrected = np.clip(corrected, 0, 255).reshape(h, w, 3).astype(np.uint8)

    return corrected


# ===== QUALITY ASSESSMENT =====

def get_skin_brightness_v(image_rgb: np.ndarray, roi_config: Dict) -> float:
    """Get brightness (V in HSV) of skin region."""
    crop = crop_roi(image_rgb, shrink_roi(roi_config["skin_center"], get_shrink_ratio("skin_center")))
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    return float(np.median(hsv[..., 2]))


def gray_level_spread_score(gray_df: pd.DataFrame) -> float:
    """Compute std of gray levels across patches."""
    gray_levels = gray_df[["r", "g", "b"]].mean(axis=1).to_numpy(dtype=np.float32)
    return float(np.std(gray_levels))


def choose_calibration_mode(metrics: Dict) -> Tuple[str, str, int, Dict]:
    """
    Choose preprocessing mode based on quality metrics.
    
    Returns:
        (selected_mode, quality_label, quality_score, quality_flags_dict)
    """
    exposure_ok = EXPOSURE_V_RANGE[0] <= metrics["skin_v_median"] <= EXPOSURE_V_RANGE[1]
    placement_ok = metrics["gray_level_spread_raw"] <= GRAY_SPREAD_MAX
    wb_improves_gray = metrics["gray_std_wb"] <= (metrics["gray_std_raw"] - WB_GRAY_IMPROVEMENT_MIN)
    final_improves_color = metrics["patch_mae_final"] <= (
        min(metrics["patch_mae_raw"], metrics["patch_mae_wb"]) - FINAL_COLOR_IMPROVEMENT_MIN
    )
    final_keeps_gray_stable = metrics["gray_std_final"] <= (metrics["gray_std_wb"] + FINAL_GRAY_DEGRADATION_TOL)

    quality_flags = {
        "exposure_ok": exposure_ok,
        "placement_ok": placement_ok,
        "wb_improves_gray": wb_improves_gray,
        "final_improves_color": final_improves_color,
    }
    quality_score = int(sum(quality_flags.values()) * 25)

    # Decide mode based on flags
    if exposure_ok and placement_ok and final_improves_color and final_keeps_gray_stable:
        mode = "wb_plus_palette"
    elif placement_ok and wb_improves_gray:
        mode = "white_balance_only"
    else:
        mode = "raw_aligned"

    # Quality label
    if quality_score >= 75:
        quality_label = "high"
    elif quality_score >= 50:
        quality_label = "medium"
    else:
        quality_label = "low"

    return mode, quality_label, quality_score, quality_flags


# ===== MAIN PREPROCESSING CLASS =====

class BilirubinPreprocessor:
    """
    Complete preprocessing pipeline for bilirubin images.
    
    Handles: card detection, alignment, white balance, palette correction, quality assessment.
    """

    def __init__(
        self,
        roi_config: Dict = None,
        reference_palette_df: pd.DataFrame = None,
        gray_reference_df: pd.DataFrame = None,
    ):
        self.roi_config = ROI_CONFIG if roi_config is None else roi_config
        self.reference_palette_df = REFERENCE_PALETTE_DF if reference_palette_df is None else reference_palette_df
        self.gray_reference_df = GRAY_PATCHES_REFERENCE_DF if gray_reference_df is None else gray_reference_df
        self.last_error = None

    def preprocess_image(
        self,
        image_bgr: np.ndarray,
        return_diagnostics: bool = False
    ) -> Tuple[Optional[np.ndarray], str, Dict]:
        """
        Complete preprocessing pipeline: detect card -> align -> assess quality -> apply corrections.
        
        Returns:
            (preprocessed_image_rgb or None, applied_mode_string, diagnostics_dict)
        """
        try:
            def gate_failure(mode: str, message: str, diagnostics: Dict) -> Tuple[None, str, Dict]:
                self.last_error = message
                # Preserve palette_detected from quality_flags if already computed;
                # don't unconditionally overwrite to False when something else failed.
                palette_val = diagnostics.get("quality_flags", {}).get("palette_detected", False)
                diagnostics.update({
                    "error": message,
                    "selected_mode": mode,
                    "quality_label": "failed",
                    "quality_score": 0,
                    "gatecheck_passed": False,
                    "palette_detected": palette_val,
                })
                return None, mode, diagnostics if return_diagnostics else {"error": message}

            # Step 1: Card detection and alignment
            corners, edges = detect_card_corners(image_bgr)
            if corners is None:
                diagnostics = {
                    "gatecheck_errors": ["Kartu kalibrasi tidak terdeteksi."],
                    "gatecheck_warnings": [],
                    "metrics": {},
                    "quality_flags": {"card_detected": False},
                }
                return gate_failure("card_not_detected", "Card not detected", diagnostics)

            # Step 2: Perspective warp and orientation
            warped_bgr, _, _ = warp_card(image_bgr, corners, output_size=WARP_SIZE)
            oriented_bgr, detected_side, side_scores = orient_card_by_checkerboard(
                warped_bgr, target_side=TARGET_CHECKERBOARD_SIDE
            )
            side_scores = {side: float(score) for side, score in side_scores.items()}
            aligned_rgb = cv2.cvtColor(oriented_bgr, cv2.COLOR_BGR2RGB)
            raw_rgb = aligned_rgb.copy()

            # Step 3: Gatecheck and quality assessment
            gatecheck_errors: List[str] = []
            gatecheck_warnings: List[str] = []

            checkerboard_score_max = max(side_scores.values()) if side_scores else 0.0
            if checkerboard_score_max < GATECHECK_MIN_CHECKERBOARD_SCORE:
                gatecheck_errors.append("Checkerboard pada kartu kalibrasi tidak cukup jelas.")
            else:
                sorted_scores = sorted(side_scores.values(), reverse=True)
                if len(sorted_scores) >= 2 and sorted_scores[1] > 0:
                    if sorted_scores[0] / sorted_scores[1] < 1.5:
                        gatecheck_warnings.append(
                            "Orientasi kartu tidak pasti — posisikan checkerboard agar lebih terlihat jelas di salah satu sisi."
                        )

            blur_score = blur_score_laplacian(raw_rgb)
            if blur_score < GATECHECK_MIN_BLUR_SCORE:
                gatecheck_errors.append("Foto terlalu blur. Ulangi capture dengan kamera lebih stabil.")

            skin_crop = crop_roi(raw_rgb, shrink_roi(self.roi_config["skin_center"], get_shrink_ratio("skin_center")))
            if skin_crop.size == 0:
                gatecheck_errors.append("Area kulit bayi tidak valid atau berada di luar kartu.")
                skin_v_median = 0.0
            else:
                skin_v_median = get_skin_brightness_v(raw_rgb, self.roi_config)
                if not (EXPOSURE_V_RANGE[0] <= skin_v_median <= EXPOSURE_V_RANGE[1]):
                    gatecheck_errors.append("Exposure foto tidak sesuai. Atur pencahayaan lalu ambil ulang.")

            gray_raw = extract_gray_patch_summary(raw_rgb, self.roi_config, GRAY_PATCHES)
            if len(gray_raw) < GATECHECK_MIN_GRAY_PATCHES:
                gatecheck_errors.append("Gray patches pada kartu kalibrasi tidak cukup terbaca.")

            color_raw = extract_patch_medians(raw_rgb, self.roi_config, COLOR_PATCHES)
            if len(color_raw) < GATECHECK_MIN_COLOR_PATCHES:
                gatecheck_errors.append("Color palette pada kartu kalibrasi tidak cukup terbaca.")

            metrics = {
                "checkerboard_score": checkerboard_score_max,
                "blur_score": blur_score,
                "skin_v_median": skin_v_median,
                "gray_patch_count": int(len(gray_raw)),
                "color_patch_count": int(len(color_raw)),
            }

            patch_mae_raw = None
            palette_detected = False
            if len(color_raw) >= GATECHECK_MIN_COLOR_PATCHES:
                err_raw = evaluate_patch_error(color_raw, self.reference_palette_df)
                if len(err_raw) > 0:
                    patch_mae_raw = float(err_raw["mean_abs_err_rgb"].mean())
                    palette_detected = patch_mae_raw <= GATECHECK_MAX_RAW_PALETTE_MAE
                    metrics["patch_mae_raw"] = patch_mae_raw

            if GATECHECK_REQUIRE_PALETTE and not palette_detected:
                gatecheck_errors.append("Color palette tidak terdeteksi atau tidak cocok dengan referensi.")

            quality_flags = {
                "card_detected": True,
                "checkerboard_ok": checkerboard_score_max >= GATECHECK_MIN_CHECKERBOARD_SCORE,
                "blur_ok": blur_score >= GATECHECK_MIN_BLUR_SCORE,
                "skin_roi_ok": skin_crop.size > 0,
                "exposure_ok": EXPOSURE_V_RANGE[0] <= skin_v_median <= EXPOSURE_V_RANGE[1],
                "gray_patches_ok": len(gray_raw) >= GATECHECK_MIN_GRAY_PATCHES,
                "palette_patches_ok": len(color_raw) >= GATECHECK_MIN_COLOR_PATCHES,
                "palette_detected": palette_detected,
            }

            if gatecheck_errors:
                diagnostics = {
                    "gatecheck_errors": gatecheck_errors,
                    "gatecheck_warnings": gatecheck_warnings,
                    "metrics": metrics,
                    "quality_flags": quality_flags,
                    "detected_checkerboard_side": detected_side,
                    "side_scores": side_scores,
                }
                return gate_failure("gatecheck_failed", "Capture gatecheck failed", diagnostics)

            gray_std_raw = gray_neutrality_score(gray_raw)
            gray_level_spread_raw = gray_level_spread_score(gray_raw)

            # White balance only
            wb_gains, _gray_obs = fit_gray_white_balance(
                raw_rgb, self.roi_config, GRAY_PATCHES, self.gray_reference_df
            )
            wb_rgb = apply_channel_gains(raw_rgb, wb_gains)
            # Measure neutrality on the WB-corrected image (not pre-WB observations)
            gray_wb = extract_gray_patch_summary(wb_rgb, self.roi_config, GRAY_PATCHES)
            gray_std_wb = gray_neutrality_score(gray_wb)

            err_wb = evaluate_patch_error(
                extract_patch_medians(wb_rgb, self.roi_config, COLOR_PATCHES),
                self.reference_palette_df
            )
            patch_mae_wb = float(err_wb["mean_abs_err_rgb"].mean())

            # White balance + palette
            palette_transform, _ = fit_palette_transform_regularized(
                extract_patch_medians(wb_rgb, self.roi_config, COLOR_PATCHES),
                self.reference_palette_df,
                correction_strength=PALETTE_CORRECTION_STRENGTH
            )
            final_rgb = apply_palette_transform(wb_rgb, palette_transform)
            gray_final = extract_gray_patch_summary(final_rgb, self.roi_config, GRAY_PATCHES)
            gray_std_final = gray_neutrality_score(gray_final)

            err_final = evaluate_patch_error(
                extract_patch_medians(final_rgb, self.roi_config, COLOR_PATCHES),
                self.reference_palette_df
            )
            patch_mae_final = float(err_final["mean_abs_err_rgb"].mean())

            metrics.update({
                "gray_std_raw": gray_std_raw,
                "gray_std_wb": gray_std_wb,
                "gray_std_final": gray_std_final,
                "gray_level_spread_raw": gray_level_spread_raw,
                "patch_mae_wb": patch_mae_wb,
                "patch_mae_final": patch_mae_final,
            })

            # Step 4: Choose calibration mode
            selected_mode, quality_label, quality_score, calibration_flags = choose_calibration_mode(metrics)
            quality_flags.update(calibration_flags)
            if quality_label == "low":
                gatecheck_warnings.append("Kualitas foto rendah meski lolos gatecheck.")

            # Step 5: Return selected image
            if selected_mode == "raw_aligned":
                output_rgb = raw_rgb
            elif selected_mode == "white_balance_only":
                output_rgb = wb_rgb
            else:  # wb_plus_palette
                output_rgb = final_rgb

            diagnostics = {
                "error": None,
                "selected_mode": selected_mode,
                "quality_label": quality_label,
                "quality_score": quality_score,
                "quality_flags": quality_flags,
                "gatecheck_passed": True,
                "gatecheck_errors": gatecheck_errors,
                "gatecheck_warnings": gatecheck_warnings,
                "palette_detected": palette_detected,
                "metrics": metrics,
                "detected_checkerboard_side": detected_side,
                "side_scores": side_scores,
            } if return_diagnostics else {}

            return output_rgb, selected_mode, diagnostics

        except Exception as e:
            self.last_error = str(e)
            return None, "error", {"error": self.last_error}

    def preprocess_image_file(
        self,
        image_path: str,
        return_diagnostics: bool = False
    ) -> Tuple[Optional[np.ndarray], str, Dict]:
        """Preprocess image from file path."""
        try:
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                self.last_error = f"Failed to read image: {image_path}"
                return None, "file_read_error", {"error": self.last_error}
            
            return self.preprocess_image(image_bgr, return_diagnostics=return_diagnostics)
        
        except Exception as e:
            self.last_error = str(e)
            return None, "error", {"error": self.last_error}
