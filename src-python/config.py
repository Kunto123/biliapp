# config.py
# Configuration file for Bilirubin Prediction System.

from pathlib import Path
import os


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_resolution(name: str, default: tuple[int, int]) -> tuple[int, int]:
    value = os.getenv(name)
    if not value:
        return default
    normalized = value.lower().replace(",", "x").replace(" ", "")
    try:
        width, height = normalized.split("x", 1)
        return int(width), int(height)
    except (ValueError, TypeError):
        return default


def _env_rotation(name: str, default: int) -> int:
    value = _env_int(name, default)
    return value if value in {0, 90, 180, 270} else default

# ===== PATHS =====
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
IMAGES_DIR = PROJECT_ROOT / "data" / "captures"
MODELS_DIR = PROJECT_ROOT

# ===== DEVICE PROFILE =====
DEVICE_PROFILE = os.getenv("BILIRUBIN_DEVICE", "desktop").strip().lower()
IS_RASPBERRY_PI = DEVICE_PROFILE in {"raspi5", "raspberrypi5", "raspberry_pi_5", "pi5"}

# Model paths
MODEL_STAGE1_PATH = MODELS_DIR / "best_model_stage1.keras"
MODEL_STAGE2_PATH = MODELS_DIR / "best_model_stage2.keras"
MODEL_STAGE1_TFLITE_PATH = MODELS_DIR / "models" / "best_model_stage1.tflite"
MODEL_STAGE2_TFLITE_PATH = MODELS_DIR / "models" / "best_model_stage2.tflite"

# ===== MODEL CONFIGURATION =====
MODEL_BACKEND = os.getenv("BILIRUBIN_MODEL_BACKEND", "tflite" if IS_RASPBERRY_PI else "keras").strip().lower()
USE_STAGE2 = _env_bool("BILIRUBIN_USE_STAGE2", not IS_RASPBERRY_PI)
MODEL_INPUT_SIZE = (224, 224)  # Input size for EfficientNetB0

# ===== CAMERA CONFIGURATION =====
CAMERA_TYPE = os.getenv("BILIRUBIN_CAMERA_TYPE", "libcamera" if IS_RASPBERRY_PI else "opencv").strip().lower()
CAMERA_INDEX = _env_int("BILIRUBIN_CAMERA_INDEX", 0)
CAMERA_RESOLUTION = _env_resolution(
    "BILIRUBIN_CAMERA_RESOLUTION",
    (1920, 1080) if IS_RASPBERRY_PI else (3840, 2160),
)
CAMERA_PREVIEW_RESOLUTION = _env_resolution("BILIRUBIN_CAMERA_PREVIEW_RESOLUTION", (640, 480))
CAMERA_ROTATION = _env_rotation("BILIRUBIN_CAMERA_ROTATION", 180)
CAMERA_AUTO_EXPOSURE = _env_bool("BILIRUBIN_CAMERA_AUTO_EXPOSURE", True)
CAMERA_BRIGHTNESS = _env_float("BILIRUBIN_CAMERA_BRIGHTNESS", 0.0)
CAMERA_TIMEOUT_SECONDS = _env_float("BILIRUBIN_CAMERA_TIMEOUT_SECONDS", 8.0 if IS_RASPBERRY_PI else 20.0)
PREVIEW_POLL_MS = _env_int("BILIRUBIN_PREVIEW_POLL_MS", 33 if IS_RASPBERRY_PI else 250)
PREVIEW_JPEG_QUALITY = _env_int("BILIRUBIN_PREVIEW_JPEG_QUALITY", 65 if IS_RASPBERRY_PI else 70)
PREVIEW_FPS = _env_int("BILIRUBIN_PREVIEW_FPS", 30)
PREVIEW_MIN_FPS = _env_int("BILIRUBIN_PREVIEW_MIN_FPS", 30)

# ===== PREPROCESSING =====
PREPROCESSING_TARGET_SIZE = 512  # Warp card to 512x512
PREPROCESSING_CHECKERBOARD_SIDE = "top"

# ===== LOGGING =====
USE_CSV_LOGGING = True
USE_SQLITE_LOGGING = False
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

# ===== UI CONFIGURATION =====
UI_WINDOW_WIDTH = 800
UI_WINDOW_HEIGHT = 600
UI_FONT_SIZE_LARGE = 18
UI_FONT_SIZE_MEDIUM = 14
UI_FONT_SIZE_SMALL = 12

# ===== CLEANUP POLICY =====
CLEANUP_IMAGES_OLDER_THAN_DAYS = 7  # Delete images older than this
AUTO_CLEANUP_ON_STARTUP = False

# ===== QUALITY THRESHOLDS =====
QUALITY_SCORE_HIGH = 75  # >=75 is "high" quality
QUALITY_SCORE_MEDIUM = 50  # >=50 is "medium" quality
QUALITY_SCORE_LOW = 0   # <50 is "low" quality

# ===== GATECHECK SETTINGS =====
GATECHECK_REQUIRE_PALETTE = _env_bool("BILIRUBIN_REQUIRE_PALETTE", True)
GATECHECK_MIN_GRAY_PATCHES = _env_int("BILIRUBIN_MIN_GRAY_PATCHES", 2)
GATECHECK_MIN_COLOR_PATCHES = _env_int("BILIRUBIN_MIN_COLOR_PATCHES", 4)
GATECHECK_MIN_BLUR_SCORE = _env_float("BILIRUBIN_MIN_BLUR_SCORE", 60.0)
GATECHECK_MAX_RAW_PALETTE_MAE = _env_float("BILIRUBIN_MAX_RAW_PALETTE_MAE", 95.0)
GATECHECK_MIN_CHECKERBOARD_SCORE = _env_float("BILIRUBIN_MIN_CHECKERBOARD_SCORE", 35.0)

# ===== INFERENCE SETTINGS =====
ENABLE_INFERENCE_TIME_LOGGING = True  # Log inference latency
BATCH_SIZE = 1  # For inference

print("[Config] Configuration loaded from:", __file__)
print(
    "[Config] device=%s backend=%s camera=%s resolution=%sx%s rotation=%s"
    % (
        DEVICE_PROFILE,
        MODEL_BACKEND,
        CAMERA_TYPE,
        CAMERA_RESOLUTION[0],
        CAMERA_RESOLUTION[1],
        CAMERA_ROTATION,
    )
)
