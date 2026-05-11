"""
Runtime camera settings persisted outside source-controlled config.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from config import (
    CAMERA_INDEX,
    CAMERA_PREVIEW_RESOLUTION,
    CAMERA_RESOLUTION,
    CAMERA_ROTATION,
    CAMERA_TYPE,
    PREVIEW_FPS,
    PREVIEW_JPEG_QUALITY,
    PREVIEW_MIN_FPS,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS_PATH = BASE_DIR / "data" / "camera_settings.json"
VALID_CAMERA_TYPES = {"libcamera", "opencv", "pi_legacy"}
VALID_ROTATIONS = {0, 90, 180, 270}
MAX_RESOLUTION_DIMENSION = 8192
MAX_FPS = 120


def _resolution_dict(resolution: tuple[int, int]) -> dict[str, int]:
    return {"width": int(resolution[0]), "height": int(resolution[1])}


def default_camera_settings() -> dict[str, Any]:
    return {
        "camera_type": CAMERA_TYPE,
        "camera_index": int(CAMERA_INDEX),
        "capture_resolution": _resolution_dict(CAMERA_RESOLUTION),
        "preview_resolution": _resolution_dict(CAMERA_PREVIEW_RESOLUTION),
        "fps": int(PREVIEW_FPS),
        "min_fps": int(PREVIEW_MIN_FPS),
        "jpeg_quality": int(PREVIEW_JPEG_QUALITY),
        "rotation": int(CAMERA_ROTATION),
    }


def _coerce_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be an integer")


def _validate_resolution(value: Any, field: str) -> dict[str, int]:
    if isinstance(value, dict):
        width = _coerce_int(value.get("width"), f"{field}.width")
        height = _coerce_int(value.get("height"), f"{field}.height")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        width = _coerce_int(value[0], f"{field}.width")
        height = _coerce_int(value[1], f"{field}.height")
    else:
        raise ValueError(f"{field} must contain width and height")

    if width <= 0 or height <= 0:
        raise ValueError(f"{field} dimensions must be positive")
    if width > MAX_RESOLUTION_DIMENSION or height > MAX_RESOLUTION_DIMENSION:
        raise ValueError(f"{field} dimensions are too large")

    return {"width": width, "height": height}


def normalize_camera_settings(raw: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    settings = default_camera_settings()
    if raw:
        settings.update({k: v for k, v in raw.items() if v is not None})

    camera_type = str(settings.get("camera_type", "")).strip().lower()
    if camera_type not in VALID_CAMERA_TYPES:
        raise ValueError(f"camera_type must be one of: {', '.join(sorted(VALID_CAMERA_TYPES))}")

    camera_index = _coerce_int(settings.get("camera_index"), "camera_index")
    if camera_index < 0:
        raise ValueError("camera_index must be >= 0")

    fps = _coerce_int(settings.get("fps"), "fps")
    if fps < 0 or fps > MAX_FPS:
        raise ValueError(f"fps must be between 0 and {MAX_FPS}")

    min_fps = _coerce_int(settings.get("min_fps"), "min_fps")
    if min_fps < 1 or min_fps > MAX_FPS:
        raise ValueError(f"min_fps must be between 1 and {MAX_FPS}")

    jpeg_quality = _coerce_int(settings.get("jpeg_quality"), "jpeg_quality")
    if jpeg_quality < 1 or jpeg_quality > 100:
        raise ValueError("jpeg_quality must be between 1 and 100")

    rotation = _coerce_int(settings.get("rotation"), "rotation")
    if rotation not in VALID_ROTATIONS:
        raise ValueError("rotation must be one of: 0, 90, 180, 270")

    return {
        "camera_type": camera_type,
        "camera_index": camera_index,
        "capture_resolution": _validate_resolution(settings.get("capture_resolution"), "capture_resolution"),
        "preview_resolution": _validate_resolution(settings.get("preview_resolution"), "preview_resolution"),
        "fps": fps,
        "min_fps": min_fps,
        "jpeg_quality": jpeg_quality,
        "rotation": rotation,
    }


def load_camera_settings(path: Path = DEFAULT_SETTINGS_PATH) -> tuple[dict[str, Any], str]:
    if not path.exists():
        return normalize_camera_settings(), "defaults"

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return normalize_camera_settings(raw), "file"


def get_camera_settings(path: Path = DEFAULT_SETTINGS_PATH) -> dict[str, Any]:
    settings, _source = load_camera_settings(path)
    return settings


def save_camera_settings(settings: dict[str, Any], path: Path = DEFAULT_SETTINGS_PATH) -> dict[str, Any]:
    normalized = normalize_camera_settings(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=2)
        handle.write("\n")
    return normalized


def resolution_tuple(value: dict[str, int]) -> tuple[int, int]:
    return int(value["width"]), int(value["height"])
