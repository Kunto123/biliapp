"""
api_server.py

FastAPI REST server untuk Tauri Bilirubin frontend.
Jalankan dari root bili-app/:  python src-python/api_server.py
Port: 127.0.0.1:7878
"""

import sys
import base64
import cv2
import time
import threading
from pathlib import Path
from typing import Optional

# src-python/ ada di sys.path agar semua modul pipeline bisa diimport
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

# BASE_DIR = bili-app/ (parent dari src-python/)
BASE_DIR = SRC_DIR.parent

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from main_pipeline import BilirubinPredictionPipeline
from config import (
    CAMERA_PREVIEW_RESOLUTION,
    CAMERA_TYPE,
    DEVICE_PROFILE,
    GATECHECK_MIN_BLUR_SCORE,
    MODEL_BACKEND,
    MODEL_STAGE1_TFLITE_PATH,
    MODEL_STAGE2_TFLITE_PATH,
    PREVIEW_JPEG_QUALITY,
    PREVIEW_POLL_MS,
    USE_STAGE2,
)

app = FastAPI(title="Bilirubin API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PORT = 7878

pipeline: Optional[BilirubinPredictionPipeline] = None
preview_cache_b64: Optional[str] = None
preview_cache_at: float = 0.0
preview_focus_score: Optional[float] = None
preview_focus_ok: Optional[bool] = None
camera_lock = threading.RLock()
capture_in_progress = False


def _camera_is_available() -> bool:
    return pipeline is not None and pipeline.camera is not None and pipeline.camera.is_open


def _prepare_preview_frame(frame):
    width, height = CAMERA_PREVIEW_RESOLUTION
    if frame.shape[1] != width or frame.shape[0] != height:
        return cv2.resize(frame, (width, height))
    return frame


def _calculate_focus_score(frame) -> float:
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _encode_preview_frame(frame):
    frame = _prepare_preview_frame(frame)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY])
    return base64.b64encode(buf).decode()


def _update_preview_cache(frame, timestamp: Optional[float] = None) -> None:
    global preview_cache_b64, preview_cache_at, preview_focus_score, preview_focus_ok

    preview_frame = _prepare_preview_frame(frame)
    preview_focus_score = _calculate_focus_score(preview_frame)
    preview_focus_ok = preview_focus_score >= GATECHECK_MIN_BLUR_SCORE
    _, buf = cv2.imencode(".jpg", preview_frame, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY])
    preview_cache_b64 = base64.b64encode(buf).decode()
    preview_cache_at = timestamp if timestamp is not None else time.monotonic()


def _preview_payload(frame_b64: Optional[str], available: bool, **extra):
    payload = {
        "frame": frame_b64,
        "available": available,
        "focus_score": preview_focus_score,
        "focus_ok": preview_focus_ok,
    }
    payload.update(extra)
    return payload


def _reset_camera_if_needed(force: bool = False) -> bool:
    if pipeline is None:
        return False

    camera = getattr(pipeline, "camera", None)
    if camera is not None and camera.is_open and not force:
        return True

    try:
        if camera is not None:
            camera.release()
        pipeline.camera = pipeline._init_configured_camera()
        return _camera_is_available()
    except Exception as exc:
        pipeline.last_error = f"Camera reconnect failed: {exc}"
        return False


@app.on_event("startup")
async def startup():
    global pipeline
    m1 = BASE_DIR / "best_model_stage1.keras"
    m2 = BASE_DIR / "best_model_stage2.keras"
    print(f"[api] BASE_DIR : {BASE_DIR}")
    print(f"[api] Stage1   : {m1} (exists={m1.exists()})")
    print(f"[api] Stage2   : {m2} (exists={m2.exists()})")
    try:
        pipeline = BilirubinPredictionPipeline(
            model_stage1_path=str(m1),
            model_stage2_path=str(m2) if m2.exists() else None,
            use_stage2=USE_STAGE2 and m2.exists(),
            logs_dir=str(BASE_DIR / "logs"),
            images_dir=str(BASE_DIR / "data" / "captures"),
            model_backend=MODEL_BACKEND,
            tflite_stage1_path=str(MODEL_STAGE1_TFLITE_PATH),
            tflite_stage2_path=str(MODEL_STAGE2_TFLITE_PATH),
        )
        print("[api] ✓ Pipeline initialized")
    except Exception as e:
        print(f"[api] ✗ Pipeline init failed: {e}")

@app.on_event("shutdown")
async def shutdown():
    with camera_lock:
        if pipeline:
            pipeline.cleanup()


# ── Status ────────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    if pipeline is None:
        return {"initialized": False, "error": "Pipeline not ready"}
    status = pipeline.get_system_status()
    status["initialized"] = True
    status["runtime_config"] = {
        "device_profile": DEVICE_PROFILE,
        "model_backend": MODEL_BACKEND,
        "camera_type": CAMERA_TYPE,
        "preview_poll_ms": PREVIEW_POLL_MS,
        "preview_resolution": CAMERA_PREVIEW_RESOLUTION,
        "preview_jpeg_quality": PREVIEW_JPEG_QUALITY,
        "use_stage2": USE_STAGE2,
    }
    # Pastikan serializable
    for k, v in list(status.items()):
        if not isinstance(v, (str, int, float, bool, dict, list, type(None))):
            status[k] = str(v)
    return status


# ── Camera ────────────────────────────────────────────────────────────────────

@app.get("/api/camera/frame")
async def get_camera_frame():
    now = time.monotonic()
    min_interval = PREVIEW_POLL_MS / 1000.0

    if preview_cache_b64 and (now - preview_cache_at) < min_interval:
        return _preview_payload(preview_cache_b64, True, cached=True)

    if capture_in_progress:
        return _preview_payload(preview_cache_b64, preview_cache_b64 is not None, busy=True)

    if not camera_lock.acquire(blocking=False):
        return _preview_payload(preview_cache_b64, preview_cache_b64 is not None, busy=True)

    try:
        if not _camera_is_available() and not _reset_camera_if_needed():
            return _preview_payload(None, False)

        frame = pipeline.camera.capture_image()
        if frame is None:
            error = pipeline.camera.error_message
            _reset_camera_if_needed(force=True)
            return _preview_payload(
                preview_cache_b64,
                preview_cache_b64 is not None,
                error=error,
            )

        _update_preview_cache(frame, now)
        return _preview_payload(preview_cache_b64, True, cached=False)
    finally:
        camera_lock.release()


@app.post("/api/camera/reconnect")
async def reconnect_camera():
    global preview_cache_b64, preview_cache_at, preview_focus_score, preview_focus_ok
    with camera_lock:
        preview_cache_b64 = None
        preview_cache_at = 0.0
        preview_focus_score = None
        preview_focus_ok = None
        if pipeline is not None:
            pipeline.cleanup()
            pipeline.camera = pipeline._init_configured_camera()
        ok = _camera_is_available()
        return {"success": ok}


# ── Prediction ────────────────────────────────────────────────────────────────

@app.post("/api/capture")
async def capture_and_predict():
    global capture_in_progress
    if pipeline is None:
        return {"success": False, "error": "Pipeline tidak diinisialisasi"}

    with camera_lock:
        capture_in_progress = True
        try:
            prediction, result = pipeline.capture_and_predict()
            if result.get("timestamp"):
                result["timestamp"] = result["timestamp"].isoformat()

            if result.get("image_path") and Path(result["image_path"]).exists():
                img = cv2.imread(result["image_path"])
                if img is not None:
                    _update_preview_cache(img)
                    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    result["image_b64"] = base64.b64encode(buf).decode()

            if not result.get("success"):
                error_text = str(result.get("error") or "").lower()
                camera_failed = (
                    error_text.startswith("camera")
                    or "capture failed" in error_text
                    or "rpicam" in error_text
                    or "libcamera" in error_text
                )
                result["camera_recovered"] = _reset_camera_if_needed(force=camera_failed)

            return result
        finally:
            capture_in_progress = False


# ── History & Stats ───────────────────────────────────────────────────────────

@app.get("/api/history")
async def get_history(limit: int = 10):
    if pipeline is None:
        return {"records": []}
    return {"records": pipeline.get_last_results(num=limit)}


@app.get("/api/stats")
async def get_stats():
    if pipeline is None:
        return {"total_predictions": 0, "successful": 0, "failed": 0, "mean_bilirubin": None}
    stats = pipeline.get_statistics()
    # Hapus NaN
    import math
    if stats.get("mean_bilirubin") is not None:
        try:
            if math.isnan(stats["mean_bilirubin"]):
                stats["mean_bilirubin"] = None
        except TypeError:
            stats["mean_bilirubin"] = None
    return stats


# ── Settings ──────────────────────────────────────────────────────────────────

class ModelSettings(BaseModel):
    use_stage2: bool


@app.post("/api/settings/model")
async def update_model(settings: ModelSettings):
    if pipeline is None:
        return {"success": False, "error": "Pipeline tidak diinisialisasi"}
    try:
        pipeline.prediction_engine.use_stage2 = settings.use_stage2
        return {"success": True, "use_stage2": settings.use_stage2}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/images/cleanup")
async def cleanup_images():
    if pipeline is None:
        return {"success": False, "deleted": 0}
    try:
        deleted = pipeline.storage.cleanup_old_images(days_to_keep=7)
        return {"success": True, "deleted": deleted}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
