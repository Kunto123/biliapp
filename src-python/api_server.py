"""
api_server.py

FastAPI REST server untuk Tauri Bilirubin frontend.
Jalankan dari root bili-app/:  python src-python/api_server.py
Port: 127.0.0.1:7878
"""

import sys
import asyncio
import base64
import cv2
import numpy as np
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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from camera_manager import CameraPreviewStream, CameraType
from main_pipeline import BilirubinPredictionPipeline
from config import (
    CAMERA_PREVIEW_RESOLUTION,
    CAMERA_ROTATION,
    CAMERA_TYPE,
    DEVICE_PROFILE,
    GATECHECK_MIN_BLUR_SCORE,
    MODEL_BACKEND,
    MODEL_STAGE1_TFLITE_PATH,
    MODEL_STAGE2_TFLITE_PATH,
    PREVIEW_FPS,
    PREVIEW_JPEG_QUALITY,
    PREVIEW_MIN_FPS,
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
preview_stream: Optional[CameraPreviewStream] = None
preview_clients = 0
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


def _decode_jpeg(jpeg: bytes):
    image_array = np.frombuffer(jpeg, dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


def _update_preview_cache(frame, timestamp: Optional[float] = None) -> None:
    global preview_cache_b64, preview_cache_at, preview_focus_score, preview_focus_ok

    preview_frame = _prepare_preview_frame(frame)
    preview_focus_score = _calculate_focus_score(preview_frame)
    preview_focus_ok = preview_focus_score >= GATECHECK_MIN_BLUR_SCORE
    _, buf = cv2.imencode(".jpg", preview_frame, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY])
    preview_cache_b64 = base64.b64encode(buf).decode()
    preview_cache_at = timestamp if timestamp is not None else time.monotonic()


def _update_preview_cache_from_jpeg(jpeg: bytes, timestamp: Optional[float] = None) -> None:
    global preview_cache_b64, preview_cache_at, preview_focus_score, preview_focus_ok

    frame = _decode_jpeg(jpeg)
    if frame is not None:
        preview_frame = _prepare_preview_frame(frame)
        preview_focus_score = _calculate_focus_score(preview_frame)
        preview_focus_ok = preview_focus_score >= GATECHECK_MIN_BLUR_SCORE

    preview_cache_b64 = base64.b64encode(jpeg).decode()
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


def _configured_camera_type() -> CameraType:
    try:
        return CameraType(CAMERA_TYPE)
    except ValueError:
        camera = getattr(pipeline, "camera", None)
        return getattr(camera, "camera_type", CameraType.OPENCV)


def _create_preview_stream() -> CameraPreviewStream:
    camera = getattr(pipeline, "camera", None)
    camera_type = getattr(camera, "camera_type", _configured_camera_type())
    camera_index = getattr(camera, "camera_index", 0)
    rotation = getattr(camera, "rotation", CAMERA_ROTATION)

    return CameraPreviewStream(
        camera_type=camera_type,
        camera_index=camera_index,
        resolution=CAMERA_PREVIEW_RESOLUTION,
        fps=PREVIEW_FPS,
        min_fps=PREVIEW_MIN_FPS,
        rotation=rotation,
        jpeg_quality=PREVIEW_JPEG_QUALITY,
    )


def _ensure_preview_stream() -> bool:
    global preview_stream
    if pipeline is None or not _camera_is_available():
        return False

    if preview_stream is None:
        preview_stream = _create_preview_stream()

    return preview_stream.start()


def _stop_preview_stream() -> None:
    if preview_stream is not None:
        preview_stream.stop()


def _reset_camera_if_needed(force: bool = False) -> bool:
    global preview_stream
    if pipeline is None:
        return False

    camera = getattr(pipeline, "camera", None)
    if camera is not None and camera.is_open and not force:
        return True

    try:
        _stop_preview_stream()
        preview_stream = None
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
        _stop_preview_stream()
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
        "camera_rotation": CAMERA_ROTATION,
        "preview_poll_ms": PREVIEW_POLL_MS,
        "preview_resolution": CAMERA_PREVIEW_RESOLUTION,
        "preview_fps": PREVIEW_FPS,
        "preview_min_fps": PREVIEW_MIN_FPS,
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

        _ensure_preview_stream()
        jpeg = preview_stream.get_latest_jpeg() if preview_stream is not None else None
        if jpeg is None:
            return _preview_payload(
                preview_cache_b64,
                preview_cache_b64 is not None,
                warming_up=True,
            )

        _update_preview_cache_from_jpeg(jpeg, now)
        return _preview_payload(preview_cache_b64, True, cached=False)
    finally:
        camera_lock.release()


@app.get("/api/camera/stream")
async def stream_camera():
    global preview_clients

    if not camera_lock.acquire(blocking=False):
        async def busy_stream():
            while capture_in_progress:
                await asyncio.sleep(0.05)
                yield b""
        return StreamingResponse(busy_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

    try:
        if not _camera_is_available() and not _reset_camera_if_needed():
            async def empty_stream():
                yield b""
            return StreamingResponse(empty_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

        _ensure_preview_stream()
        preview_clients += 1
    finally:
        camera_lock.release()

    async def generate():
        global preview_clients
        last_frame = None
        sleep_s = max(1.0 / max(PREVIEW_FPS, 1), 0.005)

        try:
            while True:
                stream = preview_stream
                jpeg = stream.get_latest_jpeg() if stream is not None else None
                if jpeg is not None and jpeg != last_frame:
                    last_frame = jpeg
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Cache-Control: no-store\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )
                await asyncio.sleep(sleep_s)
        finally:
            with camera_lock:
                preview_clients = max(0, preview_clients - 1)
                if preview_clients == 0 and not capture_in_progress:
                    _stop_preview_stream()

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/camera/preview/status")
async def get_preview_status():
    if not capture_in_progress and camera_lock.acquire(blocking=False):
        try:
            if _camera_is_available():
                _ensure_preview_stream()
        finally:
            camera_lock.release()

    stream = preview_stream
    status = stream.status() if stream is not None else {
        "available": False,
        "running": False,
        "fps": None,
        "fps_ok": False,
        "target_fps": PREVIEW_FPS,
        "min_fps": PREVIEW_MIN_FPS,
        "frame_size": CAMERA_PREVIEW_RESOLUTION,
        "updated_at": 0.0,
        "error": None,
    }

    jpeg = stream.get_latest_jpeg() if stream is not None else None
    if jpeg is not None:
        _update_preview_cache_from_jpeg(jpeg)

    status.update({
        "available": bool(status.get("available")) and not capture_in_progress,
        "busy": capture_in_progress,
        "focus_score": preview_focus_score,
        "focus_ok": preview_focus_ok,
    })
    return status


@app.post("/api/camera/reconnect")
async def reconnect_camera():
    global preview_cache_b64, preview_cache_at, preview_focus_score, preview_focus_ok, preview_stream
    with camera_lock:
        _stop_preview_stream()
        preview_stream = None
        preview_cache_b64 = None
        preview_cache_at = 0.0
        preview_focus_score = None
        preview_focus_ok = None
        if pipeline is not None:
            pipeline.cleanup()
            pipeline.camera = pipeline._init_configured_camera()
        ok = _camera_is_available()
        if ok:
            _ensure_preview_stream()
        return {"success": ok}


# ── Prediction ────────────────────────────────────────────────────────────────

@app.post("/api/capture")
async def capture_and_predict():
    global capture_in_progress
    if pipeline is None:
        return {"success": False, "error": "Pipeline tidak diinisialisasi"}

    with camera_lock:
        restart_preview = preview_stream is not None and preview_stream.is_running
        _stop_preview_stream()
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
            if restart_preview and _camera_is_available():
                _ensure_preview_stream()


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
