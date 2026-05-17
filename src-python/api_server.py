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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from camera_manager import CameraPreviewStream, CameraType, scan_opencv_devices
from camera_settings import (
    DEFAULT_SETTINGS_PATH,
    get_camera_settings,
    load_camera_settings,
    normalize_camera_settings,
    resolution_tuple,
    save_camera_settings,
)
from main_pipeline import BilirubinPredictionPipeline
from config import (
    DEVICE_PROFILE,
    GATECHECK_MIN_BLUR_SCORE,
    MODEL_BACKEND,
    MODEL_STAGE1_TFLITE_PATH,
    MODEL_STAGE2_TFLITE_PATH,
    PREVIEW_POLL_MS,
    USE_STAGE2,
)
from gpio_manager import gpio_manager

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
preview_focus_frame_id: int = 0
preview_focus_at: float = 0.0
preview_stream: Optional[CameraPreviewStream] = None
preview_clients = 0
camera_lock = threading.RLock()
capture_in_progress = False


def _camera_is_available() -> bool:
    return pipeline is not None and pipeline.camera is not None and pipeline.camera.is_open


def _active_camera_settings() -> dict:
    return get_camera_settings()


def _preview_resolution() -> tuple[int, int]:
    return resolution_tuple(_active_camera_settings()["preview_resolution"])


def _prepare_preview_frame(frame):
    width, height = _preview_resolution()
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
    _, buf = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, _active_camera_settings()["jpeg_quality"]],
    )
    return base64.b64encode(buf).decode()


def _decode_jpeg(jpeg: bytes):
    image_array = np.frombuffer(jpeg, dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


def _update_preview_cache(frame, timestamp: Optional[float] = None) -> None:
    global preview_cache_b64, preview_cache_at, preview_focus_score, preview_focus_ok

    preview_frame = _prepare_preview_frame(frame)
    preview_focus_score = _calculate_focus_score(preview_frame)
    preview_focus_ok = preview_focus_score >= GATECHECK_MIN_BLUR_SCORE
    _, buf = cv2.imencode(
        ".jpg",
        preview_frame,
        [cv2.IMWRITE_JPEG_QUALITY, _active_camera_settings()["jpeg_quality"]],
    )
    preview_cache_b64 = base64.b64encode(buf).decode()
    preview_cache_at = timestamp if timestamp is not None else time.monotonic()


def _update_preview_cache_from_jpeg(
    jpeg: bytes,
    timestamp: Optional[float] = None,
    frame_id: Optional[int] = None,
    force_focus: bool = False,
) -> None:
    global preview_cache_b64, preview_cache_at, preview_focus_score, preview_focus_ok
    global preview_focus_frame_id, preview_focus_at

    now = timestamp if timestamp is not None else time.monotonic()
    should_update_focus = force_focus
    if frame_id is None:
        should_update_focus = should_update_focus or (now - preview_focus_at) >= 0.25
    else:
        should_update_focus = should_update_focus or (
            frame_id != preview_focus_frame_id and (now - preview_focus_at) >= 0.25
        )

    if should_update_focus:
        frame = _decode_jpeg(jpeg)
        if frame is not None:
            preview_frame = _prepare_preview_frame(frame)
            preview_focus_score = _calculate_focus_score(preview_frame)
            preview_focus_ok = preview_focus_score >= GATECHECK_MIN_BLUR_SCORE
            preview_focus_at = now
            preview_focus_frame_id = frame_id or preview_focus_frame_id

    preview_cache_b64 = base64.b64encode(jpeg).decode()
    preview_cache_at = now


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
        return CameraType(_active_camera_settings()["camera_type"])
    except ValueError:
        camera = getattr(pipeline, "camera", None)
        return getattr(camera, "camera_type", CameraType.OPENCV)


def _create_preview_stream() -> CameraPreviewStream:
    camera = getattr(pipeline, "camera", None)
    settings = _active_camera_settings()
    camera_type = getattr(camera, "camera_type", _configured_camera_type())
    camera_index = getattr(camera, "camera_index", settings["camera_index"])
    rotation = getattr(camera, "rotation", settings["rotation"])

    return CameraPreviewStream(
        camera_type=camera_type,
        camera_index=camera_index,
        resolution=resolution_tuple(settings["preview_resolution"]),
        fps=settings["fps"],
        min_fps=settings["min_fps"],
        rotation=rotation,
        jpeg_quality=settings["jpeg_quality"],
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


def _clear_preview_cache() -> None:
    global preview_cache_b64, preview_cache_at, preview_focus_score, preview_focus_ok
    global preview_focus_frame_id, preview_focus_at
    preview_cache_b64 = None
    preview_cache_at = 0.0
    preview_focus_score = None
    preview_focus_ok = None
    preview_focus_frame_id = 0
    preview_focus_at = 0.0


def _preview_sleep_seconds(stream: Optional[CameraPreviewStream] = None) -> float:
    target_fps = 0
    if stream is not None:
        status = stream.status()
        for value in (status.get("target_fps"), status.get("detected_fps"), status.get("fps")):
            if isinstance(value, (int, float)) and value > 0:
                target_fps = int(value)
                break
    if target_fps <= 0:
        target_fps = _active_camera_settings()["fps"]
    if target_fps > 0:
        return max(1.0 / min(target_fps, 120), 0.005)
    return max(PREVIEW_POLL_MS / 1000.0, 0.005)


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
    stage2_available = m2.exists() or MODEL_STAGE2_TFLITE_PATH.exists()
    print(f"[api] BASE_DIR : {BASE_DIR}")
    print(f"[api] Stage1   : {m1} (exists={m1.exists()})")
    print(f"[api] Stage2   : {m2} (keras={m2.exists()}, tflite={MODEL_STAGE2_TFLITE_PATH.exists()})")
    try:
        pipeline = BilirubinPredictionPipeline(
            model_stage1_path=str(m1),
            model_stage2_path=str(m2) if m2.exists() else None,
            use_stage2=USE_STAGE2 and stage2_available,
            logs_dir=str(BASE_DIR / "logs"),
            images_dir=str(BASE_DIR / "data" / "captures"),
            model_backend=MODEL_BACKEND,
            tflite_stage1_path=str(MODEL_STAGE1_TFLITE_PATH),
            tflite_stage2_path=str(MODEL_STAGE2_TFLITE_PATH),
        )
        print("[api] ✓ Pipeline initialized")
    except Exception as e:
        print(f"[api] ✗ Pipeline init failed: {e}")

    gpio_manager.start()
    print(f"[api] GPIO available: {gpio_manager.available}")

@app.on_event("shutdown")
async def shutdown():
    with camera_lock:
        _stop_preview_stream()
        if pipeline:
            pipeline.cleanup()
    gpio_manager.stop()


# ── Status ────────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    if pipeline is None:
        return {"initialized": False, "error": "Pipeline not ready"}
    status = pipeline.get_system_status()
    camera_settings, settings_source = load_camera_settings()
    status["initialized"] = True
    status["runtime_config"] = {
        "device_profile": DEVICE_PROFILE,
        "model_backend": MODEL_BACKEND,
        "camera_type": camera_settings["camera_type"],
        "camera_index": camera_settings["camera_index"],
        "camera_rotation": camera_settings["rotation"],
        "preview_poll_ms": PREVIEW_POLL_MS,
        "preview_resolution": [
            camera_settings["preview_resolution"]["width"],
            camera_settings["preview_resolution"]["height"],
        ],
        "capture_resolution": [
            camera_settings["capture_resolution"]["width"],
            camera_settings["capture_resolution"]["height"],
        ],
        "preview_fps": camera_settings["fps"],
        "preview_min_fps": camera_settings["min_fps"],
        "preview_jpeg_quality": camera_settings["jpeg_quality"],
        "camera_settings_source": settings_source,
        "use_stage2": USE_STAGE2,
    }
    # Pastikan serializable
    for k, v in list(status.items()):
        if not isinstance(v, (str, int, float, bool, dict, list, type(None))):
            status[k] = str(v)
    return status


# ── Camera ────────────────────────────────────────────────────────────────────

class CameraSettingsPayload(BaseModel):
    camera_type: Optional[str] = None
    camera_index: Optional[int] = None
    capture_resolution: Optional[dict] = None
    preview_resolution: Optional[dict] = None
    fps: Optional[int] = None
    min_fps: Optional[int] = None
    jpeg_quality: Optional[int] = None
    rotation: Optional[int] = None


@app.get("/api/camera/config")
async def get_camera_config():
    settings, source = load_camera_settings()
    return {
        "success": True,
        "settings": settings,
        "source": source,
        "path": str(DEFAULT_SETTINGS_PATH),
    }


@app.put("/api/camera/config")
async def update_camera_config(payload: CameraSettingsPayload):
    global preview_stream
    if pipeline is None:
        return {"success": False, "error": "Pipeline tidak diinisialisasi"}

    current = get_camera_settings()
    merged = current.copy()
    if hasattr(payload, "model_dump"):
        merged.update(payload.model_dump(exclude_unset=True))
    else:
        merged.update(payload.dict(exclude_unset=True))

    try:
        normalized = normalize_camera_settings(merged)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    with camera_lock:
        if capture_in_progress:
            return {"success": False, "error": "Kamera sedang capture"}

        try:
            save_camera_settings(normalized)
            _stop_preview_stream()
            preview_stream = None
            _clear_preview_cache()
            camera = getattr(pipeline, "camera", None)
            if camera is not None:
                camera.release()
            pipeline.camera = pipeline._init_configured_camera()
            ok = _camera_is_available()
            if ok:
                _ensure_preview_stream()
            return {"success": ok, "settings": normalized, "error": None if ok else pipeline.last_error}
        except Exception as exc:
            return {"success": False, "error": str(exc), "settings": normalized}


@app.get("/api/camera/devices")
async def get_camera_devices(max_index: int = 5):
    max_index = max(0, min(int(max_index), 10))
    devices = scan_opencv_devices(max_index=max_index)
    return {"success": True, "devices": devices, "max_index": max_index}


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
        frame_id, jpeg, _updated_at = (
            preview_stream.get_latest() if preview_stream is not None else (0, None, 0.0)
        )
        if jpeg is None:
            return _preview_payload(
                preview_cache_b64,
                preview_cache_b64 is not None,
                warming_up=True,
            )

        _update_preview_cache_from_jpeg(jpeg, now, frame_id=frame_id)
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
        last_frame_id = 0

        try:
            while True:
                stream = preview_stream
                frame_id, jpeg, _updated_at = stream.get_latest() if stream is not None else (0, None, 0.0)
                if jpeg is not None and frame_id != last_frame_id:
                    last_frame_id = frame_id
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Cache-Control: no-store\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )
                await asyncio.sleep(_preview_sleep_seconds(stream))
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

    settings = _active_camera_settings()
    stream = preview_stream
    status = stream.status() if stream is not None else {
        "available": False,
        "running": False,
        "fps": None,
        "fps_ok": False,
        "target_fps": settings["fps"],
        "min_fps": settings["min_fps"],
        "frame_size": resolution_tuple(settings["preview_resolution"]),
        "updated_at": 0.0,
        "frame_id": 0,
        "error": None,
    }

    frame_id, jpeg, _updated_at = stream.get_latest() if stream is not None else (0, None, 0.0)
    if jpeg is not None:
        _update_preview_cache_from_jpeg(jpeg, frame_id=frame_id)

    status.update({
        "available": bool(status.get("available")) and not capture_in_progress,
        "busy": capture_in_progress,
        "focus_score": preview_focus_score,
        "focus_ok": preview_focus_ok,
    })
    return status


@app.post("/api/camera/reconnect")
async def reconnect_camera():
    global preview_stream
    with camera_lock:
        _stop_preview_stream()
        preview_stream = None
        _clear_preview_cache()
        if pipeline is not None:
            pipeline.cleanup()
            pipeline.camera = pipeline._init_configured_camera()
        ok = _camera_is_available()
        if ok:
            _ensure_preview_stream()
        return {"success": ok}


# ── Prediction ────────────────────────────────────────────────────────────────

async def _execute_capture() -> dict:
    """Core capture+predict logic. Flash dikontrol langsung oleh switch di gpio_manager."""
    global capture_in_progress

    if capture_in_progress:
        return {"success": False, "error": "Capture sedang berlangsung", "busy": True}
    capture_in_progress = True

    try:
        with camera_lock:
            restart_preview = preview_stream is not None and preview_stream.is_running
            _stop_preview_stream()
            gpio_manager.mark_captured()   # blokir re-capture sampai switch dilepas
            gpio_manager.set_flash(True)   # nyalakan flash agar AE warmup terang
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
                gpio_manager.set_flash(False)  # matikan flash setelah capture selesai
                if restart_preview and _camera_is_available():
                    _ensure_preview_stream()
    finally:
        capture_in_progress = False


@app.post("/api/capture")
async def capture_and_predict():
    if pipeline is None:
        return {"success": False, "error": "Pipeline tidak diinisialisasi"}

    # Consume any GPIO trigger flag set by the limit switch monitor
    gpio_manager.consume_trigger()

    # When GPIO is wired: block if switch has not returned HIGH since last capture
    if gpio_manager.available and not gpio_manager.capture_ready:
        return {
            "success": False,
            "error": "Menunggu sensor — lepaskan limit switch (GPIO 8) terlebih dahulu",
            "gpio_blocked": True,
        }

    return await _execute_capture()


# ── GPIO ─────────────────────────────────────────────────────────────────────

@app.get("/api/gpio/status")
async def get_gpio_status():
    return gpio_manager.get_status()


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
