"""
camera_manager.py

Camera management for ArduCam Hawkeye 64MP on Raspberry Pi.
Supports libcamera and fallback to OpenCV VideoCapture.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple
from enum import Enum
from collections import deque
import shutil
import subprocess
import threading
import time

VALID_CAMERA_ROTATIONS = {0, 90, 180, 270}
JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"


def normalize_camera_rotation(rotation: int) -> int:
    try:
        value = int(rotation)
    except (TypeError, ValueError):
        return 0
    return value if value in VALID_CAMERA_ROTATIONS else 0


def extract_jpeg_frames(buffer: bytes) -> tuple[list[bytes], bytes]:
    """Extract complete JPEG images from an MJPEG byte buffer."""
    frames = []

    while True:
        start = buffer.find(JPEG_SOI)
        if start < 0:
            return frames, b""

        if start > 0:
            buffer = buffer[start:]

        end = buffer.find(JPEG_EOI, 2)
        if end < 0:
            return frames, buffer

        frame_end = end + len(JPEG_EOI)
        frames.append(buffer[:frame_end])
        buffer = buffer[frame_end:]


class CameraType(Enum):
    """Supported camera types."""
    LIBCAMERA = "libcamera"     # ArduCam via libcamera (recommended for Pi)
    OPENCV = "opencv"             # Generic USB/CSI via OpenCV
    PI_LEGACY = "pi_legacy"        # Legacy picamera (Pi < 5)


class CameraPreviewStream:
    """Continuously capture lightweight MJPEG preview frames."""

    def __init__(
        self,
        camera_type: CameraType,
        camera_index: int = 0,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
        min_fps: int = 30,
        rotation: int = 0,
        jpeg_quality: int = 65,
    ):
        self.camera_type = camera_type
        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = max(1, int(fps))
        self.min_fps = max(1, int(min_fps))
        self.rotation = normalize_camera_rotation(rotation)
        self.jpeg_quality = max(1, min(100, int(jpeg_quality)))

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen] = None
        self._cap = None
        self._latest_jpeg: Optional[bytes] = None
        self._latest_at: float = 0.0
        self._frame_times = deque(maxlen=max(self.fps * 3, 30))
        self._error_message: Optional[str] = None

    @staticmethod
    def find_video_command() -> Optional[str]:
        return shutil.which("rpicam-vid") or shutil.which("libcamera-vid")

    def build_libcamera_command(self) -> Optional[list[str]]:
        video_cmd = self.find_video_command()
        if not video_cmd:
            return None

        width, height = self.resolution
        return [
            video_cmd,
            "-n",
            "-t",
            "0",
            "--codec",
            "mjpeg",
            "--width",
            str(width),
            "--height",
            str(height),
            "--framerate",
            str(self.fps),
            "--rotation",
            str(self.rotation),
            "-o",
            "-",
        ]

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> bool:
        if self.is_running:
            return True

        self.stop()
        self._stop_event.clear()
        self._error_message = None

        target = self._run_libcamera if self.camera_type == CameraType.LIBCAMERA else self._run_opencv
        self._thread = threading.Thread(target=target, name="camera-preview", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()

        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=1.5)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass

        cap = self._cap
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)

        self._thread = None
        self._process = None
        self._cap = None

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def status(self) -> dict:
        with self._lock:
            fps = self._calculate_fps_locked()
            return {
                "available": self._latest_jpeg is not None,
                "running": self.is_running,
                "fps": fps,
                "fps_ok": fps >= self.min_fps if fps is not None else False,
                "target_fps": self.fps,
                "min_fps": self.min_fps,
                "frame_size": self.resolution,
                "updated_at": self._latest_at,
                "error": self._error_message,
            }

    def _store_jpeg(self, jpeg: bytes) -> None:
        now = time.monotonic()
        with self._lock:
            self._latest_jpeg = jpeg
            self._latest_at = now
            self._frame_times.append(now)
            self._error_message = None

    def _calculate_fps_locked(self) -> Optional[float]:
        if len(self._frame_times) < 2:
            return None
        elapsed = self._frame_times[-1] - self._frame_times[0]
        if elapsed <= 0:
            return None
        return round((len(self._frame_times) - 1) / elapsed, 1)

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._error_message = message

    def _run_libcamera(self) -> None:
        cmd = self.build_libcamera_command()
        if not cmd:
            self._set_error("rpicam-vid/libcamera-vid not found")
            return

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except Exception as exc:
            self._set_error(f"Failed to start preview stream: {exc}")
            return

        buffer = b""
        stdout = self._process.stdout
        if stdout is None:
            self._set_error("Preview stream stdout unavailable")
            return

        while not self._stop_event.is_set():
            try:
                chunk = stdout.read(8192)
            except Exception as exc:
                self._set_error(f"Preview stream read failed: {exc}")
                break

            if not chunk:
                if self._process and self._process.poll() is not None:
                    self._set_error(f"Preview stream stopped ({self._process.returncode})")
                    break
                time.sleep(0.005)
                continue

            buffer += chunk
            frames, buffer = extract_jpeg_frames(buffer)
            for frame in frames:
                self._store_jpeg(frame)

    def _run_opencv(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        self._cap = cap
        if not cap.isOpened():
            self._set_error(f"Failed to open preview camera at index {self.camera_index}")
            return

        width, height = self.resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        frame_interval = 1.0 / self.fps
        while not self._stop_event.is_set():
            started = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                self._set_error("Failed to read preview frame")
                time.sleep(0.05)
                continue

            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            frame = self._apply_rotation_for_preview(frame)

            ok, buf = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            )
            if ok:
                self._store_jpeg(buf.tobytes())

            elapsed = time.monotonic() - started
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    def _apply_rotation_for_preview(self, frame: np.ndarray) -> np.ndarray:
        if self.rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if self.rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame


class CameraManager:
    """
    Manage camera capture from ArduCam Hawkeye 64MP on Raspberry Pi.
    
    Attempts libcamera first, falls back to OpenCV VideoCapture.
    """

    def __init__(
        self,
        camera_type: CameraType = CameraType.OPENCV,
        camera_index: int = 0,
        resolution: Tuple[int, int] = (3840, 2160),  # 4K
        brightness: float = 0.0,
        auto_exposure: bool = True,
        timeout_seconds: float = 20.0,
        rotation: int = 0,
    ):
        """
        Initialize camera.
        
        Args:
            camera_type: Type of camera device
            camera_index: Camera device index (0 for primary)
            resolution: (width, height) tuple
            brightness: Brightness adjustment (-1.0 to 1.0)
            auto_exposure: Enable auto exposure
            rotation: Camera rotation in degrees (0, 90, 180, 270)
        """
        self.camera_type = camera_type
        self.camera_index = camera_index
        self.resolution = resolution
        self.brightness = brightness
        self.auto_exposure = auto_exposure
        self.timeout_seconds = timeout_seconds
        self.rotation = self._normalize_rotation(rotation)
        
        self.cap = None
        self._rpicam_cmd = None
        self.is_open = False
        self.error_message = None
        self._init_camera()

    @staticmethod
    def _normalize_rotation(rotation: int) -> int:
        return normalize_camera_rotation(rotation)

    def _apply_rotation(self, frame: np.ndarray) -> np.ndarray:
        if self.rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if self.rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _init_camera(self) -> bool:
        """Initialize camera connection."""
        try:
            if self.camera_type == CameraType.OPENCV:
                return self._init_opencv()
            elif self.camera_type == CameraType.LIBCAMERA:
                return self._init_libcamera()
            elif self.camera_type == CameraType.PI_LEGACY:
                return self._init_pi_legacy()
            else:
                self.error_message = f"Unknown camera type: {self.camera_type}"
                return False
        except Exception as e:
            self.error_message = str(e)
            return False

    def _init_opencv(self) -> bool:
        """Initialize OpenCV VideoCapture."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                self.error_message = f"Failed to open camera at index {self.camera_index}"
                return False

            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            # Set FPS
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Auto exposure
            if self.auto_exposure:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

            # Brightness
            if -1.0 <= self.brightness <= 1.0:
                brightness_val = int((self.brightness + 1.0) * 127.5)
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness_val)

            self.is_open = True
            return True

        except Exception as e:
            self.error_message = str(e)
            return False

    def _init_libcamera(self) -> bool:
        """
        Initialize via rpicam/libcamera CLI tools (for Raspberry Pi 5).
        """
        try:
            # Prefer rpicam-still on new Raspberry Pi OS, fallback to libcamera-still.
            rpicam_cmd = shutil.which("rpicam-still")
            libcamera_cmd = shutil.which("libcamera-still")

            if rpicam_cmd:
                self._rpicam_cmd = rpicam_cmd
            elif libcamera_cmd:
                self._rpicam_cmd = libcamera_cmd
            else:
                self.error_message = (
                "rpicam-still/libcamera-still not found. "
                    "Install rpicam/libcamera tools and ArduCam camera support."
                )
                return False

            self.cap = None
            self.is_open = True
            return True

        except Exception as e:
            self.error_message = str(e)
            return False

    def _capture_libcamera_frame(self) -> Optional[np.ndarray]:
        """Capture one JPEG frame using rpicam/libcamera command line tools."""
        if not self._rpicam_cmd:
            self.error_message = "libcamera backend not initialized"
            return None

        width, height = self.resolution
        cmd = [
            self._rpicam_cmd,
            "-n",
            "--immediate",
            "--width",
            str(width),
            "--height",
            str(height),
            "--rotation",
            str(self.rotation),
            "--encoding",
            "jpg",
            "-o",
            "-",
        ]

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            self.error_message = "Timed out while capturing frame via rpicam"
            return None
        except Exception as e:
            self.error_message = f"rpicam invocation failed: {e}"
            return None

        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
            self.error_message = f"rpicam failed ({proc.returncode}): {stderr}"
            return None

        if not proc.stdout:
            self.error_message = "rpicam returned empty output"
            return None

        image_array = np.frombuffer(proc.stdout, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame is None:
            self.error_message = "Failed to decode frame from rpicam output"
            return None

        return frame

    def _init_pi_legacy(self) -> bool:
        """
        Initialize via legacy picamera (not recommended for Pi 5).
        """
        try:
            # Fallback to OpenCV
            return self._init_opencv()
        except Exception as e:
            self.error_message = str(e)
            return False

    def capture_image(self) -> Optional[np.ndarray]:
        """
        Capture single frame from camera.
        
        Returns:
            Image in BGR format (cv2 convention), or None on failure
        """
        if not self.is_open:
            self.error_message = "Camera not initialized"
            return None

        if self.camera_type == CameraType.LIBCAMERA:
            return self._capture_libcamera_frame()

        if self.cap is None:
            self.error_message = "OpenCV camera handle not initialized"
            return None

        try:
            ret, frame = self.cap.read()
            
            if not ret:
                self.error_message = "Failed to capture frame"
                return None

            return self._apply_rotation(frame)  # BGR format

        except Exception as e:
            self.error_message = str(e)
            return None

    def capture_multiple(self, num_frames: int = 5, interval_ms: int = 100) -> list:
        """
        Capture multiple frames with interval.
        
        Args:
            num_frames: Number of frames to capture
            interval_ms: Milliseconds between captures
        
        Returns:
            List of images in BGR format
        """
        frames = []
        
        for i in range(num_frames):
            frame = self.capture_image()
            if frame is not None:
                frames.append(frame)
            
            # Wait between captures (except last one)
            if i < num_frames - 1:
                import time
                time.sleep(interval_ms / 1000.0)

        return frames

    def get_frame_size(self) -> Optional[Tuple[int, int]]:
        """Get actual frame size captured from camera."""
        if not self.is_open:
            return None

        if self.camera_type == CameraType.LIBCAMERA:
            return self.resolution

        if self.cap is None:
            return None

        try:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        except Exception:
            return None

    def set_brightness(self, brightness: float) -> bool:
        """Set camera brightness (-1.0 to 1.0)."""
        if not self.is_open:
            return False

        if self.camera_type == CameraType.LIBCAMERA:
            # rpicam CLI exposes more advanced controls; keep this as a no-op state update.
            self.brightness = brightness
            return True

        if self.cap is None:
            return False

        try:
            brightness_val = int((brightness + 1.0) * 127.5)
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness_val)
            self.brightness = brightness
            return True
        except Exception:
            return False

    def set_resolution(self, width: int, height: int) -> bool:
        """Set camera resolution."""
        if not self.is_open:
            return False

        if self.camera_type == CameraType.LIBCAMERA:
            self.resolution = (width, height)
            return True

        if self.cap is None:
            return False

        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.resolution = (width, height)
            return True
        except Exception:
            return False

    def get_camera_info(self) -> dict:
        """Get camera information."""
        if not self.is_open:
            return {"status": "not_initialized", "error": self.error_message}

        try:
            frame_size = self.get_frame_size()
            fps = None
            if self.cap is not None:
                fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            return {
                "status": "open",
                "camera_type": self.camera_type.value,
                "frame_size": frame_size,
                "fps": fps,
                "brightness": self.brightness,
                "auto_exposure": self.auto_exposure,
                "timeout_seconds": self.timeout_seconds,
                "camera_rotation": self.rotation,
                "capture_command": self._rpicam_cmd,
                "error": None
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def release(self):
        """Release camera resource."""
        cap = getattr(self, "cap", None)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_open = False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()

    def __del__(self):
        """Cleanup on deletion."""
        self.release()


def auto_detect_camera(rotation: int = 0) -> Optional[CameraManager]:
    """
    Auto-detect available camera and initialize.
    
    Returns:
        CameraManager instance or None if no camera found
    """
    # Try rpicam/libcamera first on Raspberry Pi.
    try:
        cam = CameraManager(camera_type=CameraType.LIBCAMERA, rotation=rotation)
        if cam.is_open:
            return cam
    except Exception:
        pass

    # Fallback to OpenCV (works well for many USB cameras).
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.release()
            return CameraManager(camera_type=CameraType.OPENCV, camera_index=0, rotation=rotation)
    except Exception:
        pass

    # Try libcamera if on Raspberry Pi 5
    try:
        return CameraManager(camera_type=CameraType.LIBCAMERA, rotation=rotation)
    except Exception:
        pass

    return None
