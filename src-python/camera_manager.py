"""
camera_manager.py

Camera management for ArduCam Hawkeye 64MP on Raspberry Pi.
Supports libcamera and fallback to OpenCV VideoCapture.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from enum import Enum
import shutil
import subprocess
import warnings


class CameraType(Enum):
    """Supported camera types."""
    LIBCAMERA = "libcamera"     # ArduCam via libcamera (recommended for Pi)
    OPENCV = "opencv"             # Generic USB/CSI via OpenCV
    PI_LEGACY = "pi_legacy"        # Legacy picamera (Pi < 5)


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
    ):
        """
        Initialize camera.
        
        Args:
            camera_type: Type of camera device
            camera_index: Camera device index (0 for primary)
            resolution: (width, height) tuple
            brightness: Brightness adjustment (-1.0 to 1.0)
            auto_exposure: Enable auto exposure
        """
        self.camera_type = camera_type
        self.camera_index = camera_index
        self.resolution = resolution
        self.brightness = brightness
        self.auto_exposure = auto_exposure
        self.timeout_seconds = timeout_seconds
        
        self.cap = None
        self._rpicam_cmd = None
        self.is_open = False
        self.error_message = None
        self._init_camera()

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

            return frame  # BGR format

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
                "capture_command": self._rpicam_cmd,
                "error": None
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def release(self):
        """Release camera resource."""
        if self.cap is not None:
            self.cap.release()
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


def auto_detect_camera() -> Optional[CameraManager]:
    """
    Auto-detect available camera and initialize.
    
    Returns:
        CameraManager instance or None if no camera found
    """
    # Try rpicam/libcamera first on Raspberry Pi.
    try:
        cam = CameraManager(camera_type=CameraType.LIBCAMERA)
        if cam.is_open:
            return cam
    except Exception:
        pass

    # Fallback to OpenCV (works well for many USB cameras).
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.release()
            return CameraManager(camera_type=CameraType.OPENCV, camera_index=0)
    except Exception:
        pass

    # Try libcamera if on Raspberry Pi 5
    try:
        return CameraManager(camera_type=CameraType.LIBCAMERA)
    except Exception:
        pass

    return None
