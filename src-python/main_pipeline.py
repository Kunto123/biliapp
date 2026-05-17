"""
main_pipeline.py

Main pipeline orchestrator: camera -> preprocess -> predict -> log -> return results
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict

from camera_manager import CameraManager, CameraType, auto_detect_camera
from preprocessing import BilirubinPreprocessor
from prediction_engine import BilirubinPredictor
from logger import PredictionLogger
from image_storage import ImageStorage
from config import (
    CAMERA_AUTO_EXPOSURE,
    CAMERA_BRIGHTNESS,
    CAMERA_TIMEOUT_SECONDS,
    MODEL_BACKEND,
    MODEL_INPUT_SIZE,
    MODEL_STAGE1_TFLITE_PATH,
    MODEL_STAGE2_TFLITE_PATH,
)
from camera_settings import get_camera_settings, resolution_tuple


class BilirubinPredictionPipeline:
    """
    Complete pipeline: capture image -> preprocess -> predict -> log results.
    """

    def __init__(
        self,
        model_stage1_path: str,
        model_stage2_path: Optional[str] = None,
        use_stage2: bool = True,
        logs_dir: str = "logs",
        images_dir: str = "data/captures",
        camera: Optional[CameraManager] = None,
        model_backend: str = MODEL_BACKEND,
        tflite_stage1_path: Optional[str] = None,
        tflite_stage2_path: Optional[str] = None,
    ):
        """
        Initialize complete pipeline.
        
        Args:
            model_stage1_path: Path to stage1 model
            model_stage2_path: Path to stage2 model (optional)
            use_stage2: Use stage2 if available
            logs_dir: Directory for logs
            images_dir: Directory for captured images
            camera: CameraManager instance (auto-detect if None)
        """
        self.logs_dir = logs_dir
        self.images_dir = images_dir
        self.camera = None
        self.last_error = None

        # Initialize components
        self.prediction_engine = BilirubinPredictor(
            model_stage1_path,
            model_stage2_path,
            use_stage2=use_stage2,
            target_size=MODEL_INPUT_SIZE,
            model_backend=model_backend,
            tflite_stage1_path=tflite_stage1_path or str(MODEL_STAGE1_TFLITE_PATH),
            tflite_stage2_path=tflite_stage2_path or str(MODEL_STAGE2_TFLITE_PATH),
        )
        self.logger = PredictionLogger(log_dir=logs_dir, use_csv=True, use_sqlite=False)
        self.storage = ImageStorage(base_dir=images_dir)

        # Camera - either provided or auto-detect
        if camera is not None:
            self.camera = camera
        else:
            self.camera = self._init_configured_camera()

    def _init_configured_camera(self) -> Optional[CameraManager]:
        """Initialize the configured camera first, then fall back to auto-detect."""
        try:
            settings = get_camera_settings()
            camera_type = CameraType(settings["camera_type"])
            camera = CameraManager(
                camera_type=camera_type,
                camera_index=settings["camera_index"],
                resolution=resolution_tuple(settings["capture_resolution"]),
                brightness=CAMERA_BRIGHTNESS,
                auto_exposure=CAMERA_AUTO_EXPOSURE,
                timeout_seconds=CAMERA_TIMEOUT_SECONDS,
                rotation=settings["rotation"],
                fps=settings["fps"],
            )
            if camera.is_open:
                return camera
            self.last_error = camera.error_message
        except Exception as exc:
            self.last_error = str(exc)

        try:
            settings = get_camera_settings()
            return auto_detect_camera(rotation=settings["rotation"])
        except Exception:
            return auto_detect_camera()

    def capture_and_predict(self) -> Tuple[Optional[float], Dict]:
        """
        Single-shot: capture image -> predict -> log.
        
        Returns:
            (predicted_bilirubin_value, result_info_dict)
        """
        result = {
            "success": False,
            "bilirubin_prediction": None,
            "image_path": None,
            "preprocessing_mode": None,
            "quality_label": None,
            "quality_score": None,
            "gatecheck_passed": None,
            "gatecheck_errors": [],
            "gatecheck_warnings": [],
            "palette_detected": False,
            "quality_flags": {},
            "model_backend": self.prediction_engine.model_backend,
            "model_used": None,
            "inference_time_ms": None,
            "error": None,
            "timestamp": datetime.now()
        }

        try:
            # Step 1: Capture image
            if self.camera is None:
                result["error"] = "Camera not initialized"
                self.last_error = result["error"]
                return None, result

            image_bgr = self.camera.capture_image()
            result["timestamp"] = datetime.now()  # Timestamp set after actual capture
            if image_bgr is None:
                result["error"] = f"Camera capture failed: {self.camera.error_message}"
                self.last_error = result["error"]
                return None, result

            # Step 2: Predict
            prediction, pred_info = self.prediction_engine.predict_from_image(image_bgr, return_diagnostics=True)

            if prediction is None:
                result["error"] = pred_info.get("error", "Prediction failed")
                result["success"] = False
                result["preprocessing_mode"] = pred_info.get("preprocessing_mode", "unknown")
                result["quality_label"] = pred_info.get("quality_label", "failed")
                result["quality_score"] = pred_info.get("quality_score", 0)
                result["gatecheck_passed"] = pred_info.get("gatecheck_passed")
                result["gatecheck_errors"] = pred_info.get("gatecheck_errors", [])
                result["gatecheck_warnings"] = pred_info.get("gatecheck_warnings", [])
                result["palette_detected"] = pred_info.get("palette_detected", False)
                result["quality_flags"] = pred_info.get("quality_flags", {})
                result["model_backend"] = pred_info.get("model_backend", self.prediction_engine.model_backend)

                self.last_error = result["error"]
                return None, result

            # Step 3: Save and log successful prediction only
            result["success"] = True
            result["bilirubin_prediction"] = prediction
            result["preprocessing_mode"] = pred_info.get("preprocessing_mode", "unknown")
            result["quality_label"] = pred_info.get("quality_label", "unknown")
            result["quality_score"] = pred_info.get("quality_score", 0)
            result["quality_flags"] = pred_info.get("quality_flags", {})
            result["gatecheck_passed"] = pred_info.get("gatecheck_passed", True)
            result["gatecheck_errors"] = pred_info.get("gatecheck_errors", [])
            result["gatecheck_warnings"] = pred_info.get("gatecheck_warnings", [])
            result["palette_detected"] = pred_info.get("palette_detected", False)
            result["model_backend"] = pred_info.get("model_backend", self.prediction_engine.model_backend)
            result["model_used"] = pred_info.get("model_used")
            result["inference_time_ms"] = pred_info.get("inference_time_ms")
            result["error"] = None

            timestamp = result["timestamp"]
            save_ok, image_path = self.storage.save_image(image_bgr, prefix="capture", timestamp=timestamp)
            if save_ok:
                result["image_path"] = image_path

            log_ok = self.logger.log_prediction(
                timestamp=timestamp,
                image_filename=Path(image_path).name if save_ok else "",
                image_path=image_path if save_ok else "",
                bilirubin_prediction=prediction,
                preprocessing_mode=result["preprocessing_mode"],
                quality_label=result["quality_label"],
                quality_score=int(result["quality_score"]),
                success=True,
                error_message=None,
                model_version=f"bilirubin_v1_{result['model_backend']}",
                notes=f"model_used={result['model_used']}; inference_ms={result['inference_time_ms']}"
            )
            if not log_ok:
                result["log_warning"] = f"Gagal menulis log: {self.logger.last_write_error}"

            return prediction, result

        except Exception as e:
            self.last_error = str(e)
            result["error"] = self.last_error
            return None, result

    def predict_from_file(self, image_path: str) -> Tuple[Optional[float], Dict]:
        """
        Predict from existing image file (without camera capture).
        
        Returns:
            (predicted_bilirubin_value, result_info_dict)
        """
        result = {
            "success": False,
            "bilirubin_prediction": None,
            "image_path": image_path,
            "preprocessing_mode": None,
            "quality_label": None,
            "quality_score": None,
            "error": None,
            "timestamp": datetime.now()
        }

        try:
            # Read image
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                result["error"] = f"Failed to read image: {image_path}"
                return None, result

            # Predict
            prediction, pred_info = self.prediction_engine.predict_from_image(image_bgr, return_diagnostics=True)

            if prediction is None:
                result["error"] = pred_info.get("error", "Prediction failed")
                return None, result

            # Build result
            result["success"] = True
            result["bilirubin_prediction"] = prediction
            result["preprocessing_mode"] = pred_info.get("preprocessing_mode", "unknown")
            result["quality_label"] = pred_info.get("quality_label", "unknown")
            result["quality_score"] = pred_info.get("quality_score", 0)

            # Log
            log_ok = self.logger.log_prediction(
                timestamp=result["timestamp"],
                image_filename=Path(image_path).name,
                image_path=image_path,
                bilirubin_prediction=prediction,
                preprocessing_mode=result["preprocessing_mode"],
                quality_label=result["quality_label"],
                quality_score=int(result["quality_score"]),
                success=True,
                error_message=None,
                model_version="bilirubin_v1"
            )
            if not log_ok:
                result["log_warning"] = f"Gagal menulis log: {self.logger.last_write_error}"

            return prediction, result

        except Exception as e:
            self.last_error = str(e)
            result["error"] = self.last_error
            return None, result

    def get_last_results(self, num: int = 5) -> list:
        """Get last N prediction results from log."""
        return self.logger.get_last_predictions(num)

    def get_statistics(self) -> Dict:
        """Get statistics from logged predictions."""
        return self.logger.get_statistics()

    def get_system_status(self) -> Dict:
        """Get current system status."""
        return {
            "camera": self.camera.get_camera_info() if self.camera else {"status": "not_initialized"},
            "models": self.prediction_engine.get_model_info(),
            "logs_directory": str(self.logs_dir),
            "images_directory": str(self.images_dir),
            "total_captures": self.storage.get_capture_count(),
            "last_error": self.last_error
        }

    def cleanup(self):
        """Release resources."""
        camera = getattr(self, "camera", None)
        if camera is not None:
            camera.release()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass
