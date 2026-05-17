"""
prediction_engine.py

Load and run bilirubin prediction models with preprocessing.
Supports Keras for desktop/development and TensorFlow Lite for Raspberry Pi.
"""

import os
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from data_artifacts import ROI_CONFIG, REFERENCE_PALETTE_DF, GRAY_PATCHES_REFERENCE_DF
from preprocessing import BilirubinPreprocessor

# Suppress TensorFlow warnings when the Keras fallback is used.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


def _numpy_major_version() -> int:
    try:
        return int(np.__version__.split(".", 1)[0])
    except (AttributeError, ValueError):
        return 0


class BilirubinPredictor:
    """
    Predict bilirubin level from image using trained models.

    The `keras` backend loads `.keras` models and is intended for desktop/dev.
    The `tflite` backend loads `.tflite` models and is intended for Raspberry Pi.
    """

    def __init__(
        self,
        model_stage1_path: str,
        model_stage2_path: Optional[str] = None,
        use_stage2: bool = True,
        target_size: Tuple[int, int] = (224, 224),
        model_backend: str = "keras",
        tflite_stage1_path: Optional[str] = None,
        tflite_stage2_path: Optional[str] = None,
        allow_backend_fallback: bool = True,
    ):
        self.model_stage1_path = model_stage1_path
        self.model_stage2_path = model_stage2_path
        self.tflite_stage1_path = tflite_stage1_path
        self.tflite_stage2_path = tflite_stage2_path
        self.use_stage2 = use_stage2
        self.target_size = target_size
        self.requested_model_backend = (model_backend or "keras").lower()
        self.model_backend = self.requested_model_backend
        self.allow_backend_fallback = allow_backend_fallback

        self.model_stage1 = None
        self.model_stage2 = None
        self.last_error = None
        self.last_inference_time_ms = None
        self.tflite_runtime = None

        self.preprocessor = BilirubinPreprocessor(
            roi_config=ROI_CONFIG,
            reference_palette_df=REFERENCE_PALETTE_DF,
            gray_reference_df=GRAY_PATCHES_REFERENCE_DF,
        )

        self._load_models()

    def ensure_models_loaded(self) -> bool:
        """Reload models if the predictor is alive but model objects are missing."""
        if self.model_stage1 is not None:
            return True
        self.last_error = "Models not loaded; attempting reload"
        return self._load_models()

    def _load_models(self) -> bool:
        """Load models for the selected backend."""
        backend = self.requested_model_backend
        if backend == "tflite":
            if self._load_tflite_models():
                return True
            if not self.allow_backend_fallback:
                return False
            print("[model] TFLite unavailable, trying Keras fallback")
            self.model_backend = "keras"
            return self._load_keras_models()

        self.model_backend = "keras"
        return self._load_keras_models()

    def _load_keras_models(self) -> bool:
        """Load Keras models from disk."""
        try:
            from tensorflow import keras

            if not Path(self.model_stage1_path).exists():
                self.last_error = f"Stage 1 Keras model not found: {self.model_stage1_path}"
                return False

            self.model_stage1 = keras.models.load_model(self.model_stage1_path)
            print(f"[model] Loaded Keras stage 1: {self.model_stage1_path}")

            if self.use_stage2 and self.model_stage2_path and Path(self.model_stage2_path).exists():
                self.model_stage2 = keras.models.load_model(self.model_stage2_path)
                print(f"[model] Loaded Keras stage 2: {self.model_stage2_path}")
            elif self.use_stage2:
                print(f"[model] Stage 2 Keras model not found, using stage 1 only: {self.model_stage2_path}")
                self.use_stage2 = False

            self.last_error = None
            return True

        except Exception as exc:
            self.last_error = f"Failed to load Keras models: {exc}"
            print(f"[model] {self.last_error}")
            return False

    def _load_tflite_models(self) -> bool:
        """Load TensorFlow Lite interpreters from disk."""
        try:
            if not self.tflite_stage1_path or not Path(self.tflite_stage1_path).exists():
                self.last_error = f"Stage 1 TFLite model not found: {self.tflite_stage1_path}"
                return False

            if _numpy_major_version() >= 2:
                self.last_error = (
                    "tflite-runtime is incompatible with NumPy "
                    f"{np.__version__}. Reinstall the Raspberry Pi environment with "
                    "`pip install --force-reinstall 'numpy>=1.26,<2'` and then reinstall requirements-rpi.txt."
                )
                return False

            Interpreter = self._get_tflite_interpreter_class()
            self.model_stage1 = Interpreter(model_path=str(self.tflite_stage1_path))
            self.model_stage1.allocate_tensors()
            print(f"[model] Loaded TFLite stage 1: {self.tflite_stage1_path}")

            if self.use_stage2 and self.tflite_stage2_path and Path(self.tflite_stage2_path).exists():
                self.model_stage2 = Interpreter(model_path=str(self.tflite_stage2_path))
                self.model_stage2.allocate_tensors()
                print(f"[model] Loaded TFLite stage 2: {self.tflite_stage2_path}")
            elif self.use_stage2:
                print(f"[model] Stage 2 TFLite model not found, using stage 1 only: {self.tflite_stage2_path}")
                self.use_stage2 = False

            self.model_backend = "tflite"
            self.last_error = None
            return True

        except Exception as exc:
            self.last_error = f"Failed to load TFLite models: {exc}"
            print(f"[model] {self.last_error}")
            self.model_stage1 = None
            self.model_stage2 = None
            return False

    def _get_tflite_interpreter_class(self):
        """Resolve an Interpreter without requiring full TensorFlow on Raspberry Pi."""
        try:
            from tflite_runtime.interpreter import Interpreter

            self.tflite_runtime = "tflite_runtime"
            return Interpreter
        except ImportError:
            import tensorflow as tf

            Interpreter = getattr(tf.lite, "Interpreter", None)
            if Interpreter is None:
                from tensorflow.lite.python.interpreter import Interpreter

            self.tflite_runtime = "tensorflow.lite"
            return Interpreter

    def _preprocess_model_input(self, input_batch: np.ndarray) -> np.ndarray:
        """
        Apply the same model input preprocessing as the training path.

        Keras EfficientNet preprocessing is effectively a pass-through in modern
        tf.keras, but use it when TensorFlow is available to preserve parity.
        """
        try:
            from tensorflow.keras.applications.efficientnet import preprocess_input

            return preprocess_input(input_batch)
        except Exception:
            return input_batch

    def _run_keras_inference(self, model, input_batch: np.ndarray) -> float:
        output = model.predict(input_batch, verbose=0)
        return float(np.ravel(output)[0])

    def _run_tflite_inference(self, interpreter, input_batch: np.ndarray) -> float:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_info = input_details[0]

        expected_shape = tuple(int(v) for v in input_info["shape"])
        if expected_shape != tuple(input_batch.shape):
            interpreter.resize_tensor_input(input_info["index"], input_batch.shape, strict=False)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            input_info = input_details[0]

        model_input = input_batch
        input_dtype = input_info["dtype"]
        if input_dtype != np.float32:
            scale, zero_point = input_info.get("quantization", (0.0, 0))
            if not scale or scale <= 0:
                raise ValueError(
                    f"Model TFLite bertipe {input_dtype} tapi parameter quantization tidak valid "
                    f"(scale={scale}, zero_point={zero_point}). "
                    "Pastikan model di-export dengan quantization parameter yang benar."
                )
            model_input = np.round(input_batch / scale + zero_point)
            model_input = np.clip(model_input, np.iinfo(input_dtype).min, np.iinfo(input_dtype).max)
            model_input = model_input.astype(input_dtype)
        else:
            model_input = model_input.astype(np.float32)

        interpreter.set_tensor(input_info["index"], model_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

        output_dtype = output_details[0]["dtype"]
        if output_dtype != np.float32:
            scale, zero_point = output_details[0].get("quantization", (0.0, 0))
            if scale and scale > 0:
                output = (output.astype(np.float32) - zero_point) * scale

        return float(np.ravel(output)[0])

    def _run_model(self, model, input_batch: np.ndarray) -> float:
        if self.model_backend == "tflite":
            return self._run_tflite_inference(model, input_batch)
        return self._run_keras_inference(model, input_batch)

    def predict_from_image(
        self,
        image_bgr: np.ndarray,
        return_diagnostics: bool = False,
    ) -> Tuple[Optional[float], Dict]:
        """
        Predict bilirubin from image (BGR format from camera).

        Returns:
            (prediction_value, info_dict)
        """
        try:
            if not self.ensure_models_loaded():
                return None, {
                    "error": self.last_error or "Models not loaded",
                    "success": False,
                    "model_backend": self.model_backend,
                    "model_loaded": False,
                }

            preprocessed_rgb, preprocess_mode, preprocess_diag = self.preprocessor.preprocess_image(
                image_bgr,
                return_diagnostics=True,
            )

            if preprocessed_rgb is None:
                return None, {
                    "error": self.preprocessor.last_error,
                    "success": False,
                    "preprocessing_mode": preprocess_mode,
                    "model_backend": self.model_backend,
                    "diagnostics": preprocess_diag,
                    "gatecheck_passed": preprocess_diag.get("gatecheck_passed", False),
                    "gatecheck_errors": preprocess_diag.get("gatecheck_errors", []),
                    "gatecheck_warnings": preprocess_diag.get("gatecheck_warnings", []),
                    "palette_detected": preprocess_diag.get("palette_detected", False),
                    "quality_label": preprocess_diag.get("quality_label", "failed"),
                    "quality_score": preprocess_diag.get("quality_score", 0),
                    "quality_flags": preprocess_diag.get("quality_flags", {}),
                }

            resized_rgb = cv2.resize(preprocessed_rgb, self.target_size)
            input_batch = np.expand_dims(resized_rgb, axis=0).astype(np.float32)
            input_batch = self._preprocess_model_input(input_batch)

            started = time.perf_counter()
            INFERENCE_TIMEOUT = 30.0  # detik
            if self.use_stage2 and self.model_stage2 is not None:
                pred_stage1 = self._run_model(self.model_stage1, input_batch)
                if time.perf_counter() - started > INFERENCE_TIMEOUT:
                    raise TimeoutError(f"Inference stage 1 melebihi {INFERENCE_TIMEOUT}s")
                pred_stage2 = self._run_model(self.model_stage2, input_batch)
                if time.perf_counter() - started > INFERENCE_TIMEOUT:
                    raise TimeoutError(f"Inference stage 2 melebihi {INFERENCE_TIMEOUT}s")
                bilirubin_prediction = (pred_stage1 + pred_stage2) / 2.0
                model_used = "stage1_stage2_average"
            else:
                bilirubin_prediction = self._run_model(self.model_stage1, input_batch)
                if time.perf_counter() - started > INFERENCE_TIMEOUT:
                    raise TimeoutError(f"Inference melebihi {INFERENCE_TIMEOUT}s")
                model_used = "stage1_only"

            self.last_inference_time_ms = round((time.perf_counter() - started) * 1000.0, 2)

            if not (0.0 <= bilirubin_prediction <= 30.0):
                raise ValueError(
                    f"Prediksi di luar rentang valid: {bilirubin_prediction:.2f} mg/dL. "
                    "Periksa kalibrasi model atau kualitas gambar input."
                )

            result = {
                "success": True,
                "bilirubin_prediction": bilirubin_prediction,
                "model_backend": self.model_backend,
                "model_used": model_used,
                "inference_time_ms": self.last_inference_time_ms,
                "preprocessing_mode": preprocess_mode,
                "quality_label": preprocess_diag.get("quality_label", "unknown"),
                "quality_score": preprocess_diag.get("quality_score", 0),
                "quality_flags": preprocess_diag.get("quality_flags", {}),
                "gatecheck_passed": preprocess_diag.get("gatecheck_passed", True),
                "gatecheck_errors": preprocess_diag.get("gatecheck_errors", []),
                "gatecheck_warnings": preprocess_diag.get("gatecheck_warnings", []),
                "palette_detected": preprocess_diag.get("palette_detected", False),
                "error": None,
            }

            if return_diagnostics:
                result["diagnostics"] = preprocess_diag
                result["metrics"] = preprocess_diag.get("metrics", {})

            return bilirubin_prediction, result

        except Exception as exc:
            self.last_error = str(exc)
            return None, {
                "success": False,
                "error": self.last_error,
                "bilirubin_prediction": None,
                "model_backend": self.model_backend,
            }

    def predict_from_file(
        self,
        image_path: str,
        return_diagnostics: bool = False,
    ) -> Tuple[Optional[float], Dict]:
        """Predict bilirubin from image file path."""
        try:
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                return None, {
                    "success": False,
                    "error": f"Failed to read image: {image_path}",
                }

            prediction, info = self.predict_from_image(image_bgr, return_diagnostics=return_diagnostics)
            info["image_path"] = image_path
            return prediction, info

        except Exception as exc:
            self.last_error = str(exc)
            return None, {"success": False, "error": self.last_error}

    def batch_predict(self, image_list: list, return_diagnostics: bool = False) -> list:
        """Predict on multiple images or image paths."""
        results = []
        for item in image_list:
            if isinstance(item, str):
                pred, info = self.predict_from_file(item, return_diagnostics=return_diagnostics)
            else:
                pred, info = self.predict_from_image(item, return_diagnostics=return_diagnostics)
            results.append((pred, info))
        return results

    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        return {
            "requested_backend": self.requested_model_backend,
            "model_backend": self.model_backend,
            "tflite_runtime": self.tflite_runtime,
            "stage1_loaded": self.model_stage1 is not None,
            "stage2_loaded": self.model_stage2 is not None,
            "using_stage2": self.use_stage2 and self.model_stage2 is not None,
            "stage1_path": self.model_stage1_path if self.model_backend == "keras" else self.tflite_stage1_path,
            "stage2_path": (
                self.model_stage2_path if self.model_backend == "keras" else self.tflite_stage2_path
            ) if self.use_stage2 else None,
            "target_size": self.target_size,
            "last_inference_time_ms": self.last_inference_time_ms,
            "error": self.last_error,
        }

    def __del__(self):
        """Cleanup hook."""
        pass
