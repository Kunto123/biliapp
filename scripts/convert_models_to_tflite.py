"""
Convert the desktop Keras models to TensorFlow Lite artifacts for Raspberry Pi.

Run from the project root:
    python scripts/convert_models_to_tflite.py
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
CONVERSIONS = [
    (ROOT / "best_model_stage1.keras", MODELS_DIR / "best_model_stage1.tflite"),
    (ROOT / "best_model_stage2.keras", MODELS_DIR / "best_model_stage2.tflite"),
]


def convert_model(src: Path, dest: Path) -> bool:
    if not src.exists():
        print(f"[skip] Missing source model: {src}")
        return False

    import tensorflow as tf

    model = tf.keras.models.load_model(src)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(tflite_model)
    print(f"[ok] {src.name} -> {dest}")
    return True


def main() -> int:
    converted = 0
    for src, dest in CONVERSIONS:
        if convert_model(src, dest):
            converted += 1

    if converted == 0:
        print("[error] No models were converted.")
        return 1

    print(f"[done] Converted {converted} model(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
