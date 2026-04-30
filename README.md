# Tauri + Vanilla

This template should help get you started developing with Tauri in vanilla HTML, CSS and Javascript.

## Recommended IDE Setup

- [VS Code](https://code.visualstudio.com/) + [Tauri](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode) + [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)

## Raspberry Pi 5 Ubuntu ARM64 Setup

Target runtime: Raspberry Pi 5, Ubuntu 64-bit, ArduCam Hawkeye 64MP through libcamera/rpicam, and TensorFlow Lite inference.

1. Install system packages:
   ```bash
   sudo apt update
   sudo apt install -y \
     python3 python3-venv python3-pip python3-opencv \
     rpicam-apps libcamera-apps \
     build-essential curl pkg-config libssl-dev \
     libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev
   ```

2. Install Node.js, Rust, and Tauri CLI as required by Tauri v2.

3. Create the Pi virtualenv from the app root:
   ```bash
   python3 -m venv .venv-lin --system-site-packages
   . .venv-lin/bin/activate
   pip install -U pip
   pip install -r requirements-rpi.txt
   ```

4. Convert model artifacts on a desktop/dev machine with TensorFlow installed:
   ```bash
   python scripts/convert_models_to_tflite.py
   ```
   Copy the generated `models/best_model_stage1.tflite` and `models/best_model_stage2.tflite` to the Raspberry Pi.

5. Run with Pi defaults:
   ```bash
   export BILIRUBIN_DEVICE=raspi5
   export BILIRUBIN_CAMERA_TYPE=libcamera
   export BILIRUBIN_MODEL_BACKEND=tflite
   export BILIRUBIN_USE_STAGE2=0
   python src-python/api_server.py
   ```

6. Smoke test:
   ```bash
   curl http://127.0.0.1:7878/api/status
   curl http://127.0.0.1:7878/api/camera/frame
   curl -X POST http://127.0.0.1:7878/api/capture
   ```

7. Run the Tauri app:
   ```bash
   npm run tauri dev
   ```

Useful Raspberry Pi environment overrides:

- `BILIRUBIN_CAMERA_RESOLUTION=1920x1080`
- `BILIRUBIN_CAMERA_PREVIEW_RESOLUTION=640x480`
- `BILIRUBIN_PREVIEW_POLL_MS=1000`
- `BILIRUBIN_MIN_BLUR_SCORE=60`
- `BILIRUBIN_MAX_RAW_PALETTE_MAE=95`

Capture gatecheck rejects images before inference when the card, checkerboard, gray patches, color palette, exposure, blur, or skin ROI is not acceptable.
