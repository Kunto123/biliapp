#!/bin/bash
# launch-gui.sh — tunggu backend siap, lalu buka GUI

BINARY="/home/bilirubin/BiliApp/biliapp/src-tauri/target/release/bili-app"
API_URL="http://localhost:7878/api/gpio/status"
MAX_WAIT=30  # detik maksimal menunggu backend

echo "[gui] Menunggu backend siap..."
elapsed=0
until curl -s "$API_URL" > /dev/null 2>&1; do
    sleep 1
    elapsed=$((elapsed + 1))
    if [ "$elapsed" -ge "$MAX_WAIT" ]; then
        echo "[gui] Backend tidak merespons setelah ${MAX_WAIT}s, tetap lanjut..."
        break
    fi
done

echo "[gui] Backend siap (${elapsed}s). Tunggu window manager..."
sleep 3  # beri waktu window manager fully ready

echo "[gui] Membuka GUI..."
exec "$BINARY" --window-size=480,854
