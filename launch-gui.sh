#!/bin/bash
# launch-gui.sh
# Urutan: tunggu Wayland compositor → tunggu backend API → buka GUI

BINARY="/home/bilirubin/BiliApp/biliapp/src-tauri/target/release/bili-app"
API_URL="http://localhost:7878/api/gpio/status"
USER_ID=$(id -u)
export XDG_RUNTIME_DIR="/run/user/${USER_ID}"

# ── 1. Tunggu Wayland compositor siap ─────────────────────────────────────────
echo "[gui] Menunggu Wayland compositor..."
WAYLAND_FOUND=0
ELAPSED=0

while [ $ELAPSED -lt 60 ]; do
    for NAME in wayland-1 wayland-0; do
        if [ -S "${XDG_RUNTIME_DIR}/${NAME}" ]; then
            export WAYLAND_DISPLAY="${NAME}"
            WAYLAND_FOUND=1
            echo "[gui] Compositor siap: ${NAME} (${ELAPSED}s)"
            break 2
        fi
    done
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

if [ $WAYLAND_FOUND -eq 0 ]; then
    echo "[gui] Compositor timeout — coba lanjut tanpa Wayland"
fi

# ── 2. Beri jeda agar compositor stabil menerima window request ───────────────
sleep 4

# ── 3. Tunggu backend API siap ────────────────────────────────────────────────
echo "[gui] Menunggu backend API..."
ELAPSED=0

until curl -s "$API_URL" > /dev/null 2>&1; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
    if [ $ELAPSED -ge 30 ]; then
        echo "[gui] Backend timeout (${ELAPSED}s) — tetap lanjut"
        break
    fi
done

[ $ELAPSED -lt 30 ] && echo "[gui] Backend siap (${ELAPSED}s)"

# ── 4. Buka GUI ───────────────────────────────────────────────────────────────
echo "[gui] Membuka GUI..."
exec "$BINARY"
