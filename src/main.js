/**
 * main.js
 * Bilirubin Detection — Tauri Frontend
 * Screen navigation + API calls + camera preview
 */

const API = 'http://127.0.0.1:7878';
const DEFAULT_PREVIEW_POLL_MS = 33;
const DEFAULT_PREVIEW_STATUS_MS = 500;
const CAMERA_CONTROLS_SPACE = 118;
const RISK_BANDS = [
  { min: 17, className: 'sev-err', label: 'TINGGI - perlu evaluasi klinis' },
  { min: 12, className: 'sev-warn', label: 'MENINGKAT - perlu konfirmasi' },
  { min: 0, className: 'sev-ok', label: 'RENDAH - interpretasikan sesuai usia bayi' },
];

// ── State ─────────────────────────────────────────────────────────────────
const state = {
  currentScreen: 'screen-splash',
  cameraTimer: null,
  cameraStatusTimer: null,
  useStage2: true,
  lastImageB64: null,
  lastPrediction: null,
  previewPollMs: DEFAULT_PREVIEW_POLL_MS,
  previewStatusMs: DEFAULT_PREVIEW_STATUS_MS,
  backendStatus: null,
  isCapturing: false,
  lastFocusOk: null,
  lastFocusScore: null,
  screenMetrics: null,
  nativeDisplayMetrics: null,
  gpioAvailable: false,  // true when RPi GPIO is active
  gpioReady: true,       // false = waiting for limit switch to return HIGH
};

// ── Runtime viewport measurement ─────────────────────────────────────────
function getScreenMetrics() {
  const vv = window.visualViewport;
  const dpr = window.devicePixelRatio || 1;
  const native = state.nativeDisplayMetrics;
  const cssWidth = Math.round(native?.css_width || vv?.width || window.innerWidth || document.documentElement.clientWidth || screen.width);
  const cssHeight = Math.round(native?.css_height || vv?.height || window.innerHeight || document.documentElement.clientHeight || screen.height);
  const offsetLeft = Math.round(vv?.offsetLeft || 0);
  const offsetTop = Math.round(vv?.offsetTop || 0);
  return {
    css_width: cssWidth,
    css_height: cssHeight,
    physical_width: Math.round(native?.monitor_width || cssWidth * dpr),
    physical_height: Math.round(native?.monitor_height || cssHeight * dpr),
    screen_width: screen.width,
    screen_height: screen.height,
    device_pixel_ratio: native?.scale_factor || dpr,
    offset_left: offsetLeft,
    offset_top: offsetTop,
    orientation: cssWidth >= cssHeight ? 'landscape' : 'portrait',
    native,
  };
}

function applyScreenMetrics() {
  const metrics = getScreenMetrics();
  state.screenMetrics = metrics;

  const root = document.documentElement;
  const controlsSpace = metrics.orientation === 'landscape'
    ? Math.min(CAMERA_CONTROLS_SPACE, Math.max(84, Math.round(metrics.css_height * 0.24)))
    : CAMERA_CONTROLS_SPACE;
  const controlsTop = Math.max(0, metrics.offset_top + metrics.css_height - controlsSpace);

  root.style.setProperty('--app-width', `${metrics.css_width}px`);
  root.style.setProperty('--app-height', `${metrics.css_height}px`);
  root.style.setProperty('--viewport-offset-left', `${metrics.offset_left}px`);
  root.style.setProperty('--viewport-offset-top', `${metrics.offset_top}px`);
  root.style.setProperty('--camera-controls-space', `${controlsSpace}px`);
  root.style.setProperty('--camera-controls-top', `${controlsTop}px`);

  return metrics;
}

function installScreenMetricsWatcher() {
  const update = () => requestAnimationFrame(applyScreenMetrics);
  applyScreenMetrics();
  window.addEventListener('resize', update);
  window.addEventListener('orientationchange', update);
  window.visualViewport?.addEventListener('resize', update);
  window.visualViewport?.addEventListener('scroll', update);
}

async function syncNativeDisplayMetrics() {
  const invoke = window.__TAURI__?.core?.invoke;
  if (!invoke) {
    return null;
  }

  try {
    const metrics = await invoke('sync_display_metrics');
    state.nativeDisplayMetrics = metrics;
    applyScreenMetrics();
    return metrics;
  } catch (err) {
    console.warn('Failed to sync native display metrics:', err);
    return null;
  }
}

// ── API helpers ───────────────────────────────────────────────────────────
async function apiFetch(method, path, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const r = await fetch(API + path, opts);
  return r.json();
}
const apiGet  = (path)       => apiFetch('GET',  path);
const apiPost = (path, body) => apiFetch('POST', path, body);
const apiPut  = (path, body) => apiFetch('PUT',  path, body);

async function getBackendStartStatus() {
  try {
    return await window.__TAURI__?.core?.invoke?.('get_backend_status');
  } catch {
    return null;
  }
}

// ── Screen navigation ─────────────────────────────────────────────────────
function showScreen(id, onEnter) {
  applyScreenMetrics();
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  state.currentScreen = id;
  document.getElementById(id).classList.add('active');
  if (onEnter) onEnter();
}

// ── Toast ─────────────────────────────────────────────────────────────────
let toastTimer = null;
function toast(msg, duration = 2500) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), duration);
}

// ── Camera preview ────────────────────────────────────────────────────────
function setFocusState(focusOk, focusScore) {
  const cls = focusOk === true ? 'focus-ok' : focusOk === false ? 'focus-warn' : 'focus-idle';
  state.lastFocusOk = focusOk === true ? true : focusOk === false ? false : null;
  state.lastFocusScore = typeof focusScore === 'number' ? focusScore : null;

  [document.getElementById('camera-wrap'), document.getElementById('focus-reticle')]
    .filter(Boolean)
    .forEach(el => {
      el.classList.remove('focus-ok', 'focus-warn', 'focus-idle');
      el.classList.add(cls);
    });
}

function setCameraStatus(status) {
  const el = document.getElementById('camera-status');
  if (!el) return;

  el.classList.remove('is-visible', 'is-warn', 'is-idle');
  el.textContent = '';

  if (!status) return;

  if (status.busy) {
    el.textContent = 'Kamera sedang capture';
    el.classList.add('is-visible', 'is-idle');
    return;
  }

  if (status.available === false) {
    el.textContent = 'Menunggu kamera...';
    el.classList.add('is-visible', 'is-idle');
    return;
  }

  if (status.fps_ok === false && typeof status.fps === 'number') {
    const fps = Number(status.fps).toFixed(1);
    const minFps = status.min_fps ?? 30;
    el.textContent = `Preview ${fps} FPS, target ${minFps} FPS`;
    el.classList.add('is-visible', 'is-warn');
  }
}

function updateLastThumb() {
  const img = document.getElementById('last-thumb-img');
  const empty = document.getElementById('last-thumb-empty');
  if (!img || !empty) return;

  if (state.lastImageB64) {
    img.src = 'data:image/jpeg;base64,' + state.lastImageB64;
    img.style.display = 'block';
    empty.style.display = 'none';
  } else {
    img.removeAttribute('src');
    img.style.display = 'none';
    empty.style.display = 'block';
  }
}

function resolutionValue(res) {
  if (!res) return '';
  const width = Array.isArray(res) ? res[0] : res.width;
  const height = Array.isArray(res) ? res[1] : res.height;
  return `${width}x${height}`;
}

function parseResolutionValue(value) {
  const [width, height] = String(value).split('x').map(v => parseInt(v, 10));
  return { width, height };
}

function optionHtml(value, label, selectedValue) {
  const selected = String(value) === String(selectedValue) ? ' selected' : '';
  return `<option value="${value}"${selected}>${label}</option>`;
}

function renderResolutionOptions(selectedValue, presets) {
  const values = presets.includes(selectedValue) ? presets : [selectedValue, ...presets].filter(Boolean);
  return values.map(value => optionHtml(value, value, selectedValue)).join('');
}

function updateCaptureButton() {
  const btn = document.getElementById('btn-capture');
  if (!btn) return;
  const gpioBlocked = state.gpioAvailable && !state.gpioReady;
  btn.disabled = gpioBlocked || state.isCapturing;
}

function startCamera() {
  stopCamera();
  updateLastThumb();
  setFocusState(null, null);
  setCameraStatus(null);
  const img = document.getElementById('cam-img');
  const ph  = document.getElementById('cam-placeholder');
  let streamStarted = false;

  const openPreviewStream = () => {
    if (!img || streamStarted || state.isCapturing) return;
    streamStarted = true;
    img.src = `${API}/api/camera/stream?ts=${Date.now()}`;
  };

  if (img) {
    img.onload = () => {
      img.style.display = 'block';
      ph.style.display = 'none';
    };
    img.onerror = () => {
      img.style.display = 'none';
      ph.style.display = 'flex';
      setCameraStatus({ available: false });
    };
  }

  if (state.isCapturing) {
    setCameraStatus({ busy: true });
  } else {
    openPreviewStream();
  }

  async function tickStatus() {
    if (state.currentScreen !== 'screen-home') return;
    if (state.isCapturing) {
      setCameraStatus({ busy: true });
      state.cameraStatusTimer = setTimeout(tickStatus, state.previewStatusMs);
      return;
    }
    openPreviewStream();
    try {
      const [d, gpioData] = await Promise.all([
        apiGet('/api/camera/preview/status'),
        apiGet('/api/gpio/status').catch(() => null),
      ]);

      // Update GPIO state and handle auto-trigger from limit switch
      if (gpioData) {
        state.gpioAvailable = !!gpioData.available;
        state.gpioReady = gpioData.capture_ready !== false;
        if (gpioData.capture_triggered && state.gpioReady && !state.isCapturing) {
          state.cameraStatusTimer = setTimeout(tickStatus, state.previewStatusMs);
          App.startCapture();
          return;
        }
      }
      updateCaptureButton();

      if (d.available) {
        img.style.display = 'block';
        ph.style.display  = 'none';
        setFocusState(typeof d.focus_ok === 'boolean' ? d.focus_ok : null, d.focus_score);
        setCameraStatus(d);
      } else {
        img.style.display = 'none';
        ph.style.display  = 'flex';
        setFocusState(null, null);
        setCameraStatus(d);
      }

      // GPIO waiting message overrides camera status
      if (state.gpioAvailable && !state.gpioReady) {
        const el = document.getElementById('camera-status');
        if (el) {
          el.textContent = 'Menunggu sensor — lepaskan limit switch (GPIO 8)';
          el.classList.remove('is-warn');
          el.classList.add('is-visible', 'is-idle');
        }
      }
    } catch {
      setFocusState(null, null);
      setCameraStatus({ available: false });
    }
    state.cameraStatusTimer = setTimeout(tickStatus, state.previewStatusMs);
  }
  tickStatus();
}

function stopCamera() {
  clearTimeout(state.cameraTimer);
  clearTimeout(state.cameraStatusTimer);
  state.cameraTimer = null;
  state.cameraStatusTimer = null;
  const img = document.getElementById('cam-img');
  if (img) {
    img.onload = null;
    img.onerror = null;
    img.removeAttribute('src');
    img.style.display = 'none';
  }
  setCameraStatus(null);
}

// ── App public API ────────────────────────────────────────────────────────
const App = {

  // ── Home ────────────────────────────────────────────────────────────────
  goHome() {
    showScreen('screen-home', () => {
      updateLastThumb();
      startCamera();
    });
  },

  // ── Menu ────────────────────────────────────────────────────────────────
  goMenu() {
    stopCamera();
    showScreen('screen-menu');
  },

  // ── Capture ─────────────────────────────────────────────────────────────
  async startCapture() {
    if (state.isCapturing) return;
    if (state.gpioAvailable && !state.gpioReady) {
      toast('Sensor belum siap — tunggu limit switch kembali ke posisi awal');
      return;
    }
    state.isCapturing = true;
    stopCamera();
    updateCaptureButton();

    // Show capture screen with loading indicator
    showScreen('screen-capture');
    const content = document.getElementById('capture-content');
    content.innerHTML = `
      <div class="capture-loading">
        <div class="mini-spinner"></div>
        Mengambil gambar dan menganalisis…
      </div>`;

    try {
      const result = await apiPost('/api/capture');
      renderCaptureResult(result);
    } catch (e) {
      content.innerHTML = `
        <div class="result-card sev-err" style="padding:20px">
          <div style="font-size:18px; font-weight:700; margin-bottom:8px">Koneksi Gagal</div>
          <div style="font-size:13px">${e.message}</div>
        </div>`;
    } finally {
      state.isCapturing = false;
      updateCaptureButton();
      if (state.currentScreen === 'screen-home') {
        startCamera();
      }
    }
  },

  // ── History ──────────────────────────────────────────────────────────────
  async goHistory() {
    showScreen('screen-history', async () => {
      const [histRes, statsRes] = await Promise.all([
        apiGet('/api/history?limit=10'),
        apiGet('/api/stats'),
      ]);

      // Stats bar
      const stats = statsRes || {};
      const meanStr = (stats.mean_bilirubin != null)
        ? parseFloat(stats.mean_bilirubin).toFixed(2) + ' mg/dL'
        : 'N/A';
      document.getElementById('history-stats').innerHTML = `
        <div class="stat-col"><div class="stat-label">Total</div><div class="stat-value">${stats.total_predictions ?? 0}</div></div>
        <div class="stat-col"><div class="stat-label">Berhasil</div><div class="stat-value">${stats.successful ?? 0}</div></div>
        <div class="stat-col"><div class="stat-label">Gagal</div><div class="stat-value">${stats.failed ?? 0}</div></div>
        <div class="stat-col"><div class="stat-label">Rata-rata</div><div class="stat-value">${meanStr}</div></div>`;

      // Table
      const records = histRes?.records ?? [];
      if (!records.length) {
        document.getElementById('history-table').textContent = 'Belum ada data prediksi.';
        return;
      }
      const header = `${'#'.padEnd(3)} ${'Waktu'.padEnd(19)} ${'mg/dL'.padEnd(7)} ${'Kualitas'.padEnd(8)} Mode\n${'─'.repeat(56)}\n`;
      const rows = [...records].reverse().slice(0, 10).map((r, i) => {
        const ts   = String(r.timestamp ?? 'N/A').slice(0, 19).replace('T', ' ');
        const bili = r.bilirubin_prediction != null ? parseFloat(r.bilirubin_prediction).toFixed(2) : 'N/A';
        const q    = String(r.quality_label ?? 'N/A');
        const m    = String(r.preprocessing_mode ?? 'N/A');
        return `${String(i + 1).padEnd(3)} ${ts.padEnd(19)} ${bili.padEnd(7)} ${q.padEnd(8)} ${m}`;
      }).join('\n');
      document.getElementById('history-table').textContent = header + rows;
    });
  },

  // ── Last image ────────────────────────────────────────────────────────────
  goLastImage() {
    stopCamera();
    showScreen('screen-image', () => {
      const content = document.getElementById('image-content');
      if (!state.lastImageB64) {
        content.innerHTML = `<div style="text-align:center; padding:40px; color:var(--text-sub); font-size:14px">
          Belum ada foto.<br>Lakukan Capture terlebih dahulu.</div>`;
        return;
      }
      content.innerHTML = `
        <div class="card" style="padding:8px; margin-bottom:8px">
          <img src="data:image/jpeg;base64,${state.lastImageB64}" class="image-preview" alt="Last capture" />
        </div>
        <div style="text-align:center; font-size:12px; color:var(--text-sub)">Foto Terakhir</div>`;
    });
  },

  // ── System info ───────────────────────────────────────────────────────────
  async goSysInfo() {
    showScreen('screen-sysinfo', async () => {
      const content = document.getElementById('sysinfo-content');
      content.innerHTML = `<div style="padding:20px; text-align:center; color:var(--text-sub)">Memuat…</div>`;
      try {
        const s = await apiGet('/api/status');
        const cam = s.camera ?? {};
        const mdl = s.models ?? {};
        const runtime = s.runtime_config ?? {};
        content.innerHTML = buildInfoSections([
          {
            title: 'KAMERA',
            rows: [
              ['Status',    cam.status ?? '?'],
              ['Tipe',      cam.camera_type ?? '?'],
              ['Resolusi',  cam.frame_size ? JSON.stringify(cam.frame_size) : '?'],
              ['FPS',       cam.fps != null ? String(parseFloat(cam.fps).toFixed(0)) : '?'],
            ],
          },
          {
            title: 'MODEL',
            rows: [
              ['Backend',   mdl.model_backend ?? runtime.model_backend ?? '?'],
              ['Stage 1',   mdl.stage1_loaded ? 'Dimuat ✓' : 'Tidak tersedia'],
              ['Stage 2',   mdl.stage2_loaded ? 'Dimuat ✓' : 'Tidak tersedia'],
              ['Digunakan', mdl.using_stage2  ? 'Stage 1 + 2' : 'Stage 1 saja'],
              ['Latency',   mdl.last_inference_time_ms != null ? `${mdl.last_inference_time_ms} ms` : '-'],
            ],
          },
          {
            title: 'RUNTIME',
            rows: [
              ['Device',    runtime.device_profile ?? 'desktop'],
              ['Preview',   runtime.preview_fps != null ? (runtime.preview_fps === 0 ? 'Auto' : `${runtime.preview_fps} FPS`) : '?'],
              ['Server',    s.initialized ? 'Aktif' : 'Tidak aktif'],
            ],
          },
          {
            title: 'PENYIMPANAN',
            rows: [
              ['Dir. Log',    String(s.logs_directory  ?? '?')],
              ['Dir. Gambar', String(s.images_directory ?? '?')],
              ['Total Foto',  String(s.total_captures  ?? 0)],
            ],
          },
        ]);
      } catch {
        content.innerHTML = `<div style="padding:20px; color:var(--err)">Gagal memuat status sistem.</div>`;
      }
    });
  },

  // ── Settings ──────────────────────────────────────────────────────────────
  goSettings() {
    stopCamera();
    showScreen('screen-settings');
  },

  // ── Camera config ─────────────────────────────────────────────────────────
  async goCameraConfig() {
    showScreen('screen-camera-config', async () => {
      const content = document.getElementById('camera-config-content');
      content.innerHTML = `<div style="padding:20px; text-align:center; color:var(--text-sub)">Memuat…</div>`;
      try {
        const [configResult, devicesResult, statusResult] = await Promise.allSettled([
          apiGet('/api/camera/config'),
          apiGet('/api/camera/devices'),
          apiGet('/api/status'),
        ]);
        if (configResult.status !== 'fulfilled') {
          throw configResult.reason;
        }
        const configResp = configResult.value;
        const devicesResp = devicesResult.status === 'fulfilled' ? devicesResult.value : { devices: [], error: devicesResult.reason?.message };
        const statusResp = statusResult.status === 'fulfilled' ? statusResult.value : { camera: {} };
        const settings = configResp.settings ?? {};
        const devices = devicesResp.devices ?? [];
        const cam = statusResp.camera ?? {};
        const scanWarning = devicesResp.success === false || devicesResult.status !== 'fulfilled'
          ? `<div class="info-panel" style="margin-top:10px">Scan kamera tidak lengkap: ${esc(devicesResp.error || 'kamera sedang dipakai')}</div>`
          : '';
        const previewValue = resolutionValue(settings.preview_resolution);
        const captureValue = resolutionValue(settings.capture_resolution);
        const fpsValue = settings.fps ?? 0;
        const cameraIndex = settings.camera_index ?? 0;
        const deviceOptions = devices.length
          ? devices.map(d => {
              const details = d.width && d.height ? ` - ${d.width}x${d.height}${d.fps ? ` @ ${d.fps} FPS` : ''}` : '';
              return optionHtml(d.index, `${d.name ?? `Camera ${d.index}`}${details}`, cameraIndex);
            }).join('')
          : optionHtml(cameraIndex, `Camera ${cameraIndex}`, cameraIndex);
        content.innerHTML = `
          <div class="card">
            ${infoRow('Status',   cam.status   ?? '?')}
            ${infoRow('Tipe',     cam.camera_type ?? '?')}
            ${infoRow('Resolusi', cam.frame_size ? JSON.stringify(cam.frame_size) : '?')}
            ${infoRow('FPS',      cam.fps != null ? String(parseFloat(cam.fps).toFixed(0)) : '?')}
          </div>
          <div class="card camera-form">
            <label class="field-row">
              <span>Kamera</span>
              <select id="camera-index-select">${deviceOptions}</select>
            </label>
            <label class="field-row">
              <span>Resolusi preview</span>
              <select id="preview-resolution-select">
                ${renderResolutionOptions(previewValue, ['320x240', '640x480', '1280x720'])}
              </select>
            </label>
            <label class="field-row">
              <span>Resolusi capture</span>
              <select id="capture-resolution-select">
                ${renderResolutionOptions(captureValue, ['1280x720', '1920x1080', '3840x2160'])}
              </select>
            </label>
            <label class="field-row">
              <span>FPS</span>
              <select id="camera-fps-select">
                ${optionHtml(0, 'Auto', fpsValue)}
                ${optionHtml(15, '15 FPS', fpsValue)}
                ${optionHtml(24, '24 FPS', fpsValue)}
                ${optionHtml(30, '30 FPS', fpsValue)}
                ${optionHtml(60, '60 FPS', fpsValue)}
              </select>
            </label>
          </div>
          <div class="button-stack">
            <button class="btn btn-primary" onclick="App.saveCameraConfig()">Simpan & Terapkan</button>
            <button class="btn btn-secondary" onclick="App.goCameraConfig()">Scan Ulang Kamera</button>
          </div>
          <button class="btn btn-primary" style="width:100%; margin-top:4px" onclick="App.reconnectCamera()">
            🔄 Sambung Ulang Kamera
          </button>
          <div class="info-panel" style="margin-top:10px">
            Setting hanya tersimpan sementara di memori dan akan hilang saat server di-restart. Untuk perubahan permanen, edit config.py lalu restart server.
          </div>
          ${scanWarning}`;
      } catch (err) {
        content.innerHTML = `<div style="padding:20px; color:var(--err)">Gagal memuat info kamera: ${esc(err?.message || err || 'error tidak diketahui')}</div>`;
      }
    });
  },

  async saveCameraConfig() {
    const cameraIndexEl = document.getElementById('camera-index-select');
    const previewEl = document.getElementById('preview-resolution-select');
    const captureEl = document.getElementById('capture-resolution-select');
    const fpsEl = document.getElementById('camera-fps-select');
    if (!cameraIndexEl || !previewEl || !captureEl || !fpsEl) return;

    const payload = {
      camera_index: parseInt(cameraIndexEl.value, 10),
      preview_resolution: parseResolutionValue(previewEl.value),
      capture_resolution: parseResolutionValue(captureEl.value),
      fps: parseInt(fpsEl.value, 10),
    };

    try {
      const r = await apiPut('/api/camera/config', payload);
      if (r.success) {
        toast('Setting kamera diterapkan');
        await this.goCameraConfig();
      } else {
        toast(r.error || r.detail || 'Gagal menerapkan setting kamera');
      }
    } catch {
      toast('Gagal menghubungi server');
    }
  },

  async reconnectCamera() {
    try {
      const r = await apiPost('/api/camera/reconnect');
      toast(r.success ? '✓ Kamera disambungkan ulang' : '✗ Kamera tidak ditemukan');
    } catch {
      toast('✗ Gagal menghubungi server');
    }
  },

  // ── Model select ──────────────────────────────────────────────────────────
  goModelSelect() {
    showScreen('screen-model', () => {
      const r2 = document.getElementById('radio-stage2');
      const r1 = document.getElementById('radio-stage1');
      if (state.useStage2) { r2.checked = true; } else { r1.checked = true; }
      document.getElementById('model-msg').textContent = '';
    });
  },

  async applyModelSettings() {
    const useStage2 = document.getElementById('radio-stage2').checked;
    try {
      const r = await apiPost('/api/settings/model', { use_stage2: useStage2 });
      if (r.success) {
        state.useStage2 = useStage2;
        const label = useStage2 ? 'Stage 1 + Stage 2' : 'Stage 1 saja';
        document.getElementById('model-msg').textContent = `✓ Mode inferensi: ${label}`;
      } else {
        document.getElementById('model-msg').style.color = 'var(--err)';
        document.getElementById('model-msg').textContent = '✗ Gagal menerapkan pengaturan';
      }
    } catch {
      document.getElementById('model-msg').style.color = 'var(--err)';
      document.getElementById('model-msg').textContent = '✗ Gagal menghubungi server';
    }
  },

  // ── Logging prefs ─────────────────────────────────────────────────────────
  async goLoggingPrefs() {
    showScreen('screen-logging', async () => {
      const content = document.getElementById('logging-content');
      content.innerHTML = `<div style="padding:20px; text-align:center; color:var(--text-sub)">Memuat…</div>`;
      try {
        const s = await apiGet('/api/status');
        content.innerHTML = `
          <div class="card">
            ${infoRow('Dir. Log',    String(s.logs_directory   ?? '?'))}
            ${infoRow('Dir. Gambar', String(s.images_directory ?? '?'))}
            ${infoRow('Total Foto',  String(s.total_captures  ?? 0))}
            ${infoRow('Format',      'CSV (.csv)')}
          </div>
          <div class="divider" style="margin:12px 0"></div>
          <button class="btn btn-soft" style="width:100%;text-align:left;padding-left:16px" onclick="App.cleanupImages()">
            🗑 Bersihkan Gambar Lama (&gt; 7 hari)
          </button>`;
      } catch {
        content.innerHTML = `<div style="padding:20px; color:var(--err)">Gagal memuat info logging.</div>`;
      }
    });
  },

  // ── Cleanup images ────────────────────────────────────────────────────────
  async cleanupImages() {
    try {
      const r = await apiPost('/api/images/cleanup');
      toast(r.success ? `✓ ${r.deleted} gambar lama dihapus` : '✗ Gagal membersihkan gambar');
    } catch {
      toast('✗ Gagal menghubungi server');
    }
  },

  // ── Exit ──────────────────────────────────────────────────────────────────
  async exitApp() {
    if (window.__TAURI__?.core) {
      try {
        await window.__TAURI__.core.invoke('exit_app');
      } catch {
        window.close();
      }
    } else {
      window.close();
    }
  },
};

// ── Build helpers ─────────────────────────────────────────────────────────

function infoRow(label, value) {
  return `<div class="info-row"><div class="info-label">${label}</div><div class="info-value">${esc(value)}</div></div>`;
}

function buildInfoSections(sections) {
  return sections.map(sec => `
    <div class="section-hdr">${sec.title}</div>
    <div class="card" style="border-radius:0 0 10px 10px; margin-bottom:0">
      ${sec.rows.map(([l, v]) => infoRow(l, v)).join('')}
    </div>`).join('');
}

function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function classifyBilirubin(value) {
  if (!Number.isFinite(value)) {
    return { className: 'sev-err', label: 'HASIL TIDAK VALID' };
  }
  return RISK_BANDS.find(band => value >= band.min) ?? RISK_BANDS[RISK_BANDS.length - 1];
}

function renderCaptureResult(result) {
  const content = document.getElementById('capture-content');
  if (result?.image_b64) {
    state.lastImageB64 = result.image_b64;
    updateLastThumb();
  }

  if (!result || !result.success) {
    const errMsg = result?.error ?? 'Error tidak diketahui';
    const gateErrors = result?.gatecheck_errors ?? [];
    const gateWarnings = result?.gatecheck_warnings ?? [];
    const title = result?.gatecheck_passed === false ? 'Foto Ditolak' : 'Prediksi Gagal';
    const helper = result?.gatecheck_passed === false
      ? 'Pastikan kartu kalibrasi, color palette, dan area kulit terlihat jelas.'
      : 'Periksa status kamera dan model, lalu coba capture ulang.';
    const detail = gateErrors.length
      ? `<ul class="gate-list">${gateErrors.map(e => `<li>${esc(e)}</li>`).join('')}</ul>`
      : `<div style="font-size:13px">${esc(errMsg)}</div>`;
    const warnings = gateWarnings.length
      ? `<div class="gate-warn">${gateWarnings.map(esc).join('<br>')}</div>`
      : '';
    content.innerHTML = `
      <div class="result-card sev-err" style="padding:16px 20px">
        <div style="font-size:18px; font-weight:700; margin-bottom:8px">${title}</div>
        ${detail}
        ${warnings}
        <div style="font-size:12px; margin-top:10px">${helper}</div>
      </div>`;
    return;
  }

  const bili = Number.parseFloat(result.bilirubin_prediction);
  if (!Number.isFinite(bili)) {
    content.innerHTML = `
      <div class="result-card sev-err" style="padding:16px 20px">
        <div style="font-size:18px; font-weight:700; margin-bottom:8px">Prediksi Gagal</div>
        <div style="font-size:13px">Nilai bilirubin dari server tidak valid.</div>
      </div>`;
    return;
  }
  const risk = classifyBilirubin(bili);
  const sevClass = risk.className;
  const level = risk.label;

  const ts   = result.timestamp ? result.timestamp.slice(0, 19).replace('T', '  ') : '-';
  const qual = `${String(result.quality_label ?? '?').toUpperCase()}  (${result.quality_score ?? 0}/100)`;
  const mode = result.preprocessing_mode ?? '?';
  const palette = result.palette_detected ? 'Terdeteksi' : 'Tidak terdeteksi';
  const inference = `${result.model_backend ?? '?'} / ${result.model_used ?? '?'}`;
  const latency = result.inference_time_ms != null ? `${Number(result.inference_time_ms).toFixed(1)} ms` : '-';

  const rawAlignedReason = result.palette_detected
    ? 'mode raw_aligned - koreksi warna tidak diterapkan karena kualitas kalibrasi belum cukup stabil'
    : 'palette tidak terdeteksi - mode raw_aligned';
  const rawAlignedBanner = mode === 'raw_aligned'
    ? `<div class="result-mode-warn">Peringatan: ${rawAlignedReason}, akurasi prediksi lebih rendah</div>`
    : '';
  const logWarnBanner = result.log_warning
    ? `<div class="result-mode-warn">Peringatan: ${esc(result.log_warning)}</div>`
    : '';

  content.innerHTML = `
    <div class="result-card ${sevClass}">
      <div class="result-num">${bili.toFixed(2)}</div>
      <div class="result-unit">mg/dL</div>
      <hr class="result-hr" style="background:currentColor">
      <div class="result-level">${level}</div>
    </div>
    <div class="result-clinical-note">Skrining awal. Interpretasi klinis tetap perlu mempertimbangkan usia bayi dalam jam, berat badan, prematuritas, dan pemeriksaan tenaga kesehatan.</div>
    ${rawAlignedBanner}
    ${logWarnBanner}
    <div class="card">
      ${infoRow('Waktu',    ts)}
      ${infoRow('Kualitas', qual)}
      ${infoRow('Palette',  palette)}
      ${infoRow('Mode',     mode)}
      ${infoRow('Inferensi', inference)}
      ${infoRow('Latency',  latency)}
    </div>`;

  state.lastPrediction = bili;
}

// ── Startup — wait for server ─────────────────────────────────────────────
async function waitForServer() {
  const statusEl = document.getElementById('splash-status');
  let attempts = 0;
  const maxAttempts = 60; // 30 seconds

  while (attempts < maxAttempts) {
    attempts++;
    statusEl.textContent = `Menghubungkan ke server… (${attempts})`;
    try {
      const s = await apiGet('/api/status');
      if (s) {
        const runtime = s.runtime_config ?? {};
        state.previewPollMs = runtime.preview_poll_ms ?? DEFAULT_PREVIEW_POLL_MS;
        state.previewStatusMs = Math.max(DEFAULT_PREVIEW_STATUS_MS, runtime.preview_poll_ms ?? DEFAULT_PREVIEW_STATUS_MS);
        statusEl.textContent = '✓ Terhubung!';
        await new Promise(r => setTimeout(r, 400));
        App.goHome();
        return;
      }
    } catch { /* server not ready */ }
    await new Promise(r => setTimeout(r, 500));
  }

  state.backendStatus = await getBackendStartStatus();
  if (state.backendStatus?.error) {
    statusEl.textContent = `Server gagal start: ${state.backendStatus.error}`;
  } else {
    statusEl.textContent = '✗ Server tidak merespons. Coba restart aplikasi.';
  }
}

// ── Init ──────────────────────────────────────────────────────────────────
window.App = App;
window.getScreenMetrics = getScreenMetrics;
window.syncNativeDisplayMetrics = syncNativeDisplayMetrics;

// Append toast element
const toastEl = document.createElement('div');
toastEl.id = 'toast';
document.body.appendChild(toastEl);

// Keyboard shortcuts (F11 / Escape for fullscreen toggle via Tauri)
document.addEventListener('keydown', e => {
  if (e.key === 'F11') e.preventDefault();
  if (e.key === 'Escape' && state.currentScreen === 'screen-home') {
    // do nothing on home
  }
});

installScreenMetricsWatcher();
syncNativeDisplayMetrics().finally(waitForServer);
