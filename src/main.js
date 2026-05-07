/**
 * main.js
 * Bilirubin Detection — Tauri Frontend
 * Screen navigation + API calls + camera preview
 */

const API = 'http://127.0.0.1:7878';
const DEFAULT_PREVIEW_POLL_MS = 33;
const DEFAULT_PREVIEW_STATUS_MS = 500;

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
};

// ── API helpers ───────────────────────────────────────────────────────────
async function apiFetch(method, path, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const r = await fetch(API + path, opts);
  return r.json();
}
const apiGet  = (path)       => apiFetch('GET',  path);
const apiPost = (path, body) => apiFetch('POST', path, body);

async function getBackendStartStatus() {
  try {
    return await window.__TAURI__?.core?.invoke?.('get_backend_status');
  } catch {
    return null;
  }
}

// ── Screen navigation ─────────────────────────────────────────────────────
function showScreen(id, onEnter) {
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

function startCamera() {
  stopCamera();
  updateLastThumb();
  setFocusState(null, null);
  setCameraStatus(null);
  const img = document.getElementById('cam-img');
  const ph  = document.getElementById('cam-placeholder');

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
    img.src = `${API}/api/camera/stream?ts=${Date.now()}`;
  }

  async function tickStatus() {
    if (state.currentScreen !== 'screen-home') return;
    if (state.isCapturing) {
      state.cameraStatusTimer = setTimeout(tickStatus, state.previewStatusMs);
      return;
    }
    try {
      const d = await apiGet('/api/camera/preview/status');
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
    state.isCapturing = true;
    stopCamera();
    const btn = document.getElementById('btn-capture');
    btn.disabled = true;

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
      btn.disabled = false;
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
              ['Preview',   runtime.preview_fps ? `${runtime.preview_fps} FPS` : '?'],
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
        const s   = await apiGet('/api/status');
        const cam = s.camera ?? {};
        const runtime = s.runtime_config ?? {};
        content.innerHTML = `
          <div class="card">
            ${infoRow('Status',   cam.status   ?? '?')}
            ${infoRow('Tipe',     cam.camera_type ?? '?')}
            ${infoRow('Resolusi', cam.frame_size ? JSON.stringify(cam.frame_size) : '?')}
            ${infoRow('FPS',      cam.fps != null ? String(parseFloat(cam.fps).toFixed(0)) : '?')}
            ${infoRow('Preview',  runtime.preview_fps ? `${runtime.preview_fps} FPS` : '?')}
          </div>
          <button class="btn btn-primary" style="width:100%; margin-top:4px" onclick="App.reconnectCamera()">
            🔄 Sambung Ulang Kamera
          </button>
          <div class="info-panel" style="margin-top:10px">
            Konfigurasi lanjutan (resolusi, brightness) diatur di file src/config.py.
          </div>`;
      } catch {
        content.innerHTML = `<div style="padding:20px; color:var(--err)">Gagal memuat info kamera.</div>`;
      }
    });
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
    if (confirm('Yakin ingin keluar dari aplikasi?')) {
      if (window.__TAURI__?.core) {
        try {
          await window.__TAURI__.core.invoke('exit_app');
        } catch {
          window.close();
        }
      } else {
        window.close();
      }
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
        <div style="font-size:12px; margin-top:10px">Pastikan kartu kalibrasi, color palette, dan area kulit terlihat jelas.</div>
      </div>`;
    return;
  }

  const bili = parseFloat(result.bilirubin_prediction);
  let sevClass, level;
  if (bili >= 17) {
    sevClass = 'sev-err';
    level = 'TINGGI — Konsultasi Dokter';
  } else if (bili >= 12) {
    sevClass = 'sev-warn';
    level = 'PERHATIAN — Pantau Lebih Lanjut';
  } else {
    sevClass = 'sev-ok';
    level = 'NORMAL — Dalam Batas Aman';
  }

  const ts   = result.timestamp ? result.timestamp.slice(0, 19).replace('T', '  ') : '-';
  const qual = `${String(result.quality_label ?? '?').toUpperCase()}  (${result.quality_score ?? 0}/100)`;
  const mode = result.preprocessing_mode ?? '?';
  const palette = result.palette_detected ? 'Terdeteksi' : 'Tidak terdeteksi';
  const inference = `${result.model_backend ?? '?'} / ${result.model_used ?? '?'}`;
  const latency = result.inference_time_ms != null ? `${Number(result.inference_time_ms).toFixed(1)} ms` : '-';

  content.innerHTML = `
    <div class="result-card ${sevClass}">
      <div class="result-num">${bili.toFixed(2)}</div>
      <div class="result-unit">mg/dL</div>
      <hr class="result-hr" style="background:currentColor">
      <div class="result-level">${level}</div>
    </div>
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

waitForServer();
