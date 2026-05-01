/**
 * ECHO Frontend
 * Connects to the Flask SSE stream, renders waveform, speaker bars,
 * timeline, event log, and latency chart in real time.
 */

const API_BASE = "http://127.0.0.1:5000";

// ── State ──────────────────────────────────────────────────────────────────────
const state = {
  running: false,
  eventSource: null,
  speakerTime: { "Speaker 1": 0, "Speaker 2": 0 },
  latencyHistory: [],
  chunkCount: 0,
  speechCount: 0,
  silenceCount: 0,
  lastChannels: "—",
};

// ── DOM refs ───────────────────────────────────────────────────────────────────
const micDot    = document.getElementById("mic-dot");
const micLabel  = document.getElementById("mic-label");
const btnStart  = document.getElementById("btn-start");
const btnStop   = document.getElementById("btn-stop");

const statChunks   = document.getElementById("stat-chunks");
const statSpeech   = document.getElementById("stat-speech");
const statSilence  = document.getElementById("stat-silence");
const statLatency  = document.getElementById("stat-latency");
const statChannels = document.getElementById("stat-channels");
const statSaved    = document.getElementById("stat-saved");

const storageInfo  = document.getElementById("storage-info");
const recordingsLink = document.getElementById("recordings-link");

const s1Bar  = document.getElementById("speaker-1-bar");
const s2Bar  = document.getElementById("speaker-2-bar");
const s1Time = document.getElementById("speaker-1-time");
const s2Time = document.getElementById("speaker-2-time");

const timeline = document.getElementById("timeline");
const eventLog = document.getElementById("event-log");

const waveCanvas   = document.getElementById("waveform-canvas");
const latencyCanvas = document.getElementById("latency-canvas");
const wCtx  = waveCanvas.getContext("2d");
const lCtx  = latencyCanvas.getContext("2d");

// ── Canvas sizing ──────────────────────────────────────────────────────────────
function resizeCanvases() {
  waveCanvas.width    = waveCanvas.offsetWidth;
  waveCanvas.height   = waveCanvas.offsetHeight;
  latencyCanvas.width  = latencyCanvas.offsetWidth;
  latencyCanvas.height = latencyCanvas.offsetHeight;
}
window.addEventListener("resize", resizeCanvases);
resizeCanvases();

// ── Mic status helpers ─────────────────────────────────────────────────────────
function setMicStatus(status) {
  micDot.className = `dot ${status}`;
  const labels = { listening: "Listening", processing: "Processing", idle: "Idle" };
  micLabel.textContent = labels[status] || status;
}

// ── Pipeline control ───────────────────────────────────────────────────────────
async function startPipeline() {
  try {
    const res = await fetch(`${API_BASE}/pipeline/start`, { method: "POST" });
    const data = await res.json();
    if (data.status === "started" || data.status === "already_running") {
      state.running = true;
      btnStart.disabled = true;
      btnStop.disabled  = false;
      setMicStatus("listening");
      appendLog("sys", "▶ Pipeline started");
      connectSSE();
    }
  } catch (err) {
    appendLog("sys", `✗ Could not start pipeline: ${err.message}`);
  }
}

async function stopPipeline() {
  try {
    const res = await fetch(`${API_BASE}/pipeline/stop`, { method: "POST" });
    const data = await res.json();
    state.running = false;
    btnStart.disabled = false;
    btnStop.disabled  = true;
    setMicStatus("idle");
    appendLog("sys", `■ Pipeline stopped — ${JSON.stringify(data.stats ?? {})}`);
    if (state.eventSource) {
      state.eventSource.close();
      state.eventSource = null;
    }
  } catch (err) {
    appendLog("sys", `✗ Could not stop pipeline: ${err.message}`);
  }
}

btnStart.addEventListener("click", startPipeline);
btnStop.addEventListener("click",  stopPipeline);

// ── SSE connection ─────────────────────────────────────────────────────────────
function connectSSE() {
  if (state.eventSource) state.eventSource.close();

  const es = new EventSource(`${API_BASE}/stream`);
  state.eventSource = es;

  es.onmessage = (e) => {
    try {
      const event = JSON.parse(e.data);
      handleEvent(event);
    } catch (_) {}
  };

  es.onerror = () => {
    appendLog("sys", "⚠ SSE connection lost — retrying…");
    setMicStatus("idle");
  };
}

// ── Event handler ──────────────────────────────────────────────────────────────
function handleEvent(event) {
  if (event.type === "heartbeat" || event.type === "connected") return;

  if (event.type === "pipeline_status") {
    appendLog("sys", `Pipeline → ${event.status}`);
    setMicStatus(event.status === "started" ? "listening" : "idle");
    return;
  }

  if (event.type === "chunk") {
    setMicStatus("processing");
    processChunk(event);
    setTimeout(() => { if (state.running) setMicStatus("listening"); }, 400);
  }
}

function processChunk(chunk) {
  const { chunk_id, start_time, end_time, speaker, latency, quality, channels, segments } = chunk;
  const duration = (end_time - start_time) || 1.5;

  // ── Stats ──────────────────────────────────────────────────────────────────
  state.chunkCount++;
  statChunks.textContent = state.chunkCount;

  if (quality === "clean") {
    state.speechCount++;
    statSpeech.textContent = state.speechCount;
  } else {
    state.silenceCount++;
    statSilence.textContent = state.silenceCount;
  }

  if (channels) {
    state.lastChannels = channels === 1 ? "Mono" : "Stereo";
    statChannels.textContent = state.lastChannels;
  }

  // ── Latency ────────────────────────────────────────────────────────────────
  state.latencyHistory.push(latency * 1000); // convert to ms
  if (state.latencyHistory.length > 60) state.latencyHistory.shift();
  const avgMs = state.latencyHistory.reduce((a, b) => a + b, 0) / state.latencyHistory.length;
  statLatency.textContent = `${avgMs.toFixed(0)} ms`;
  drawLatencyChart();

  // ── Speaker time ───────────────────────────────────────────────────────────
  state.speakerTime[speaker] = (state.speakerTime[speaker] || 0) + duration;
  updateSpeakerBars();

  // ── Waveform (simulated from latency/quality) ──────────────────────────────
  drawWaveform(quality, latency);

  // ── Timeline ───────────────────────────────────────────────────────────────
  addTimelineItem(start_time, end_time, speaker, quality, latency);

  // ── Event log ──────────────────────────────────────────────────────────────
  const cls = speaker === "Speaker 1" ? "s1" : "s2";
  appendLog(cls, `[${fmt(start_time)}–${fmt(end_time)}] ${speaker} speaking  (${quality}, ${(latency * 1000).toFixed(0)} ms)`);
}

// ── Speaker bars ───────────────────────────────────────────────────────────────
function updateSpeakerBars() {
  const t1 = state.speakerTime["Speaker 1"] || 0;
  const t2 = state.speakerTime["Speaker 2"] || 0;
  const total = t1 + t2 || 1;

  s1Bar.style.width  = `${(t1 / total * 100).toFixed(1)}%`;
  s2Bar.style.width  = `${(t2 / total * 100).toFixed(1)}%`;
  s1Time.textContent = `${t1.toFixed(1)} s`;
  s2Time.textContent = `${t2.toFixed(1)} s`;
}

// ── Timeline ───────────────────────────────────────────────────────────────────
function addTimelineItem(start, end, speaker, quality, latency) {
  const cls = speaker === "Speaker 1" ? "s1" : "s2";
  const div = document.createElement("div");
  div.className = `timeline-item ${cls}`;
  div.innerHTML = `
    <span class="ts">[${fmt(start)} – ${fmt(end)}]</span>
    <span class="spk">${speaker}</span>
    <span class="quality-badge ${quality}">${quality}</span>
    <span class="latency-tag">${(latency * 1000).toFixed(0)} ms</span>
  `;
  timeline.prepend(div);

  // Keep last 50 items
  while (timeline.children.length > 50) timeline.lastChild.remove();
}

// ── Event log ──────────────────────────────────────────────────────────────────
function appendLog(cls, text) {
  const div = document.createElement("div");
  div.className = `log-line ${cls}`;
  div.textContent = text;
  eventLog.prepend(div);
  while (eventLog.children.length > 100) eventLog.lastChild.remove();
}

// ── Waveform canvas ────────────────────────────────────────────────────────────
function drawWaveform(quality, latency) {
  const w = waveCanvas.width;
  const h = waveCanvas.height;
  const mid = h / 2;

  // Shift existing content left by 2px per 10s chunk (slower scroll)
  const imageData = wCtx.getImageData(2, 0, w - 2, h);
  wCtx.putImageData(imageData, 0, 0);
  wCtx.clearRect(w - 2, 0, 2, h);

  // Draw new bar
  const amplitude = quality === "clean"
    ? (0.3 + Math.random() * 0.5) * mid
    : (0.05 + Math.random() * 0.15) * mid;

  const color = quality === "clean" ? "#4f8ef7" : "#7a8499";
  wCtx.fillStyle = color;
  wCtx.fillRect(w - 2, mid - amplitude, 2, amplitude * 2);
}

// ── Latency chart ──────────────────────────────────────────────────────────────
function drawLatencyChart() {
  const w = latencyCanvas.width;
  const h = latencyCanvas.height;
  const data = state.latencyHistory;

  lCtx.clearRect(0, 0, w, h);

  if (data.length < 2) return;

  const maxVal = Math.max(...data, 500);
  const step = w / (data.length - 1);

  // Fill area
  lCtx.beginPath();
  lCtx.moveTo(0, h);
  data.forEach((v, i) => {
    const x = i * step;
    const y = h - (v / maxVal) * (h - 8);
    i === 0 ? lCtx.lineTo(x, y) : lCtx.lineTo(x, y);
  });
  lCtx.lineTo(w, h);
  lCtx.closePath();
  lCtx.fillStyle = "rgba(79,142,247,0.15)";
  lCtx.fill();

  // Line
  lCtx.beginPath();
  data.forEach((v, i) => {
    const x = i * step;
    const y = h - (v / maxVal) * (h - 8);
    i === 0 ? lCtx.moveTo(x, y) : lCtx.lineTo(x, y);
  });
  lCtx.strokeStyle = "#4f8ef7";
  lCtx.lineWidth = 2;
  lCtx.stroke();

  // 2-second threshold line
  const threshY = h - (2000 / maxVal) * (h - 8);
  if (threshY > 0 && threshY < h) {
    lCtx.beginPath();
    lCtx.setLineDash([4, 4]);
    lCtx.moveTo(0, threshY);
    lCtx.lineTo(w, threshY);
    lCtx.strokeStyle = "rgba(247,201,72,0.5)";
    lCtx.lineWidth = 1;
    lCtx.stroke();
    lCtx.setLineDash([]);
  }
}

// ── Helpers ────────────────────────────────────────────────────────────────────
function fmt(sec) {
  const s = Math.floor(sec);
  const ms = Math.round((sec - s) * 10);
  return `${s}.${ms}s`;
}

// ── Poll /status on load to sync state ────────────────────────────────────────
async function syncStatus() {
  try {
    const res = await fetch(`${API_BASE}/status`);
    const data = await res.json();

    if (data.status === "running") {
      state.running = true;
      btnStart.disabled = true;
      btnStop.disabled  = false;
      setMicStatus("listening");
      connectSSE();
      appendLog("sys", "↺ Reconnected to running pipeline");
    }

    // Replay recent chunks into stats
    (data.recent_chunks || []).forEach(chunk => {
      state.chunkCount++;
      const spk = chunk.speaker || "Speaker 1";
      const dur = (chunk.end_time - chunk.start_time) || 1.5;
      state.speakerTime[spk] = (state.speakerTime[spk] || 0) + dur;
      if (chunk.latency) state.latencyHistory.push(chunk.latency * 1000);
    });

    statChunks.textContent = state.chunkCount;
    updateSpeakerBars();
    if (state.latencyHistory.length) drawLatencyChart();

    if (data.pipeline?.avg_latency) {
      statLatency.textContent = `${(data.pipeline.avg_latency * 1000).toFixed(0)} ms`;
    }

    // Restore storage info
    if (data.storage?.session_dir) {
      updateStorageUI(data.storage.manifest);
    }
  } catch (_) {
    appendLog("sys", "⚠ Could not reach ECHO server — is it running?");
  }
}

// ── Storage UI ─────────────────────────────────────────────────────────────────
function updateStorageUI(manifest) {
  if (!manifest || !manifest.session_id) return;

  const speakers = manifest.speakers || {};
  const total = manifest.total_chunks || 0;
  statSaved.textContent = total;

  const speakerBadges = Object.entries(speakers).map(([spk, info]) => {
    const cls = spk === "Speaker 1" ? "s1" : "s2";
    return `<span class="storage-spk-badge ${cls}">${spk}: ${info.chunk_count} chunks (${info.total_duration}s)</span>`;
  }).join("");

  storageInfo.innerHTML = `
    <div class="storage-session">
      <div><strong>Session:</strong> ${manifest.session_id}</div>
      <div class="storage-path">${manifest.session_id}/</div>
      <div class="storage-speakers">${speakerBadges || '<span style="color:var(--text-muted)">No chunks saved yet</span>'}</div>
    </div>
  `;

  recordingsLink.style.display = "inline-block";
}

// Poll storage info every 5 seconds while running
setInterval(async () => {
  if (!state.running) return;
  try {
    const res = await fetch(`${API_BASE}/status`);
    const data = await res.json();
    if (data.storage?.manifest) updateStorageUI(data.storage.manifest);
    if (data.pipeline) {
      statChunks.textContent = data.pipeline.chunks_captured || state.chunkCount;
      if (data.pipeline.avg_latency) {
        statLatency.textContent = `${(data.pipeline.avg_latency * 1000).toFixed(0)} ms`;
      }
    }
  } catch (_) {}
}, 5000);

syncStatus();
