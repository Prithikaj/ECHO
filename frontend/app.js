/**
 * ECHO Frontend - AI Emergency Call Intelligence Dashboard
 * Gemini 2.5 Flash powered: transcription, language detection,
 * emotion analysis, intent classification, crisis detection,
 * verification loop, TTS responses, human-in-the-loop dashboard.
 */

/** Same host as the Flask app when served over HTTP; fallback if opening file:// */
const API_BASE = (window.location.protocol === 'http:' || window.location.protocol === 'https:')
  ? window.location.origin
  : 'http://127.0.0.1:5000';

// State
const state = {
  running: false,
  eventSource: null,
  speakerTime: { 'Speaker 1': 0, 'Speaker 2': 0 },
  latencyHistory: [],
  aiLatencyHistory: [],
  urgencyHistory: [],
  emotionHistory: [],
  chunkCount: 0,
  speechCount: 0,
  silenceCount: 0,
  transcriptionCount: 0,
  crisisActive: false,
  currentLanguage: null,
  currentEmotion: 'neutral',
  currentUrgency: 0,
  currentIntent: null,
  lastChannels: '-',
  _lastDisplayedTranscript: '',
  /** debounce id for Chrome speechSynthesis (cancel + speak must be async) */
  _ttsSchedule: null,
};

/** Prime voice list (Chrome loads voices asynchronously). */
function loadVoices() {
  if (!window.speechSynthesis) return [];
  return window.speechSynthesis.getVoices();
}

/**
 * Pick a SpeechSynthesis voice using a fresh getVoices() call (required on Windows/Chrome).
 */
function pickVoiceForLanguage(langKey) {
  const voices = loadVoices();
  if (!voices || !voices.length) return null;
  const norm = function (v) {
    return (v.lang || '').toLowerCase().replace(/_/g, '-');
  };
  const key = (langKey || 'english').toLowerCase();
  const byPrefix = function (prefixes) {
    for (var i = 0; i < prefixes.length; i++) {
      var p = prefixes[i].toLowerCase();
      var found = voices.find(function (v) {
        var l = norm(v);
        return l.startsWith(p) || (p.length <= 2 && l.split(/[-_]/)[0] === p);
      });
      if (found) return found;
    }
    return null;
  };
  if (key === 'hindi' || key === 'code_mixed') {
    var hv = byPrefix(['hi-in', 'hi']);
    if (hv) return hv;
    hv = voices.find(function (x) {
      return /hindi|hemant|kalpana|swara|sapna|microsoft.*hi/i.test(x.name);
    });
    if (hv) return hv;
  }
  if (key === 'kannada') {
    var kv = byPrefix(['kn-in', 'kn']);
    if (kv) return kv;
    kv = voices.find(function (x) { return /kannada/i.test(x.name); });
    if (kv) return kv;
  }
  if (key === 'english') {
    var ev = byPrefix(['en-in', 'en-gb', 'en-us', 'en']);
    if (ev) return ev;
  }
  return voices[0] || null;
}

const EMO_EMOJI = { fear:'😨', panic:'😰', anger:'😡', confusion:'😕', calm:'😌', distress:'😢', sadness:'😔', neutral:'😐' };
const EMO_COLOR = { fear:'#f7c948', panic:'#f76f6f', anger:'#f79a4f', confusion:'#a259ff', calm:'#3ecf8e', distress:'#f76f6f', sadness:'#4f8ef7', neutral:'#7a8499' };
const URGENCY_COLOR = s => s>=0.8?'#f76f6f':s>=0.6?'#f79a4f':s>=0.4?'#f7c948':'#3ecf8e';

// DOM refs
const micDot       = document.getElementById('mic-dot');
const micLabel     = document.getElementById('mic-label');
const btnStart     = document.getElementById('btn-start');
const btnStop      = document.getElementById('btn-stop');
const statChunks   = document.getElementById('stat-chunks');
const statSpeech   = document.getElementById('stat-speech');
const statSilence  = document.getElementById('stat-silence');
const statLatency  = document.getElementById('stat-latency');
const statAiLat    = document.getElementById('stat-ai-latency');
const statLanguage = document.getElementById('stat-language');
const statChannels = document.getElementById('stat-channels');
const statSaved    = document.getElementById('stat-saved');
const crisisbanner = document.getElementById('crisis-banner');
const crisisText   = document.getElementById('crisis-text');
const crisisTypeBadge  = document.getElementById('crisis-type-badge');
const crisisEscalation = document.getElementById('crisis-escalation');
const langBadge    = document.getElementById('lang-badge');
const confBadge    = document.getElementById('conf-badge');
const transcriptFeed = document.getElementById('transcript-feed');
const dialectNote  = document.getElementById('dialect-note');
const verificationBox = document.getElementById('verification-box');
const emotionIcon  = document.getElementById('emotion-icon');
const emotionLabel = document.getElementById('emotion-label');
const urgencyBar   = document.getElementById('urgency-bar');
const urgencyValue = document.getElementById('urgency-value');
const sentimentBadge   = document.getElementById('sentiment-badge');
const trajectoryBadge  = document.getElementById('trajectory-badge');
const implicitMeaning  = document.getElementById('implicit-meaning');
const intentBadge  = document.getElementById('intent-badge');
const riskBadge    = document.getElementById('risk-badge');
const intentConf   = document.getElementById('intent-conf');
const entitiesGrid = document.getElementById('entities-grid');
const missingInfo  = document.getElementById('missing-info');
const s1Bar  = document.getElementById('speaker-1-bar');
const s2Bar  = document.getElementById('speaker-2-bar');
const s1Time = document.getElementById('speaker-1-time');
const s2Time = document.getElementById('speaker-2-time');
const dashRisk     = document.getElementById('dash-risk');
const dashCrisis   = document.getElementById('dash-crisis');
const dashEscalation = document.getElementById('dash-escalation');
const dashLanguage = document.getElementById('dash-language');
const dashTranscriptions = document.getElementById('dash-transcriptions');
const dashTurns    = document.getElementById('dash-turns');
const ttsBox       = document.getElementById('tts-box');
const timeline     = document.getElementById('timeline');
const eventLog     = document.getElementById('event-log');
const storageInfo  = document.getElementById('storage-info');
const recordingsLink = document.getElementById('recordings-link');
const waveCanvas   = document.getElementById('waveform-canvas');
const latencyCanvas = document.getElementById('latency-canvas');
const emotionCanvas = document.getElementById('emotion-chart');
const wCtx  = waveCanvas.getContext('2d');
const lCtx  = latencyCanvas.getContext('2d');
const eCtx  = emotionCanvas.getContext('2d');

function resizeCanvases() {
  waveCanvas.width    = waveCanvas.offsetWidth;
  waveCanvas.height   = waveCanvas.offsetHeight;
  latencyCanvas.width  = latencyCanvas.offsetWidth;
  latencyCanvas.height = latencyCanvas.offsetHeight;
  emotionCanvas.width  = emotionCanvas.offsetWidth;
  emotionCanvas.height = emotionCanvas.offsetHeight || 60;
}
window.addEventListener('resize', resizeCanvases);
resizeCanvases();
if (window.speechSynthesis) {
  loadVoices();
  window.speechSynthesis.onvoiceschanged = loadVoices;
}

// Mic status
function setMicStatus(status) {
  micDot.className = 'dot ' + status;
  const labels = { listening:'Listening', processing:'Processing', idle:'Idle' };
  micLabel.textContent = labels[status] || status;
}

// Pipeline control
async function startPipeline() {
  try {
    const res = await fetch(API_BASE + '/pipeline/start', { method: 'POST' });
    const data = await res.json();
    if (data.status === 'started' || data.status === 'already_running') {
      state.running = true;
      btnStart.disabled = true;
      btnStop.disabled  = false;
      setMicStatus('listening');
      appendLog('sys', 'Pipeline started');
      connectSSE();
    }
  } catch (err) { appendLog('sys', 'Could not start: ' + err.message); }
}

async function stopPipeline() {
  try {
    const res = await fetch(API_BASE + '/pipeline/stop', { method: 'POST' });
    const data = await res.json();
    state.running = false;
    btnStart.disabled = false;
    btnStop.disabled  = true;
    setMicStatus('idle');
    appendLog('sys', 'Pipeline stopped — waiting for final AI results...');
    // Keep SSE open for 30 seconds to catch late-arriving AI results
    setTimeout(() => {
      if (state.eventSource) { state.eventSource.close(); state.eventSource = null; }
      appendLog('sys', 'Session complete');
    }, 30000);
    if (data.post_call_summary) showPostCallSummary(data.post_call_summary);
    // Also poll /status once after 5s and 15s to catch any missed chunks
    setTimeout(() => fetchAndDisplayLatestChunks(), 5000);
    setTimeout(() => fetchAndDisplayLatestChunks(), 15000);
    setTimeout(() => fetchAndDisplayLatestChunks(), 25000);
  } catch (err) { appendLog('sys', 'Could not stop: ' + err.message); }
}

async function fetchAndDisplayLatestChunks() {
  try {
    var res = await fetch(API_BASE + '/status');
    var data = await res.json();
    var chunks = data.recent_chunks || [];
    if (chunks.length === 0) return;
    var latest = chunks[chunks.length - 1];
    // Only display if it has AI data we haven't shown yet
    if (latest.transcript && latest.transcript !== state._lastDisplayedTranscript) {
      state._lastDisplayedTranscript = latest.transcript;
      updateTranscription(
        latest.speaker || 'Speaker 1',
        latest.transcript,
        latest.normalized_text || latest.transcript,
        latest.language,
        latest.transcription_confidence,
        latest.is_code_mixed,
        latest.dialect_notes
      );
      if (latest.language) {
        var langDisplay = latest.language.replace('_', ' ').toUpperCase();
        statLanguage.textContent = langDisplay;
        langBadge.textContent = langDisplay;
        dashLanguage.textContent = langDisplay;
      }
      if (latest.emotion) updateEmotionPanel(latest.emotion, latest.emotion_confidence, latest.urgency_level, latest.urgency_score, latest.sentiment, latest.emotion_trajectory, latest.implicit_meaning);
      if (latest.intent) updateIntentPanel(latest.intent, latest.intent_confidence, latest.entities, latest.risk_level, latest.missing_critical_info);
      if (latest.crisis_activated) triggerCrisisMode(latest.crisis_type, latest.crisis_severity, latest.escalation_path, latest.bypass_ai);
      if (latest.verification_action) updateVerification(latest.verification_action, latest.verification_statement, latest.clarification_question);
      if (latest.tts_text) updateTTS(latest.tts_text, latest.tts_tone, latest.tts_language || latest.language);
      updateDashboard(latest.risk_level, latest.crisis_activated, latest.crisis_type, latest.escalation_path, latest.conversation_context);
      addTimelineItem(latest.start_time, latest.end_time, latest.speaker, latest.quality, latest.latency, latest.language, latest.emotion);
      appendLog(latest.speaker === 'Speaker 1' ? 's1' : 's2', latest.speaker + ': ' + latest.transcript.substring(0, 80));
      state.transcriptionCount++;
      dashTranscriptions.textContent = state.transcriptionCount;
    }
  } catch(e) {}
}

btnStart.addEventListener('click', startPipeline);
btnStop.addEventListener('click',  stopPipeline);
document.getElementById('btn-override').addEventListener('click', () => {
  appendLog('crisis', 'AGENT OVERRIDE — transferring to human agent');
  alert('Transferring call to human agent...');
});

document.getElementById('btn-connect-human').addEventListener('click', async () => {
  try {
    const res = await fetch(API_BASE + '/connect_human', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: null }),
    });
    const data = await res.json();
    if (data.status === 'ok') {
      appendLog('crisis', 'Human professional connected — ' + (data.details || 'notification sent'));
      alert('Human professional has been alerted.');
    } else {
      appendLog('sys', 'Human connection failed: ' + (data.reason || JSON.stringify(data)));
      alert('Could not connect human professional. See log for details.');
    }
  } catch (err) {
    appendLog('sys', 'Human connection request failed: ' + err.message);
    alert('Connection request failed.');
  }
});

document.getElementById('crisis-dismiss').addEventListener('click', () => {
  crisisbanner.classList.add('hidden');
});

// SSE connection
function connectSSE() {
  if (state.eventSource) state.eventSource.close();
  const es = new EventSource(API_BASE + '/stream');
  state.eventSource = es;
  es.onmessage = (e) => { try { handleEvent(JSON.parse(e.data)); } catch(_) {} };
  es.onerror = () => { appendLog('sys', 'SSE lost - retrying...'); setMicStatus('idle'); };
}

// Main event handler
function handleEvent(event) {
  if (event.type === 'heartbeat' || event.type === 'connected') return;
  if (event.type === 'pipeline_status') {
    appendLog('sys', 'Pipeline: ' + event.status);
    setMicStatus(event.status === 'started' ? 'listening' : 'idle');
    return;
  }
  if (event.type === 'crisis_alert') {
    handleCrisisAlert(event);
    return;
  }
  if (event.type === 'human_connection') {
    appendLog('human', 'Human professional requested: ' + (event.message || 'notification sent'));
    if (event.success === false) {
      appendLog('sys', 'Human professional connection failed: ' + (event.reason || 'unknown'));
    }
    return;
  }
  if (event.type === 'post_call_summary') {
    showPostCallSummary(event.summary);
    return;
  }
  if (event.type === 'chunk') {
    setMicStatus('processing');
    processChunk(event);
    setTimeout(() => { if (state.running) setMicStatus('listening'); }, 400);
  }
}

// Process a chunk - the main AI data handler
function processChunk(chunk) {
  const { chunk_id, start_time, end_time, speaker, latency, quality, channels,
    transcript, language, language_confidence, normalized_text, transcription_confidence,
    dialect_notes, is_code_mixed,
    emotion, emotion_confidence, urgency_level, urgency_score, sentiment,
    implicit_meaning, emotion_trajectory,
    intent, intent_confidence, entities, risk_level, requires_immediate_action, missing_critical_info,
    crisis_activated, crisis_type, crisis_severity, bypass_ai, escalation_path,
    verification_action, verification_statement, clarification_question,
    tts_text, tts_tone, tts_language, assistant_response,
    conversation_context, ai_latency
  } = chunk;

  const duration = (end_time - start_time) || 1.5;

  // Stats
  state.chunkCount++;
  statChunks.textContent = state.chunkCount;
  if (quality === 'clean') { state.speechCount++; statSpeech.textContent = state.speechCount; }
  else { state.silenceCount++; statSilence.textContent = state.silenceCount; }
  if (channels) statChannels.textContent = channels === 1 ? 'Mono' : 'Stereo';

  // Latency
  state.latencyHistory.push(latency * 1000);
  if (state.latencyHistory.length > 60) state.latencyHistory.shift();
  const avgMs = state.latencyHistory.reduce((a,b)=>a+b,0) / state.latencyHistory.length;
  statLatency.textContent = avgMs.toFixed(0) + ' ms';
  if (ai_latency) {
    state.aiLatencyHistory.push(ai_latency * 1000);
    if (state.aiLatencyHistory.length > 60) state.aiLatencyHistory.shift();
    const aiAvg = state.aiLatencyHistory.reduce((a,b)=>a+b,0) / state.aiLatencyHistory.length;
    statAiLat.textContent = aiAvg.toFixed(0) + ' ms';
  }
  drawLatencyChart();

  // Speaker time
  state.speakerTime[speaker] = (state.speakerTime[speaker] || 0) + duration;
  updateSpeakerBars();

  // Waveform
  drawWaveform(quality, latency);

  // Timeline
  addTimelineItem(start_time, end_time, speaker, quality, latency, language, emotion);

  // AI: Transcription
  if (transcript) {
    state.transcriptionCount++;
    state._lastDisplayedTranscript = transcript;
    updateTranscription(speaker, transcript, normalized_text, language, transcription_confidence, is_code_mixed, dialect_notes);
    dashTranscriptions.textContent = state.transcriptionCount;
  }

  // AI: Language
  if (language) {
    state.currentLanguage = language;
    // Display primary language prominently (not 'CODE MIXED')
    const langMap = {
      'hindi': 'HINDI',
      'kannada': 'KANNADA',
      'english': 'ENGLISH',
      'code_mixed': 'CODE MIXED (HINDI)',
      'unknown': 'UNKNOWN'
    };
    const langDisplay = langMap[language] || language.replace('_', ' ').toUpperCase();
    statLanguage.textContent = langDisplay;
    langBadge.textContent = langDisplay;
    dashLanguage.textContent = langDisplay;
    if (language_confidence) confBadge.textContent = (language_confidence * 100).toFixed(0) + '% conf';
  }

  // AI: Emotion
  if (emotion) {
    state.currentEmotion = emotion;
    state.currentUrgency = urgency_score || 0;
    state.urgencyHistory.push(urgency_score || 0);
    state.emotionHistory.push(emotion);
    if (state.urgencyHistory.length > 20) state.urgencyHistory.shift();
    if (state.emotionHistory.length > 20) state.emotionHistory.shift();
    updateEmotionPanel(emotion, emotion_confidence, urgency_level, urgency_score, sentiment, emotion_trajectory, implicit_meaning);
    drawEmotionChart();
  }

  // AI: Intent + Risk
  if (intent) {
    state.currentIntent = intent;
    updateIntentPanel(intent, intent_confidence, entities, risk_level, missing_critical_info);
    // Also update dashboard risk immediately when received
    if (risk_level && dashRisk) {
      dashRisk.textContent = risk_level.toUpperCase();
      dashRisk.className = 'dash-value risk-value ' + risk_level.toLowerCase();
    }
  }

  // AI: Crisis
  if (crisis_activated) {
    state.crisisActive = true;
    triggerCrisisMode(crisis_type, crisis_severity, escalation_path, bypass_ai);
  }

  // AI: Verification
  if (verification_action) {
    updateVerification(verification_action, verification_statement, clarification_question);
  }

  // AI: TTS
  if (tts_text) {
    updateTTS(tts_text, tts_tone || 'calm', tts_language || language);
  } else if (assistant_response) {
    updateTTS(assistant_response, 'calm', language);
  } else if (transcript && verification_statement) {
    // Fallback: speak the verification statement
    updateTTS(verification_statement, 'calm', language);
  }

  // Dashboard (always pass a string risk so the agent panel is never blank)
  updateDashboard(risk_level || 'low', crisis_activated, crisis_type, escalation_path, conversation_context);

  // Event log
  const cls = speaker === 'Speaker 1' ? 's1' : 's2';
  let logMsg = '[' + fmt(start_time) + '-' + fmt(end_time) + '] ' + speaker;
  if (transcript) logMsg += ': ' + transcript.substring(0, 60) + (transcript.length > 60 ? '...' : '');
  else logMsg += ' (no transcript)';
  appendLog(cls, logMsg);
  if (emotion) appendLog('ai', '  emotion=' + emotion + ' urgency=' + (urgency_level||'?') + ' intent=' + (intent||'?') + ' risk=' + (risk_level||'?'));
}

// Update transcription feed
function updateTranscription(speaker, transcript, normalized, language, confidence, isCodeMixed, dialectNotes) {
  const cls = speaker === 'Speaker 1' ? 's1' : 's2';
  const div = document.createElement('div');
  div.className = 'transcript-item ' + cls;
  const langTag = language ? '<span class="t-tag lang">' + language.replace('_',' ') + '</span>' : '';
  const mixTag = isCodeMixed ? '<span class="t-tag code-mixed">code-mixed</span>' : '';
  const confText = confidence ? (confidence * 100).toFixed(0) + '% conf' : '';
  const normText = (normalized && normalized !== transcript) ? '<div class="t-normalized">Normalized: ' + normalized + '</div>' : '';
  div.innerHTML = '<div class="t-speaker">' + speaker + ' &bull; ' + confText + '</div>' +
    '<div class="t-text">' + transcript + '</div>' +
    normText +
    '<div class="t-meta">' + langTag + mixTag + '</div>';
  transcriptFeed.prepend(div);
  while (transcriptFeed.children.length > 30) transcriptFeed.lastChild.remove();
  if (dialectNotes) {
    dialectNote.textContent = 'Dialect: ' + dialectNotes;
    dialectNote.classList.remove('hidden');
  }
}

// Update emotion panel
function updateEmotionPanel(emotion, confidence, urgencyLevel, urgencyScore, sentiment, trajectory, implicit) {
  const emoji = EMO_EMOJI[emotion] || 'question';
  const color = EMO_COLOR[emotion] || '#7a8499';
  emotionIcon.textContent = emoji;
  emotionLabel.textContent = emotion || 'neutral';
  emotionLabel.style.color = color;
  const pct = Math.round((urgencyScore || 0) * 100);
  urgencyBar.style.width = pct + '%';
  urgencyBar.style.background = URGENCY_COLOR(urgencyScore || 0);
  urgencyValue.textContent = pct + '%';
  sentimentBadge.textContent = sentiment || 'neutral';
  sentimentBadge.className = 'sentiment-badge ' + (sentiment || 'neutral');
  trajectoryBadge.textContent = trajectory || 'stable';
  trajectoryBadge.className = 'trajectory-badge ' + (trajectory || 'stable').replace('-','');
  if (implicit) {
    implicitMeaning.textContent = 'Implicit: ' + implicit;
    implicitMeaning.classList.remove('hidden');
  } else { implicitMeaning.classList.add('hidden'); }
}

// Update intent panel
function updateIntentPanel(intent, confidence, entities, riskLevel, missingInfo) {
  intentBadge.textContent = (intent || 'unknown').replace(/_/g, ' ');
  riskBadge.textContent = riskLevel || 'low';
  riskBadge.className = 'risk-badge ' + (riskLevel || 'low');
  intentConf.textContent = confidence ? 'Confidence: ' + (confidence * 100).toFixed(0) + '%' : '';
  entitiesGrid.innerHTML = '';
  if (entities) {
    const show = ['location','incident_type','people_involved','time_mentioned','area_landmark'];
    show.forEach(key => {
      const val = entities[key];
      if (!val || (Array.isArray(val) && !val.length)) return;
      const d = document.createElement('div');
      d.className = 'entity-item';
      d.innerHTML = '<div class="entity-label">' + key.replace(/_/g,' ') + '</div>' +
        '<div class="entity-value">' + (Array.isArray(val) ? val.join(', ') : val) + '</div>';
      entitiesGrid.appendChild(d);
    });
  }
  if (missingInfo && missingInfo.length) {
    missingInfo.textContent = 'Missing: ' + missingInfo.join(', ');
    missingInfo.classList.remove('hidden');
  } else { missingInfo.classList.add('hidden'); }
}

// Crisis mode
function triggerCrisisMode(crisisType, severity, escalationPath, bypassAi) {
  crisisbanner.classList.remove('hidden');
  crisisText.textContent = 'CRISIS DETECTED - Severity ' + (severity || '?') + '/10';
  crisisTypeBadge.textContent = (crisisType || 'unknown').replace(/_/g,' ').toUpperCase();
  crisisEscalation.textContent = escalationPath ? 'Escalate to: ' + escalationPath.replace(/_/g,' ').toUpperCase() : '';
  dashCrisis.textContent = (crisisType || 'active').replace(/_/g,' ');
  dashCrisis.style.color = '#f76f6f';
  if (bypassAi) appendLog('crisis', 'BYPASS AI - transferring to human agent immediately');
  appendLog('crisis', 'CRISIS: ' + crisisType + ' severity=' + severity + ' escalate=' + escalationPath);
}

// Verification loop
function updateVerification(action, statement, question) {
  const div = document.createElement('div');
  div.className = 'verification-item';
  const actionColor = action === 'proceed' ? 'proceed' : action === 'clarify' ? 'clarify' : 'escalate';
  div.innerHTML = '<div class="v-action ' + actionColor + '">' + (action || '').toUpperCase() + '</div>' +
    (statement ? '<div class="v-statement">' + statement + '</div>' : '') +
    (question  ? '<div class="v-question">' + question  + '</div>' : '');
  verificationBox.innerHTML = '';
  verificationBox.appendChild(div);
}

// TTS response (SpeechSynthesis — use DOM button + deferred speak for Hindi/Windows Chrome)
function updateTTS(text, tone, language) {
  var raw = String(text || '').trim();
  if (!raw) return;
  var lang = (language || 'english').toLowerCase();
  var t = tone || 'calm';
  ttsBox.innerHTML = '';
  var wrap = document.createElement('div');
  wrap.className = 'tts-item';
  var hdr = document.createElement('div');
  hdr.className = 'tts-lang';
  hdr.textContent = lang.toUpperCase() + ' • tone: ' + t;
  var body = document.createElement('div');
  body.className = 'tts-text';
  body.textContent = raw;
  var btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'btn-speak';
  btn.textContent = '🔊 Speak';
  btn.addEventListener('click', function () { speakText(raw, lang, t); });
  wrap.appendChild(hdr);
  wrap.appendChild(body);
  wrap.appendChild(btn);
  ttsBox.appendChild(wrap);
  speakText(raw, lang, t);
}

function speakText(text, language, tone) {
  if (!window.speechSynthesis) {
    appendLog('sys', 'TTS not supported in this browser');
    return;
  }
  var raw = String(text || '').trim();
  if (!raw) return;
  loadVoices();
  var synth = window.speechSynthesis;
  synth.cancel();
  if (state._ttsSchedule) {
    window.clearTimeout(state._ttsSchedule);
    state._ttsSchedule = null;
  }
  var langKey = (language || 'english').toLowerCase();
  var toneVal = (tone || 'calm').toLowerCase();
  state._ttsSchedule = window.setTimeout(function () {
    state._ttsSchedule = null;
    loadVoices();
    try { synth.resume(); } catch (e) {}
    var utter = new SpeechSynthesisUtterance(raw);
    var langMap = { hindi: 'hi-IN', kannada: 'kn-IN', english: 'en-IN', code_mixed: 'hi-IN', unknown: 'en-IN' };
    utter.lang = langMap[langKey] || 'en-IN';
    var voice = pickVoiceForLanguage(langKey);
    if (voice) utter.voice = voice;
    if (toneVal === 'urgent' || toneVal === 'fast') { utter.rate = 1.12; utter.pitch = 1.05; }
    else if (toneVal === 'calm' || toneVal === 'slow') { utter.rate = 0.92; utter.pitch = 1.0; }
    else if (toneVal === 'empathetic' || toneVal === 'reassuring') { utter.rate = 0.92; utter.pitch = 1.0; }
    else { utter.rate = 1.0; utter.pitch = 1.0; }
    utter.volume = 1.0;
    utter.onstart = function () {
      appendLog('ai', 'TTS started (' + utter.lang + (voice ? ', ' + voice.name : '') + ')');
    };
    utter.onend = function () { appendLog('ai', 'TTS finished'); };
    utter.onerror = function (e) {
      appendLog('sys', 'TTS error: ' + (e && e.error ? e.error : 'unknown') + ' — try the Speak button or check Windows speech voices');
    };
    try {
      synth.speak(utter);
    } catch (err) {
      appendLog('sys', 'TTS speak() failed: ' + err);
    }
    appendLog('ai', 'Speaking: ' + raw.substring(0, 80) + (raw.length > 80 ? '...' : ''));
  }, 150);
}

window.speakText = speakText;

// Agent dashboard
function updateDashboard(riskLevel, crisisActivated, crisisType, escalationPath, ctx) {
  var rl = (riskLevel != null && riskLevel !== '') ? String(riskLevel) : 'low';
  dashRisk.textContent = rl.toUpperCase();
  dashRisk.className = 'dash-value risk-value ' + rl.toLowerCase().replace(/\s+/g, '');
  if (!crisisActivated) {
    dashCrisis.textContent = 'None';
    dashCrisis.style.color = '';
  }
  if (escalationPath) dashEscalation.textContent = escalationPath.replace(/_/g,' ');
  if (ctx) {
    if (ctx.total_turns !== undefined) dashTurns.textContent = ctx.total_turns;
    if (ctx.languages_detected && ctx.languages_detected.length > 0) {
      dashLanguage.textContent = ctx.languages_detected.join(', ').toUpperCase();
    }
  }
}

// Post-call summary
function showPostCallSummary(summary) {
  if (!summary) return;
  appendLog('sys', '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  appendLog('sys', '📋 POST-CALL SUMMARY');
  appendLog('sys', '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  if (summary.summary) appendLog('sys', '📝 ' + summary.summary);
  if (summary.issue) appendLog('sys', '🎯 Issue: ' + summary.issue);
  if (summary.primary_emotion_overall) appendLog('sys', '❤️ Primary emotion: ' + summary.primary_emotion_overall);
  if (summary.case_category) appendLog('sys', '📂 Category: ' + summary.case_category);
  if (summary.risk_assessment) appendLog('sys', '⚠️ Risk level: ' + summary.risk_assessment);
  if (summary.actions_taken && summary.actions_taken.length) appendLog('sys', '✅ Actions: ' + summary.actions_taken.join(', '));
  if (summary.follow_up_required) appendLog('sys', '🔔 Follow-up required: ' + (summary.follow_up_notes || 'Yes'));
  if (summary.languages_used && summary.languages_used.length) appendLog('sys', '🗣️ Languages: ' + summary.languages_used.join(', '));
  appendLog('sys', '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
}

// Crisis alert event
function handleCrisisAlert(event) {
  triggerCrisisMode(event.crisis_type, event.crisis_severity, event.escalation_path, event.bypass_ai);
  if (event.immediate_response) {
    updateTTS(event.immediate_response, 'urgent', event.tts_language || state.currentLanguage || 'english');
  }
}


// Speaker bars
function updateSpeakerBars() {
  var t1 = state.speakerTime['Speaker 1'] || 0;
  var t2 = state.speakerTime['Speaker 2'] || 0;
  var total = t1 + t2 || 1;
  s1Bar.style.width = (t1/total*100).toFixed(1) + '%';
  s2Bar.style.width = (t2/total*100).toFixed(1) + '%';
  s1Time.textContent = t1.toFixed(1) + ' s';
  s2Time.textContent = t2.toFixed(1) + ' s';
}

// Timeline
function addTimelineItem(start, end, speaker, quality, latency, language, emotion) {
  var cls = speaker === 'Speaker 1' ? 's1' : 's2';
  var div = document.createElement('div');
  div.className = 'timeline-item ' + cls;
  var emoColor = EMO_COLOR[emotion] || '#7a8499';
  var langTag = language ? '<span class="tl-lang">' + language.replace('_',' ') + '</span>' : '';
  var emoTag = emotion ? '<span class="tl-emotion" style="color:' + emoColor + '">' + emotion + '</span>' : '';
  div.innerHTML = '<span class="ts">[' + fmt(start) + '-' + fmt(end) + ']</span>' +
    '<span class="spk">' + speaker + '</span>' + langTag + emoTag +
    '<span class="quality-badge ' + quality + '">' + quality + '</span>' +
    '<span class="latency-tag">' + (latency*1000).toFixed(0) + 'ms</span>';
  timeline.prepend(div);
  while (timeline.children.length > 50) timeline.lastChild.remove();
}

// Event log
function appendLog(cls, text) {
  var div = document.createElement('div');
  div.className = 'log-line ' + cls;
  div.textContent = text;
  eventLog.prepend(div);
  while (eventLog.children.length > 150) eventLog.lastChild.remove();
}

// Waveform — continuous animation loop
var waveAnimFrame = null;
var waveActive = false;

function startWaveformAnimation() {
  waveActive = true;
  function animate() {
    if (!waveActive) return;
    var w = waveCanvas.width, h = waveCanvas.height, mid = h / 2;
    var img = wCtx.getImageData(2, 0, w - 2, h);
    wCtx.putImageData(img, 0, 0);
    wCtx.clearRect(w - 2, 0, 2, h);
    if (state.running) {
      var color = URGENCY_COLOR(state.currentUrgency);
      var amp = (0.15 + Math.random() * 0.7) * mid;
      wCtx.fillStyle = color;
      wCtx.globalAlpha = 0.5 + Math.random() * 0.5;
      wCtx.fillRect(w - 2, mid - amp, 2, amp * 2);
      wCtx.globalAlpha = 1.0;
    } else {
      wCtx.fillStyle = '#3a3f4b';
      wCtx.fillRect(w - 2, mid - 1, 2, 2);
    }
    waveAnimFrame = requestAnimationFrame(animate);
  }
  animate();
}

function stopWaveformAnimation() {
  waveActive = false;
  if (waveAnimFrame) cancelAnimationFrame(waveAnimFrame);
}

// Start animation immediately
startWaveformAnimation();

function drawWaveform(quality, latency) {
  // Boost amplitude on speech chunks
  if (quality === 'clean') {
    state.currentUrgency = state.currentUrgency || 0.3;
  }
}

// Latency chart
function drawLatencyChart() {
  var w = latencyCanvas.width, h = latencyCanvas.height;
  var data = state.latencyHistory;
  lCtx.clearRect(0,0,w,h);
  if (data.length < 1) return;
  var maxVal = Math.max.apply(null, data.concat([500]));
  if (data.length === 1) {
    var v = data[0];
    var y = h - (v / maxVal) * (h - 8);
    lCtx.beginPath();
    lCtx.moveTo(0, h);
    lCtx.lineTo(w * 0.5, y);
    lCtx.lineTo(w, h);
    lCtx.closePath();
    lCtx.fillStyle = 'rgba(79,142,247,0.15)';
    lCtx.fill();
    lCtx.beginPath();
    lCtx.moveTo(0, y);
    lCtx.lineTo(w, y);
    lCtx.strokeStyle = '#4f8ef7';
    lCtx.lineWidth = 2;
    lCtx.stroke();
    return;
  }
  var step = w/(data.length-1);
  lCtx.beginPath(); lCtx.moveTo(0,h);
  data.forEach(function(v,i){ lCtx.lineTo(i*step, h-(v/maxVal)*(h-8)); });
  lCtx.lineTo(w,h); lCtx.closePath();
  lCtx.fillStyle='rgba(79,142,247,0.15)'; lCtx.fill();
  lCtx.beginPath();
  data.forEach(function(v,i){ var x=i*step,y=h-(v/maxVal)*(h-8); i===0?lCtx.moveTo(x,y):lCtx.lineTo(x,y); });
  lCtx.strokeStyle='#4f8ef7'; lCtx.lineWidth=2; lCtx.stroke();
  var ty = h-(2000/maxVal)*(h-8);
  if (ty>0 && ty<h) {
    lCtx.beginPath(); lCtx.setLineDash([4,4]);
    lCtx.moveTo(0,ty); lCtx.lineTo(w,ty);
    lCtx.strokeStyle='rgba(247,201,72,0.5)'; lCtx.lineWidth=1; lCtx.stroke();
    lCtx.setLineDash([]);
  }
}

// Emotion/urgency chart
function drawEmotionChart() {
  var w = emotionCanvas.width, h = emotionCanvas.height || 60;
  var data = state.urgencyHistory;
  eCtx.clearRect(0,0,w,h);
  if (data.length < 2) return;
  var step = w/(data.length-1);
  eCtx.beginPath();
  data.forEach(function(v,i){ var x=i*step,y=h-(v*(h-4)+2); i===0?eCtx.moveTo(x,y):eCtx.lineTo(x,y); });
  eCtx.strokeStyle = URGENCY_COLOR(data[data.length-1]);
  eCtx.lineWidth=2; eCtx.stroke();
}

// Format seconds
function fmt(sec) {
  var s=Math.floor(sec), ms=Math.round((sec-s)*10);
  return s+'.'+ms+'s';
}

// Storage UI
function updateStorageUI(manifest) {
  if (!manifest || !manifest.session_id) return;
  var speakers = manifest.speakers || {};
  statSaved.textContent = manifest.total_chunks || 0;
  var badges = Object.entries(speakers).map(function(e){
    var spk=e[0], info=e[1], cls=spk==='Speaker 1'?'s1':'s2';
    return '<span class="storage-spk-badge '+cls+'">'+spk+': '+info.chunk_count+' chunks</span>';
  }).join('');
  storageInfo.innerHTML = '<div class="storage-session"><div><strong>Session:</strong> '+manifest.session_id+'</div><div class="storage-path">'+manifest.session_id+'/</div><div class="storage-speakers">'+badges+'</div></div>';
  recordingsLink.style.display = 'inline-block';
}

// Sync on load
async function syncStatus() {
  try {
    var res = await fetch(API_BASE + '/status');
    var data = await res.json();
    if (data.status === 'running') {
      state.running = true;
      btnStart.disabled = true;
      btnStop.disabled = false;
      setMicStatus('listening');
      connectSSE();
      appendLog('sys', 'Reconnected to running pipeline');
    }
    // Replay all recent chunks to populate the dashboard
    var chunks = data.recent_chunks || [];
    chunks.forEach(function(chunk) {
      state.chunkCount++;
      var spk = chunk.speaker || 'Speaker 1';
      state.speakerTime[spk] = (state.speakerTime[spk] || 0) + ((chunk.end_time - chunk.start_time) || 1.5);
      if (chunk.latency) state.latencyHistory.push(chunk.latency * 1000);
      // Display AI data from each chunk
      if (chunk.transcript) {
        state._lastDisplayedTranscript = chunk.transcript;
        state.transcriptionCount++;
        updateTranscription(spk, chunk.transcript, chunk.normalized_text, chunk.language, chunk.transcription_confidence, chunk.is_code_mixed, chunk.dialect_notes);
        dashTranscriptions.textContent = state.transcriptionCount;
      }
      if (chunk.language) {
        var langDisplay = chunk.language.replace('_', ' ').toUpperCase();
        statLanguage.textContent = langDisplay;
        langBadge.textContent = langDisplay;
        dashLanguage.textContent = langDisplay;
        if (chunk.language_confidence) confBadge.textContent = (chunk.language_confidence * 100).toFixed(0) + '% conf';
      }
      if (chunk.emotion) {
        state.currentEmotion = chunk.emotion;
        state.currentUrgency = chunk.urgency_score || 0;
        state.urgencyHistory.push(chunk.urgency_score || 0);
        updateEmotionPanel(chunk.emotion, chunk.emotion_confidence, chunk.urgency_level, chunk.urgency_score, chunk.sentiment, chunk.emotion_trajectory, chunk.implicit_meaning);
      }
      if (chunk.intent) updateIntentPanel(chunk.intent, chunk.intent_confidence, chunk.entities, chunk.risk_level, chunk.missing_critical_info);
      if (chunk.crisis_activated) triggerCrisisMode(chunk.crisis_type, chunk.crisis_severity, chunk.escalation_path, chunk.bypass_ai);
      if (chunk.verification_action) updateVerification(chunk.verification_action, chunk.verification_statement, chunk.clarification_question);
      if (chunk.tts_text) updateTTS(chunk.tts_text, chunk.tts_tone, chunk.tts_language || chunk.language);
      updateDashboard(chunk.risk_level || 'low', chunk.crisis_activated, chunk.crisis_type, chunk.escalation_path, chunk.conversation_context);
      if (chunk.start_time !== undefined) addTimelineItem(chunk.start_time, chunk.end_time, spk, chunk.quality || 'clean', chunk.latency || 0, chunk.language, chunk.emotion);
    });
    statChunks.textContent = state.chunkCount;
    updateSpeakerBars();
    if (state.latencyHistory.length) drawLatencyChart();
    if (state.urgencyHistory.length) drawEmotionChart();
    if (data.pipeline && data.pipeline.avg_latency)
      statLatency.textContent = (data.pipeline.avg_latency * 1000).toFixed(0) + ' ms';
    if (data.storage && data.storage.manifest) updateStorageUI(data.storage.manifest);
    if (data.ai_context) {
      var ctx = data.ai_context;
      if (ctx.dominant_emotion) updateEmotionPanel(ctx.dominant_emotion, 0, ctx.current_urgency > 0.6 ? 'high' : 'low', ctx.current_urgency, 'neutral', ctx.emotion_trajectory, '');
      if (ctx.crisis_active) triggerCrisisMode(ctx.crisis_type, 0, '', ctx.bypass_ai);
      if (ctx.total_turns) dashTurns.textContent = ctx.total_turns;
      if (ctx.languages_detected && ctx.languages_detected.length) dashLanguage.textContent = ctx.languages_detected.join(', ').toUpperCase();
    }
  } catch(e) { appendLog('sys', 'Cannot reach ECHO server - is it running?'); }
}

// Poll every 5s
setInterval(async function() {
  if (!state.running) return;
  try {
    var res = await fetch(API_BASE + '/status');
    var data = await res.json();
    if (data.storage && data.storage.manifest) updateStorageUI(data.storage.manifest);
    if (data.pipeline) {
      statChunks.textContent = data.pipeline.chunks_captured || state.chunkCount;
      if (data.pipeline.avg_latency) statLatency.textContent = (data.pipeline.avg_latency*1000).toFixed(0)+' ms';
      if (data.pipeline.transcriptions) dashTranscriptions.textContent = data.pipeline.transcriptions;
    }
  } catch(e) {}
}, 5000);

// Agent Edit Mode
var agentModeActive = false;
document.getElementById('btn-agent-mode').addEventListener('click', function() {
  if (!agentModeActive) {
    var pin = prompt('Enter agent PIN to enable edit mode:');
    if (pin !== '1234') { alert('Incorrect PIN'); return; }
    agentModeActive = true;
    document.getElementById('agent-edit-panel').classList.remove('hidden');
    document.getElementById('btn-agent-mode').textContent = '🔓 Agent Mode ON';
    document.getElementById('btn-agent-mode').style.background = '#2a7a2a';
    appendLog('sys', 'Agent edit mode activated');
  } else {
    agentModeActive = false;
    document.getElementById('agent-edit-panel').classList.add('hidden');
    document.getElementById('btn-agent-mode').textContent = '🔐 Agent Edit Mode';
    document.getElementById('btn-agent-mode').style.background = '';
    appendLog('sys', 'Agent edit mode deactivated');
  }
});

document.getElementById('btn-save-edit').addEventListener('click', function() {
  var corrections = {
    transcript: document.getElementById('edit-transcript').value,
    intent: document.getElementById('edit-intent').value,
    location: document.getElementById('edit-location').value,
    risk: document.getElementById('edit-risk').value,
    notes: document.getElementById('edit-notes').value,
    saved_at: new Date().toISOString()
  };
  // Apply corrections to dashboard
  if (corrections.intent) {
    intentBadge.textContent = corrections.intent.replace(/_/g, ' ');
    appendLog('sys', 'Agent corrected intent → ' + corrections.intent);
  }
  if (corrections.risk) {
    riskBadge.textContent = corrections.risk;
    riskBadge.className = 'risk-badge ' + corrections.risk;
    dashRisk.textContent = corrections.risk;
    appendLog('sys', 'Agent corrected risk → ' + corrections.risk);
  }
  if (corrections.location) {
    appendLog('sys', 'Agent corrected location → ' + corrections.location);
  }
  if (corrections.transcript) {
    appendLog('sys', 'Agent corrected transcript');
  }
  if (corrections.notes) {
    appendLog('sys', 'Agent notes: ' + corrections.notes);
  }
  // Show saved confirmation
  var msg = document.getElementById('edit-saved-msg');
  msg.classList.remove('hidden');
  setTimeout(function() { msg.classList.add('hidden'); }, 3000);
  // POST corrections to backend for logging
  fetch(API_BASE + '/agent_correction', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(corrections)
  }).catch(function() {});
});

syncStatus();