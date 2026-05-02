/**
 * ECHO Frontend - AI Emergency Call Intelligence Dashboard
 * Gemini 2.5 Flash powered: transcription, language detection,
 * emotion analysis, intent classification, crisis detection,
 * verification loop, TTS responses, human-in-the-loop dashboard.
 */

const API_BASE = 'http://127.0.0.1:5000';

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
};

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
    appendLog('sys', 'Pipeline stopped');
    if (state.eventSource) { state.eventSource.close(); state.eventSource = null; }
    if (data.post_call_summary) showPostCallSummary(data.post_call_summary);
  } catch (err) { appendLog('sys', 'Could not stop: ' + err.message); }
}

btnStart.addEventListener('click', startPipeline);
btnStop.addEventListener('click',  stopPipeline);
document.getElementById('btn-override').addEventListener('click', () => {
  appendLog('crisis', 'AGENT OVERRIDE — transferring to human agent');
  alert('Transferring call to human agent...');
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
    tts_text, tts_tone, tts_language,
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
    updateTranscription(speaker, transcript, normalized_text, language, transcription_confidence, is_code_mixed, dialect_notes);
    dashTranscriptions.textContent = state.transcriptionCount;
  }

  // AI: Language
  if (language && language !== 'unknown') {
    state.currentLanguage = language;
    const langDisplay = language.replace('_', ' ').toUpperCase();
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

  // AI: Intent
  if (intent) {
    state.currentIntent = intent;
    updateIntentPanel(intent, intent_confidence, entities, risk_level, missing_critical_info);
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
  if (tts_text) updateTTS(tts_text, tts_tone, tts_language || language);

  // Dashboard
  updateDashboard(risk_level, crisis_activated, crisis_type, escalation_path, conversation_context);

  // Event log
  const cls = speaker === 'Speaker 1' ? 's1' : 's2';
  let logMsg = '[' + fmt(start_time) + '-' + fmt(end_time) + '] ' + speaker;
  if (transcript) logMsg += ': ' + transcript.substring(0, 60) + (transcript.length > 60 ? '...' : '');
  else logMsg += ' (no transcript)';
  appendLog(cls, logMsg);
  if (emotion) appendLog('ai', '  emotion=' + emotion + ' urgency=' + (urgency_level||'?') + ' intent=' + (intent||'?'));
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

// TTS response
function updateTTS(text, tone, language) {
  const div = document.createElement('div');
  div.className = 'tts-item';
  div.innerHTML = '<div class="tts-lang">' + (language || 'en').toUpperCase() + ' &bull; tone: ' + (tone || 'calm') + '</div>' +
    '<div class="tts-text">' + text + '</div>';
  ttsBox.innerHTML = '';
  ttsBox.appendChild(div);
}

// Agent dashboard
function updateDashboard(riskLevel, crisisActivated, crisisType, escalationPath, ctx) {
  dashRisk.textContent = riskLevel || '-';
  dashRisk.className = 'dash-value risk-value ' + (riskLevel || 'low');
  if (!crisisActivated) {
    dashCrisis.textContent = 'None';
    dashCrisis.style.color = '';
  }
  dashEscalation.textContent = escalationPath ? escalationPath.replace(/_/g,' ') : '-';
  if (ctx) {
    if (ctx.total_turns) dashTurns.textContent = ctx.total_turns;
  }
}

// Post-call summary
function showPostCallSummary(summary) {
  if (!summary || !summary.summary) return;
  appendLog('sys', '--- POST-CALL SUMMARY ---');
  appendLog('sys', summary.summary);
  if (summary.risk_assessment) appendLog('sys', 'Risk: ' + summary.risk_assessment);
  if (summary.case_category)   appendLog('sys', 'Category: ' + summary.case_category);
  if (summary.follow_up_required) appendLog('sys', 'Follow-up required: ' + (summary.follow_up_notes || 'yes'));
}

// Crisis alert event
function handleCrisisAlert(event) {
  triggerCrisisMode(event.crisis_type, event.crisis_severity, event.escalation_path, event.bypass_ai);
  if (event.immediate_response) updateTTS(event.immediate_response, 'urgent', 'en');
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

// Waveform
function drawWaveform(quality, latency) {
  var w = waveCanvas.width, h = waveCanvas.height, mid = h/2;
  var img = wCtx.getImageData(2,0,w-2,h);
  wCtx.putImageData(img,0,0);
  wCtx.clearRect(w-2,0,2,h);
  var amp = quality==='clean' ? (0.3+Math.random()*0.5)*mid : (0.05+Math.random()*0.15)*mid;
  wCtx.fillStyle = quality==='clean' ? URGENCY_COLOR(state.currentUrgency) : '#7a8499';
  wCtx.fillRect(w-2, mid-amp, 2, amp*2);
}

// Latency chart
function drawLatencyChart() {
  var w = latencyCanvas.width, h = latencyCanvas.height;
  var data = state.latencyHistory;
  lCtx.clearRect(0,0,w,h);
  if (data.length < 2) return;
  var maxVal = Math.max.apply(null, data.concat([500]));
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
    (data.recent_chunks || []).forEach(function(chunk){
      state.chunkCount++;
      var spk = chunk.speaker || 'Speaker 1';
      state.speakerTime[spk] = (state.speakerTime[spk]||0) + ((chunk.end_time-chunk.start_time)||1.5);
      if (chunk.latency) state.latencyHistory.push(chunk.latency*1000);
    });
    statChunks.textContent = state.chunkCount;
    updateSpeakerBars();
    if (state.latencyHistory.length) drawLatencyChart();
    if (data.pipeline && data.pipeline.avg_latency)
      statLatency.textContent = (data.pipeline.avg_latency*1000).toFixed(0)+' ms';
    if (data.storage && data.storage.manifest) updateStorageUI(data.storage.manifest);
    if (data.ai_context) {
      var ctx = data.ai_context;
      if (ctx.dominant_emotion) updateEmotionPanel(ctx.dominant_emotion,0,ctx.current_urgency>0.6?'high':'low',ctx.current_urgency,'neutral',ctx.emotion_trajectory,'');
      if (ctx.crisis_active) triggerCrisisMode(ctx.crisis_type,0,'',ctx.bypass_ai);
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

syncStatus();