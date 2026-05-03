"""
ECHO - Flask API Server
Receives AI-enriched audio chunks from the pipeline and serves real-time
status/events to the frontend via Server-Sent Events (SSE).

New endpoints:
  POST /audio_chunk        — receive enriched chunk (now includes AI fields)
  GET  /stream             — SSE for real-time frontend updates
  GET  /status             — pipeline + AI stats
  GET  /context            — current conversation context
  POST /pipeline/start     — start pipeline
  POST /pipeline/stop      — stop pipeline + get post-call summary
  GET  /recordings         — list saved sessions
  GET  /recordings/<id>/manifest — session manifest
  POST /analyze_text       — on-demand text analysis via Gemini
  GET  /health             — health check
"""

import os
import json
import queue
import threading
import time
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from utils import get_logger

# Load .env from project root (one level up from backend/)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logger = get_logger("app")

app = Flask(__name__, static_folder=None)
CORS(app)

# ── In-memory event bus for SSE ────────────────────────────────────────────────
_event_queue: queue.Queue = queue.Queue(maxsize=200)
_chunk_log: list[dict] = []
_chunk_log_lock = threading.Lock()
MAX_LOG = 200

# ── Pipeline reference ─────────────────────────────────────────────────────────
_pipeline = None


# ── Helper ────────────────────────────────────────────────────────────────────
def _push_event(data: dict):
    """Push a dict to the SSE event queue (non-blocking)."""
    try:
        _event_queue.put_nowait(data)
    except queue.Full:
        pass


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/audio_chunk", methods=["POST"])
def receive_audio_chunk():
    """
    Receive an AI-enriched audio chunk from the pipeline.

    Required fields: chunk_id, start_time, end_time, speaker, audio, latency, quality
    AI fields (optional): transcript, language, emotion, intent, crisis_activated, etc.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error", "message": "Invalid JSON"}), 400

    required = {"chunk_id", "start_time", "end_time", "speaker", "audio", "latency", "quality"}
    missing = required - data.keys()
    if missing:
        return jsonify({"status": "error", "message": f"Missing fields: {missing}"}), 400

    # Strip raw audio from log entry
    log_entry = {k: v for k, v in data.items() if k != "audio"}
    log_entry["received_at"] = round(time.time(), 3)

    with _chunk_log_lock:
        _chunk_log.append(log_entry)
        if len(_chunk_log) > MAX_LOG:
            _chunk_log.pop(0)

    # Log with AI fields if present
    ai_info = ""
    if data.get("transcript"):
        ai_info = (
            f" | transcript='{data['transcript'][:50]}…'" if len(data.get('transcript','')) > 50
            else f" | transcript='{data.get('transcript')}'"
        )
    if data.get("language"):
        ai_info += f" | lang={data['language']}"
    if data.get("emotion"):
        ai_info += f" | emotion={data['emotion']}"
    if data.get("intent"):
        ai_info += f" | intent={data['intent']}"
    if data.get("crisis_activated"):
        ai_info += f" | 🚨CRISIS={data.get('crisis_type')}"

    logger.info(
        f"[Chunk {data['chunk_id']}] {data['speaker']} | "
        f"{data['start_time']:.2f}–{data['end_time']:.2f}s | "
        f"latency={data['latency']}s | quality={data['quality']}{ai_info}"
    )

    # Broadcast full enriched chunk to SSE clients (NOT just chunk metadata, but ALL enriched AI fields)
    _push_event({"type": "chunk", **log_entry})

    # Save chunk to MongoDB (non-blocking, fire and forget)
    try:
        from mongo_db import save_chunk
        session_id = ""
        if _pipeline and hasattr(_pipeline, "_session_id"):
            session_id = _pipeline._session_id
        import threading
        threading.Thread(target=save_chunk, args=(session_id, log_entry), daemon=True).start()
    except Exception:
        pass

    # If crisis detected, push a separate high-priority crisis event
    if data.get("crisis_activated"):
        _push_event({
            "type": "crisis_alert",
            "chunk_id": data["chunk_id"],
            "crisis_type": data.get("crisis_type", "unknown"),
            "crisis_severity": data.get("crisis_severity", 0),
            "bypass_ai": data.get("bypass_ai", False),
            "escalation_path": data.get("escalation_path", ""),
            "immediate_response": data.get("tts_text", ""),
            "tts_language": data.get("tts_language") or data.get("language", "english"),
            "timestamp": log_entry["received_at"],
        })

    return jsonify({"status": "ok", "chunk_id": data["chunk_id"]}), 200


@app.route("/status", methods=["GET"])
def status():
    """Return pipeline status, AI stats, and recent chunk log."""
    pipeline_stats = _pipeline.get_stats() if _pipeline else {}
    with _chunk_log_lock:
        recent = list(_chunk_log[-50:])

    storage_info = {}
    if _pipeline and hasattr(_pipeline, "_storage") and _pipeline._storage:
        storage_info = {
            "session_dir": _pipeline._storage.session_dir,
            "is_active": _pipeline._storage.is_active,
            "manifest": _pipeline._storage.get_manifest(),
        }

    # AI context
    ai_context = {}
    if _pipeline and hasattr(_pipeline, "_intelligence") and _pipeline._intelligence:
        ai_context = _pipeline._intelligence.get_context() or {}

    return jsonify({
        "status": "running" if (_pipeline and _pipeline._running) else "idle",
        "pipeline": pipeline_stats,
        "recent_chunks": recent,
        "storage": storage_info,
        "ai_context": ai_context,
    })


@app.route("/context", methods=["GET"])
def get_context():
    """Return current conversation context and AI analysis state."""
    if _pipeline and hasattr(_pipeline, "_intelligence") and _pipeline._intelligence:
        ctx = _pipeline._intelligence.get_context()
        return jsonify(ctx or {"status": "no_active_session"})
    return jsonify({"status": "ai_not_enabled"})


@app.route("/stream", methods=["GET"])
def stream():
    """Server-Sent Events endpoint for real-time frontend updates."""
    def event_generator():
        yield "data: {\"type\": \"connected\"}\n\n"
        while True:
            try:
                event = _event_queue.get(timeout=15)
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                yield "data: {\"type\": \"heartbeat\"}\n\n"

    return Response(
        event_generator(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/pipeline/start", methods=["POST"])
def pipeline_start():
    """Start the audio pipeline."""
    if _pipeline is None:
        return jsonify({"status": "error", "message": "Pipeline not embedded"}), 400
    if _pipeline._running:
        return jsonify({"status": "already_running"}), 200
    _pipeline.start()
    _push_event({"type": "pipeline_status", "status": "started"})
    return jsonify({"status": "started"}), 200


@app.route("/pipeline/stop", methods=["POST"])
def pipeline_stop():
    """Stop the audio pipeline and return post-call summary."""
    if _pipeline is None:
        return jsonify({"status": "error", "message": "Pipeline not embedded"}), 400
    post_call = _pipeline.stop()
    _push_event({"type": "pipeline_status", "status": "stopped"})
    _push_event({"type": "post_call_summary", "summary": post_call})

    # Save session end to MongoDB
    try:
        from mongo_db import save_session_end
        import threading
        threading.Thread(
            target=save_session_end,
            args=(_pipeline._session_id, post_call, _pipeline.get_stats()),
            daemon=True
        ).start()
    except Exception:
        pass
    return jsonify({
        "status": "stopped",
        "stats": _pipeline.get_stats(),
        "post_call_summary": post_call,
    }), 200


@app.route("/analyze_text", methods=["POST"])
def analyze_text():
    """
    On-demand text analysis via Gemini.
    Body: { "text": "...", "language": "english" }
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    try:
        from gemini_ai import classify_intent_and_extract, analyze_emotion, normalize_dialect
        text = data["text"]
        language = data.get("language", "english")

        emotion = analyze_emotion(text)
        intent = classify_intent_and_extract(text, language)
        normalized = normalize_dialect(text, language)

        return jsonify({
            "text": text,
            "language": language,
            "emotion": emotion,
            "intent": intent,
            "normalized": normalized,
        })
    except Exception as e:
        logger.error(f"Text analysis failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/recordings", methods=["GET"])
def list_recordings():
    """List all saved recording sessions."""
    from storage import RECORDINGS_ROOT

    sessions = []
    if os.path.isdir(RECORDINGS_ROOT):
        for entry in sorted(os.listdir(RECORDINGS_ROOT), reverse=True):
            session_path = os.path.join(RECORDINGS_ROOT, entry)
            manifest_path = os.path.join(session_path, "manifest.json")
            if os.path.isfile(manifest_path):
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                    sessions.append({
                        "session_id": manifest.get("session_id", entry),
                        "started_at": manifest.get("started_at"),
                        "ended_at": manifest.get("ended_at"),
                        "total_chunks": manifest.get("total_chunks", 0),
                        "speakers": list(manifest.get("speakers", {}).keys()),
                        "path": session_path,
                    })
                except Exception:
                    pass

    return jsonify({"sessions": sessions, "count": len(sessions)})


@app.route("/recordings/<session_id>/manifest", methods=["GET"])
def get_manifest(session_id):
    """Return the full manifest for a specific session."""
    from storage import RECORDINGS_ROOT

    manifest_path = os.path.join(RECORDINGS_ROOT, session_id, "manifest.json")
    if not os.path.isfile(manifest_path):
        return jsonify({"error": "Session not found"}), 404

    with open(manifest_path, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.route("/agent_correction", methods=["POST"])
def agent_correction():
    """Log agent corrections to a file for audit trail."""
    data = request.get_json(silent=True) or {}
    data["logged_at"] = time.time()
    corrections_file = os.path.join(os.path.dirname(__file__), "..", "recordings", "agent_corrections.jsonl")
    os.makedirs(os.path.dirname(corrections_file), exist_ok=True)
    try:
        with open(corrections_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
        logger.info(f"Agent correction logged: intent={data.get('intent')} risk={data.get('risk')}")
        # Also save to MongoDB
        try:
            from mongo_db import save_correction
            import threading
            threading.Thread(target=save_correction, args=(data,), daemon=True).start()
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Failed to log correction: {e}")
    return jsonify({"status": "ok"}), 200


@app.route("/connect_human", methods=["POST"])
def connect_human():
    """Request a human professional connection when crisis needs human takeover."""
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id") or (getattr(_pipeline, "_session_id", "") if _pipeline else "")
    last_chunk = _chunk_log[-1] if _chunk_log else {}
    crisis_type = data.get("crisis_type") or last_chunk.get("crisis_type") or "life_threat"
    location = data.get("location") or ""
    transcript = data.get("transcript") or last_chunk.get("transcript") or ""
    urgency_score = float(data.get("urgency_score") or last_chunk.get("urgency_score") or 0.0)
    entities = data.get("entities") or last_chunk.get("entities") or {}

    result = {"status": "not_configured", "message": "Human connection not sent."}
    try:
        from twilio_helpline import send_whatsapp_alert
        send_result = send_whatsapp_alert(
            escalation_path="human_agent",
            crisis_type=crisis_type,
            location=location,
            transcript=transcript,
            session_id=session_id or "unknown_session",
            emotion=data.get("emotion", last_chunk.get("emotion", "unknown")),
            urgency_score=urgency_score,
            entities=entities,
        )
        result = {"status": "ok", "details": send_result}
        _push_event({
            "type": "human_connection",
            "success": True,
            "message": "Human professional notified.",
            "details": send_result,
            "timestamp": time.time(),
        })
    except Exception as e:
        result = {"status": "error", "reason": str(e)}
        _push_event({
            "type": "human_connection",
            "success": False,
            "message": "Human professional connection failed.",
            "reason": str(e),
            "timestamp": time.time(),
        })

    return jsonify(result), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "ECHO",
        "ai_enabled": bool(_pipeline and getattr(_pipeline, "enable_ai", False)),
    }), 200


# ── Serve frontend static files ────────────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:filename>", methods=["GET"])
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# ── Entry point ────────────────────────────────────────────────────────────────
def create_app(embed_pipeline: bool = False, enable_ai: bool = True):
    global _pipeline
    if embed_pipeline:
        from pipeline import EchoPipeline
        _pipeline = EchoPipeline(enable_ai=enable_ai)
        _pipeline.start()
        logger.info(f"Embedded pipeline started (AI: {enable_ai}).")
    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ECHO Flask API")
    parser.add_argument("--embed-pipeline", action="store_true",
                        help="Run the audio pipeline inside this process")
    parser.add_argument("--no-ai", action="store_true",
                        help="Disable Gemini AI analysis")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    application = create_app(
        embed_pipeline=args.embed_pipeline,
        enable_ai=not args.no_ai,
    )
    logger.info(f"Starting ECHO server on {args.host}:{args.port} (AI: {not args.no_ai})")
    application.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=False,
        threaded=True,
    )
