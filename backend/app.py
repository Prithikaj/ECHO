"""
ECHO - Flask API Server
Receives processed audio chunks from the pipeline and serves real-time
status/events to the frontend via Server-Sent Events (SSE).
"""

import os
import json
import queue
import threading
import time
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from utils import get_logger

logger = get_logger("app")

app = Flask(__name__, static_folder=None)
CORS(app)

# ── In-memory event bus for SSE ────────────────────────────────────────────────
_event_queue: queue.Queue = queue.Queue(maxsize=100)
_chunk_log: list[dict] = []          # last N chunks for /status
_chunk_log_lock = threading.Lock()
MAX_LOG = 200

# ── Pipeline reference (set when pipeline is embedded) ────────────────────────
_pipeline = None


# ── Helper ────────────────────────────────────────────────────────────────────
def _push_event(data: dict):
    """Push a dict to the SSE event queue (non-blocking)."""
    try:
        _event_queue.put_nowait(data)
    except queue.Full:
        pass  # drop oldest — frontend will catch up


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/audio_chunk", methods=["POST"])
def receive_audio_chunk():
    """
    Receive a processed audio chunk from the pipeline.

    Expected JSON body:
    {
        "chunk_id":   int,
        "start_time": float,
        "end_time":   float,
        "speaker":    "Speaker 1" | "Speaker 2",
        "audio":      "<base64-encoded WAV>",
        "latency":    float,
        "quality":    "clean" | "noisy",
        "channels":   int,          (optional)
        "segments":   [...]         (optional diarization segments)
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error", "message": "Invalid JSON"}), 400

    # Validate required fields
    required = {"chunk_id", "start_time", "end_time", "speaker", "audio", "latency", "quality"}
    missing = required - data.keys()
    if missing:
        return jsonify({"status": "error", "message": f"Missing fields: {missing}"}), 400

    # Strip audio from log entry (keep it lean)
    log_entry = {k: v for k, v in data.items() if k != "audio"}
    log_entry["received_at"] = round(time.time(), 3)

    with _chunk_log_lock:
        _chunk_log.append(log_entry)
        if len(_chunk_log) > MAX_LOG:
            _chunk_log.pop(0)

    logger.info(
        f"[Chunk {data['chunk_id']}] {data['speaker']} | "
        f"{data['start_time']:.2f}–{data['end_time']:.2f}s | "
        f"latency={data['latency']}s | quality={data['quality']}"
    )

    # Broadcast to SSE clients
    _push_event({"type": "chunk", **log_entry})

    return jsonify({"status": "ok", "chunk_id": data["chunk_id"]}), 200


@app.route("/status", methods=["GET"])
def status():
    """Return pipeline status and recent chunk log."""
    pipeline_stats = _pipeline.get_stats() if _pipeline else {}
    with _chunk_log_lock:
        recent = list(_chunk_log[-50:])

    # Include storage info if pipeline has an active session
    storage_info = {}
    if _pipeline and hasattr(_pipeline, "_storage") and _pipeline._storage:
        storage_info = {
            "session_dir": _pipeline._storage.session_dir,
            "is_active": _pipeline._storage.is_active,
            "manifest": _pipeline._storage.get_manifest(),
        }

    return jsonify({
        "status": "running" if (_pipeline and _pipeline._running) else "idle",
        "pipeline": pipeline_stats,
        "recent_chunks": recent,
        "storage": storage_info,
    })


@app.route("/stream", methods=["GET"])
def stream():
    """
    Server-Sent Events endpoint.
    The frontend connects here to receive real-time chunk events.
    """
    def event_generator():
        # Send a heartbeat immediately so the browser knows the connection is live
        yield "data: {\"type\": \"connected\"}\n\n"
        while True:
            try:
                event = _event_queue.get(timeout=15)
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                # Heartbeat to keep connection alive
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
    """Start the audio pipeline (if embedded)."""
    if _pipeline is None:
        return jsonify({"status": "error", "message": "Pipeline not embedded"}), 400
    if _pipeline._running:
        return jsonify({"status": "already_running"}), 200
    _pipeline.start()
    _push_event({"type": "pipeline_status", "status": "started"})
    return jsonify({"status": "started"}), 200


@app.route("/pipeline/stop", methods=["POST"])
def pipeline_stop():
    """Stop the audio pipeline (if embedded)."""
    if _pipeline is None:
        return jsonify({"status": "error", "message": "Pipeline not embedded"}), 400
    _pipeline.stop()
    _push_event({"type": "pipeline_status", "status": "stopped"})
    return jsonify({"status": "stopped", "stats": _pipeline.get_stats()}), 200


@app.route("/recordings", methods=["GET"])
def list_recordings():
    """
    List all saved recording sessions.
    Returns session folders with their manifest summaries.
    """
    import os
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
    import os
    from storage import RECORDINGS_ROOT

    manifest_path = os.path.join(RECORDINGS_ROOT, session_id, "manifest.json")
    if not os.path.isfile(manifest_path):
        return jsonify({"error": "Session not found"}), 404

    with open(manifest_path, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "ECHO"}), 200


# ── Serve frontend static files ────────────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:filename>", methods=["GET"])
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# ── Entry point ────────────────────────────────────────────────────────────────
def create_app(embed_pipeline: bool = False):
    """
    Factory function.
    If embed_pipeline=True, the pipeline runs inside the same process.
    Otherwise, run pipeline.py separately.
    """
    global _pipeline
    if embed_pipeline:
        from pipeline import EchoPipeline
        _pipeline = EchoPipeline()
        _pipeline.start()
        logger.info("Embedded pipeline started.")
    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ECHO Flask API")
    parser.add_argument("--embed-pipeline", action="store_true",
                        help="Run the audio pipeline inside this process")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    application = create_app(embed_pipeline=args.embed_pipeline)
    logger.info(f"Starting ECHO server on {args.host}:{args.port}")
    application.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=False,   # reloader breaks background threads
        threaded=True,
    )
