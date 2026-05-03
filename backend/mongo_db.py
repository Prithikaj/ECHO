"""
ECHO - MongoDB Integration
Persists sessions, chunks, and agent corrections to MongoDB.
Runs silently alongside the existing file-based storage — does not replace it.

Collections:
  sessions       — one document per call session (start/end, summary, stats)
  chunks         — one document per audio chunk (transcript, emotion, intent, crisis)
  corrections    — agent corrections with audit trail

Setup:
  1. pip install pymongo
  2. Create free cluster at https://cloud.mongodb.com
  3. Add MONGODB_URI to .env
"""

import os
import time
import threading
from datetime import datetime
from dotenv import load_dotenv
from utils import get_logger

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
logger = get_logger("mongo_db")

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = "echo_db"

_client = None
_db = None
_lock = threading.Lock()
_enabled = False


def _connect():
    """Lazy connect to MongoDB. Returns True if connected."""
    global _client, _db, _enabled
    with _lock:
        if _client is not None:
            return _enabled
        if not MONGODB_URI or "your_user" in MONGODB_URI:
            logger.warning("MongoDB URI not configured — skipping DB storage.")
            _enabled = False
            return False
        try:
            from pymongo import MongoClient
            from pymongo.server_api import ServerApi
            _client = MongoClient(MONGODB_URI, server_api=ServerApi("1"), serverSelectionTimeoutMS=5000)
            _client.admin.command("ping")  # test connection
            _db = _client[DB_NAME]
            _enabled = True
            logger.info(f"MongoDB connected → {DB_NAME}")
        except Exception as e:
            logger.warning(f"MongoDB connection failed: {e} — continuing without DB.")
            _enabled = False
    return _enabled


# ── Sessions ───────────────────────────────────────────────────────────────────

def save_session_start(session_id: str) -> None:
    """Called when a new session starts."""
    if not _connect():
        return
    try:
        _db.sessions.insert_one({
            "session_id": session_id,
            "started_at": datetime.utcnow(),
            "ended_at": None,
            "status": "active",
            "total_chunks": 0,
            "languages_detected": [],
            "crisis_events": 0,
            "transcriptions": 0,
        })
        logger.info(f"MongoDB: session started → {session_id}")
    except Exception as e:
        logger.warning(f"MongoDB save_session_start failed: {e}")


def save_session_end(session_id: str, post_call_summary: dict, stats: dict) -> None:
    """Called when a session ends — updates the session document with summary."""
    if not _connect():
        return
    try:
        _db.sessions.update_one(
            {"session_id": session_id},
            {"$set": {
                "ended_at": datetime.utcnow(),
                "status": "completed",
                "post_call_summary": post_call_summary,
                "pipeline_stats": stats,
                "total_chunks": stats.get("chunks_sent", 0),
                "transcriptions": stats.get("transcriptions", 0),
                "crisis_events": stats.get("crisis_events", 0),
                "languages_detected": list(stats.get("languages_detected", {}).keys()),
            }},
            upsert=True,
        )
        logger.info(f"MongoDB: session ended → {session_id}")
    except Exception as e:
        logger.warning(f"MongoDB save_session_end failed: {e}")


# ── Chunks ─────────────────────────────────────────────────────────────────────

def save_chunk(session_id: str, chunk_data: dict) -> None:
    """
    Save an AI-enriched chunk to MongoDB.
    Strips the raw audio (base64) before saving.
    """
    if not _connect():
        return
    try:
        doc = {k: v for k, v in chunk_data.items() if k != "audio"}
        doc["session_id"] = session_id
        doc["saved_at"] = datetime.utcnow()
        _db.chunks.insert_one(doc)
    except Exception as e:
        logger.warning(f"MongoDB save_chunk failed: {e}")


# ── Agent corrections ──────────────────────────────────────────────────────────

def save_correction(correction_data: dict) -> None:
    """Save an agent correction to MongoDB."""
    if not _connect():
        return
    try:
        doc = dict(correction_data)
        doc["saved_at"] = datetime.utcnow()
        _db.corrections.insert_one(doc)
        logger.info(f"MongoDB: correction saved")
    except Exception as e:
        logger.warning(f"MongoDB save_correction failed: {e}")


# ── Query helpers (for future use) ────────────────────────────────────────────

def get_all_sessions(limit: int = 50) -> list:
    """Return recent sessions from MongoDB."""
    if not _connect():
        return []
    try:
        return list(_db.sessions.find(
            {}, {"_id": 0}
        ).sort("started_at", -1).limit(limit))
    except Exception as e:
        logger.warning(f"MongoDB get_all_sessions failed: {e}")
        return []


def get_session_chunks(session_id: str) -> list:
    """Return all chunks for a session."""
    if not _connect():
        return []
    try:
        return list(_db.chunks.find(
            {"session_id": session_id}, {"_id": 0}
        ).sort("chunk_id", 1))
    except Exception as e:
        logger.warning(f"MongoDB get_session_chunks failed: {e}")
        return []
