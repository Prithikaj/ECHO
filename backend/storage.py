"""
ECHO - Audio Storage
Saves clean, speaker-labeled audio chunks to disk so downstream
processes can pick them up without re-running the pipeline.

Directory layout:
    recordings/
    └── session_<YYYYMMDD_HHMMSS>/
        ├── manifest.json          ← full session index (all chunks)
        ├── Speaker_1/
        │   ├── chunk_001_0.0-1.5s.wav
        │   └── ...
        └── Speaker_2/
            └── ...
"""

import os
import json
import time
import threading
import numpy as np
import soundfile as sf
from datetime import datetime
from utils import get_logger

logger = get_logger("storage")

# Root folder for all recordings (relative to project root)
RECORDINGS_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "recordings"
)


class AudioStorage:
    """
    Thread-safe storage manager for a single recording session.

    Usage:
        store = AudioStorage()
        store.start_session()
        store.save_chunk(audio, sr, chunk_id, start_time, end_time,
                         speaker, latency, quality)
        store.end_session()
    """

    def __init__(self, recordings_root: str = RECORDINGS_ROOT):
        self.recordings_root = recordings_root
        self._session_dir: str | None = None
        self._manifest_path: str | None = None
        self._manifest: dict = {}
        self._lock = threading.Lock()
        self._active = False

    # ── Session lifecycle ──────────────────────────────────────────────────────

    def start_session(self) -> str:
        """
        Create a new timestamped session directory.
        Returns the session directory path.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = os.path.join(self.recordings_root, f"session_{ts}")
        os.makedirs(self._session_dir, exist_ok=True)

        self._manifest = {
            "session_id": f"session_{ts}",
            "started_at": datetime.now().isoformat(),
            "ended_at": None,
            "total_chunks": 0,
            "speakers": {},
            "chunks": [],
        }
        self._manifest_path = os.path.join(self._session_dir, "manifest.json")
        self._write_manifest()
        self._active = True

        logger.info(f"Session started → {self._session_dir}")
        return self._session_dir

    def end_session(self) -> dict:
        """
        Finalise the session manifest and return it.
        """
        if not self._active:
            return {}

        with self._lock:
            self._manifest["ended_at"] = datetime.now().isoformat()
            self._write_manifest()
            self._active = False

        logger.info(
            f"Session ended — {self._manifest['total_chunks']} chunks saved "
            f"across {len(self._manifest['speakers'])} speaker(s)"
        )
        return self._manifest

    # ── Chunk saving ───────────────────────────────────────────────────────────

    def save_chunk(
        self,
        audio: np.ndarray,
        sample_rate: int,
        chunk_id: int,
        start_time: float,
        end_time: float,
        speaker: str,
        latency: float = 0.0,
        quality: str = "clean",
        channels: int = 1,
    ) -> str | None:
        """
        Save a single audio chunk to disk.

        Returns the relative file path (relative to session dir),
        or None if storage is not active.
        """
        if not self._active or self._session_dir is None:
            logger.warning("save_chunk called but no active session.")
            return None

        # Only persist clean speech (skip noisy/silence chunks)
        if quality != "clean":
            logger.debug(f"Chunk {chunk_id} skipped (quality={quality})")
            return None

        # Build speaker sub-directory
        speaker_dir = os.path.join(
            self._session_dir, speaker.replace(" ", "_")
        )
        os.makedirs(speaker_dir, exist_ok=True)

        # Filename: chunk_001_0.0-1.5s.wav
        filename = f"chunk_{chunk_id:04d}_{start_time:.1f}-{end_time:.1f}s.wav"
        filepath = os.path.join(speaker_dir, filename)
        rel_path = os.path.relpath(filepath, self._session_dir)

        # Write WAV (float32 → PCM_16 for broad compatibility)
        try:
            sf.write(
                filepath,
                audio.astype(np.float32),
                sample_rate,
                format="WAV",
                subtype="PCM_16",
            )
        except Exception as exc:
            logger.error(f"Failed to write chunk {chunk_id}: {exc}")
            return None

        # Update manifest
        chunk_entry = {
            "chunk_id": chunk_id,
            "file": rel_path,
            "speaker": speaker,
            "start_time": round(start_time, 3),
            "end_time": round(end_time, 3),
            "duration": round(end_time - start_time, 3),
            "voiced_duration": round(len(audio) / sample_rate, 3),
            "sample_rate": sample_rate,
            "channels": channels,
            "latency": round(latency, 4),
            "quality": quality,
            "saved_at": datetime.now().isoformat(),
        }

        with self._lock:
            self._manifest["chunks"].append(chunk_entry)
            self._manifest["total_chunks"] += 1

            # Per-speaker summary
            spk = self._manifest["speakers"].setdefault(
                speaker,
                {"chunk_count": 0, "total_duration": 0.0, "files": []},
            )
            spk["chunk_count"] += 1
            spk["total_duration"] = round(
                spk["total_duration"] + chunk_entry["duration"], 3
            )
            spk["files"].append(rel_path)

            self._write_manifest()

        logger.debug(f"Saved chunk {chunk_id} → {rel_path}")
        return rel_path

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _write_manifest(self):
        """Write manifest.json (called under lock or during init)."""
        try:
            with open(self._manifest_path, "w", encoding="utf-8") as f:
                json.dump(self._manifest, f, indent=2)
        except Exception as exc:
            logger.error(f"Failed to write manifest: {exc}")

    @property
    def session_dir(self) -> str | None:
        return self._session_dir

    @property
    def is_active(self) -> bool:
        return self._active

    def get_manifest(self) -> dict:
        with self._lock:
            return dict(self._manifest)
