"""
ECHO - Intelligence Layer
Manages conversation context, emotion trajectory tracking,
verification loop, crisis mode, and the human-in-the-loop dashboard data.

This module sits between the pipeline and the API, enriching each chunk
with full AI analysis before broadcasting to the frontend.
"""

import threading
import time
from collections import deque
from typing import Optional
from utils import get_logger
from gemini_ai import (
    full_analysis,
    generate_verification,
    detect_crisis,
    generate_post_call_summary,
    generate_tts_response,
    normalize_dialect,
)

logger = get_logger("intelligence")


class ConversationContext:
    """
    Maintains rolling conversation history and emotion trajectory
    for a single call session.
    """

    def __init__(self, session_id: str, max_history: int = 50):
        self.session_id = session_id
        self.max_history = max_history
        self._history: deque = deque(maxlen=max_history)
        self._lock = threading.Lock()

        # Emotion trajectory (last 10 readings)
        self._emotion_history: deque = deque(maxlen=10)
        self._urgency_history: deque = deque(maxlen=10)

        # Crisis state
        self.crisis_active = False
        self.crisis_type = "none"
        self.bypass_ai = False
        self.silent_assist_mode = False

        # Verification state
        self.pending_verification = False
        self.verification_retries = 0
        self.max_verification_retries = 3

        # Aggregated stats
        self.total_turns = 0
        self.languages_detected = set()
        self.intents_seen = []

    def add_turn(self, analysis: dict):
        """Add a processed turn to conversation history."""
        with self._lock:
            turn = {
                "turn_id": self.total_turns + 1,
                "timestamp": time.time(),
                "speaker": analysis.get("speaker", "Unknown"),
                "transcript": analysis.get("transcription", {}).get("transcript", ""),
                "language": analysis.get("transcription", {}).get("language", "unknown"),
                "emotion": analysis.get("emotion", {}),
                "intent": analysis.get("intent", {}),
                "crisis": analysis.get("crisis", {}),
            }
            self._history.append(turn)
            self.total_turns += 1

            # Track languages
            lang = turn["language"]
            if lang and lang != "unknown":
                self.languages_detected.add(lang)

            # Track intents
            intent = turn["intent"].get("intent")
            if intent:
                self.intents_seen.append(intent)

            # Update emotion trajectory
            emotion = turn["emotion"]
            if emotion:
                self._emotion_history.append(emotion.get("primary_emotion", "neutral"))
                self._urgency_history.append(emotion.get("urgency_score", 0.0))

            # Update crisis state
            crisis = turn["crisis"]
            if crisis.get("crisis_activated"):
                self.crisis_active = True
                self.crisis_type = crisis.get("crisis_type", "unknown")
                self.bypass_ai = crisis.get("bypass_ai", False)
                self.silent_assist_mode = crisis.get("silent_assist_mode", False)

    def get_history(self) -> list:
        with self._lock:
            return list(self._history)

    def get_emotion_trajectory(self) -> str:
        """Determine if emotion is escalating, stable, or de-escalating."""
        if len(self._urgency_history) < 2:
            return "stable"
        recent = list(self._urgency_history)
        trend = recent[-1] - recent[0]
        if trend > 0.2:
            return "escalating"
        elif trend < -0.2:
            return "de-escalating"
        return "stable"

    def get_current_urgency(self) -> float:
        if not self._urgency_history:
            return 0.0
        return list(self._urgency_history)[-1]

    def get_dominant_emotion(self) -> str:
        if not self._emotion_history:
            return "neutral"
        emotions = list(self._emotion_history)
        return max(set(emotions), key=emotions.count)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "total_turns": self.total_turns,
            "languages_detected": list(self.languages_detected),
            "crisis_active": self.crisis_active,
            "crisis_type": self.crisis_type,
            "bypass_ai": self.bypass_ai,
            "silent_assist_mode": self.silent_assist_mode,
            "emotion_trajectory": self.get_emotion_trajectory(),
            "current_urgency": self.get_current_urgency(),
            "dominant_emotion": self.get_dominant_emotion(),
            "pending_verification": self.pending_verification,
            "verification_retries": self.verification_retries,
        }


class IntelligenceEngine:
    """
    Main intelligence orchestrator.
    Processes each audio chunk through the full AI pipeline.
    """

    def __init__(self):
        self._context: Optional[ConversationContext] = None
        self._lock = threading.Lock()
        self._processing_queue = []
        logger.info("Intelligence Engine initialized.")

    def start_session(self, session_id: str):
        """Start a new conversation session."""
        with self._lock:
            self._context = ConversationContext(session_id)
        logger.info(f"Intelligence session started: {session_id}")

    def end_session(self) -> dict:
        """End session and generate post-call summary."""
        with self._lock:
            if not self._context:
                return {}
            ctx = self._context

        history = ctx.get_history()
        if not history:
            return {"summary": "No conversation recorded."}

        try:
            summary = generate_post_call_summary(history, ctx.session_id)
            summary["session_stats"] = ctx.to_dict()
            return summary
        except Exception as e:
            logger.error(f"Post-call summary failed: {e}")
            return {"error": str(e), "session_stats": ctx.to_dict()}

    def process_chunk(self, audio_b64: str, speaker: str, chunk_id: int) -> dict:
        """
        Full AI processing pipeline for one audio chunk.

        Returns enriched analysis dict ready for SSE broadcast.
        """
        start_time = time.perf_counter()

        with self._lock:
            ctx = self._context
            history = ctx.get_history() if ctx else []

        # ── Step 1: Full Gemini analysis (transcription + emotion + intent + crisis) ──
        try:
            analysis = full_analysis(audio_b64, speaker=speaker, conversation_history=history)
        except Exception as e:
            logger.error(f"Gemini full_analysis failed for chunk {chunk_id}: {e}")
            analysis = {
                "transcription": {"transcript": "", "language": "unknown", "confidence_score": 0.0},
                "emotion": {"primary_emotion": "neutral", "urgency_score": 0.0, "is_crisis": False},
                "intent": {"intent": "unknown", "risk_level": "low"},
                "crisis": {"crisis_activated": False},
                "error": str(e),
            }

        transcript = analysis.get("transcription", {}).get("transcript", "")
        language = analysis.get("transcription", {}).get("language", "english")
        confidence = analysis.get("transcription", {}).get("confidence_score", 0.5)
        emotion = analysis.get("emotion", {})
        intent = analysis.get("intent", {})
        crisis_data = analysis.get("crisis", {})

        # ── Step 2: Update conversation context ──────────────────────────────
        if ctx:
            ctx.add_turn(analysis)

        # ── Step 3: Crisis check ──────────────────────────────────────────────
        crisis_result = {}
        if emotion.get("urgency_score", 0) > 0.7 or emotion.get("is_crisis", False):
            try:
                crisis_result = detect_crisis(transcript, emotion, intent)
                if crisis_result.get("crisis_activated") and ctx:
                    ctx.crisis_active = True
                    ctx.bypass_ai = crisis_result.get("bypass_ai", False)
            except Exception as e:
                logger.warning(f"Crisis detection failed: {e}")

        # ── Step 4: Verification loop ─────────────────────────────────────────
        verification = {}
        if transcript and not crisis_data.get("crisis_activated") and not (crisis_result.get("bypass_ai")):
            try:
                verification = generate_verification(
                    transcript=transcript,
                    intent=intent,
                    emotion=emotion,
                    language=language,
                    confidence=confidence,
                )
            except Exception as e:
                logger.warning(f"Verification generation failed: {e}")

        # ── Step 5: TTS response ──────────────────────────────────────────────
        tts = {}
        if verification.get("tts_text") or crisis_result.get("immediate_response"):
            tts_message = (
                crisis_result.get("immediate_response")
                or verification.get("tts_text", "")
            )
            try:
                tts = generate_tts_response(
                    message=tts_message,
                    language=language,
                    emotion_context=emotion.get("primary_emotion", "neutral"),
                    is_crisis=bool(crisis_result.get("crisis_activated")),
                )
            except Exception as e:
                logger.warning(f"TTS generation failed: {e}")
                tts = {"tts_text": tts_message, "language": language}

        # ── Step 6: Emotion trajectory ────────────────────────────────────────
        emotion_trajectory = ctx.get_emotion_trajectory() if ctx else "stable"

        # ── Assemble enriched result ──────────────────────────────────────────
        ai_latency = round(time.perf_counter() - start_time, 3)

        enriched = {
            "chunk_id": chunk_id,
            "speaker": speaker,
            # Transcription
            "transcript": transcript,
            "language": language,
            "language_confidence": analysis.get("transcription", {}).get("language_confidence", 0),
            "normalized_text": analysis.get("transcription", {}).get("normalized_text", transcript),
            "transcription_confidence": confidence,
            "dialect_notes": analysis.get("transcription", {}).get("dialect_notes", ""),
            "is_code_mixed": analysis.get("transcription", {}).get("is_code_mixed", False),
            # Emotion
            "emotion": emotion.get("primary_emotion", "neutral"),
            "emotion_confidence": emotion.get("emotion_confidence", 0),
            "urgency_level": emotion.get("urgency_level", "low"),
            "urgency_score": emotion.get("urgency_score", 0),
            "sentiment": emotion.get("sentiment", "neutral"),
            "implicit_meaning": emotion.get("implicit_meaning", ""),
            "emotion_trajectory": emotion_trajectory,
            # Intent
            "intent": intent.get("intent", "unknown"),
            "intent_confidence": intent.get("intent_confidence", 0),
            "entities": intent.get("entities", {}),
            "risk_level": intent.get("risk_level", "low"),
            "requires_immediate_action": intent.get("requires_immediate_action", False),
            "missing_critical_info": intent.get("missing_critical_info", []),
            # Crisis
            "crisis_activated": crisis_result.get("crisis_activated", False),
            "crisis_type": crisis_result.get("crisis_type", "none"),
            "crisis_severity": crisis_result.get("crisis_severity", 0),
            "bypass_ai": crisis_result.get("bypass_ai", False),
            "silent_assist_mode": crisis_result.get("silent_assist_mode", False),
            "escalation_path": crisis_result.get("escalation_path", ""),
            # Verification
            "verification_action": verification.get("action", ""),
            "verification_statement": verification.get("verification_statement", ""),
            "clarification_question": verification.get("clarification_question", ""),
            # TTS
            "tts_text": tts.get("tts_text", ""),
            "tts_tone": tts.get("tone", "calm"),
            "tts_language": tts.get("language", language),
            # Context
            "conversation_context": ctx.to_dict() if ctx else {},
            "ai_latency": ai_latency,
        }

        logger.info(
            f"[Chunk {chunk_id}] {speaker} | lang={language} | "
            f"emotion={emotion.get('primary_emotion')} | urgency={emotion.get('urgency_level')} | "
            f"intent={intent.get('intent')} | crisis={crisis_result.get('crisis_activated', False)} | "
            f"ai_latency={ai_latency}s"
        )

        return enriched

    def get_context(self) -> Optional[dict]:
        with self._lock:
            return self._context.to_dict() if self._context else None


# ── Singleton ──────────────────────────────────────────────────────────────────
_engine: Optional[IntelligenceEngine] = None
_engine_lock = threading.Lock()


def get_engine() -> IntelligenceEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = IntelligenceEngine()
    return _engine
