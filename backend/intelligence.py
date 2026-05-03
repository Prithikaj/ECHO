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
            tb = analysis.get("transcription")
            if not isinstance(tb, dict):
                tb = {}
            turn = {
                "turn_id": self.total_turns + 1,
                "timestamp": time.time(),
                "speaker": analysis.get("speaker", "Unknown"),
                "transcript": (tb.get("transcript") or tb.get("text") or "").strip(),
                "language": tb.get("language") or "unknown",
                "emotion": analysis.get("emotion") if isinstance(analysis.get("emotion"), dict) else {},
                "intent": analysis.get("intent") if isinstance(analysis.get("intent"), dict) else {},
                "crisis": analysis.get("crisis") if isinstance(analysis.get("crisis"), dict) else {},
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

        tblock = analysis.get("transcription")
        if not isinstance(tblock, dict):
            tblock = {}
        transcript = (tblock.get("transcript") or tblock.get("text") or "").strip()
        language = tblock.get("language") or "english"
        primary_language = tblock.get("primary_language")
        if tblock.get("is_code_mixed") and primary_language:
            language = primary_language
        if not language or language == "unknown":
            language = "english"
        confidence = tblock.get("confidence_score")
        if confidence is None:
            confidence = 0.5
        emotion = analysis.get("emotion")
        if not isinstance(emotion, dict):
            emotion = {}
        intent = analysis.get("intent")
        if not isinstance(intent, dict):
            intent = {}
        crisis_data = analysis.get("crisis")
        if not isinstance(crisis_data, dict):
            crisis_data = {}

        # ── Step 2: Update conversation context ──────────────────────────────
        if ctx:
            ctx.add_turn(analysis)

        # ── Step 3: Crisis check ──────────────────────────────────────────────
        crisis_result = {}
        uq = float(emotion.get("urgency_score") or 0)
        if uq > 0.7 or emotion.get("is_crisis", False):
            try:
                crisis_result = detect_crisis(transcript, emotion, intent)
                if crisis_result.get("crisis_activated") and ctx:
                    ctx.crisis_active = True
                    ctx.bypass_ai = crisis_result.get("bypass_ai", False)
                    # ── Twilio WhatsApp alert ─────────────────────────────────
                    try:
                        from twilio_helpline import send_whatsapp_to_all_relevant
                        ents = intent.get("entities", {})
                        location = ents.get("location", "") or ""
                        wa_results = send_whatsapp_to_all_relevant(
                            crisis_type=crisis_result.get("crisis_type", "unknown"),
                            escalation_path=crisis_result.get("escalation_path", "police"),
                            location=location,
                            transcript=transcript[:150],
                            session_id=ctx.session_id,
                            emotion=emotion.get("primary_emotion", "unknown"),
                            urgency_score=emotion.get("urgency_score", 0.0),
                            entities=ents,
                        )
                        logger.info(f"WhatsApp alerts: {wa_results}")
                    except Exception as te:
                        logger.warning(f"WhatsApp alert skipped: {te}")
            except Exception as e:
                logger.warning(f"Crisis detection failed: {e}")

        # ── Step 4: Verification loop ─────────────────────────────────────────
        verification = {}
        assistant_response = (analysis.get("assistant_response") or "").strip()
        if transcript:
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
        elif assistant_response:
            verification = {
                "action": "clarify",
                "verification_statement": assistant_response,
                "clarification_question": "",
                "tts_text": assistant_response,
            }

        if not isinstance(verification, dict):
            verification = {}
        # ── Step 5: TTS response ──────────────────────────────────────────────
        tts = {}
        tts_message = (
            crisis_result.get("immediate_response")
            or assistant_response
            or verification.get("tts_text", "")
        )
        # If still no TTS message but we have a transcript, generate a default response
        if not tts_message and transcript:
            tts_message = f"I heard you. {verification.get('verification_statement', 'Can you please provide more details?')}"

        if tts_message:
            # Don't call Gemini for TTS if possible - use cached response
            try:
                # Check if we already have a cached TTS response for this exact message
                # to avoid redundant Gemini API calls within the same crisis event
                if "immediate_response" in crisis_result:  # Already from crisis detection
                    tts = {"tts_text": tts_message, "language": language, "tone": "urgent"}
                else:
                    tts = generate_tts_response(
                        message=tts_message,
                        language=language,
                        emotion_context=emotion.get("primary_emotion", "neutral"),
                        is_crisis=bool(crisis_result.get("crisis_activated")),
                    )
                    if not isinstance(tts, dict) or not (str(tts.get("tts_text") or "").strip()):
                        tts = {"tts_text": tts_message, "language": language, "tone": "calm"}
            except Exception as e:
                logger.warning(f"TTS generation failed: {e}")
                tts = {"tts_text": tts_message, "language": language, "tone": "calm"}

        # ── Step 6: Emotion trajectory ────────────────────────────────────────
        emotion_trajectory = ctx.get_emotion_trajectory() if ctx else "stable"

        # ── Assemble enriched result ──────────────────────────────────────────
        ai_latency = round(time.perf_counter() - start_time, 3)

        ents = intent.get("entities")
        if not isinstance(ents, dict):
            ents = {}
        missing = intent.get("missing_critical_info")
        if not isinstance(missing, list):
            missing = []

        enriched = {
            "chunk_id": chunk_id,
            "speaker": speaker,
            # Transcription
            "transcript": transcript,
            "language": language,
            "language_confidence": float(tblock.get("language_confidence") or 0),
            "normalized_text": (tblock.get("normalized_text") or transcript or "").strip(),
            "transcription_confidence": float(confidence or 0),
            "dialect_notes": (tblock.get("dialect_notes") or "").strip(),
            "is_code_mixed": bool(tblock.get("is_code_mixed", False)),
            # Emotion (JSON null → real defaults so the frontend always gets strings/numbers)
            "emotion": emotion.get("primary_emotion") or "neutral",
            "emotion_confidence": float(emotion.get("emotion_confidence") or 0),
            "urgency_level": emotion.get("urgency_level") or "low",
            "urgency_score": float(emotion.get("urgency_score") or 0),
            "sentiment": emotion.get("sentiment") or "neutral",
            "implicit_meaning": (emotion.get("implicit_meaning") or "").strip(),
            "emotion_trajectory": emotion_trajectory,
            # Intent
            "intent": intent.get("intent") or "unknown",
            "intent_confidence": float(intent.get("intent_confidence") or 0),
            "entities": ents,
            "risk_level": str(intent.get("risk_level") or "low").lower(),
            "requires_immediate_action": bool(intent.get("requires_immediate_action", False)),
            "missing_critical_info": missing,
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
