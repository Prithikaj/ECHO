"""
ECHO - Gemini AI Integration
Uses Gemini 2.5 Flash for:
  - Language detection
  - Multilingual transcription (Kannada, Hindi, English + code-mix)
  - Emotion & sentiment analysis
  - Intent classification + entity extraction
  - Verification loop generation
  - Crisis detection
  - Post-call summary generation
  - Dialect normalization & cultural phrase interpretation
"""

import os
import json
import base64
import threading
import time
from typing import Optional
from google import genai
from google.genai import types
from utils import get_logger

logger = get_logger("gemini_ai")

# ── API Key ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = "gemini-2.5-flash"

# ── Client (new SDK uses a client object, not a global configure) ──────────────
_client = None
_client_lock = threading.Lock()


def _get_client() -> genai.Client:
    global _client
    with _client_lock:
        if _client is None:
            _client = genai.Client(api_key=GEMINI_API_KEY)
            logger.info("Gemini 2.5 Flash client initialised.")
    return _client


# ── Core JSON caller ───────────────────────────────────────────────────────────
def _call_gemini(prompt: str, audio_b64: Optional[str] = None, retries: int = 2) -> dict:
    """
    Call Gemini and parse JSON response.
    Optionally attach base64 WAV audio as inline data.
    """
    client = _get_client()

    # Build content parts
    parts = []
    if audio_b64:
        parts.append(
            types.Part.from_bytes(
                data=base64.b64decode(audio_b64),
                mime_type="audio/wav",
            )
        )
    parts.append(types.Part.from_text(text=prompt))

    config = types.GenerateContentConfig(
        temperature=0.2,
        top_p=0.95,
        max_output_tokens=2048,
        response_mime_type="application/json",
    )

    for attempt in range(retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=parts,
                config=config,
            )
            text = response.text.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Gemini JSON parse error (attempt {attempt+1}): {e}")
            if attempt == retries:
                return {"error": "json_parse_failed", "raw": getattr(response, "text", "")[:200]}
        except Exception as e:
            logger.error(f"Gemini API error (attempt {attempt+1}): {e}")
            if attempt == retries:
                return {"error": str(e)}
            time.sleep(1.5 * (attempt + 1))

    return {"error": "max_retries_exceeded"}


# ══════════════════════════════════════════════════════════════════════════════
# 1. TRANSCRIPTION + LANGUAGE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def transcribe_and_detect(audio_b64: str, speaker: str = "Speaker 1") -> dict:
    """
    Transcribe audio and detect language in one Gemini call.

    Returns:
    {
      "transcript": "...",
      "language": "kannada|hindi|english|code_mixed",
      "language_confidence": 0.95,
      "dialect_notes": "...",
      "normalized_text": "...",
      "confidence_score": 0.88,
      "word_confidences": [{"word": "...", "confidence": 0.9}, ...]
    }
    """
    prompt = f"""You are an expert multilingual speech transcription system for an emergency call center in India.
The speaker is: {speaker}

Listen to this audio and return a JSON object with:
{{
  "transcript": "exact transcription of what was said",
  "language": "one of: kannada, hindi, english, code_mixed, unknown",
  "language_confidence": 0.0-1.0,
  "dialect_notes": "any dialect or accent observations (e.g. North Karnataka Kannada, Hyderabadi Hindi)",
  "normalized_text": "standardized version removing slang/dialect, in the detected language",
  "confidence_score": 0.0-1.0,
  "word_confidences": [{{"word": "word", "confidence": 0.0-1.0}}],
  "is_code_mixed": true/false,
  "code_mix_languages": ["list of languages mixed if code_mixed"]
}}

Handle Kannada, Hindi, English, and code-mixed speech. If audio is silent or unclear, set transcript to "" and confidence_score to 0.0."""

    result = _call_gemini(prompt, audio_b64=audio_b64)
    result["speaker"] = speaker
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 2. EMOTION & SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_emotion(transcript: str, audio_b64: Optional[str] = None) -> dict:
    """
    Analyze emotion, sentiment, and urgency from transcript (+ optional audio).

    Returns:
    {
      "primary_emotion": "fear|panic|anger|confusion|calm|distress|neutral",
      "emotion_confidence": 0.95,
      "secondary_emotions": [...],
      "sentiment": "negative|neutral|positive",
      "urgency_level": "critical|high|medium|low",
      "urgency_score": 0.0-1.0,
      "is_crisis": true/false,
      "crisis_indicators": [...],
      "voice_cues": {"pitch": "high/normal/low", "speed": "fast/normal/slow", "stress": "high/normal/low"},
      "emotion_trajectory": "escalating|stable|de-escalating"
    }
    """
    prompt = f"""You are an expert emotion analysis system for an emergency call center.
Analyze the following transcript for emotional state and urgency.

Transcript: "{transcript}"

Return a JSON object:
{{
  "primary_emotion": "one of: fear, panic, anger, confusion, calm, distress, sadness, neutral",
  "emotion_confidence": 0.0-1.0,
  "secondary_emotions": ["list of other detected emotions"],
  "sentiment": "negative|neutral|positive",
  "urgency_level": "critical|high|medium|low",
  "urgency_score": 0.0-1.0,
  "is_crisis": true if life-threatening emergency detected,
  "crisis_indicators": ["list of specific words/phrases indicating crisis"],
  "voice_cues": {{
    "pitch": "high|normal|low",
    "speed": "fast|normal|slow",
    "stress": "high|normal|low"
  }},
  "emotion_trajectory": "escalating|stable|de-escalating",
  "implicit_meaning": "any hidden meaning, e.g. hesitation suggesting fear, understatement of severity"
}}

Consider cultural context: In Indian languages, 'he is not well' may mean distress, not illness.
Hesitation, repetition, or incomplete sentences may indicate fear or trauma."""

    return _call_gemini(prompt, audio_b64=audio_b64)


# ══════════════════════════════════════════════════════════════════════════════
# 3. INTENT CLASSIFICATION + ENTITY EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def classify_intent_and_extract(transcript: str, language: str = "english") -> dict:
    """
    Classify caller intent and extract key entities.

    Returns:
    {
      "intent": "emergency|complaint|inquiry|report|request_help|...",
      "intent_confidence": 0.92,
      "sub_intent": "...",
      "entities": {
        "location": "...",
        "people_involved": [...],
        "incident_type": "...",
        "time_mentioned": "...",
        "phone_numbers": [...],
        "names": [...]
      },
      "risk_level": "critical|high|medium|low",
      "requires_immediate_action": true/false,
      "predicted_next_intent": "..."
    }
    """
    prompt = f"""You are an intent classification system for an Indian emergency/complaint call center.
Language: {language}

Transcript: "{transcript}"

Return a JSON object:
{{
  "intent": "one of: emergency, complaint, inquiry, report_crime, request_help, provide_information, harassment_report, medical_emergency, fire_emergency, accident_report, domestic_violence, missing_person, other",
  "intent_confidence": 0.0-1.0,
  "sub_intent": "more specific intent description",
  "entities": {{
    "location": "extracted location or null",
    "area_landmark": "nearby landmark if mentioned",
    "people_involved": ["list of people mentioned"],
    "incident_type": "type of incident",
    "time_mentioned": "any time reference",
    "phone_numbers": ["any phone numbers mentioned"],
    "names": ["any names mentioned"],
    "vehicle_numbers": ["any vehicle numbers"]
  }},
  "risk_level": "critical|high|medium|low",
  "requires_immediate_action": true/false,
  "predicted_next_intent": "what the caller is likely to say/need next",
  "missing_critical_info": ["list of important info not yet provided, e.g. exact location, nature of emergency"]
}}"""

    return _call_gemini(prompt)


# ══════════════════════════════════════════════════════════════════════════════
# 4. VERIFICATION LOOP GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_verification(
    transcript: str,
    intent: dict,
    emotion: dict,
    language: str = "english",
    confidence: float = 0.8,
) -> dict:
    """
    Generate AI verification/clarification response.

    Returns:
    {
      "action": "proceed|clarify|escalate",
      "verification_statement": "...",
      "clarification_question": "...",
      "response_language": "...",
      "simplified_version": "...",
      "tts_text": "...",
      "confidence_assessment": "high|medium|low"
    }
    """
    urgency = emotion.get("urgency_level", "medium")
    intent_type = intent.get("intent", "unknown")
    entities = intent.get("entities", {})

    if confidence >= 0.85:
        action_hint = "proceed — restate and confirm"
    elif confidence >= 0.6:
        action_hint = "clarify — ask one focused question"
    else:
        action_hint = "escalate — too unclear, transfer to human agent"

    prompt = f"""You are a compassionate AI call center assistant for an Indian emergency helpline.
Caller language: {language}
Detected intent: {intent_type}
Urgency: {urgency}
Extracted info: {json.dumps(entities)}
Confidence: {confidence:.0%}
Recommended action: {action_hint}

Transcript: "{transcript}"

Generate a verification response. Return JSON:
{{
  "action": "proceed|clarify|escalate",
  "verification_statement": "Restate what you understood in {language} (e.g. 'You are reporting X near Y location, is that correct?')",
  "clarification_question": "If clarify needed: one simple focused question in {language}",
  "response_language": "{language}",
  "simplified_version": "Same message in very simple words for distressed callers",
  "tts_text": "Text to be spoken aloud — calm, clear, empathetic tone",
  "confidence_assessment": "high|medium|low",
  "retry_count_recommendation": 1-3
}}

Rules:
- For panic/fear: use very short, calm sentences
- For crisis: skip verification, go straight to action
- Always respond in the caller's language ({language})
- Never ask more than one question at a time"""

    return _call_gemini(prompt)


# ══════════════════════════════════════════════════════════════════════════════
# 5. CRISIS MODE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_crisis(transcript: str, emotion: dict, intent: dict) -> dict:
    """
    Determine if crisis mode should be activated.

    Returns:
    {
      "crisis_activated": true/false,
      "crisis_type": "...",
      "crisis_severity": 1-10,
      "immediate_response": "...",
      "escalation_path": "...",
      "silent_assist_mode": true/false
    }
    """
    urgency_score = emotion.get("urgency_score", 0)
    is_crisis_emotion = emotion.get("is_crisis", False)
    risk_level = intent.get("risk_level", "low")

    prompt = f"""You are a crisis detection system for an emergency call center.

Transcript: "{transcript}"
Urgency score: {urgency_score}
Emotion crisis flag: {is_crisis_emotion}
Risk level: {risk_level}
Primary emotion: {emotion.get('primary_emotion', 'unknown')}
Intent: {intent.get('intent', 'unknown')}

Determine if CRISIS MODE should be activated. Return JSON:
{{
  "crisis_activated": true/false,
  "crisis_type": "life_threat|violence|medical|fire|missing_person|harassment|none",
  "crisis_severity": 1-10,
  "immediate_response": "Short actionable response to give caller RIGHT NOW",
  "escalation_path": "police|ambulance|fire|women_helpline|child_helpline|human_agent",
  "silent_assist_mode": true if caller cannot speak safely (e.g. domestic violence),
  "silent_prompts": ["Yes/No questions if silent assist mode is on"],
  "bypass_ai": true if human agent must take over immediately
}}

Crisis triggers:
- Urgency score > 0.8
- Keywords: help, bachao, save me, dying, fire, blood, attack, rape, kidnap
- Caller stops speaking mid-sentence (possible danger)
- Extreme panic or fear detected"""

    return _call_gemini(prompt)


# ══════════════════════════════════════════════════════════════════════════════
# 6. FULL PIPELINE ANALYSIS (single call for efficiency)
# ══════════════════════════════════════════════════════════════════════════════

def full_analysis(audio_b64: str, speaker: str = "Speaker 1", conversation_history: list = None) -> dict:
    """
    Single Gemini call that does transcription + language + emotion + intent.
    More efficient than 4 separate calls.

    Returns combined analysis dict.
    """
    history_text = ""
    if conversation_history:
        recent = conversation_history[-5:]  # last 5 turns
        history_text = "\n".join([
            f"{h.get('speaker', 'Unknown')}: {h.get('transcript', '')}"
            for h in recent
        ])

    prompt = f"""You are ECHO, an AI system for an Indian emergency call center.
Analyze this audio from {speaker}.

{"Previous conversation context:" + chr(10) + history_text if history_text else ""}

Return a comprehensive JSON analysis:
{{
  "transcription": {{
    "transcript": "exact text spoken",
    "language": "kannada|hindi|english|code_mixed|unknown",
    "language_confidence": 0.0-1.0,
    "normalized_text": "standardized version",
    "confidence_score": 0.0-1.0,
    "dialect_notes": "any dialect observations",
    "is_code_mixed": true/false
  }},
  "emotion": {{
    "primary_emotion": "fear|panic|anger|confusion|calm|distress|sadness|neutral",
    "emotion_confidence": 0.0-1.0,
    "urgency_level": "critical|high|medium|low",
    "urgency_score": 0.0-1.0,
    "is_crisis": true/false,
    "crisis_indicators": [],
    "implicit_meaning": "hidden meaning if any"
  }},
  "intent": {{
    "intent": "emergency|complaint|inquiry|report_crime|request_help|harassment_report|medical_emergency|fire_emergency|accident_report|domestic_violence|missing_person|other",
    "intent_confidence": 0.0-1.0,
    "entities": {{
      "location": null,
      "area_landmark": null,
      "people_involved": [],
      "incident_type": null,
      "time_mentioned": null,
      "names": [],
      "phone_numbers": []
    }},
    "risk_level": "critical|high|medium|low",
    "requires_immediate_action": false,
    "missing_critical_info": []
  }},
  "crisis": {{
    "crisis_activated": false,
    "crisis_type": "none",
    "crisis_severity": 0,
    "bypass_ai": false,
    "silent_assist_mode": false
  }}
}}

Handle Kannada, Hindi, English, and code-mixed speech.
Consider Indian cultural context for phrase interpretation.
If audio is silent/unclear, set transcript to "" and all confidence scores to 0."""

    result = _call_gemini(prompt, audio_b64=audio_b64)
    result["speaker"] = speaker
    result["analyzed_at"] = time.time()
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 7. POST-CALL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def generate_post_call_summary(conversation_history: list, session_id: str) -> dict:
    """
    Generate structured post-call report from full conversation history.

    Returns:
    {
      "summary": "...",
      "issue": "...",
      "primary_emotion_overall": "...",
      "actions_taken": [...],
      "timeline": [...],
      "risk_assessment": "...",
      "follow_up_required": true/false,
      "case_category": "...",
      "structured_report": {...}
    }
    """
    transcript_text = "\n".join([
        f"[{h.get('timestamp', '')}] {h.get('speaker', 'Unknown')}: {h.get('transcript', '')}"
        for h in conversation_history
        if h.get('transcript')
    ])

    prompt = f"""You are generating a post-call intelligence report for an Indian emergency call center.
Session ID: {session_id}

Full conversation transcript:
{transcript_text}

Generate a comprehensive case report as JSON:
{{
  "summary": "2-3 sentence summary of the entire call",
  "issue": "primary issue reported by caller",
  "primary_emotion_overall": "dominant emotion throughout call",
  "emotion_progression": ["list of emotion changes during call"],
  "actions_taken": ["list of actions taken or recommended"],
  "timeline": [
    {{"time": "timestamp", "event": "what happened", "speaker": "who"}}
  ],
  "risk_assessment": "critical|high|medium|low",
  "follow_up_required": true/false,
  "follow_up_notes": "what follow-up is needed",
  "case_category": "emergency|complaint|inquiry|harassment|medical|fire|accident|domestic_violence|other",
  "languages_used": ["list of languages detected"],
  "key_entities": {{
    "location": null,
    "people_involved": [],
    "incident_type": null
  }},
  "structured_report": {{
    "caller_profile": "brief description",
    "incident_description": "detailed incident description",
    "response_quality": "assessment of AI response quality",
    "escalation_history": []
  }},
  "agent_notes_placeholder": "Space for human agent to add notes"
}}"""

    return _call_gemini(prompt)


# ══════════════════════════════════════════════════════════════════════════════
# 8. TEXT-TO-SPEECH TEXT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_tts_response(
    message: str,
    language: str,
    emotion_context: str = "neutral",
    is_crisis: bool = False,
) -> dict:
    """
    Generate appropriate TTS text with tone guidance.

    Returns:
    {
      "tts_text": "...",
      "tone": "calm|urgent|empathetic|clear",
      "language": "...",
      "speaking_rate": "slow|normal|fast",
      "emphasis_words": [...]
    }
    """
    prompt = f"""Generate a voice response for an emergency call center AI.
Target language: {language}
Caller emotion: {emotion_context}
Crisis mode: {is_crisis}
Message to convey: "{message}"

Return JSON:
{{
  "tts_text": "Natural spoken text in {language} — appropriate for the emotional context",
  "tone": "calm|urgent|empathetic|clear|reassuring",
  "language": "{language}",
  "speaking_rate": "slow|normal|fast",
  "emphasis_words": ["words to emphasize"],
  "pause_after": true/false,
  "english_fallback": "English version if language generation fails"
}}

Rules:
- Crisis/panic: very short sentences, calm reassuring tone, slow rate
- Normal inquiry: clear, professional, medium rate
- Always end with a clear question or instruction
- For Kannada/Hindi: use simple, widely understood vocabulary"""

    return _call_gemini(prompt)


# ══════════════════════════════════════════════════════════════════════════════
# 9. DIALECT NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def normalize_dialect(text: str, language: str) -> dict:
    """
    Normalize dialect/slang to standard form and interpret cultural phrases.

    Returns:
    {
      "normalized": "...",
      "cultural_interpretations": [...],
      "slang_resolved": {...}
    }
    """
    prompt = f"""You are a linguistic expert in Indian languages and dialects.
Language: {language}
Text: "{text}"

Normalize dialect/slang and interpret cultural phrases. Return JSON:
{{
  "normalized": "standardized version of the text",
  "cultural_interpretations": [
    {{"phrase": "original phrase", "standard_meaning": "what it actually means", "context": "cultural context"}}
  ],
  "slang_resolved": {{"slang_word": "standard_meaning"}},
  "regional_dialect": "identified dialect if any",
  "formality_level": "formal|informal|distressed"
}}

Examples of cultural phrases to watch for:
- "he is not well" → may mean distress/danger, not illness
- "something happened" → may be euphemism for violence/assault
- Hesitation or incomplete sentences → possible fear/trauma"""

    return _call_gemini(prompt)
