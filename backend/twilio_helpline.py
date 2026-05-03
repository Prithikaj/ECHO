"""
ECHO - Twilio WhatsApp Helpline Integration

Sends WhatsApp messages to emergency helpline numbers when a crisis is detected.
Messages only — no voice calls.

Free tier: ~$15.50 credit, ~3000 WhatsApp messages, no auto-billing ever.

Setup:
1. pip install twilio
2. Create account at https://www.twilio.com/try-twilio (free, no credit card needed)
3. Enable WhatsApp Sandbox:
   - Twilio Console → Messaging → Try it out → Send a WhatsApp message
   - Note the sandbox number (e.g. +14155238886) and join code
   - Each recipient WhatsApps "join <code>" to that number once
4. Add to .env:
   TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   TWILIO_AUTH_TOKEN=your_auth_token
   TWILIO_FROM_NUMBER=whatsapp:+14155238886
   HELPLINE_POLICE=whatsapp:+91XXXXXXXXXX
   HELPLINE_AMBULANCE=whatsapp:+91XXXXXXXXXX
   HELPLINE_FIRE=whatsapp:+91XXXXXXXXXX
   HELPLINE_WOMEN=whatsapp:+91XXXXXXXXXX
"""

import os
import datetime
from dotenv import load_dotenv
from utils import get_logger

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
logger = get_logger("twilio_helpline")

# ── Helpline number mapping ────────────────────────────────────────────────────
HELPLINE_MAP = {
    "police":         os.getenv("HELPLINE_POLICE", ""),
    "ambulance":      os.getenv("HELPLINE_AMBULANCE", ""),
    "fire":           os.getenv("HELPLINE_FIRE", ""),
    "women_helpline": os.getenv("HELPLINE_WOMEN", ""),
    "child_helpline": os.getenv("HELPLINE_WOMEN", ""),
    "human_agent":    os.getenv("HELPLINE_POLICE", ""),
}

# ── Emoji map for crisis types ─────────────────────────────────────────────────
CRISIS_EMOJI = {
    "life_threat":    "🆘",
    "violence":       "🚨",
    "medical":        "🏥",
    "fire":           "🔥",
    "missing_person": "🔍",
    "harassment":     "⚠️",
    "none":           "📢",
}

# ── Suggested responses per crisis type ───────────────────────────────────────
RESPONSE_GUIDE = {
    "life_threat":    "Dispatch nearest unit immediately. Stay on line with caller.",
    "violence":       "Send police patrol. Do not alert suspect. Keep caller calm.",
    "medical":        "Dispatch ambulance. Ask caller about consciousness and breathing.",
    "fire":           "Alert fire brigade. Evacuate nearby residents. Check for injuries.",
    "missing_person": "Get physical description and last known location. Issue alert.",
    "harassment":     "Log complaint. Offer safe house if needed. Send patrol.",
    "none":           "Review call recording and assess situation.",
}


def send_whatsapp_alert(
    escalation_path: str,
    crisis_type: str,
    location: str,
    transcript: str,
    session_id: str,
    emotion: str = "unknown",
    urgency_score: float = 0.0,
    entities: dict = None,
) -> dict:
    """
    Send a WhatsApp alert message to the appropriate helpline.

    Args:
        escalation_path: 'police' | 'ambulance' | 'fire' | 'women_helpline'
        crisis_type:     type of crisis detected by AI
        location:        extracted location from entities
        transcript:      last transcript snippet from caller
        session_id:      current session ID for reference
        emotion:         detected emotion of caller
        urgency_score:   0.0-1.0 urgency score
        entities:        full entities dict from AI

    Returns:
        {"status": "sent", "to": number, "sid": message_sid}
        or {"status": "not_configured"} if Twilio not set up
        or {"status": "error", "reason": ...}
    """
    sid         = os.getenv("TWILIO_ACCOUNT_SID", "")
    token       = os.getenv("TWILIO_AUTH_TOKEN", "")
    from_number = os.getenv("TWILIO_FROM_NUMBER", "")
    to_number   = HELPLINE_MAP.get(escalation_path, "")

    if not all([sid, token, from_number, to_number]):
        logger.warning(
            "Twilio not configured — WhatsApp alert not sent. "
            "Add TWILIO_* and HELPLINE_* keys to .env"
        )
        return {"status": "not_configured", "reason": "Missing Twilio credentials in .env"}

    entities = entities or {}
    now = datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")
    emoji = CRISIS_EMOJI.get(crisis_type, "🚨")
    response_guide = RESPONSE_GUIDE.get(crisis_type, "Assess situation immediately.")
    urgency_pct = int(urgency_score * 100)

    # ── Build the WhatsApp message ─────────────────────────────────────────────
    lines = [
        f"{emoji} *ECHO EMERGENCY ALERT*",
        f"━━━━━━━━━━━━━━━━━━━━━━",
        f"🕐 *Time:* {now}",
        f"🆔 *Session:* `{session_id}`",
        f"",
        f"🔴 *Crisis Type:* {crisis_type.replace('_', ' ').upper()}",
        f"📊 *Urgency:* {urgency_pct}%",
        f"😰 *Caller Emotion:* {emotion.capitalize()}",
        f"",
        f"📍 *Location:* {location or 'Not specified — ask caller'}",
    ]

    # Add extra entities if available
    if entities.get("area_landmark"):
        lines.append(f"🏛 *Landmark:* {entities['area_landmark']}")
    if entities.get("people_involved") and entities["people_involved"]:
        lines.append(f"👥 *People involved:* {', '.join(entities['people_involved'])}")
    if entities.get("incident_type"):
        lines.append(f"📋 *Incident:* {entities['incident_type']}")

    lines += [
        f"",
        f"🗣 *Caller said:*",
        f'_"{transcript[:150]}{"..." if len(transcript) > 150 else ""}"_',
        f"",
        f"━━━━━━━━━━━━━━━━━━━━━━",
        f"✅ *Recommended Response:*",
        f"{response_guide}",
        f"",
        f"⚡ *Escalated to:* {escalation_path.replace('_', ' ').upper()}",
        f"━━━━━━━━━━━━━━━━━━━━━━",
        f"_Sent by ECHO AI Emergency Intelligence System_",
    ]

    body = "\n".join(lines)

    try:
        from twilio.rest import Client
        client = Client(sid, token)

        message = client.messages.create(
            body=body,
            from_=from_number,
            to=to_number,
        )
        logger.info(
            f"WhatsApp alert sent to {escalation_path} ({to_number}): {message.sid}"
        )
        return {
            "status": "sent",
            "to": to_number,
            "sid": message.sid,
            "escalation_path": escalation_path,
        }

    except Exception as e:
        logger.error(f"Twilio WhatsApp failed: {e}")
        return {"status": "error", "reason": str(e)}


def send_whatsapp_to_all_relevant(
    crisis_type: str,
    escalation_path: str,
    location: str,
    transcript: str,
    session_id: str,
    emotion: str = "unknown",
    urgency_score: float = 0.0,
    entities: dict = None,
) -> list:
    """
    Send alerts to multiple helplines if the crisis warrants it.
    E.g. a life_threat sends to both police AND ambulance.
    """
    # Determine which helplines to notify
    notify = [escalation_path]

    # Cross-notify logic
    if crisis_type == "life_threat":
        notify = ["police", "ambulance"]
    elif crisis_type == "violence":
        notify = ["police"]
    elif crisis_type == "medical":
        notify = ["ambulance"]
    elif crisis_type == "fire":
        notify = ["fire", "ambulance"]

    results = []
    for path in notify:
        result = send_whatsapp_alert(
            escalation_path=path,
            crisis_type=crisis_type,
            location=location,
            transcript=transcript,
            session_id=session_id,
            emotion=emotion,
            urgency_score=urgency_score,
            entities=entities,
        )
        results.append(result)

    return results
