import json
import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Free-tier quotas are counted PER MODEL, so we keep a list of models and fall back to the
# next one when a model has used up its daily quota (or does not exist for your account).
# Put the model IDs you see on https://aistudio.google.com/rate-limit in .env like this:
#   GEMINI_MODELS=model-one,model-two,model-three
# Models with the highest free "requests per day" are best for development, so list them first.
DEFAULT_MODELS = (
    "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.8-flash,"
    "gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash"
)
MODELS = [
    m.strip()
    for m in os.environ.get("GEMINI_MODELS", DEFAULT_MODELS).split(",")
    if m.strip()
]

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

AMANAT_PERSONA = """You are Amanat, a calm, respectful voice assistant supporting a Lady Health Worker (LHW)
during home visits. You speak in simple Urdu written in Urdu script, formal enough to show respect
(use "aap"), warm but brief -- the LHW is busy and often has her hands full. Never sound robotic or
overly cheerful. When escalating a case, sound steady and reassuring, not alarming. When asking a
follow-up question, keep it short and specific. Keep every reply under two short sentences.
Write EVERY word in Urdu script (Nastaliq), with no Latin/English letters anywhere in your reply,
even for medical or technical terms -- write those in Urdu script too (e.g. "بی پی" for BP,
"ٹمپریچر" for temperature), never spell them out in English letters. This matters because your
reply is read aloud by a text-to-speech voice, and any English-letter word in the middle of an
Urdu sentence gets mispronounced."""

# DRAFT red-flag list. STILL NOT clinically reviewed — get sign-off before relying on this.
# Original 5 lines: from the team's earlier draft.
# Two lines below added from WHO IMCI's general danger signs and a Pakistan-specific LHW
# training gap (pre-eclampsia/eclampsia) identified when comparing against WHO's checklist
# and published LHW-program research — see the team's hackathon notes for sources.
# This is still a draft: get a clinician or LHW supervisor to review before the final submission.
RED_FLAGS = """- Very high fever (about 103 F / 39.5 C or higher)
- Difficulty breathing, fast or labored breathing, blue or bluish lips
- Severe or heavy bleeding (including heavy bleeding after delivery)
- Unconscious, unresponsive, fainting, or convulsions/seizures
- Low blood pressure (about 90/60 or lower) together with dizziness or weakness
- Baby or patient unable to drink or breastfeed at all (WHO IMCI general danger sign)
- High blood pressure during pregnancy together with severe headache, swelling, or
  vision changes (possible pre-eclampsia/eclampsia -- a leading cause of maternal
  death in Pakistan, and an area LHW supervisors have flagged as under-trained)"""

VALID_INTENTS = {"new_visit", "follow_up", "emergency", "off_topic"}


# ---------------------------------------------------------------- errors
class QuotaExhausted(Exception):
    """Every configured model has used its free daily quota (or is not available)."""


class NetworkDown(Exception):
    """The internet connection to Google failed."""


# ---------------------------------------------------------------- helpers
_dead_models = set()  # models to skip for the rest of this run
_last_model = None


def _network_error(msg):
    text = msg.lower()
    return any(s in text for s in ("getaddrinfo", "connecterror", "connection", "timed out", "timeout"))


def call_llm(system_prompt, user_text, json_mode=False):
    """Send one request to Gemini.
    - per-minute limit  -> wait a little and retry
    - per-day limit     -> give up on that model, try the next one in MODELS
    - model not found   -> skip it
    - no internet       -> retry a few times, then raise NetworkDown"""
    global _last_model
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.2,
        response_mime_type="application/json" if json_mode else "text/plain",
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    for model in MODELS:
        if model in _dead_models:
            continue
        busy_tries = 0
        net_tries = 0
        while True:
            try:
                resp = client.models.generate_content(
                    model=model, contents=user_text, config=config
                )
                if model != _last_model:
                    print(f"  [using model: {model}]")
                    _last_model = model
                return (resp.text or "").strip()
            except Exception as e:
                msg = str(e)
                if "RESOURCE_EXHAUSTED" in msg:
                    if "PerDay" in msg:
                        print(f"  (daily free quota used up for {model} - trying next model)")
                        _dead_models.add(model)
                        break
                    busy_tries += 1
                    if busy_tries > 3:
                        raise
                    wait = 10 * busy_tries
                    print(f"  (per-minute limit on {model}, waiting {wait}s...)")
                    time.sleep(wait)
                    continue
                if "NOT_FOUND" in msg:
                    print(f"  (model {model} not available for your account - skipping)")
                    _dead_models.add(model)
                    break
                if "UNAVAILABLE" in msg or "503" in msg:
                    busy_tries += 1
                    if busy_tries > 2:
                        # Temporary overload, not exhausted quota - don't blacklist the model,
                        # just use a different one for THIS request; it may work again shortly.
                        # Fails over quickly (~5s) rather than burning ~30s retrying one model.
                        print(f"  ({model} still unavailable - trying next model)")
                        break
                    time.sleep(3 * busy_tries)
                    continue
                if _network_error(msg):
                    net_tries += 1
                    if net_tries > 3:
                        raise NetworkDown(msg)
                    print("  (no connection to Google, retrying in 5s...)")
                    time.sleep(5)
                    continue
                raise
    raise QuotaExhausted(
        "All models are out of free quota for today (or unavailable): " + ", ".join(MODELS)
    )


def parse_json(text, fallback):
    """Turn the model's text into a Python object, tolerating extra text around the JSON."""
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return fallback


# ---------------------------------------------------------------- extraction shape
EMPTY_EXTRACTION = {
    "patient_name": None,
    "symptoms": [],
    "vitals": {"temp": None, "bp": None},
    "notes": "",
}


def normalise_extraction(data):
    """Make sure the extraction always has the exact shape the backend expects."""
    out = json.loads(json.dumps(EMPTY_EXTRACTION))
    if not isinstance(data, dict):
        return out
    name = data.get("patient_name")
    out["patient_name"] = name.strip() if isinstance(name, str) and name.strip() else None
    symptoms = data.get("symptoms")
    if isinstance(symptoms, list):
        out["symptoms"] = [str(s).strip() for s in symptoms if str(s).strip()]
    vitals = data.get("vitals")
    if isinstance(vitals, dict):
        for key in ("temp", "bp"):
            value = vitals.get(key)
            out["vitals"][key] = str(value).strip() if value not in (None, "") else None
    notes = data.get("notes")
    out["notes"] = notes.strip() if isinstance(notes, str) else ""
    return out


EXTRACTION_RULES = """- patient_name: the name exactly as spoken, e.g. "Fatima" or "Ahmed's baby". If she corrects
  herself, use the corrected version. If no name is given (only "a house", "she", "bhabi",
  "her father"), use null.
- symptoms: short English terms, e.g. ["fever", "cough"]. Empty list if none mentioned.
- vitals.temp: the temperature as stated. If a number is given, write it as stated ("104 F"). If
  only a description is given with no number ("high", "normal", "mild fever"), write that
  description instead. If temperature is not mentioned at all, use null.
- vitals.bp: blood pressure as stated. If a number is given, write it as "systolic/diastolic"
  ("130/85"). If only a description is given with no number ("low", "high", "normal"), write that
  description instead ("low"). If blood pressure is not mentioned at all, use null.
  NEVER return null just because there was no number -- a word like "low" is still useful information.
- notes: short English note ONLY for clinical details that do not fit the other fields (weight,
  feeding, medicine given, how the patient looks, duration). NEVER write that a visit happened or
  that something was "checked" (e.g. do not write "Visited Rukhsana's house" or "Checked
  temperature"), and never repeat symptoms or vitals already in the other fields. If there is no
  clinical detail beyond the other fields, notes must be an empty string "".
If information is unclear or garbled, use null rather than guessing. Never invent details."""

INTENT_RULES = """- "new_visit": she describes visiting a patient / patient details (even if incomplete)
- "follow_up": she describes a repeat visit or progress of an earlier case
- "emergency": she says it is an emergency or urgent help is needed
- "off_topic": no patient or visit information at all (weather, travel, small talk)
If the text is garbled but mentions symptoms or a visit, do NOT use off_topic."""

TRIAGE_RULES = f"""Escalate if ANY of these red flags is present:
{RED_FLAGS}
Do NOT escalate ordinary mild illness (mild fever, cough, headache, stomach ache, weakness that
does not come with a red flag). If a red flag is clearly described, always escalate."""


# ---------------------------------------------------------------- the separate agents (guide structure)
def router_agent(transcript):
    prompt = f"""You classify a Lady Health Worker's spoken visit note.
The text may be Urdu script, Roman Urdu, Punjabi, English, or a mix, and may be noisy.
Intents:
{INTENT_RULES}
Return ONLY JSON: {{"intent": "new_visit" | "follow_up" | "emergency" | "off_topic"}}"""
    result = parse_json(call_llm(prompt, transcript, json_mode=True), {})
    if not isinstance(result, dict) or result.get("intent") not in VALID_INTENTS:
        return {"intent": "new_visit"}  # safe default: never silently drop a visit
    return result


def extraction_agent(transcript):
    prompt = f"""Extract patient visit details from this Lady Health Worker transcript
(Urdu script / Roman Urdu / Punjabi / English mixed).
Rules:
{EXTRACTION_RULES}
Return ONLY JSON: {{"patient_name": str|null, "symptoms": [str], "vitals": {{"temp": str|null, "bp": str|null}}, "notes": str}}"""
    data = parse_json(call_llm(prompt, transcript, json_mode=True), None)
    return normalise_extraction(data)


def triage_agent(transcript, extracted, intent="new_visit"):
    """Decides if the case must be escalated. Runs BEFORE verification: a dangerous case
    must never wait for a missing patient name."""
    if intent == "emergency":
        return {"escalate": True, "reason": "LHW reported an emergency"}
    prompt = f"""You are a safety checker for Lady Health Worker visit notes in Pakistan.
Decide whether this case must be escalated to a doctor/supervisor right now.
{TRIAGE_RULES}
Return ONLY JSON: {{"escalate": true | false, "reason": "<short English reason>"}}"""
    user = (
        f"Original transcript: {transcript}\n"
        f"Extracted data: {json.dumps(extracted, ensure_ascii=False)}"
    )
    data = parse_json(call_llm(prompt, user, json_mode=True), None)
    if isinstance(data, dict) and isinstance(data.get("escalate"), bool):
        return {"escalate": data["escalate"], "reason": str(data.get("reason", ""))}
    return {"escalate": True, "reason": "triage output unclear - escalating to be safe"}


# ---------------------------------------------------------------- combined agent (saves quota)
def combined_agent(transcript):
    """Router + extraction + triage in ONE Gemini call (1 request instead of 3).
    Returns None if the answer can't be understood, so the caller can fall back."""
    prompt = f"""You analyse one spoken visit note from a Lady Health Worker (LHW) in Pakistan.
The text may be Urdu script, Roman Urdu, Punjabi, English, or a mix, and may be noisy.
Do three jobs and return them together.

JOB 1 - intent. One of:
{INTENT_RULES}

JOB 2 - extracted. Rules:
{EXTRACTION_RULES}

JOB 3 - triage. Decide whether this case must be escalated to a doctor/supervisor right now.
{TRIAGE_RULES}
Also escalate when the intent is "emergency".

Return ONLY JSON in exactly this shape:
{{"intent": "new_visit" | "follow_up" | "emergency" | "off_topic",
  "extracted": {{"patient_name": str|null, "symptoms": [str], "vitals": {{"temp": str|null, "bp": str|null}}, "notes": str}},
  "triage": {{"escalate": true | false, "reason": "<short English reason>"}}}}"""
    data = parse_json(call_llm(prompt, transcript, json_mode=True), None)
    if not isinstance(data, dict) or data.get("intent") not in VALID_INTENTS:
        return None
    extracted = normalise_extraction(data.get("extracted"))
    triage_raw = data.get("triage") if isinstance(data.get("triage"), dict) else {}
    escalate = triage_raw.get("escalate") is True or data["intent"] == "emergency"
    reason = str(triage_raw.get("reason", "")) or ("LHW reported an emergency" if escalate else "")
    return {
        "intent": data["intent"],
        "extracted": extracted,
        "triage": {"escalate": escalate, "reason": reason},
    }


# ---------------------------------------------------------------- verification
def find_missing(extracted):
    missing = []
    if not extracted.get("patient_name"):
        missing.append("patient name")
    vitals = extracted.get("vitals") or {}
    has_details = (
        extracted.get("symptoms") or any(vitals.values()) or extracted.get("notes")
    )
    if not has_details:
        missing.append("symptoms or health details")
    return missing


def verification_agent(extracted):
    missing = find_missing(extracted)
    if not missing:
        return {"ok": True, "missing": []}
    known = json.dumps(extracted, ensure_ascii=False)
    question = call_llm(
        AMANAT_PERSONA,
        "Ask ONE short, respectful follow-up question in Urdu script to get ONLY the missing "
        f"information: {', '.join(missing)}.\n"
        f"What is already known so far: {known}\n"
        "Base the question strictly on what is known above. Do NOT assume or mention a baby, "
        "child, pregnancy, or any other detail that was not already stated - if nothing else is "
        "known, keep the question completely general. Reply with only the question.",
    )
    return {"ok": False, "missing": missing, "follow_up_question": question}


# ---------------------------------------------------------------- reply
def reply_agent(extracted, triage):
    name = extracted.get("patient_name") or "the patient (name not known yet)"
    if triage["escalate"]:
        task = (
            f"Confirm that the visit for {name} is recorded, and calmly say this case looks "
            "serious and she should contact her supervisor or the nearest health facility "
            f"right away. Reason: {triage['reason']}."
        )
    else:
        task = f"Confirm briefly that the visit for {name} has been recorded."
    return call_llm(
        AMANAT_PERSONA, task + " Reply with only the spoken sentences, in Urdu script."
    )


# ---------------------------------------------------------------- text-only pipeline
def run_agents(transcript, combined=True):
    """Runs the agents on TEXT (no audio).
    combined=True  -> 1 Gemini call for router+extraction+triage (saves free quota)
    combined=False -> 3 separate calls, like the original guide
    outcome is one of: "skip", "escalate", "ask", "record"."""
    analysis = combined_agent(transcript) if combined else None

    if analysis is None:  # combined answer unusable (or combined=False): use separate agents
        intent = router_agent(transcript)["intent"]
        if intent == "off_topic":
            return {"intent": intent, "outcome": "skip"}
        extracted = extraction_agent(transcript)
        triage = triage_agent(transcript, extracted, intent)
        analysis = {"intent": intent, "extracted": extracted, "triage": triage}

    intent = analysis["intent"]
    extracted = analysis["extracted"]
    triage = analysis["triage"]

    if intent == "off_topic":
        return {"intent": intent, "outcome": "skip"}

    if triage["escalate"]:
        missing = find_missing(extracted)
        verification = {"ok": not missing, "missing": missing}
        outcome = "escalate"
    else:
        verification = verification_agent(extracted)
        outcome = "record" if verification["ok"] else "ask"

    return {
        "intent": intent,
        "extracted": extracted,
        "triage": triage,
        "verification": verification,
        "outcome": outcome,
    }
