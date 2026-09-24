import asyncio
import json
import time

from agents import (
    AMANAT_PERSONA,
    call_llm,
    parse_json,
    reply_agent,
    run_agents,
    verification_agent,
)
from web_tts import synthesize
import care
import db

MAX_FOLLOWUPS = 2          # how many times Amanat asks a question about the same visit
PENDING_TIMEOUT = 120       # seconds; a question awaiting an answer expires after this
                            # (raised from 90 — was too tight against the pipeline's current latency)
CONTINUATION_WINDOW = 45    # seconds; a visit just recorded can still receive one more sentence
                            # (raised from 20 — a slow reply could outlast the old window entirely,
                            # making a genuine continuation look like a brand-new visit. Revisit
                            # once the pipeline's per-step latency is fixed — this is a safety
                            # margin, not the real solution to a slow round trip.)
CLARIFY_TIMEOUT = 60        # seconds to wait for an answer to "same patient, or a new one?"


class ClientGone(Exception):
    """The browser disconnected mid-pipeline. Raised so the pipeline stops right away
    instead of spending more Gemini/TTS calls on a reply nobody will hear."""


def _merge_extracted(new: dict, prior: dict) -> dict:
    """Fill gaps in a fresh extraction using a known-good extraction from earlier in the same
    conversation. Needed because a continuation is built by concatenating raw transcript text
    and re-running extraction on the WHOLE thing (see process_sync) — that re-extraction can
    silently drop a detail (a name, a symptom) that was caught correctly the first time,
    especially on quieter or mumbled audio. This backfills from what's already confirmed,
    it never overwrites something the new extraction did find."""
    merged = dict(new)
    if not merged.get("patient_name") and prior.get("patient_name"):
        merged["patient_name"] = prior["patient_name"]
    prior_symptoms = prior.get("symptoms") or []
    new_symptoms = merged.get("symptoms") or []
    for s in prior_symptoms:
        if s not in new_symptoms:
            new_symptoms.append(s)
    merged["symptoms"] = new_symptoms
    prior_vitals = prior.get("vitals") or {}
    new_vitals = merged.get("vitals") or {}
    for key in ("temp", "bp"):
        if not new_vitals.get(key) and prior_vitals.get(key):
            new_vitals[key] = prior_vitals[key]
    merged["vitals"] = new_vitals
    if merged.get("visit_type", "general") == "general" and prior.get("visit_type"):
        merged["visit_type"] = prior["visit_type"]
    details = dict(prior.get("details") or {})
    details.update(merged.get("details") or {})
    merged["details"] = details
    return merged


class Session:
    """One connected browser = one Session. Mirrors orchestrator.py's process() exactly,
    except output goes over a WebSocket instead of print()/local speaker playback.
    process_sync() is called from the background STT thread (see server.py) — that's fine,
    since every call inside it (Gemini, edge-tts) is a normal blocking call."""

    def __init__(self, loop: asyncio.AbstractEventLoop, websocket):
        self.loop = loop
        self.websocket = websocket
        self._pending = None
        self._recent = None
        self._clarify = None
        self.closed = False    # set by server.py when the browser disconnects

    def _status(self, step: str):
        """Tell the browser which pipeline step is running: understanding / replying / listening."""
        if not self.closed:
            self._send(self.websocket.send_json({"type": "status", "step": step}))

    # ---- output helpers: schedule the actual send onto the asyncio loop and wait for it ----
    def _send(self, coro):
        """Run one websocket send from this background thread. Any failure means the browser
        is gone, so it's turned into ClientGone to end the pipeline quietly."""
        if self.closed:
            coro.close()
            raise ClientGone()
        try:
            asyncio.run_coroutine_threadsafe(coro, self.loop).result()
        except Exception:
            self.closed = True
            raise ClientGone()

    def _send_json(self, payload: dict):
        self._send(self.websocket.send_text(json.dumps(payload, ensure_ascii=False)))

    def _speak(self, text: str):
        """Synthesize and send audio bytes. The BROWSER decides when to unmute the mic again
        (based on actual playback duration there), so no timing coordination happens here."""
        if not text:
            return
        if self.closed:
            raise ClientGone()
        self._status("replying")
        audio_bytes = synthesize(text)
        self._send(self.websocket.send_bytes(audio_bytes))

    def _reply(self, result, extracted, triage):
        """Use the reply combined_agent already wrote (saves a whole Gemini round trip).
        Falls back to a fresh reply_agent call if there wasn't one, or if merging with an
        earlier sentence changed the patient's name since it was written. For a child's
        visit, the vaccine reminder is added after it."""
        if result.get("reply") and extracted.get("patient_name") == result["extracted"].get("patient_name"):
            reply = result["reply"]
        else:
            reply = reply_agent(extracted, triage)
        vaccines, _ = self._vaccines(extracted)
        return f"{reply} {vaccines}".strip()

    @staticmethod
    def _vaccines(extracted):
        """(Urdu sentence, English note) about vaccines due for a child's visit, else ("", "")."""
        if extracted.get("visit_type") != "child":
            return "", ""
        return care.vaccine_reminder((extracted.get("details") or {}).get("child_age_weeks"))

    def _emit(self, result, extracted, triage, visit_id):
        """Save the visit (updating it if this conversation already saved it) and show it
        in the browser. Returns the visit id."""
        visit_id, patient_id = db.save_visit(
            extracted, triage, patient_id=result.get("patient_id"), visit_id=visit_id
        )
        history = result.get("history") or []
        self._send_json({
            "type": "record",
            "extracted": extracted,
            "triage": triage,
            "visit_type": extracted.get("visit_type", "general"),
            "visit_number": len(history) + 1 if patient_id else None,
            "last_seen": care.days_ago(history[0]["recorded_at"]) if history else None,
            "vaccine_note": self._vaccines(extracted)[1],
        })
        return visit_id

    # ---- same-patient disambiguation (identical to orchestrator.py) ----
    def _same_patient(self, known_name, candidate_name):
        if not known_name or not candidate_name:
            return True
        a, b = known_name.strip().lower(), candidate_name.strip().lower()
        return a == b or a in b or b in a

    def _ask_same_or_new(self, recent_name):
        name_clause = f" ({recent_name})" if recent_name else ""
        return call_llm(
            AMANAT_PERSONA,
            "Ask ONE short, polite question in Urdu script: is what she is about to say about the "
            f"same patient as before{name_clause}, or a different, new patient? "
            "Reply with only the question.",
        )

    def _classify_same_or_new(self, recent_name, answer_text):
        prompt = """You are reading a Lady Health Worker's answer to the question "is this the same
patient as before, or a new patient?" (asked in Urdu). Her answer may be Urdu script, Roman
Urdu, Punjabi, or English, and may be short.
Return ONLY JSON: {"same": true | false | null}   (use null only if her answer truly does not
indicate either way)."""
        user = f"Previous patient's name (may be unknown): {recent_name or '(unknown)'}\nHer answer: {answer_text}"
        data = parse_json(call_llm(prompt, user, json_mode=True), {})
        return data.get("same") if isinstance(data, dict) else None

    def process_sync(self, transcript: str):
        """Handle one finished sentence from the Lady Health Worker. Identical control flow
        to orchestrator.py's process() — see that file's comments for the reasoning."""
        transcript = (transcript or "").strip()
        if not transcript:
            return
        self._status("understanding")
        try:
            self._process_sync_inner(transcript)
        finally:
            if not self.closed:
                self._status("listening")

    def _process_sync_inner(self, transcript: str):

        count, already_escalated = 0, False
        prior_extracted = None  # known-good extraction from earlier in this conversation, if any
        visit_id = None         # this visit's saved row, once there is one (updated, never duplicated)

        clarify, self._clarify = self._clarify, None
        if clarify and time.time() - clarify["time"] <= CLARIFY_TIMEOUT:
            same = self._classify_same_or_new(clarify["recent_name"], transcript)
            if same is True:
                text = f"{clarify['recent_text']} {clarify['held']} {transcript}"
                prior_extracted = clarify.get("recent_extracted")
                visit_id = clarify.get("recent_visit_id")
            elif same is False:
                text = f"{clarify['held']} {transcript}"
            else:
                text = f"{clarify['recent_text']} {clarify['held']} {transcript}"
                prior_extracted = clarify.get("recent_extracted")
                visit_id = clarify.get("recent_visit_id")

        else:
            pending, self._pending = self._pending, None
            if pending and time.time() - pending["time"] > PENDING_TIMEOUT:
                pending = None

            if pending:
                text = f"{pending['text']} {transcript}"
                count, already_escalated = pending["count"], pending["escalated"]
                prior_extracted = pending.get("extracted")
                visit_id = pending.get("visit_id")

            else:
                recent = self._recent
                if recent and time.time() - recent["time"] > CONTINUATION_WINDOW:
                    recent = None
                    self._recent = None

                if recent:
                    solo = run_agents(transcript, current_visit_id=recent.get("visit_id"))
                    if solo["outcome"] == "skip":
                        self._send_json({"type": "skip"})
                        return
                    candidate_name = solo["extracted"].get("patient_name")
                    if candidate_name:
                        if self._same_patient(recent["name"], candidate_name):
                            text = f"{recent['text']} {transcript}"
                            prior_extracted = recent.get("extracted")
                            visit_id = recent.get("visit_id")
                        else:
                            self._recent = None
                            text = transcript
                    else:
                        self._speak(self._ask_same_or_new(recent["name"]))
                        self._clarify = {
                            "recent_text": recent["text"],
                            "recent_name": recent["name"],
                            "recent_extracted": recent.get("extracted"),
                            "recent_visit_id": recent.get("visit_id"),
                            "held": transcript,
                            "time": time.time(),
                        }
                        return
                else:
                    text = transcript

        result = run_agents(text, current_visit_id=visit_id)
        outcome = result["outcome"]

        if outcome == "skip":
            self._send_json({"type": "skip"})
            return

        extracted = result["extracted"]
        if prior_extracted:
            extracted = _merge_extracted(extracted, prior_extracted)
        triage = result["triage"]
        name = extracted.get("patient_name")

        if outcome == "record":
            visit_id = self._emit(result, extracted, triage, visit_id)
            self._speak(self._reply(result, extracted, triage))
            self._recent = {"text": text, "name": name, "extracted": extracted, "visit_id": visit_id, "time": time.time()}
            return

        if outcome == "escalate":
            visit_id = self._emit(result, extracted, triage, visit_id)
            missing = result["verification"]["missing"]
            if already_escalated:
                if not missing:
                    self._speak(call_llm(
                        AMANAT_PERSONA,
                        "Confirm briefly, in Urdu script, that the patient's details have been "
                        "added to the urgent case.",
                    ))
                    self._recent = {"text": text, "name": name, "extracted": extracted, "visit_id": visit_id, "time": time.time()}
                    return
            else:
                self._speak(self._reply(result, extracted, triage))  # danger first, questions after
                if not missing:
                    self._recent = {"text": text, "name": name, "extracted": extracted, "visit_id": visit_id, "time": time.time()}
                    return
            # Re-check against the merged extraction: an earlier sentence may have filled the gap.
            question = verification_agent(
                extracted, result["verification"].get("follow_up_question", "")
            ).get("follow_up_question")
            if question and count < MAX_FOLLOWUPS:
                self._speak(question)
                self._pending = {"text": text, "count": count + 1, "escalated": True, "extracted": extracted, "visit_id": visit_id, "time": time.time()}
            else:
                self._recent = {"text": text, "name": name, "extracted": extracted, "visit_id": visit_id, "time": time.time()}
            return

        if outcome == "ask":
            if count < MAX_FOLLOWUPS:
                self._speak(result["verification"]["follow_up_question"])
                self._pending = {"text": text, "count": count + 1, "escalated": False, "extracted": extracted, "visit_id": visit_id, "time": time.time()}
            else:
                visit_id = self._emit(result, extracted, triage, visit_id)
                self._speak(self._reply(result, extracted, triage))
                self._recent = {"text": text, "name": name, "extracted": extracted, "visit_id": visit_id, "time": time.time()}
