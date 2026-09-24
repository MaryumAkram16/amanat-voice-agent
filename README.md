# Amanat — Voice Assistant for Lady Health Workers

*Built for the AssemblyAI Voice Agent Hackathon (lablab.ai)*

**🔴 Live demo:** [maryumakram16.github.io/amanat-voice-agent](https://maryumakram16.github.io/amanat-voice-agent/)

Amanat lets a Lady Health Worker (LHW) record a home-visit report by speaking
instead of writing it down.

- **Speaks naturally** — Urdu, Punjabi, English, or any mix of the three, in the same sentence
- **Listens in real time** and pulls out the patient's name and health details
- **Checks every visit for danger signs**
- **Replies out loud**, in Urdu, with a short spoken confirmation
- **Never guesses** — asks a follow-up question when something important is missing, instead of filing an incomplete or wrong record

---

## The problem

Pakistan's Lady Health Worker Programme is one of the largest community
health systems in the world: roughly 90,000+ LHWs, each responsible for
about a thousand people, reaching close to 60% of the country's population —
mostly rural, mostly underserved.

Every one of those visits is still recorded the same way it has been for
decades: on paper, standing up, often in poor light, sometimes hours after
the visit when details have already blurred. That creates three concrete
problems:

1. **Nothing is structured.** A paper note can't be searched, aggregated, or
   used to spot a pattern across visits.
2. **Nothing is reviewed in real time.** A dangerous symptom written on paper
   sits there until someone reads it later — there's no moment where it gets
   flagged.
3. **Typing isn't realistic for the job.** An LHW's hands are often full —
   with a child, a stethoscope, or her own notebook and pen. A form-based
   digital app doesn't actually solve the problem; it just moves the same
   friction onto a screen.

## The solution

Amanat replaces the notebook with a voice conversation. The LHW speaks the
visit exactly as she would say it out loud to a colleague. Amanat:

- **Listens** in real time, handling Urdu, Punjabi, and English mixed freely
  in the same sentence — because that's how people actually speak, not how
  a form expects them to.
- **Extracts** the patient's name, symptoms, vitals, and notes — without
  guessing. If something essential is missing, it asks one short, specific
  follow-up question instead of filing an incomplete record.
- **Screens every visit** against a danger-sign list. A serious case is
  flagged immediately — before anything else happens in that turn.
- **Replies out loud**, in Urdu, with a short calm confirmation, so the LHW
  knows the visit is recorded without breaking her stride or looking at a
  screen.
- **Remembers the conversation.** A follow-up sentence about the same
  patient gets merged into one record, not filed as a second, disconnected
  entry — including the case where she starts describing a new patient
  without saying whose visit is being continued, where Amanat asks directly.

Everything gets written to a shared, live dashboard — the read side of the
system a supervisor would actually use, not just a technical demo.

---

## How it works — the pipeline

```
Browser (mic)
   │  raw audio, captured by an AudioWorklet, downsampled to 16kHz PCM16
   ▼
WebSocket ──────────────────────────────────────────────────────────────┐
   │                                                                     │
   ▼                                                                     │
AssemblyAI Realtime STT (Whisper-rt model)                               │
   │  partial transcripts stream back live (shown on screen as she talks)│
   │  finished "turns" get filtered for STT hallucination artifacts      │
   ▼                                                                     │
Multi-agent pipeline (session.py)                                        │
   │                                                                     │
   ├─ 1. Router + Extraction + Triage — one combined Gemini call         │
   │     decides the outcome: record / ask / escalate / skip             │
   │                                                                     │
   ├─ 2. Verification — if something critical is missing, composes a     │
   │     short follow-up question instead of filing an incomplete record │
   │                                                                     │
   ├─ 3. Conversation memory — tracks whether this sentence continues    │
   │     the last visit, answers a pending question, or starts fresh;    │
   │     merges structured data across turns rather than re-guessing     │
   │     everything from concatenated raw text                           │
   │                                                                     │
   └─ 4. Reply — composes a short, calm, Urdu confirmation in Amanat's   │
        persona (never robotic, steady rather than alarming on a         │
        red-flag case)                                                  │
   │                                                                     │
   ▼                                                                     │
edge-tts (Urdu neural voice) → spoken reply, sent back as audio bytes ───┘
   │
   ▼
SQLite (persisted to a Railway volume) ── every recorded/escalated visit
   │
   ▼
Admin dashboard — live table, charts, and an instant escalation alert
```

### Why these specific technical choices

- **AssemblyAI's Whisper-rt streaming model, not the flagship real-time model**
  - The flagship real-time model natively handles only 18 languages — Urdu and Punjabi aren't among them
  - Whisper-rt covers 99+ languages, including both
  - Without this choice, the project simply couldn't work in the languages its actual users speak

- **Router, extraction, and triage combined into one Gemini call, not three separate ones**
  - Deliberate trade-off to fit inside a free API tier's rate limits
  - Cost: can't show four genuinely distinct "thinking" stages in the UI
  - The pipeline status indicator shows three honest stages (Listening → Understanding → Replying) rather than fabricating a fourth step that doesn't correspond to anything actually happening

- **Verification never guesses**
  - The extraction prompt explicitly returns `null` for anything unclear, rather than inferring a plausible-sounding value
  - The follow-up-question flow exists specifically to keep a wrong guess out of a health record

- **Structured conversation state, not just concatenated text**
  - Early versions reconstructed context by pasting raw transcript fragments together and re-extracting everything from scratch
  - That could silently drop a detail (a name, a symptom) already caught correctly earlier in the conversation, especially on quieter audio
  - The current version carries the actual structured extraction forward between turns
  - A backfill step fills any gap in a fresh extraction using what's already confirmed — it never overwrites a new, correct value

---

## What a live exchange looks like

**LHW:** *"Fatima ke ghar gayi thi, usko bukhar hai aur khaansi bhi, do din
se."* — "Went to Fatima's house — she has fever and a cough, for two days."

**Amanat:** *"Fatima ka visit record ho gaya hai."* — "Fatima's visit has
been recorded."

If a danger sign is present instead — high fever with difficulty breathing,
for example — Amanat still confirms the record, but the reply is calmer and
more direct, the visit is flagged on the dashboard immediately, and the
frontend plays a distinct alert tone with a red banner.

---

## Danger-sign list — status: draft, pending clinical review

The current red-flag list lives in `agents.py` (`RED_FLAGS`). It combines an
initial team draft with two additions checked against WHO's IMCI (Integrated
Management of Childhood Illness) general danger signs and a documented gap
in Pakistan's LHW training around recognizing pre-eclampsia/eclampsia:

- Very high fever (≈103°F / 39.5°C or higher)
- Difficulty breathing, fast/labored breathing, blue or bluish lips
- Severe or heavy bleeding (including postpartum hemorrhage)
- Unconscious, unresponsive, fainting, or convulsions/seizures
- Low blood pressure (≈90/60 or lower) together with dizziness or weakness
- Unable to drink or breastfeed at all (a core WHO IMCI danger sign)
- High blood pressure in pregnancy with severe headache, swelling, or
  vision changes (possible pre-eclampsia/eclampsia)

**This list has not been reviewed by anyone with clinical training.** It's
an intentionally honest limitation, not an oversight — a wrong threshold
here has real consequences, and that call shouldn't be made by an LLM or by
non-clinicians alone.

---

## Repository structure

```
amanat-web/
├── agents.py              # Combined router/extraction/triage agent, verification, reply, persona
├── session.py             # Per-connection conversation state machine (multi-turn memory, backfill merge)
├── server.py              # FastAPI WebSocket backend: audio in, AssemblyAI, pipeline, replies out
├── db.py                  # SQLite persistence + analytics aggregation for the admin dashboard
├── web_tts.py             # Urdu text-to-speech (edge-tts), returns audio bytes for the WebSocket
├── requirements.txt
├── Procfile               # Railway start command
├── .env.example           # Required environment variables (see below)
├── DEPLOY.md              # Step-by-step Railway + GitHub Pages deployment guide
├── static/                # Minimal plain-text testing interface (served by the backend itself)
│   ├── index.html
│   ├── app.js
│   └── recorder-worklet.js
├── private/
│   └── admin.html         # Password-protected dashboard (visits table, charts, escalation alerts)
└── docs/
    └── index.html         # The actual public-facing demo page (deployed via GitHub Pages)
```

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| Speech-to-text | AssemblyAI Realtime STT (`whisper-rt`) | Only real-time model covering Urdu and Punjabi |
| Language understanding | Google Gemini (`gemini-3.5-flash-lite`, with fallback models) | Free-tier friendly, fast enough for a conversational loop |
| Text-to-speech | `edge-tts` (`ur-PK-UzmaNeural`) | Free, genuinely native Urdu-Pakistan neural voice |
| Backend | FastAPI + WebSockets, hosted on Railway | Real-time bidirectional audio/text streaming |
| Storage | SQLite, optionally persisted via a Railway volume | Zero-setup, sufficient for a hackathon's data volume |
| Frontend | Vanilla JS + Web Audio API (AudioWorklet), hosted on GitHub Pages | No framework overhead for a single interactive page |
| Admin dashboard | Chart.js (via CDN) for visit analytics | Lightweight, no build step required |

---

## Running it locally

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your own API keys
uvicorn server:app --reload
```

Open `http://localhost:8000` in Chrome (needs real browser APIs — mic access
and AudioWorklet — so this can't be tested with curl). For local testing,
temporarily point the frontend's `BACKEND_WS_URL` at
`ws://localhost:8000/ws`.

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `ASSEMBLYAI_API_KEY` | Yes | Real-time speech-to-text |
| `GEMINI_API_KEY` | Yes | Extraction, triage, verification, reply generation |
| `GEMINI_MODELS` | No | Comma-separated fallback model list; sensible default if omitted |
| `ADMIN_USER` / `ADMIN_PASS` | Yes | Protects `/admin` — this dashboard shows real patient data |
| `DB_PATH` | No | Overrides where the SQLite file lives (see `DEPLOY.md` for the Railway volume setup) |

## Deploying

Full step-by-step instructions — Railway for the backend, GitHub Pages for
the frontend, including the WebSocket URL wiring between them — are in
[`DEPLOY.md`](./DEPLOY.md).

---

## Known limitations

- **Red-flag thresholds are a draft**, as noted above — needs sign-off from
  someone with medical training before this could be trusted beyond a
  hackathon demo.
- **SQLite resets on every Railway redeploy** unless a persistent volume is
  attached at `/data` — see `DEPLOY.md`. Fine for a hackathon's lifespan,
  not a durable production database.
- **Response latency** comes from several sequential network calls in one
  turn (STT → Gemini → edge-tts). Under active investigation; the
  conversation-continuation timing windows were widened as a safety margin
  against this, but the real fix is reducing the number or cost of those
  calls.
- **STT occasionally transcribes in Hindi (Devanagari) script instead of
  Urdu (Nastaliq)** — spoken Urdu and Hindi are phonetically identical, so
  the multilingual model sometimes guesses the wrong script. Gemini
  understands both, so this appears to be cosmetic rather than a data-loss
  issue, but hasn't been exhaustively verified.
- **This is a hackathon prototype handling real patient-shaped data.** No
  authentication beyond the admin password, no encryption at rest beyond
  what Railway provides by default, and no consent flow. Testing should use
  fictional names, never real patients.

---

## Team

Built for the AssemblyAI Voice Agent Hackathon on lablab.ai.

- **Maryum Akram** — product & coordination
- **Sehrish** — voice agent (STT pipeline, agent logic, extensive voice testing)
- *(add remaining team members here)*
