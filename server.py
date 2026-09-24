import asyncio
import os
import queue
import secrets
import threading
import traceback

from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    TerminationEvent,
    TurnEvent,
)
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles

import db
from session import Session

load_dotenv()

SAMPLE_RATE = 16000

# Same list as stt_stream.py — Whisper-family models can hallucinate these from
# silence/background noise regardless of the language actually being spoken.
HALLUCINATION_PATTERNS = (
    "thanks for watching", "thank you for watching", "please subscribe", "subtitles by",
    "subtitles created by", "amara.org", "dimatorzok", "продолжение следует",
    "субтитры сделал", "obrigado", "ça va", "gracias por ver", "merci d'avoir regard",
    "i'm proud to be relevant", "here we go", "so...", "or something", "my point is",
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ---------------------------------------------------------------- admin auth
security = HTTPBasic()


def require_admin(credentials: HTTPBasicCredentials = Depends(security)):
    user_ok = secrets.compare_digest(credentials.username, os.environ.get("ADMIN_USER", "admin"))
    pass_ok = secrets.compare_digest(credentials.password, os.environ.get("ADMIN_PASS", "changeme"))
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True


@app.get("/api/visits")
def api_visits(_auth: bool = Depends(require_admin)):
    return {"stats": db.get_stats(), "visits": db.get_visits()}


@app.get("/api/analytics")
def api_analytics(_auth: bool = Depends(require_admin)):
    return db.get_analytics()


@app.get("/admin", response_class=HTMLResponse)
def admin_page(_auth: bool = Depends(require_admin)):
    # Deliberately NOT in static/ — anything there is served unauthenticated at its own path.
    return open("private/admin.html", encoding="utf-8").read()


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()
    session = Session(loop, websocket)

    def send_json_threadsafe(payload: dict):
        """For messages that don't belong to Session (partial transcripts, connection-level
        status) but still need to reach the browser from this background STT thread."""
        asyncio.run_coroutine_threadsafe(websocket.send_json(payload), loop).result()

    session.on_status = lambda step: send_json_threadsafe({"type": "status", "step": step})

    audio_queue: "queue.Queue" = queue.Queue()
    handled_turns = set()

    def audio_generator():
        """Feeds the AssemblyAI client from whatever the browser sends over the WebSocket.
        A None sentinel (put on disconnect) ends the stream cleanly."""
        while True:
            chunk = audio_queue.get()
            if chunk is None:
                return
            yield chunk

    def on_begin(client, event: BeginEvent):
        print(f"[session] started: {event.id}")

    def on_turn(client, event: TurnEvent):
        text = (event.transcript or "").strip()
        if not event.end_of_turn:
            # Live partial transcript, shown on screen as she's still speaking.
            if text:
                send_json_threadsafe({"type": "partial", "text": text})
            return
        order = getattr(event, "turn_order", None)
        if order is not None:
            if order in handled_turns:
                return
            handled_turns.add(order)
        if not text:
            return
        low = text.lower()
        if any(p in low for p in HALLUCINATION_PATTERNS):
            print(f"[HEARD] {text}  (looks like a hallucinated filler phrase - ignored)")
            return
        print(f"[HEARD] {text}")
        send_json_threadsafe({"type": "final", "text": text})
        try:
            session.process_sync(text)  # blocking: fine, this thread's only job is this session
        except Exception:
            print("[session] pipeline error:")
            traceback.print_exc()

    def on_terminated(client, event: TerminationEvent):
        print(f"[session] done: {event.audio_duration_seconds}s processed")

    def on_error(client, error: StreamingError):
        print(f"[session] STT error: {error}")

    def run_stt():
        client = StreamingClient(
            StreamingClientOptions(api_key=os.environ["ASSEMBLYAI_API_KEY"])
        )
        client.on(StreamingEvents.Begin, on_begin)
        client.on(StreamingEvents.Turn, on_turn)
        client.on(StreamingEvents.Termination, on_terminated)
        client.on(StreamingEvents.Error, on_error)
        client.connect(
            StreamingParameters(
                sample_rate=SAMPLE_RATE, speech_model="whisper-rt", format_turns=True
            )
        )
        try:
            client.stream(audio_generator())
        finally:
            client.disconnect(terminate=True)

    stt_thread = threading.Thread(target=run_stt, daemon=True)
    stt_thread.start()

    try:
        while True:
            message = await websocket.receive_bytes()
            audio_queue.put(message)
    except WebSocketDisconnect:
        print("[session] client disconnected")
    finally:
        audio_queue.put(None)  # let the STT thread exit its generator cleanly


# Serves static/index.html, app.js, recorder-worklet.js at the root path.
# Must be mounted last — routes above take priority over static files.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
