// Change this to your deployed Railway backend's WebSocket URL before publishing to GitHub Pages.
const BACKEND_WS_URL = "wss://YOUR-RAILWAY-APP.up.railway.app/ws";

const ECHO_TAIL_MS = 600; // matches ECHO_TAIL_SECONDS in the original tts.py

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");

let ws, audioCtx, workletNode, mediaStream;
let muted = false; // true while Amanat's reply is playing, so we don't stream her own voice back

function log(text) {
  const div = document.createElement("div");
  div.textContent = text;
  logEl.prepend(div);
}

async function start() {
  statusEl.textContent = "Connecting...";
  ws = new WebSocket(BACKEND_WS_URL);
  ws.binaryType = "arraybuffer";

  ws.onopen = async () => {
    statusEl.textContent = "Connected — listening...";
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      statusEl.textContent = "Microphone permission denied.";
      return;
    }
    audioCtx = new AudioContext();
    await audioCtx.audioWorklet.addModule("recorder-worklet.js");
    const source = audioCtx.createMediaStreamSource(mediaStream);
    workletNode = new AudioWorkletNode(audioCtx, "recorder-processor", {
      processorOptions: { targetSampleRate: 16000 },
    });
    workletNode.port.onmessage = (event) => {
      if (!muted && ws.readyState === WebSocket.OPEN) {
        ws.send(event.data);
      }
    };
    source.connect(workletNode);
  };

  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      const msg = JSON.parse(event.data);
      if (msg.type === "record") {
        const escalated = msg.triage.escalate ? " — ESCALATED" : "";
        log(`Recorded: ${msg.extracted.patient_name || "(no name given)"}${escalated}`);
        log(JSON.stringify(msg, null, 2));
      } else if (msg.type === "skip") {
        log("(off-topic — nothing recorded)");
      }
      return;
    }
    // Binary message = Amanat's spoken reply (mp3 bytes)
    muted = true; // stop streaming mic audio while she's talking, so she doesn't hear herself
    const blob = new Blob([event.data], { type: "audio/mpeg" });
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.addEventListener("ended", () => {
      setTimeout(() => { muted = false; }, ECHO_TAIL_MS);
      URL.revokeObjectURL(url);
    });
    audio.play();
  };

  ws.onclose = () => {
    statusEl.textContent = "Disconnected";
    startBtn.disabled = false;
    stopBtn.disabled = true;
  };

  ws.onerror = () => {
    statusEl.textContent = "Connection error — check the backend URL.";
  };

  startBtn.disabled = true;
  stopBtn.disabled = false;
}

function stop() {
  if (workletNode) workletNode.disconnect();
  if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
  if (audioCtx) audioCtx.close();
  if (ws) ws.close();
  statusEl.textContent = "Stopped";
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);
