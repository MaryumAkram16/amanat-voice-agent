import json
import sqlite3
import time
from pathlib import Path

# Railway wipes the container's own filesystem on every redeploy. If a persistent
# volume is mounted at /data (see DEPLOY.md), use it so visit history survives
# deploys; otherwise fall back to a local file next to this script for local dev.
_default_path = "/data/amanat.db" if Path("/data").exists() else str(Path(__file__).parent / "amanat.db")
import os
DB_PATH = Path(os.environ.get("DB_PATH", _default_path))


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recorded_at REAL NOT NULL,
            patient_name TEXT,
            symptoms TEXT,
            vitals_temp TEXT,
            vitals_bp TEXT,
            notes TEXT,
            escalate INTEGER NOT NULL,
            reason TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_visit(extracted: dict, triage: dict):
    conn = _connect()
    conn.execute(
        """INSERT INTO visits
           (recorded_at, patient_name, symptoms, vitals_temp, vitals_bp, notes, escalate, reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            time.time(),
            extracted.get("patient_name"),
            json.dumps(extracted.get("symptoms", []), ensure_ascii=False),
            (extracted.get("vitals") or {}).get("temp"),
            (extracted.get("vitals") or {}).get("bp"),
            extracted.get("notes", ""),
            1 if triage.get("escalate") else 0,
            triage.get("reason", ""),
        ),
    )
    conn.commit()
    conn.close()


def get_visits(limit: int = 200):
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM visits ORDER BY recorded_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats():
    conn = _connect()
    total = conn.execute("SELECT COUNT(*) AS c FROM visits").fetchone()["c"]
    escalated = conn.execute(
        "SELECT COUNT(*) AS c FROM visits WHERE escalate = 1"
    ).fetchone()["c"]
    conn.close()
    return {"total": total, "escalated": escalated}


init_db()
