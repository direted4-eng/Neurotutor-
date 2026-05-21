"""Session loop. Picks a role per mode and runs the orchestrator turn-by-turn.

Modes:
  diagnostic — 30 adaptive questions, fills mastery from scratch
  review     — pulls due (concept × bloom) from FSRS scheduler
  new        — introduces 1–2 new concepts per session
  case       — case-based learning, 6 Harvard steps
  osce       — timed OSCE-style station, examiner role
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterator

from .agent.orchestrator import run_turn
from .agent.persona import DEFAULT_PERSONA
from .db.store import connect
from .fsrs.scheduler import due_today


ROLE_BY_MODE = {
    "diagnostic": "diagnostician",
    "review": "anatomist",
    "new": "anatomist",
    "case": "clinician",
    "osce": "examiner",
    "imaging": "radiologist",
}


def start(mode: str, *, notes: str = "") -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO sessions(mode, notes) VALUES (?,?)", (mode, notes)
        )
        conn.commit()
        return cur.lastrowid


def end(session_id: int) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE sessions SET ended_at=? WHERE id=?",
            (datetime.utcnow().isoformat(), session_id),
        )
        conn.commit()


def plan_review() -> list[dict]:
    return due_today(limit=20)


def turn(mode: str, user_message: str,
         history: list[dict] | None = None,
         persona: str = DEFAULT_PERSONA) -> dict:
    role = ROLE_BY_MODE.get(mode, "anatomist")
    return run_turn(role, user_message, history=history, persona=persona)


def interactive(mode: str) -> Iterator[dict]:
    """Generator: yields agent replies; send user input via .send()."""
    history: list[dict] = []
    sid = start(mode)
    try:
        while True:
            user = (yield {"session_id": sid, "awaiting": "user"})
            if user in (None, "/quit"):
                return
            result = turn(mode, user, history=history)
            history = result["messages"][1:]  # drop system; agent re-adds it
            yield {"session_id": sid, "reply": result["reply"],
                   "trace": result["trace"]}
    finally:
        end(sid)
