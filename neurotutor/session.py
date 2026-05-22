"""Session loop. Picks a role per mode and runs the orchestrator turn-by-turn.

Modes:
  diagnostic — 30 adaptive questions, fills mastery from scratch
  review     — pulls due (concept × bloom) from FSRS scheduler
  new        — introduces 1–2 new concepts per session
  case       — case-based learning, 6 Harvard steps
  osce       — timed OSCE-style station, examiner role
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Iterator

from .agent.orchestrator import run_turn
from .agent.persona import DEFAULT_PERSONA
from .db.store import connect
from .fsrs.scheduler import due_today, pick_new

log = logging.getLogger(__name__)


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


def plan_new(limit: int = 2, domain: str | None = None) -> list[dict]:
    return pick_new(limit=limit, domain=domain)


def _inject_new_concept_hint(user_message: str, domain: str | None = None) -> str:
    """For mode=new: prepend the next unseen concept so the agent has a target."""
    nxt = pick_new(limit=1, domain=domain)
    if not nxt:
        return user_message
    c = nxt[0]
    hint = (f"[Новый концепт для введения: {c['name']} "
            f"(slug={c['slug']}, domain={c['domain']})]")
    return f"{hint}\n\n{user_message}"


def _persist_responses(session_id: int, role: str, trace: list[dict]) -> int:
    """Scan a turn's trace and write one row to `responses` per graded answer.

    Pairs each grade_answer call with the subsequent schedule_fsrs call for the
    same concept (if any) so fsrs_rating lands on the same row.
    """
    graded = [s for s in trace if s["tool"] == "grade_answer"]
    if not graded:
        return 0

    # Mastery is tracked per (concept_id, bloom_level); key the rating map
    # by the same tuple, otherwise a single turn that exercises one concept
    # across multiple Bloom levels will overwrite earlier ratings.
    fsrs_by_pair: dict[tuple[int, int], int] = {}
    for s in trace:
        if s["tool"] != "schedule_fsrs":
            continue
        args = s.get("args") or {}
        cid = args.get("concept_id")
        bl = args.get("bloom_level")
        rating = args.get("rating")
        if cid is not None and bl is not None and rating is not None:
            fsrs_by_pair[(cid, bl)] = rating

    written = 0
    with connect() as conn:
        for s in graded:
            args = s.get("args") or {}
            result = s.get("result") or {}
            concept_id = args.get("concept_id") or result.get("concept_id")
            bloom = args.get("bloom_level") or result.get("bloom_level") or 1
            grade = result.get("score")
            breakdown = result.get("breakdown")
            try:
                conn.execute(
                    """INSERT INTO responses(session_id, concept_id, bloom_level,
                                              role, prompt, answer, grade,
                                              fsrs_rating, rubric_breakdown)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        session_id,
                        concept_id,
                        bloom,
                        role,
                        args.get("prompt", ""),
                        args.get("answer", ""),
                        grade,
                        fsrs_by_pair.get((concept_id, bloom)),
                        json.dumps(breakdown, ensure_ascii=False)
                            if breakdown is not None else None,
                    ),
                )
                written += 1
            except Exception:
                log.exception("failed to persist response for concept_id=%s",
                              concept_id)
        conn.commit()
    return written


def turn(mode: str, user_message: str,
         history: list[dict] | None = None,
         persona: str = DEFAULT_PERSONA,
         session_id: int | None = None) -> dict:
    role = ROLE_BY_MODE.get(mode, "anatomist")
    if mode == "new":
        user_message = _inject_new_concept_hint(user_message)
    result = run_turn(role, user_message, history=history, persona=persona)
    if session_id is not None:
        _persist_responses(session_id, role, result.get("trace", []))
    return result


def interactive(mode: str) -> Iterator[dict]:
    """Generator: yields agent replies; send user input via .send().

    One yield per round-trip — the caller does `next(gen)` once to receive
    the initial `awaiting` payload, then `.send(msg)` to deliver each user
    message and receive the corresponding reply.
    """
    history: list[dict] = []
    sid = start(mode)
    try:
        out: dict = {"session_id": sid, "awaiting": "user"}
        while True:
            user = (yield out)
            if user in (None, "/quit"):
                return
            result = turn(mode, user, history=history, session_id=sid)
            history = result["messages"][1:]  # drop system; agent re-adds it
            out = {"session_id": sid, "reply": result["reply"],
                   "trace": result["trace"]}
    finally:
        end(sid)
