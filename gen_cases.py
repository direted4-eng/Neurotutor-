#!/usr/bin/env python3
"""Generate clinical cases from the textbook RAG for under-covered domains.

The seed bank shipped with 8 cases over 5 domains, so case drills kept
recycling the same vignettes and 9 domains had no case practice at all.
This tops every clinical domain up to TARGET_PER_DOMAIN: pick a curriculum
concept from the domain, pull textbook context from the Qdrant RAG, have the
LLM write one vignette with an OSCE-style rubric, validate that the
presentation doesn't leak the diagnosis, and store it with its sources.

Usage:
    venv/bin/python gen_cases.py [--target 3] [--domains vascular,spine]
                                 [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys

from neurotutor.agent.tools import _parse_json_loose
from neurotutor.db.store import connect
from neurotutor.llm.minimax import MiniMaxClient, extract_text
from neurotutor.rag import retriever

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("gen_cases")

# Domains where case-based vignettes make sense. anatomy/professional are
# covered by recall drills, approaches by surgical_steps scenarios.
CASE_DOMAINS = [
    "vascular", "oncology", "spine", "trauma", "hydrocephalus",
    "functional", "pediatric", "peripheral_nerve", "infection",
    "neurocritical", "radiology",
]

# Scenario-skill concepts (auto-created by drills) aren't case seeds.
_SKILL_SLUG_PREFIXES = ("surgical_steps_", "emergency_", "crisis_")


def ensure_sources_column() -> None:
    with connect() as conn:
        cols = [r[1] for r in conn.execute(
            "PRAGMA table_info(cases)").fetchall()]
        if "sources" not in cols:
            conn.execute("ALTER TABLE cases ADD COLUMN sources TEXT")


def domain_counts() -> dict[str, dict]:
    with connect() as conn:
        rows = conn.execute(
            """SELECT d.code, d.id, d.title,
                      (SELECT COUNT(*) FROM cases WHERE domain_id=d.id) AS n
               FROM domains d""").fetchall()
    return {r["code"]: dict(r) for r in rows}


def pick_concept(domain_code: str, used_ids: set[int]) -> dict | None:
    with connect() as conn:
        rows = conn.execute(
            """SELECT c.id, c.name, c.summary FROM concepts c
               JOIN domains d ON d.id = c.domain_id
               WHERE d.code=?""", (domain_code,)).fetchall()
    pool = [dict(r) for r in rows
            if r["id"] not in used_ids
            and not any(str(r["name"]).lower().startswith(p.rstrip("_"))
                        for p in _SKILL_SLUG_PREFIXES)]
    return random.choice(pool) if pool else None


def existing_titles() -> set[str]:
    with connect() as conn:
        return {r["title"].lower()
                for r in conn.execute("SELECT title FROM cases").fetchall()}


def _leaks_diagnosis(title: str, presentation: str) -> bool:
    """True if a distinctive word of the diagnosis shows up in the vignette.

    Russian inflection: compare on the first 6 letters of each long word
    («глиобластома» → «глиобл» catches «глиобластомы» too).
    """
    p = presentation.lower()
    for w in title.lower().split():
        w = w.strip("(),.;:")
        if len(w) >= 6 and w[:6] in p:
            return True
    return False


def generate_case(domain_title: str, concept: dict,
                  snippets: list[dict], avoid: set[str]) -> dict | None:
    context = "\n".join(f"- {s['title']} {s['ref']}: {s['text'][:400]}"
                        for s in snippets) or "(контекст не найден)"
    sys_prompt = (
        "Ты — нейрохирург-методист, готовишь учебные клинические кейсы для "
        "ординатора. Создай РОВНО ОДИН реалистичный кейс по заданной теме, "
        "опираясь на контекст из учебников. Требования к presentation: "
        "5–10 предложений; возраст, пол, жалобы, анамнез, неврологический "
        "статус, базовые витальные данные; НЕ называй диагноз и не используй "
        "специфические шкалы, выдающие диагноз (Hunt-Hess, Spetzler-Martin, "
        "WHO grade и т.п.) — ординатор должен дойти до диагноза сам. "
        "Верни ТОЛЬКО сырой JSON без markdown:\n"
        '{"title": "точный клинический диагноз",'
        ' "presentation": "клиническая картина без диагноза",'
        ' "difficulty": 1-5,'
        ' "workup": {"обследование": "ожидаемый результат", ...},'
        ' "differential": ["..."],'
        ' "plan": ["шаг тактики", ...],'
        ' "complications": ["..."],'
        ' "rubric": {"criteria": ["критерий оценки ответа", ...]}}'
    )
    user_prompt = (
        f"Раздел программы: {domain_title}.\n"
        f"Тема кейса: {concept['name']}.\n"
        f"Описание темы: {concept.get('summary') or '—'}\n"
        f"Уже есть кейсы (НЕ повторяй их диагнозы): {sorted(avoid)}\n\n"
        f"Контекст из учебников:\n{context}"
    )
    client = MiniMaxClient()
    try:
        resp = client.chat(
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": user_prompt}],
            temperature=0.7,
        )
        text = extract_text(resp)
    finally:
        client.close()

    case = _parse_json_loose(text)
    if not case:
        log.warning("unparseable case JSON for %s", concept["name"])
        return None
    title = (case.get("title") or "").strip()
    pres = (case.get("presentation") or "").strip()
    crit = ((case.get("rubric") or {}).get("criteria")
            if isinstance(case.get("rubric"), dict) else case.get("rubric"))
    if not title or len(pres) < 200 or not crit:
        log.warning("incomplete case for %s (title=%r, pres=%d chars)",
                    concept["name"], title, len(pres))
        return None
    if title.lower() in avoid:
        log.warning("duplicate diagnosis %r — skipping", title)
        return None
    if _leaks_diagnosis(title, pres):
        log.warning("presentation leaks diagnosis %r — skipping", title)
        return None
    case["rubric"] = {"criteria": crit}
    return case


def store_case(case: dict, domain_id: int, concept_id: int,
               snippets: list[dict]) -> int:
    sources = [{"title": s["title"], "ref": s["ref"]} for s in snippets[:3]]
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO cases(title, domain_id, difficulty, presentation,
                                 workup, differential, plan, complications,
                                 rubric, concept_ids, sources)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (case["title"], domain_id,
             int(case.get("difficulty") or 3),
             case["presentation"],
             json.dumps(case.get("workup") or {}, ensure_ascii=False),
             json.dumps(case.get("differential") or [], ensure_ascii=False),
             json.dumps(case.get("plan") or [], ensure_ascii=False),
             json.dumps(case.get("complications") or [], ensure_ascii=False),
             json.dumps(case["rubric"], ensure_ascii=False),
             json.dumps([concept_id]),
             json.dumps(sources, ensure_ascii=False)))
        return cur.lastrowid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=3,
                    help="cases per domain to top up to")
    ap.add_argument("--domains", default=",".join(CASE_DOMAINS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ensure_sources_column()
    counts = domain_counts()
    avoid = existing_titles()
    made = 0

    for code in args.domains.split(","):
        code = code.strip()
        info = counts.get(code)
        if not info:
            log.warning("unknown domain %s", code)
            continue
        need = args.target - info["n"]
        used: set[int] = set()
        attempts = 0
        while need > 0 and attempts < args.target * 3:
            attempts += 1
            concept = pick_concept(code, used)
            if not concept:
                break
            used.add(concept["id"])
            snippets = retriever.search(concept["name"], k=4)
            case = generate_case(info["title"], concept, snippets, avoid)
            if not case:
                continue
            if args.dry_run:
                log.info("[dry-run] %s: %s", code, case["title"])
            else:
                cid = store_case(case, info["id"], concept["id"], snippets)
                log.info("✓ %s: #%d %s (по теме «%s»)",
                         code, cid, case["title"], concept["name"])
            avoid.add(case["title"].lower())
            need -= 1
            made += 1

    log.info("done: %d case(s) generated", made)
    return 0


if __name__ == "__main__":
    sys.exit(main())
