"""Daily drill — the closed learning loop.

Deterministic, not agent-driven: we pick due/weak (concept × Bloom) pairs,
generate one active-recall question each (grounded in the textbook RAG),
push them, and when the student answers we grade → reschedule (FSRS) →
log the response. This does not rely on the LLM remembering to call tools.

Flow:
    generate_drill(user_id, n)  -> stores N pending_questions, returns them
    (student answers via Telegram)
    answer_pending(user_id, txt) -> grades oldest open question, reschedules,
                                    persists a `responses` row, returns feedback
"""
from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone

from ..db.store import connect
from ..fsrs.scheduler import due_today, pick_new
from ..llm.minimax import MiniMaxClient, extract_text
from ..rag import retriever as rag_retriever
from .tools import _parse_json_loose, grade_answer, schedule_fsrs

log = logging.getLogger(__name__)

BLOOM_NAMES = {
    1: "запомнить", 2: "понять", 3: "применить",
    4: "анализировать", 5: "оценить", 6: "синтезировать",
}

# --- Scenario drills: proactive, RAG-grounded practice beyond flat recall. ---
# Each kind has an instruction (how to frame the task) and a pool of seed
# topics. The LLM turns a seed + textbook context into one scenario question
# with a grading rubric.
SCENARIO_KINDS = {
    "surgical_steps": {
        "label": "🔪 Ход операции",
        "instruction": (
            "Поставь задачу описать ПОШАГОВО хирургический доступ/ход операции: "
            "положение, разрез, костный этап, твёрдая мозговая оболочка, "
            "ключевые анатомические ориентиры и опасные зоны, этап закрытия. "
            "Проси перечислить этапы по порядку."),
        "seeds": [
            "Птериональный доступ", "Ретросигмовидный доступ",
            "Транссфеноидальный эндоскопический доступ", "Far lateral доступ",
            "Субфронтальный доступ", "Орбитозигоматический доступ",
            "Срединный субокципитальный доступ",
            "Межполушарный транскаллёзный доступ",
            "Декомпрессивная гемикраниэктомия",
            "Эндоскопическая третья вентрикулостомия (ETV)"],
    },
    "emergency": {
        "label": "🚨 Экстренный алгоритм",
        "instruction": (
            "Сформулируй экстренную клиническую ситуацию и попроси описать "
            "ПОШАГОВЫЙ алгоритм неотложных действий: оценка, стабилизация, "
            "диагностика, решение об операции, сроки. Жди приоритизации."),
        "seeds": [
            "Острая обструктивная гидроцефалия",
            "Расширение зрачка и вклинение (uncal herniation)",
            "Повторный разрыв аневризмы при САК",
            "Травматическая эпидуральная гематома с латерализацией",
            "Эпистатус после нейрохирургической операции",
            "Острая травма шейного отдела с нейрогенным шоком",
            "Дисфункция вентрикулоперитонеального шунта",
            "Ликворея с менингитом после операции",
            "Злокачественный инфаркт СМА",
            "Напряжённый пневмоцефалус"],
    },
    "crisis": {
        "label": "⚡ Интраоперационный кризис",
        "instruction": (
            "Опиши КРИТИЧЕСКУЮ интраоперационную ситуацию и попроси описать "
            "немедленные действия хирурга и команды по шагам: контроль, "
            "коммуникация с анестезиологом, технические манёвры, эскалация."),
        "seeds": [
            "Интраоперационный разрыв аневризмы при клипировании",
            "Острый отёк мозга ('angry brain') во время краниотомии",
            "Воздушная эмболия в положении сидя",
            "Повреждение сонной артерии при транссфеноидальном доступе",
            "Массивное кровотечение из АВМ во время резекции",
            "Повреждение верхнего сагиттального синуса",
            "Брадикардия/асистолия при манипуляции на стволе",
            "Неконтролируемое кровотечение из опухоли",
            "Потеря моторных вызванных потенциалов",
            "Разрыв крупной кортикальной вены"],
    },
}


# --------------------------- target selection ---------------------------

def _concept_detail(concept_id: int) -> dict:
    with connect() as conn:
        row = conn.execute(
            """SELECT c.id, c.name, c.slug, c.summary, d.code AS domain
               FROM concepts c JOIN domains d ON d.id = c.domain_id
               WHERE c.id = ?""",
            (concept_id,),
        ).fetchone()
    return dict(row) if row else {}


def select_targets(n: int) -> list[dict]:
    """Pick up to n (concept, bloom) targets: due first, then new concepts.

    Due cards come ordered by next_review (FSRS surfaces weak items sooner),
    so this is naturally adaptive toward gaps.
    """
    targets: list[dict] = []
    seen: set[tuple[int, int]] = set()

    for d in due_today(limit=n):
        key = (d["concept_id"], d["bloom_level"])
        if key in seen:
            continue
        seen.add(key)
        targets.append({"concept_id": d["concept_id"],
                        "bloom_level": d["bloom_level"],
                        "name": d["name"], "domain": d["domain"]})

    if len(targets) < n:
        for c in pick_new(limit=n - len(targets)):
            key = (c["concept_id"], 1)
            if key in seen:
                continue
            seen.add(key)
            targets.append({"concept_id": c["concept_id"], "bloom_level": 1,
                            "name": c["name"], "domain": c["domain"]})

    return targets[:n]


# --------------------------- question generation ------------------------

def _generate_question(concept: dict, bloom_level: int) -> dict:
    """One focused LLM call → {prompt, rubric:[criteria]} grounded in RAG."""
    name = concept.get("name", "")
    summary = concept.get("summary") or ""
    snippets = rag_retriever.search(name, k=3)
    context = "\n".join(f"- {s['title']} {s['ref']}: {s['text'][:300]}"
                        for s in snippets) or "(контекст не найден)"

    sys_prompt = (
        "Ты — экзаменатор по нейрохирургии для ординатора. Сгенерируй РОВНО ОДИН "
        f"вопрос на активное припоминание по теме на уровне Блума {bloom_level} "
        f"({BLOOM_NAMES.get(bloom_level, '')}). Вопрос требует развёрнутого "
        "ответа (не да/нет), проверяет понимание, опирается на контекст из "
        "учебников. Верни ТОЛЬКО сырой JSON без markdown: "
        '{"question": "...", "rubric": ["критерий1", "критерий2", "критерий3"]}.'
    )
    user_prompt = (
        f"Тема: {name} (домен: {concept.get('domain','')}).\n"
        f"Краткое описание: {summary}\n\n"
        f"Контекст из учебников:\n{context}"
    )

    client = MiniMaxClient()
    try:
        resp = client.chat(
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": user_prompt}],
            temperature=0.4,
        )
        text = extract_text(resp)
    finally:
        client.close()

    parsed = _parse_json_loose(text) or {}
    question = parsed.get("question") or text.strip()
    rubric = parsed.get("rubric") or None
    return {"prompt": question, "rubric": rubric}


def _generate_scenario(kind: str) -> dict | None:
    """Generate one scenario drill (surgical_steps | emergency | crisis)."""
    spec = SCENARIO_KINDS.get(kind)
    if not spec:
        return None
    topic = random.choice(spec["seeds"])
    snippets = rag_retriever.search(topic, k=3)
    context = "\n".join(f"- {s['title']} {s['ref']}: {s['text'][:300]}"
                        for s in snippets) or "(контекст не найден)"

    sys_prompt = (
        "Ты — экзаменатор-нейрохирург для ординатора. " + spec["instruction"] +
        " Опирайся на контекст из учебников. Верни ТОЛЬКО сырой JSON без "
        'markdown: {"question": "...", "rubric": ["критерий1","критерий2","критерий3"]}.'
    )
    user_prompt = f"Тема: {topic}.\n\nКонтекст из учебников:\n{context}"

    client = MiniMaxClient()
    try:
        resp = client.chat(
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": user_prompt}],
            temperature=0.5,
        )
        text = extract_text(resp)
    finally:
        client.close()

    parsed = _parse_json_loose(text) or {}
    return {"prompt": parsed.get("question") or text.strip(),
            "rubric": parsed.get("rubric"), "label": topic}


def _generate_case() -> dict | None:
    """Pull a random clinical case and frame it as a management question."""
    with connect() as conn:
        row = conn.execute(
            "SELECT id, title, presentation, rubric, concept_ids "
            "FROM cases ORDER BY RANDOM() LIMIT 1").fetchone()
    if not row:
        return None
    case = dict(row)
    prompt = (
        f"Клинический случай: {case['title']}.\n\n{case['presentation']}\n\n"
        "Вопрос: ваш дифференциальный диагноз, план обследования и тактика "
        "ведения? Кратко обоснуйте каждый шаг."
    )
    rubric = None
    if case.get("rubric"):
        try:
            rj = json.loads(case["rubric"])
            rubric = rj.get("criteria") if isinstance(rj, dict) else rj
        except (TypeError, json.JSONDecodeError):
            pass
    if not rubric:
        rubric = ["верный дифференциал", "адекватный план обследования",
                  "правильная тактика", "учёт осложнений"]

    concept_id = None
    if case.get("concept_ids"):
        try:
            ids = json.loads(case["concept_ids"])
            if ids and isinstance(ids[0], int):
                concept_id = ids[0]
        except (TypeError, json.JSONDecodeError):
            pass
    return {"prompt": prompt, "rubric": rubric, "label": case["title"],
            "concept_id": concept_id, "bloom_level": 3}


def _store_pending(user_id: str, *, kind: str, prompt: str,
                   rubric: list | None, concept_id: int | None = None,
                   bloom_level: int = 1) -> int:
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO pending_questions(user_id, concept_id, bloom_level,
                                             prompt, rubric, kind)
               VALUES (?,?,?,?,?,?)""",
            (user_id, concept_id, bloom_level, prompt,
             json.dumps(rubric, ensure_ascii=False) if rubric else None, kind),
        )
        return cur.lastrowid


def generate_drill(user_id: str, n_recall: int = 3,
                   scenario_kinds: list[str] | None = None) -> list[dict]:
    """Build a varied drill batch: spaced-repetition recall + scenarios.

    Skips entirely if the user already has open (unanswered) questions.
    scenario_kinds defaults to one clinical case + two random scenario types
    (emergency / crisis / surgical_steps) for proactive practice.
    """
    if has_open(user_id):
        log.info("user %s has open questions, skipping generation", user_id)
        return []

    if scenario_kinds is None:
        pool = list(SCENARIO_KINDS.keys())
        scenario_kinds = ["case"] + random.sample(pool, k=min(2, len(pool)))

    created: list[dict] = []

    # 1) Spaced-repetition recall core (due + new concepts).
    for t in select_targets(n_recall):
        concept = _concept_detail(t["concept_id"])
        if not concept:
            continue
        q = _generate_question(concept, t["bloom_level"])
        qid = _store_pending(user_id, kind="recall", prompt=q["prompt"],
                             rubric=q["rubric"], concept_id=t["concept_id"],
                             bloom_level=t["bloom_level"])
        created.append({"id": qid, "kind": "recall", "label": concept["name"],
                        "prompt": q["prompt"]})

    # 2) Proactive scenarios (cases, emergencies, intraop crises, op steps).
    for kind in scenario_kinds:
        if kind == "case":
            q = _generate_case()
            if not q:
                continue
            qid = _store_pending(user_id, kind="case", prompt=q["prompt"],
                                 rubric=q["rubric"], concept_id=q.get("concept_id"),
                                 bloom_level=q.get("bloom_level", 3))
        else:
            q = _generate_scenario(kind)
            if not q:
                continue
            qid = _store_pending(user_id, kind=kind, prompt=q["prompt"],
                                 rubric=q["rubric"], concept_id=None,
                                 bloom_level=3)
        created.append({"id": qid, "kind": kind, "label": q["label"],
                        "prompt": q["prompt"]})

    return created


# --------------------------- answering ----------------------------------

def open_questions(user_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """SELECT id, concept_id, bloom_level, prompt, kind
               FROM pending_questions
               WHERE user_id=? AND answered_at IS NULL
               ORDER BY created_at ASC""",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def has_open(user_id: str) -> bool:
    return bool(open_questions(user_id))


def answer_pending(user_id: str, answer: str) -> dict | None:
    """Grade the oldest open question, reschedule, persist. Returns feedback.

    Returns None if there is no open question for this user.
    """
    with connect() as conn:
        row = conn.execute(
            """SELECT * FROM pending_questions
               WHERE user_id=? AND answered_at IS NULL
               ORDER BY created_at ASC LIMIT 1""",
            (user_id,),
        ).fetchone()
    if not row:
        return None

    rubric = json.loads(row["rubric"]) if row["rubric"] else None
    g = grade_answer(
        prompt=row["prompt"], answer=answer,
        concept_id=row["concept_id"], bloom_level=row["bloom_level"],
        rubric={"criteria": rubric} if rubric else None,
    )
    rating = int(g.get("suggested_rating") or 3)
    # Scenario drills (case/emergency/crisis/surgical_steps) aren't bound to a
    # single concept → grade & log them, but only reschedule concept-bound ones.
    sched = None
    if row["concept_id"] is not None:
        sched = schedule_fsrs(
            concept_id=row["concept_id"], bloom_level=row["bloom_level"],
            rating=rating)

    now = datetime.now(timezone.utc).isoformat()
    with connect() as conn:
        conn.execute(
            """INSERT INTO responses(session_id, concept_id, bloom_level, role,
                                     prompt, answer, grade, fsrs_rating,
                                     rubric_breakdown)
               VALUES (NULL,?,?,?,?,?,?,?,?)""",
            (row["concept_id"], row["bloom_level"], row["kind"], row["prompt"],
             answer, g.get("score"), rating,
             json.dumps(g.get("breakdown"), ensure_ascii=False)
             if g.get("breakdown") is not None else None),
        )
        conn.execute("UPDATE pending_questions SET answered_at=? WHERE id=?",
                     (now, row["id"]))

    remaining = open_questions(user_id)
    return {
        "score": g.get("score"),
        "rating": rating,
        "feedback": g.get("feedback", ""),
        "next_review": sched["next_review"] if sched else None,
        "mastery": sched["mastery"] if sched else None,
        "remaining": remaining,
        "kind": row["kind"],
    }
