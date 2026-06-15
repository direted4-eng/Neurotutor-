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
import re
from datetime import datetime, timedelta, timezone

from ..db.store import connect, get_or_create_concept, slugify
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


# Each scenario kind is anchored to a competency domain so its concepts feed
# the right slice of the mastery map (op steps → approaches; emergencies and
# intraop crises → clinical management).
SCENARIO_DOMAIN = {
    "surgical_steps": "approaches",
    "emergency": "neurocritical",
    "crisis": "neurocritical",
}


def _scenario_concept_id(kind: str, topic: str) -> int | None:
    """Resolve a scenario seed topic to a tracked concept (get-or-create).

    Without this, scenario drills carried concept_id=None → graded but never
    scheduled by FSRS and invisible in the domain mastery map. Binding each
    topic to a concept makes op-steps/emergency/crisis practice build real,
    spaced-repetition competency just like recall.
    """
    domain = SCENARIO_DOMAIN.get(kind)
    if not domain:
        return None
    slug = f"{kind}_{slugify(topic)}"[:120]
    label = SCENARIO_KINDS.get(kind, {}).get("label", kind)
    return get_or_create_concept(
        topic, domain, slug=slug, summary=f"Сценарный навык — {label}: {topic}")


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
            "rubric": parsed.get("rubric"), "label": topic,
            "concept_id": _scenario_concept_id(kind, topic), "bloom_level": 3}


# Disease-specific severity scales that would betray the diagnosis category if
# shown upfront (Hunt-Hess → SAH, WHO grade → tumour, ASIA → cord injury…).
# GCS / vitals / exam findings stay — they're generic, not diagnosis tells.
# Stripped scales remain available on demand via case_followup: the resident
# must ASK for them, the way they'd order a test in a real workup.
_SCALE_RE = re.compile(
    r"\b(?:hunt[\s-]*hess|хант[\w-]*|fisher|фишер|who\s*grade|spetzler[\s-]*martin|"
    r"спетцл[\w-]*|asia|house[\s-]*brackmann|karnofsky|карновск[\w-]*|wfns|mrs|"
    r"rankin|ранкин)\b\s*(?:grade|степень|класс|score)?\s*(?:[:=\-—]\s*)?"
    r"(?:[IVX]{1,4}|[A-E]\b|\d{1,3})?",
    re.IGNORECASE)


def _strip_scales(text: str) -> str:
    """Remove diagnosis-revealing severity-scale gradings from a case blurb."""
    cleaned = _SCALE_RE.sub("", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)            # space before punct
    cleaned = re.sub(r"([.,;:])(\s*[.,;:])+", r"\1", cleaned)   # doubled punct
    return cleaned.strip()


def _generate_case() -> dict | None:
    """Pull a random clinical case and frame it as a management question."""
    with connect() as conn:
        row = conn.execute(
            "SELECT id, title, presentation, rubric, concept_ids "
            "FROM cases ORDER BY RANDOM() LIMIT 1").fetchone()
    if not row:
        return None
    case = dict(row)
    # NB: case['title'] is the diagnosis — deliberately NOT shown. The whole
    # point of a case drill is for the resident to reach the diagnosis from the
    # clinical picture. The title is carried as `topic` (for grading & reveal),
    # never in the displayed prompt. Severity scales are stripped too (they leak
    # the diagnosis category) — available on demand via case_followup.
    prompt = (
        f"Клинический случай.\n\n{_strip_scales(case['presentation'])}\n\n"
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
                   bloom_level: int = 1, topic: str | None = None) -> int:
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO pending_questions(user_id, concept_id, bloom_level,
                                             prompt, rubric, kind, topic)
               VALUES (?,?,?,?,?,?,?)""",
            (user_id, concept_id, bloom_level, prompt,
             json.dumps(rubric, ensure_ascii=False) if rubric else None,
             kind, topic),
        )
        return cur.lastrowid


def generate_drill(user_id: str, n_recall: int = 3,
                   scenario_kinds: list[str] | None = None) -> list[dict]:
    """Build a varied drill batch: spaced-repetition recall + scenarios.

    Stale open questions (older than STALE_HOURS) are expired first, so one
    missed day never freezes the loop forever. Skips only if *fresh* open
    questions remain. scenario_kinds defaults to one clinical case + one random
    scenario type (emergency / crisis / surgical_steps): the batch is sized to
    the real answering pace, a backlog kills engagement faster than scarcity.
    """
    expire_stale(user_id)
    if has_open(user_id):
        log.info("user %s has open questions, skipping generation", user_id)
        return []

    if scenario_kinds is None:
        pool = list(SCENARIO_KINDS.keys())
        scenario_kinds = ["case"] + random.sample(pool, k=min(1, len(pool)))

    created: list[dict] = []

    # 1) Spaced-repetition recall core (due + new concepts).
    for t in select_targets(n_recall):
        concept = _concept_detail(t["concept_id"])
        if not concept:
            continue
        q = _generate_question(concept, t["bloom_level"])
        qid = _store_pending(user_id, kind="recall", prompt=q["prompt"],
                             rubric=q["rubric"], concept_id=t["concept_id"],
                             bloom_level=t["bloom_level"], topic=concept["name"])
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
                                 bloom_level=q.get("bloom_level", 3),
                                 topic=q["label"])
        else:
            q = _generate_scenario(kind)
            if not q:
                continue
            qid = _store_pending(user_id, kind=kind, prompt=q["prompt"],
                                 rubric=q["rubric"], concept_id=q.get("concept_id"),
                                 bloom_level=q.get("bloom_level", 3), topic=q["label"])
        created.append({"id": qid, "kind": kind, "label": q["label"],
                        "prompt": q["prompt"]})

    return created


def generate_calibration(user_id: str) -> list[dict]:
    """Seed the mastery map: one baseline question per competency domain.

    The cold-start problem: with zero reviewed pairs the tutor doesn't know
    the resident's actual level, so drills start blind. Calibration asks one
    never-reviewed concept per domain (Bloom 1–2); answers flow through the
    normal grade→FSRS pipeline, so every domain gets a real first data point.
    Clears the current queue first — calibration is an explicit restart.
    """
    from ..db.store import DOMAINS

    expire_all_open(user_id)
    created: list[dict] = []
    for code, _title, _target in DOMAINS:
        picks = pick_new(limit=1, domain=code)
        if not picks:
            continue
        concept = _concept_detail(picks[0]["concept_id"])
        if not concept:
            continue
        bloom = random.choice((1, 2))
        try:
            q = _generate_question(concept, bloom)
        except Exception:
            log.exception("calibration question failed for %s", concept["name"])
            continue
        qid = _store_pending(user_id, kind="recall", prompt=q["prompt"],
                             rubric=q["rubric"], concept_id=concept["id"],
                             bloom_level=bloom, topic=concept["name"])
        created.append({"id": qid, "kind": "recall", "label": concept["name"],
                        "domain": code, "prompt": q["prompt"]})
    return created


def generate_one(user_id: str, kind: str) -> dict | None:
    """Generate ONE question of a chosen kind on demand (mode selection).

    Unlike generate_drill (the proactive auto-mix), this is user-initiated:
    the resident picks a mode (recall / case / surgical_steps / emergency /
    crisis) and we produce a single matching pending question. Returns the
    created question dict, or None if nothing of that kind could be built.
    """
    if kind == "recall":
        targets = select_targets(1)
        if not targets:
            return None
        t = targets[0]
        concept = _concept_detail(t["concept_id"])
        if not concept:
            return None
        q = _generate_question(concept, t["bloom_level"])
        qid = _store_pending(user_id, kind="recall", prompt=q["prompt"],
                             rubric=q["rubric"], concept_id=t["concept_id"],
                             bloom_level=t["bloom_level"], topic=concept["name"])
        return {"id": qid, "kind": "recall", "label": concept["name"],
                "bloom_level": t["bloom_level"], "prompt": q["prompt"]}

    if kind == "case":
        q = _generate_case()
        if not q:
            return None
        qid = _store_pending(user_id, kind="case", prompt=q["prompt"],
                             rubric=q["rubric"], concept_id=q.get("concept_id"),
                             bloom_level=q.get("bloom_level", 3), topic=q["label"])
        return {"id": qid, "kind": "case", "label": q["label"],
                "prompt": q["prompt"]}

    if kind in SCENARIO_KINDS:
        q = _generate_scenario(kind)
        if not q:
            return None
        qid = _store_pending(user_id, kind=kind, prompt=q["prompt"],
                             rubric=q["rubric"], concept_id=q.get("concept_id"),
                             bloom_level=q.get("bloom_level", 3), topic=q["label"])
        return {"id": qid, "kind": kind, "label": q["label"],
                "prompt": q["prompt"]}

    return None


# --------------------------- answering ----------------------------------

# Unanswered questions older than this are expired (not graded, not a lapse):
# without a TTL one missed day used to freeze generate_drill permanently.
STALE_HOURS = 48


def _ensure_pending_columns() -> None:
    """Idempotent migration: add pending_questions columns missing on old DBs
    (expired_at / tg_message_id / saved_answer).

    Self-heals a brand-new DB: if the table doesn't exist yet (fresh deploy,
    schema never applied), build the schema first — schema.sql already declares
    every column, so no ALTERs are then needed. Without this, importing this
    module against an empty DB used to crash on the first ALTER.
    """
    with connect() as conn:
        cols = [r[1] for r in conn.execute(
            "PRAGMA table_info(pending_questions)").fetchall()]
    if not cols:
        from ..db.store import init_db
        init_db()
        return
    with connect() as conn:
        if "expired_at" not in cols:
            conn.execute(
                "ALTER TABLE pending_questions ADD COLUMN expired_at TIMESTAMP")
        if "tg_message_id" not in cols:
            # Telegram message_id of the message that DELIVERED this question.
            # Lets an answer sent as a Telegram reply bind to the exact question
            # it replies to — instead of always grading the oldest open one.
            conn.execute(
                "ALTER TABLE pending_questions ADD COLUMN tg_message_id INTEGER")
        if "saved_answer" not in cols:
            # The resident's answer, stashed when MiniMax was overloaded (529)
            # and grading had to be deferred. A re-grade pass picks these up so
            # an overload never silently swallows an answer.
            conn.execute(
                "ALTER TABLE pending_questions ADD COLUMN saved_answer TEXT")


_ensure_pending_columns()


def set_question_message(qid: int, tg_message_id: int | None) -> None:
    """Remember which Telegram message delivered a given pending question."""
    if not qid or not tg_message_id:
        return
    with connect() as conn:
        conn.execute(
            "UPDATE pending_questions SET tg_message_id=? WHERE id=?",
            (tg_message_id, qid))


def open_question_by_message(user_id: str, tg_message_id: int) -> dict | None:
    """Find the OPEN pending question delivered by a given Telegram message.

    Used for reply-binding: when the resident answers via Telegram «reply» to a
    specific question, we grade that one instead of the oldest in the queue.
    Returns None if no open question matches that message.
    """
    if not tg_message_id:
        return None
    with connect() as conn:
        row = conn.execute(
            """SELECT id, concept_id, bloom_level, prompt, kind
               FROM pending_questions
               WHERE user_id=? AND tg_message_id=?
                 AND answered_at IS NULL AND expired_at IS NULL
               LIMIT 1""",
            (user_id, tg_message_id)).fetchone()
    return dict(row) if row else None


def expire_stale(user_id: str, max_age_hours: int = STALE_HOURS) -> int:
    """Expire open questions older than the TTL. Returns how many expired.

    Expiry is neutral: no grade, no FSRS lapse — the concept just comes back
    through normal due/new selection later.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
              ).strftime("%Y-%m-%d %H:%M:%S")
    now = datetime.now(timezone.utc).isoformat()
    with connect() as conn:
        cur = conn.execute(
            """UPDATE pending_questions SET expired_at=?
               WHERE user_id=? AND answered_at IS NULL AND expired_at IS NULL
                 AND created_at <= ?""",
            (now, user_id, cutoff))
    if cur.rowcount:
        log.info("expired %d stale question(s) for user %s", cur.rowcount, user_id)
    return cur.rowcount


def expire_all_open(user_id: str) -> int:
    """Expire every open question (used when the user restarts the queue)."""
    return expire_stale(user_id, max_age_hours=0)


def skip_first(user_id: str) -> dict | None:
    """Expire just the oldest open question («пропусти»). Returns it, or None."""
    pend = open_questions(user_id)
    if not pend:
        return None
    now = datetime.now(timezone.utc).isoformat()
    with connect() as conn:
        conn.execute("UPDATE pending_questions SET expired_at=? WHERE id=?",
                     (now, pend[0]["id"]))
    return pend[0]


def open_questions(user_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """SELECT id, concept_id, bloom_level, prompt, kind
               FROM pending_questions
               WHERE user_id=? AND answered_at IS NULL AND expired_at IS NULL
               ORDER BY created_at ASC""",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def has_open(user_id: str) -> bool:
    return bool(open_questions(user_id))


def case_followup(user_id: str, query: str, qid: int | None = None) -> str | None:
    """Answer an exploratory request during an OPEN case (scale, lab, imaging).

    The resident works the case interactively — asking for findings or scale
    scores the way they'd order tests in a real workup. We answer from the full
    case record (which still holds the scales stripped from the prompt) but
    never reveal or hint at the diagnosis: that's theirs to commit via
    «заключение». Returns None if there is no open case (or on failure).

    qid pins the followup to a specific open case (reply-binding); without it we
    use the oldest open question.
    """
    open_q = open_questions(user_id)
    if not open_q:
        return None
    with connect() as conn:
        if qid is not None:
            row = conn.execute(
                "SELECT topic, prompt, kind FROM pending_questions WHERE id=? "
                "AND user_id=? AND answered_at IS NULL AND expired_at IS NULL "
                "LIMIT 1", (qid, user_id)).fetchone()
        else:
            row = conn.execute(
                "SELECT topic, prompt, kind FROM pending_questions WHERE user_id=? "
                "AND answered_at IS NULL AND expired_at IS NULL "
                "ORDER BY created_at ASC LIMIT 1",
                (user_id,)).fetchone()
        if not row or row["kind"] != "case":
            return None
        case = conn.execute(
            "SELECT presentation FROM cases WHERE title=?",
            (row["topic"],)).fetchone() if row and row["topic"] else None
    presentation = case["presentation"] if case else (row["prompt"] if row else "")

    client = MiniMaxClient()
    try:
        sys = (
            "Ты ведёшь интерактивный клинический разбор с ординатором. Он изучает "
            "случай и запрашивает данные: оценку по шкале, лабораторию, результат "
            "КТ/МРТ/ангиографии, детали осмотра. Ответь ТОЛЬКО на запрос — кратко "
            "и по делу, опираясь на случай. Если просят шкалу — посчитай по данным "
            "и дай балл с краткой расшифровкой. Если просят обследование — дай "
            "правдоподобный результат, согласованный со случаем. КАТЕГОРИЧЕСКИ не "
            "называй итоговый диагноз и не намекай на него: ординатор ставит его "
            "сам. Пиши по-русски."
        )
        user = json.dumps({"случай": presentation, "запрос": query},
                          ensure_ascii=False)
        resp = client.chat(
            [{"role": "system", "content": sys},
             {"role": "user", "content": user}], temperature=0.2)
        return extract_text(resp).strip() or None
    except Exception:
        log.exception("case_followup failed")
        return None
    finally:
        client.close()


def answer_pending(user_id: str, answer: str) -> dict | None:
    """Grade the oldest open question, reschedule, persist. Returns feedback.

    Returns None if there is no open question for this user.
    """
    with connect() as conn:
        row = conn.execute(
            """SELECT * FROM pending_questions
               WHERE user_id=? AND answered_at IS NULL AND expired_at IS NULL
               ORDER BY created_at ASC LIMIT 1""",
            (user_id,),
        ).fetchone()
    if not row:
        return None
    return _grade_pending_row(user_id, row, answer)


def answer_specific(user_id: str, qid: int, answer: str) -> dict | None:
    """Grade a SPECIFIC open question by id (reply-binding), not the oldest.

    Returns None if that question id is not an open question for this user, so
    the caller can fall back to the normal oldest-first path.
    """
    with connect() as conn:
        row = conn.execute(
            """SELECT * FROM pending_questions
               WHERE id=? AND user_id=? AND answered_at IS NULL
                 AND expired_at IS NULL LIMIT 1""",
            (qid, user_id),
        ).fetchone()
    if not row:
        return None
    return _grade_pending_row(user_id, row, answer)


def pending_regrade(user_id: str) -> list[dict]:
    """Open questions whose answer was stashed during a MiniMax overload (529).

    These are answered-but-ungraded: the resident replied, but the grader was
    down, so we kept the text in saved_answer instead of losing it.
    """
    with connect() as conn:
        rows = conn.execute(
            """SELECT id FROM pending_questions
               WHERE user_id=? AND answered_at IS NULL AND expired_at IS NULL
                 AND saved_answer IS NOT NULL
               ORDER BY created_at ASC""",
            (user_id,)).fetchall()
    return [dict(r) for r in rows]


def regrade_saved(user_id: str) -> list[dict]:
    """Re-grade every answer deferred by an overload. Returns feedback dicts.

    Each still-degraded result (MiniMax still down) is skipped — its saved_answer
    stays put so the next pass retries. Successfully graded ones flow through the
    normal grade→FSRS→responses pipeline and clear their saved_answer.
    """
    out: list[dict] = []
    for ref in pending_regrade(user_id):
        with connect() as conn:
            row = conn.execute(
                "SELECT * FROM pending_questions WHERE id=?", (ref["id"],)
            ).fetchone()
        if not row or row["saved_answer"] is None:
            continue
        fb = _grade_pending_row(user_id, row, row["saved_answer"])
        if fb and not fb.get("degraded"):
            out.append(fb)
    return out


def _grade_pending_row(user_id: str, row, answer: str) -> dict | None:
    """Grade one pending-question row, reschedule (FSRS), persist the response.

    Shared by answer_pending (oldest) and answer_specific (reply-bound) so both
    paths grade, schedule and record identically.
    """
    rubric = json.loads(row["rubric"]) if row["rubric"] else None
    # For cases the diagnosis is hidden from the prompt, so hand the grader the
    # reference diagnosis here (grading-only, never stored or shown) — otherwise
    # it can't judge whether the resident's differential actually landed.
    grade_prompt = row["prompt"]
    if row["kind"] == "case" and row["topic"]:
        grade_prompt += (
            f"\n\n[Для проверяющего, студенту НЕ показано — эталонный диагноз: "
            f"{row['topic']}. Оцени, насколько ответ к нему близок.]")
    # MiniMax-only: there is no failover model. If the grader call exhausts its
    # retries (persistent 529 / network), DON'T let the exception bubble up and
    # vanish the answer in main()'s catch-all. Stash the answer, keep the
    # question open, and signal a graceful degrade — the --regrade pass will
    # grade it once MiniMax recovers.
    try:
        g = grade_answer(
            prompt=grade_prompt, answer=answer,
            concept_id=row["concept_id"], bloom_level=row["bloom_level"],
            rubric={"criteria": rubric} if rubric else None,
        )
    except Exception:
        log.exception("grade_answer failed (overload?) for pending %s — deferring",
                      row["id"])
        with connect() as conn:
            conn.execute(
                "UPDATE pending_questions SET saved_answer=? WHERE id=?",
                (answer, row["id"]))
        return {"degraded": True, "kind": row["kind"], "topic": row["topic"],
                "question": row["prompt"]}

    # Grader couldn't be parsed → leave the question OPEN and don't touch FSRS,
    # so a transient hiccup never writes a phantom lapse into the mastery map.
    if g.get("parse_error") or g.get("score") is None:
        log.warning("answer left ungraded (parse error) for pending %s", row["id"])
        return {"ungraded": True, "feedback": g.get("feedback", ""),
                "kind": row["kind"], "topic": row["topic"],
                "question": row["prompt"]}

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
        # Clear any stashed answer from an earlier deferred (529) attempt — this
        # row is now graded for real.
        conn.execute(
            "UPDATE pending_questions SET answered_at=?, saved_answer=NULL "
            "WHERE id=?", (now, row["id"]))

    # Refresh the dashboard snapshot now that mastery moved (best-effort — a
    # missing static dir or any failure must never break grading).
    try:
        from ..dashboard.snapshot import write_snapshot
        write_snapshot()
    except Exception:
        log.exception("dashboard snapshot refresh failed")

    remaining = open_questions(user_id)
    return {
        "score": g.get("score"),
        "rating": rating,
        "feedback": g.get("feedback", ""),
        "next_review": sched["next_review"] if sched else None,
        "mastery": sched["mastery"] if sched else None,
        "remaining": remaining,
        "kind": row["kind"],
        "topic": row["topic"],
        "question": row["prompt"],
        "answer": answer,
    }
