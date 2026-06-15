"""Tool implementations exposed to the LLM via tool-calling.

Tool list mirrors the architectural spec:
  query_anatomy, interpret_imaging, lookup_classification,
  case_simulator, grade_answer, schedule_fsrs,
  rag_search, pubmed_search.

Each function takes a JSON-serializable dict and returns a
JSON-serializable dict. The orchestrator routes by tool name.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..db.store import connect
from ..fsrs import scheduler as fsrs_sched
from ..rag import retriever as rag_retriever
from ..rag import sources as med_sources

log = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.S)


def _parse_json_loose(text: str) -> dict | None:
    """Parse JSON that may be wrapped in ```json fences or have leading prose."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


# ------------------------- schemas (OpenAI-style) -------------------------

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "query_anatomy",
            "description": "Lookup a neuroanatomy concept by slug or fuzzy name. "
                           "Returns summary, parent, sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "domain": {"type": "string", "enum": [
                        "anatomy", "radiology", "vascular", "oncology", "spine",
                        "trauma", "functional", "pediatric", "hydrocephalus",
                        "peripheral_nerve", "infection", "neurocritical",
                        "professional", "approaches",
                    ]},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "interpret_imaging",
            "description": "Send one image to the vision model with a focused "
                           "prompt. Pass image_path; the loader reads → sends → "
                           "drops to keep RAM low.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string"},
                    "task": {"type": "string", "description":
                             "What to extract: structures | pattern | "
                             "differential | side"},
                    "modality": {"type": "string"},
                },
                "required": ["image_path", "task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_classification",
            "description": "Fetch a clinical classification by code: "
                           "who_cns_2021, spetzler_martin, hunt_hess, "
                           "fisher, mfisher, gcs, asia, house_brackmann.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "case_simulator",
            "description": "Draw a case viñette for the given domain and "
                           "difficulty, or step through an active case.",
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string"},
                    "difficulty": {"type": "integer", "minimum": 1, "maximum": 5},
                    "case_id": {"type": "integer"},
                    "step": {"type": "string", "enum": [
                        "presentation", "differential", "workup",
                        "interpretation", "plan", "complications"
                    ]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grade_answer",
            "description": "Grade a free-text answer against a rubric. "
                           "Returns score 0..1 and a per-criterion breakdown.",
            "parameters": {
                "type": "object",
                "properties": {
                    "concept_id": {"type": "integer"},
                    "bloom_level": {"type": "integer", "minimum": 1, "maximum": 6},
                    "prompt": {"type": "string"},
                    "answer": {"type": "string"},
                    "rubric": {"type": "object"},
                },
                "required": ["prompt", "answer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_fsrs",
            "description": "Update FSRS state for (concept × bloom_level) "
                           "after a graded review.",
            "parameters": {
                "type": "object",
                "properties": {
                    "concept_id": {"type": "integer"},
                    "bloom_level": {"type": "integer"},
                    "rating": {"type": "integer", "minimum": 1, "maximum": 4,
                               "description": "1 again 2 hard 3 good 4 easy"},
                },
                "required": ["concept_id", "bloom_level", "rating"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Semantic search over ingested textbooks "
                           "(Greenberg, Youmans, Rhoton) and saved articles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "source": {"type": "string"},
                    "k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pubmed_search",
            "description": "Search PubMed for peer-reviewed literature. "
                           "Returns titles, PMIDs, abstracts. Use for "
                           "verifying clinical statements with authoritative sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                    "filter": {"type": "string",
                               "description": "e.g. 'review[pt]', "
                               "'clinical trial[pt]', 'guideline[pt]'"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "radiopaedia_search",
            "description": "Search Radiopaedia for radiology cases with real "
                           "CT/MRI/angio images. Returns title, URL, image_url, "
                           "modality. Use in imaging mode or when the student "
                           "needs a visual example of a finding.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "e.g. 'circle of willis MRI', "
                              "'subarachnoid hemorrhage CT', 'glioblastoma'"},
                    "max_results": {"type": "integer", "default": 3},
                    "scope": {"type": "string",
                              "enum": ["cases", "articles", "all"],
                              "default": "cases"},
                },
                "required": ["query"],
            },
        },
    },
]


# --------------------------- implementations -----------------------------

def query_anatomy(query: str, domain: str | None = None) -> dict:
    with connect() as conn:
        sql = """SELECT c.id, c.name, c.slug, c.summary, c.sources,
                        d.code AS domain, p.name AS parent
                 FROM concepts c
                 JOIN domains d ON d.id = c.domain_id
                 LEFT JOIN concepts p ON p.id = c.parent_id
                 WHERE (c.slug = ? OR c.name LIKE ?)"""
        params: list[Any] = [query, f"%{query}%"]
        if domain:
            sql += " AND d.code = ?"
            params.append(domain)
        sql += " LIMIT 5"
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    return {"results": rows}


def interpret_imaging(image_path: str, task: str, modality: str | None = None) -> dict:
    """Invoke the vision model on a single image. Loaded once, dropped after."""
    from ..llm.minimax import MiniMaxClient, extract_text

    p = Path(image_path)
    if not p.exists():
        return {"error": f"image not found: {image_path}"}

    client = MiniMaxClient()
    try:
        prompt = (
            f"Modality: {modality or 'unknown'}. Task: {task}. "
            "Reply structured: structures_named, side, pattern, differential, "
            "next_step. Russian."
        )
        resp = client.vision_chat(
            [{"role": "user", "content": prompt}],
            image_path=p,
        )
        return {"text": extract_text(resp)}
    finally:
        client.close()


def lookup_classification(code: str) -> dict:
    with connect() as conn:
        row = conn.execute(
            "SELECT code, title, payload, source FROM classifications WHERE code=?",
            (code,),
        ).fetchone()
    if not row:
        return {"error": f"unknown classification: {code}"}
    return {
        "code": row["code"],
        "title": row["title"],
        "payload": json.loads(row["payload"]),
        "source": row["source"],
    }


def case_simulator(
    domain: str | None = None,
    difficulty: int | None = None,
    case_id: int | None = None,
    step: str | None = None,
) -> dict:
    with connect() as conn:
        if case_id:
            row = conn.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
        else:
            sql = "SELECT * FROM cases WHERE 1=1"
            params: list[Any] = []
            if domain:
                sql += " AND domain_id = (SELECT id FROM domains WHERE code=?)"
                params.append(domain)
            if difficulty:
                sql += " AND difficulty = ?"
                params.append(difficulty)
            rows = conn.execute(sql, params).fetchall()
            row = random.choice(rows) if rows else None

    if not row:
        return {"error": "no matching cases"}

    case = dict(row)
    for f in ("workup", "differential", "plan", "complications", "rubric",
              "concept_ids"):
        if case.get(f):
            try:
                case[f] = json.loads(case[f])
            except (TypeError, json.JSONDecodeError):
                pass

    if step:
        # Reveal only the requested step. Force the LLM to ask before showing more.
        visible = {"id": case["id"], "title": case["title"]}
        visible[step] = case.get(step)
        return visible
    return {"id": case["id"], "title": case["title"],
            "presentation": case["presentation"]}


def _rating_from_score(score: float) -> int:
    """Map a 0..1 score to an FSRS rating deterministically.

    Keeps the spaced-repetition signal consistent with the displayed score
    instead of letting the model pick a rating independently of its own marks.
    1=again, 2=hard, 3=good, 4=easy.
    """
    if score >= 0.85:
        return 4
    if score >= 0.6:
        return 3
    if score >= 0.4:
        return 2
    return 1


# Self-consistency: how many times to sample the grader and aggregate. MiniMax
# is the only judge available (no stronger cross-check model), so the lever for
# fairness is VARIANCE REDUCTION — N samples at temp>0, aggregated by median,
# cancel the run-to-run noise that reads as "unfair". Tunable via env; 1 = the
# old single deterministic call. Each extra sample is another MiniMax call (more
# latency + 529 exposure), which is why this ships only after the 529
# graceful-degrade hardening.
GRADE_SAMPLES = max(1, int(os.getenv("NEUROTUTOR_GRADE_SAMPLES", "3")))
_GRADE_SAMPLE_TEMP = 0.35


def _normalize_criteria(criteria) -> list[tuple[str, float]]:
    """Normalize a rubric's criteria into [(name, weight)].

    Accepts plain strings (weight 1.0 — the common case) or dicts carrying an
    explicit weight ({"name": ..., "weight": ...}), so a rubric can mark core
    criteria as worth more than peripheral ones without breaking old content.
    """
    out: list[tuple[str, float]] = []
    for c in criteria or []:
        if isinstance(c, dict):
            name = str(c.get("name") or c.get("criterion") or "").strip()
            raw_w = c.get("weight", 1.0)
        else:
            name, raw_w = str(c).strip(), 1.0
        try:
            w = max(0.0, float(raw_w))
        except (TypeError, ValueError):
            w = 1.0
        if name:
            out.append((name, w))
    return out or [("correctness", 1.0)]


def _weighted_mean(marks: dict, weighted_criteria: list[tuple[str, float]]):
    """Weighted mean of per-criterion marks; None if no criterion was marked.

    Equal weights (the default) reduce to a plain mean — identical to the prior
    behaviour — so existing string rubrics grade exactly as before.
    """
    num = den = 0.0
    for name, w in weighted_criteria:
        if name in marks and w > 0:
            num += marks[name] * w
            den += w
    return (num / den) if den else None


def grade_answer(
    prompt: str,
    answer: str,
    concept_id: int | None = None,
    bloom_level: int | None = None,
    rubric: dict | None = None,
    samples: int | None = None,
) -> dict:
    """Grade with the text model, then ground the score in the rubric.

    The overall score is a WEIGHTED mean of per-criterion marks (not a
    free-floating holistic number); equal weights reduce to a plain mean. To
    damp MiniMax's run-to-run dispersion — the noise that reads as unfair — the
    grader is sampled `samples` times (default GRADE_SAMPLES) at temp>0 and each
    criterion's marks are aggregated by MEDIAN before the weighted mean. The
    FSRS rating is derived deterministically from the final score. If NO sample
    yields parseable JSON we return an *ungraded* result (score=None) rather
    than a fake 0 — a parser hiccup never corrupts mastery with a phantom lapse.
    """
    from ..llm.minimax import MiniMaxClient, extract_text

    n_samples = GRADE_SAMPLES if samples is None else max(1, int(samples))

    default_rubric = {
        1: ["factual_correctness", "completeness"],
        2: ["explanation_quality", "use_of_correct_terms"],
        3: ["application_to_scenario", "step_correctness"],
        4: ["differentiation", "reasoning_depth"],
        5: ["evidence_quality", "trade_off_analysis"],
        6: ["originality", "feasibility"],
    }
    raw_criteria = (rubric and rubric.get("criteria")) or \
        default_rubric.get(bloom_level or 1, ["correctness"])
    weighted_criteria = _normalize_criteria(raw_criteria)
    criteria = [name for name, _ in weighted_criteria]   # names shown to the model

    # A single deterministic pass keeps cost minimal; multiple passes need temp>0
    # to actually diversify, otherwise the samples are near-identical.
    temperature = 0.0 if n_samples <= 1 else _GRADE_SAMPLE_TEMP

    client = MiniMaxClient()
    try:
        sys = (
            "Ты экзаменатор по нейрохирургии. Оцени ответ по каждому из "
            "заданных критериев с ЧАСТИЧНЫМ зачётом по якорной шкале: "
            "1.0 — критерий раскрыт полностью и верно; "
            "0.7 — в основном верно, мелкие пропуски; "
            "0.5 — раскрыт наполовину (главное названо, деталей нет); "
            "0.3 — затронут поверхностно, но направление мысли верное; "
            "0.0 — не раскрыт или грубая фактическая ошибка. "
            "Оценивай ЗНАНИЕ, а не стиль: за краткость не снижай, если суть "
            "верна; за уверенный, но фактически неверный ответ ставь 0. "
            "ВАЖНО: оценивай СМЫСЛ, а не форму записи. Засчитывай как ПОЛНЫЙ "
            "ответ общепринятые русские нейрохирургические аббревиатуры и "
            "синонимы наравне с полными терминами: ВСА=внутренняя сонная "
            "артерия, ПМА=передняя мозговая, СМА=средняя мозговая, ЗМА=задняя "
            "мозговая, ПСА=передняя соединительная, ЗСА=задняя соединительная, "
            "ОА=основная/базилярная, ПА=позвоночная, ТМО=твёрдая мозговая "
            "оболочка, СAК/САК=субарахноидальное кровоизлияние, ЧМТ=черепно-"
            "мозговая травма, ВЧД=внутричерепное давление, ВЧГ=внутричерепная "
            "гипертензия, ЛД=ликворное давление, ВПШ=вентрикулоперитонеальный "
            "шунт, ГЭБ, ЗЧЯ=задняя черепная ямка, ПКЯ/СЧЯ, КТ/МРТ/ДВИ/КТ-АГ и "
            "т.п. НЕ снижай балл за использование аббревиатуры или латинской/"
            "английской записи (M1/A1/P1, ICA/MCA) вместо полного названия и "
            "наоборот, если по смыслу названо верно. Не требуй дословного "
            "совпадения с формулировкой критерия — важно, что понятие названо. "
            "В feedback по-русски, строго в три блока (пустые блоки опусти): "
            "«✅ Верно: …» — что названо правильно; "
            "«⚠️ Упущено: …» — чего не хватило до полного ответа (конкретно); "
            "«❗ Ошибки: …» — фактические ошибки с исправлением. "
            "Верни ТОЛЬКО сырой JSON, без markdown и без ``` блоков. "
            "Формат: {\"breakdown\": {<каждый критерий>: 0..1}, \"feedback\": str}. "
            "Ключи breakdown — ровно из списка criteria."
        )
        user = json.dumps({"prompt": prompt, "answer": answer,
                           "criteria": criteria}, ensure_ascii=False)
        msgs = [{"role": "system", "content": sys},
                {"role": "user", "content": user}]

        def _one_sample(_ignored=None) -> str:
            return extract_text(client.chat(msgs, temperature=temperature))

        if n_samples == 1:
            texts = [_one_sample()]
        else:
            # Fire the self-consistency samples CONCURRENTLY. Sequential N× calls
            # made a grade take ~N×12s on MiniMax (a silent wait after answering);
            # httpx.Client is thread-safe, so parallel keeps latency near a single
            # call. An unrecoverable sample (529 after retries) propagates here →
            # the caller's degrade path stashes the answer for --regrade.
            with ThreadPoolExecutor(max_workers=n_samples) as ex:
                texts = list(ex.map(_one_sample, range(n_samples)))
    finally:
        client.close()

    parsed_samples = [p for p in (_parse_json_loose(t) for t in texts)
                      if p is not None]
    last_text = texts[-1] if texts else ""

    # No sample parsed → ungraded (never a fake 0): a transient parser/model
    # hiccup must not write a phantom lapse into mastery.
    if not parsed_samples:
        log.warning("grade_answer: no parseable sample; last raw: %r", last_text[:500])
        return {"score": None, "breakdown": {}, "feedback": last_text[:500].strip(),
                "suggested_rating": None, "parse_error": True,
                "concept_id": concept_id, "bloom_level": bloom_level,
                "samples_used": 0}

    # Aggregate per-criterion marks across samples by MEDIAN (robust to a single
    # outlier sample), then take the weighted mean over the rubric.
    per_criterion: dict[str, list[float]] = {name: [] for name in criteria}
    holistic: list[float] = []
    for r in parsed_samples:
        bd = r.get("breakdown") or {}
        for name in criteria:
            v = bd.get(name)
            if isinstance(v, (int, float)):
                per_criterion[name].append(float(v))
        h = r.get("score")
        if isinstance(h, (int, float)):
            holistic.append(float(h))

    median_marks = {name: statistics.median(vs)
                    for name, vs in per_criterion.items() if vs}
    score = _weighted_mean(median_marks, weighted_criteria)
    if score is None:
        # Model gave no per-criterion marks in any sample — fall back to the
        # median of its holistic score field, else 0.
        score = statistics.median(holistic) if holistic else 0.0
    score = max(0.0, min(1.0, score))

    # Representative feedback: from the sample whose own weighted score is
    # closest to the aggregate, so the prose matches the mark the student sees.
    def _sample_score(r: dict) -> float:
        bd = r.get("breakdown") or {}
        m = {n: float(bd[n]) for n in criteria
             if isinstance(bd.get(n), (int, float))}
        s = _weighted_mean(m, weighted_criteria)
        if s is None:
            s = r.get("score") if isinstance(r.get("score"), (int, float)) else score
        return float(s)

    representative = min(parsed_samples, key=lambda r: abs(_sample_score(r) - score))

    return {
        "score": score,
        "breakdown": median_marks,
        "feedback": representative.get("feedback", ""),
        "suggested_rating": _rating_from_score(score),
        "concept_id": concept_id,
        "bloom_level": bloom_level,
        "samples_used": len(parsed_samples),
    }


def schedule_fsrs(concept_id: int, bloom_level: int, rating: int) -> dict:
    out = fsrs_sched.schedule(concept_id, bloom_level, rating)
    return {
        "stability": out.stability,
        "difficulty": out.difficulty,
        "next_review": out.next_review.isoformat(),
        "mastery": out.mastery,
    }


def rag_search(query: str, source: str | None = None, k: int = 5) -> dict:
    return {"results": rag_retriever.search(query, source=source, k=k)}


def pubmed_search(query: str, max_results: int = 5,
                  filter: str | None = None) -> dict:
    return {"results": med_sources.pubmed_search(
        query, max_results=max_results, filter_=filter)}


def radiopaedia_search(query: str, max_results: int = 3,
                       scope: str = "cases") -> dict:
    return {"results": med_sources.radiopaedia_search(
        query, max_results=max_results, scope=scope)}


TOOLS: dict[str, Any] = {
    "query_anatomy": query_anatomy,
    "interpret_imaging": interpret_imaging,
    "lookup_classification": lookup_classification,
    "case_simulator": case_simulator,
    "grade_answer": grade_answer,
    "schedule_fsrs": schedule_fsrs,
    "rag_search": rag_search,
    "pubmed_search": pubmed_search,
    "radiopaedia_search": radiopaedia_search,
}


def schemas_for(tool_names: tuple[str, ...]) -> list[dict]:
    return [s for s in TOOL_SCHEMAS if s["function"]["name"] in tool_names]
