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
import random
import re
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
                        "anatomy", "pathology", "radiology", "clinical", "approaches"
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


def grade_answer(
    prompt: str,
    answer: str,
    concept_id: int | None = None,
    bloom_level: int | None = None,
    rubric: dict | None = None,
) -> dict:
    """Grade with the text model. Rubric defaults vary by Bloom level."""
    from ..llm.minimax import MiniMaxClient, extract_text

    default_rubric = {
        1: ["factual_correctness", "completeness"],
        2: ["explanation_quality", "use_of_correct_terms"],
        3: ["application_to_scenario", "step_correctness"],
        4: ["differentiation", "reasoning_depth"],
        5: ["evidence_quality", "trade_off_analysis"],
        6: ["originality", "feasibility"],
    }
    criteria = (rubric and rubric.get("criteria")) or \
               default_rubric.get(bloom_level or 1, ["correctness"])

    client = MiniMaxClient()
    try:
        sys = (
            "Ты строгий экзаменатор по нейрохирургии. Оцени ответ по критериям. "
            "Верни ТОЛЬКО сырой JSON, без markdown, без ``` блоков, без пояснений. "
            "Формат: {\"score\": 0..1, \"breakdown\": {criterion: 0..1}, "
            "\"feedback\": str, \"suggested_rating\": 1..4}. "
            "1=again, 2=hard, 3=good, 4=easy."
        )
        user = json.dumps({"prompt": prompt, "answer": answer,
                           "criteria": criteria}, ensure_ascii=False)
        resp = client.chat(
            [{"role": "system", "content": sys},
             {"role": "user", "content": user}],
            temperature=0.0,
        )
        text = extract_text(resp)
    finally:
        client.close()

    result = _parse_json_loose(text)
    if result is None:
        log.warning("grade_answer: failed to parse JSON; raw response: %r", text[:500])
        result = {"score": 0.0, "breakdown": {}, "feedback": text,
                  "suggested_rating": 1, "parse_error": True}
    result["concept_id"] = concept_id
    result["bloom_level"] = bloom_level
    return result


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


TOOLS: dict[str, Any] = {
    "query_anatomy": query_anatomy,
    "interpret_imaging": interpret_imaging,
    "lookup_classification": lookup_classification,
    "case_simulator": case_simulator,
    "grade_answer": grade_answer,
    "schedule_fsrs": schedule_fsrs,
    "rag_search": rag_search,
    "pubmed_search": pubmed_search,
}


def schemas_for(tool_names: tuple[str, ...]) -> list[dict]:
    return [s for s in TOOL_SCHEMAS if s["function"]["name"] in tool_names]
