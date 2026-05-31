"""RAG retrieval over the 13-book neurosurgery corpus.

The corpus lives in a Qdrant collection (`neurosurgery`, 30k+ chunks,
`paraphrase-multilingual-MiniLM-L12-v2`, 384-dim cosine). The model +
qdrant client are heavy and already wired up in the workspace-tutor
toolchain, so instead of pulling torch into this venv we shell out to the
proven query script and isolate its memory in a short-lived subprocess.

Returns the same contract the agent's `rag_search` tool expects:
    [{id, source, ref, title, text, score}, ...]
"""
from __future__ import annotations

import json
import logging
import subprocess

log = logging.getLogger(__name__)

# System python has sentence-transformers + qdrant access; the venv does not.
_QUERY_PY = "/root/.openclaw/workspace-tutor/scripts/rag_query.py"
_PYTHON = "/usr/bin/python3"
_COLLECTION = "neurosurgery"


def search(query: str, *, source: str | None = None, k: int = 5) -> list[dict]:
    # Over-fetch when filtering by source so post-filter still yields ~k hits.
    limit = k * 4 if source else k
    try:
        proc = subprocess.run(
            [_PYTHON, _QUERY_PY, "--collection", _COLLECTION,
             "--limit", str(limit), query],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        log.warning("rag_query timed out for %r", query)
        return []
    except Exception:
        log.exception("rag_query failed to launch")
        return []

    if proc.returncode != 0:
        # rag_query.py exits 1 with {"error": ...} on no hits — not fatal.
        log.info("rag_query rc=%s: %s", proc.returncode, proc.stderr.strip()[:200])
        return []

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        log.warning("rag_query returned non-JSON: %r", proc.stdout[:200])
        return []

    out: list[dict] = []
    for r in payload.get("results", []):
        src = r.get("source", "?")
        if source and source.lower() not in src.lower() \
                and source.lower() not in str(r.get("book_title", "")).lower():
            continue
        page = r.get("page", "?")
        out.append({
            "id": r.get("chunk_idx", "?"),
            "source": src,
            "ref": f"p.{page}" if page not in ("?", None, "") else "",
            "title": r.get("book_title", src),
            "text": r.get("text", ""),
            "score": r.get("score", 0.0),
        })
        if len(out) >= k:
            break
    return out
