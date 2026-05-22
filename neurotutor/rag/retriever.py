from __future__ import annotations

import numpy as np

from ..db.store import connect
from ..llm.minimax import MiniMaxClient


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


def search(query: str, *, source: str | None = None, k: int = 5) -> list[dict]:
    client = MiniMaxClient()
    try:
        qvec = np.asarray(
            client.embed([query], type_="query")[0], dtype=np.float32
        )
    finally:
        client.close()

    sql = "SELECT id, source, ref, title, text, embedding FROM rag_chunks"
    params: list = []
    if source:
        sql += " WHERE source = ?"
        params.append(source)

    scored: list[tuple[float, dict]] = []
    with connect() as conn:
        for row in conn.execute(sql, params):
            vec = np.frombuffer(row["embedding"], dtype=np.float32)
            if vec.size != qvec.size:
                continue
            scored.append((
                _cosine(qvec, vec),
                {"id": row["id"], "source": row["source"], "ref": row["ref"],
                 "title": row["title"], "text": row["text"]},
            ))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, item in scored[:k]:
        item["score"] = round(score, 4)
        out.append(item)
    return out
