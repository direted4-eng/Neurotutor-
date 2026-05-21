"""PDF/EPUB ingestion for Greenberg/Youmans/Rhoton.

Chunks 500–800 tokens with overlap. Embeddings computed with MiniMax in
batches. Vectors stored as float32 BLOBs in rag_chunks. sqlite-vss is
loaded if available; otherwise we use brute-force cosine at query time.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from ..db.store import connect
from ..llm.minimax import MiniMaxClient

log = logging.getLogger(__name__)

TARGET_TOKENS = 600
OVERLAP = 120


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _split_into_chunks(text: str) -> list[str]:
    # Sentence-ish split; keep paragraphs intact when possible.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for s in sentences:
        t = _approx_tokens(s)
        if buf_tokens + t > TARGET_TOKENS and buf:
            chunks.append(" ".join(buf))
            # overlap: keep tail
            tail: list[str] = []
            tail_tokens = 0
            for s2 in reversed(buf):
                tail_tokens += _approx_tokens(s2)
                tail.append(s2)
                if tail_tokens >= OVERLAP:
                    break
            buf = list(reversed(tail))
            buf_tokens = sum(_approx_tokens(x) for x in buf)
        buf.append(s)
        buf_tokens += t
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def _extract_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def ingest_pdf(path: Path, source: str, ref: str | None = None,
               title: str | None = None) -> int:
    """Ingest one PDF. Returns chunk count."""
    text = _extract_pdf(path)
    chunks = _split_into_chunks(text)
    log.info("ingesting %s: %d chunks", path.name, len(chunks))

    client = MiniMaxClient()
    inserted = 0
    try:
        # Embed in batches of 32 to stay within memory/quotas.
        batch = 32
        with connect() as conn:
            for i in range(0, len(chunks), batch):
                part = chunks[i:i + batch]
                vectors = client.embed(part)
                for chunk, vec in zip(part, vectors):
                    blob = np.asarray(vec, dtype=np.float32).tobytes()
                    conn.execute(
                        """INSERT INTO rag_chunks(source, ref, title, text,
                                                  tokens, embedding)
                           VALUES (?,?,?,?,?,?)""",
                        (source, ref, title or path.stem, chunk,
                         _approx_tokens(chunk), blob),
                    )
                    inserted += 1
            conn.commit()
    finally:
        client.close()
    return inserted
