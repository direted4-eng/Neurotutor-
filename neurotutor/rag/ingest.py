"""PDF/EPUB ingestion for Greenberg/Youmans/Rhoton.

Chunks 500–800 tokens with overlap. Embeddings computed with MiniMax in
batches. Vectors stored as float32 BLOBs in rag_chunks. sqlite-vss is
loaded if available; otherwise we use brute-force cosine at query time.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import numpy as np

from ..db.store import connect
from ..llm.minimax import MiniMaxClient

log = logging.getLogger(__name__)

TARGET_TOKENS = 600
OVERLAP = 120
BATCH_SIZE = 16          # smaller batches — easier on RPM limits
BATCH_DELAY = 2.0        # seconds between embed batches
RETRY_DELAYS = (5, 15, 30, 60)  # backoff on rate-limit


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
    return [c for c in chunks if c.strip()]


def _extract_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _embed_with_retry(client: MiniMaxClient, texts: list[str]) -> list[list[float]]:
    """Embed with exponential backoff on rate-limit (status_code 1002)."""
    for attempt, wait in enumerate((*RETRY_DELAYS, None)):
        result = client.embed(texts)
        if result:
            return result
        # empty result = rate-limit or transient error
        if wait is None:
            raise RuntimeError(f"embed failed after {len(RETRY_DELAYS)+1} retries")
        log.warning("embed returned empty (rate limit?), retry in %ds", wait)
        time.sleep(wait)
    return []  # unreachable


def ingest_pdf(path: Path, source: str, ref: str | None = None,
               title: str | None = None) -> int:
    """Ingest one PDF. Returns number of chunks inserted."""
    text = _extract_pdf(path)
    chunks = _split_into_chunks(text)
    log.info("ingesting %s: %d chunks", path.name, len(chunks))

    client = MiniMaxClient()
    inserted = 0
    try:
        with connect() as conn:
            for i in range(0, len(chunks), BATCH_SIZE):
                part = chunks[i:i + BATCH_SIZE]
                vectors = _embed_with_retry(client, part)
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
                conn.commit()  # commit per batch — resumable on crash
                if i + BATCH_SIZE < len(chunks):
                    time.sleep(BATCH_DELAY)
                log.info("  %d/%d chunks done", min(i + BATCH_SIZE, len(chunks)),
                         len(chunks))
    finally:
        client.close()
    return inserted
