"""Daily PubMed digest.

Pulls recent neurosurgery papers (configurable topics), summarizes each
abstract in 1–2 Russian sentences via MiniMax, and returns a Telegram-ready
HTML message. Topics come from NEUROTUTOR_DIGEST_TOPICS (comma-separated);
default is broad neurosurgery — narrow it to your subspecialty later.
"""
from __future__ import annotations

import logging
import os

from .llm.minimax import MiniMaxClient, extract_text
from .rag.sources import pubmed_search

log = logging.getLogger(__name__)

MAX_ARTICLES = 6   # keep the message under Telegram's 4096-char limit

# Bias results toward operative neurosurgery, not general neurology. Text-word
# (not MeSH) so it still matches very recent, not-yet-indexed papers. ANDed
# onto each topic query. Override with NEUROTUTOR_DIGEST_FILTER.
DEFAULT_SURGICAL_FILTER = (
    '(surgery[tiab] OR surgical[tiab] OR resection[tiab] OR craniotomy[tiab] '
    'OR microsurgery[tiab] OR clipping[tiab] OR operative[tiab] '
    'OR "surgical approach"[tiab] OR endovascular[tiab] OR neurosurgical[tiab])'
)


def _topics() -> list[str]:
    raw = os.getenv("NEUROTUTOR_DIGEST_TOPICS", "neurosurgery")
    return [t.strip() for t in raw.split(",") if t.strip()]


def _surgical_filter() -> str:
    return os.getenv("NEUROTUTOR_DIGEST_FILTER", DEFAULT_SURGICAL_FILTER)


def _summarize(article: dict, client: MiniMaxClient) -> str:
    abstract = (article.get("abstract") or "")[:2000]
    if not abstract:
        return ""
    try:
        resp = client.chat(
            [{"role": "system", "content":
              "Ты — нейрохирург. Сожми аннотацию в 1–2 предложения на русском: "
              "что нового и чем полезно для практики. Без воды, без вступлений."},
             {"role": "user", "content": f"{article.get('title','')}\n\n{abstract}"}],
            temperature=0.2,
        )
        return extract_text(resp).strip()
    except Exception:
        log.exception("summarize failed for pmid=%s", article.get("pmid"))
        return ""


def build_digest(per_topic: int = 3, days: int = 14) -> str:
    seen: set[str] = set()
    collected: list[dict] = []
    surgical = _surgical_filter()
    for topic in _topics():
        try:
            arts = pubmed_search(topic, max_results=per_topic,
                                 filter_=surgical, sort="date", reldate=days)
        except Exception:
            log.exception("pubmed_search failed for topic=%s", topic)
            continue
        for a in arts:
            if a.get("pmid") and a["pmid"] not in seen:
                seen.add(a["pmid"])
                collected.append(a)

    collected = collected[:MAX_ARTICLES]
    if not collected:
        return ""

    client = MiniMaxClient()
    try:
        lines = [f"📰 <b>PubMed-дайджест</b> · свежее за {days} дн.\n"]
        for a in collected:
            summ = _summarize(a, client)
            lines.append(f"<b>{a.get('title','').strip()}</b>")
            meta = " ".join(x for x in (a.get("journal", ""), a.get("year", "")) if x)
            if meta:
                lines.append(f"<i>{meta}</i>")
            if summ:
                lines.append(summ)
            lines.append(f'<a href="{a["url"]}">PMID {a["pmid"]}</a>\n')
    finally:
        client.close()

    return "\n".join(lines)
