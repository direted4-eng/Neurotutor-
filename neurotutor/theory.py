"""Remedial theory blocks.

When the resident answers a drill poorly, we generate a structured Russian
theory note on that topic — grounded in the 13-book RAG corpus — to close the
gap. Returned as Markdown, delivered as a Telegram document.
"""
from __future__ import annotations

import logging

from .llm.minimax import MiniMaxClient, extract_text
from .rag import retriever as rag_retriever

log = logging.getLogger(__name__)


def build_theory(topic: str, *, question: str = "", answer: str = "") -> str:
    """Generate a structured Markdown theory note (Russian) for a topic."""
    query = topic or question
    snippets = rag_retriever.search(query, k=6)
    context = "\n\n".join(
        f"[{s['title']} {s['ref']}]\n{s['text'][:600]}" for s in snippets
    ) or "(контекст из учебников не найден — пиши по своим знаниям, аккуратно)"
    sources = sorted({s["title"] for s in snippets})

    sys_prompt = (
        "Ты — профессор нейрохирургии. Напиши структурированный учебный "
        "конспект на русском, чтобы студент-ординатор закрыл пробел по теме. "
        "Формат — Markdown: заголовок, короткое определение, затем разделы с "
        "## подзаголовками и маркированными списками. Обязательно покрой: "
        "ключевые факты, анатомию/классификацию (если применимо), "
        "клиническую и хирургическую значимость, типичные ошибки/подводные "
        "камни. Пиши ёмко, но полно (600–1000 слов), точными терминами. "
        "Пиши СТРОГО на русском языке — без иероглифов и иных нелатинских "
        "вкраплений (латиница допустима только для терминов и сокращений). "
        "Опирайся на приведённый контекст из учебников. В конце — раздел "
        "## Источники со списком использованных книг."
    )
    user_prompt = (
        f"Тема: {topic}\n"
        + (f"Вопрос, с которым студент не справился: {question}\n" if question else "")
        + (f"Его (слабый) ответ: {answer}\n" if answer else "")
        + f"\nКонтекст из учебников:\n{context}\n\n"
        + (f"Доступные источники: {', '.join(sources)}" if sources else "")
    )

    client = MiniMaxClient(timeout=180.0)  # theory notes are long → slow gen
    try:
        resp = client.chat(
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": user_prompt}],
            temperature=0.3,
        )
        body = extract_text(resp).strip()
    finally:
        client.close()

    if sources and "Источник" not in body:
        body += "\n\n## Источники\n" + "\n".join(f"- {s}" for s in sources)
    return body
