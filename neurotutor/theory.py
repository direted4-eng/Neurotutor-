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


# --- Scenario framings: how to phrase the reference algorithm per kind. ---
_ALGO_FRAMING = {
    "surgical_steps": "эталонный ПОШАГОВЫЙ ход операции / хирургического доступа",
    "emergency": "эталонный ПОШАГОВЫЙ алгоритм неотложных действий",
    "crisis": "эталонный ПОШАГОВЫЙ алгоритм действий при интраоперационном кризисе",
}


def build_algorithm(topic: str, *, kind: str = "", question: str = "",
                    answer: str = "") -> str:
    """Generate the correct step-by-step algorithm for a scenario topic (RAG).

    Used as remediation for the algorithm modes (surgical_steps / emergency /
    crisis): instead of a general theory note, the resident gets the *reference
    algorithm itself* — ordered steps with the key rationale and danger points.
    """
    query = topic or question
    snippets = rag_retriever.search(query, k=6)
    context = "\n\n".join(
        f"[{s['title']} {s['ref']}]\n{s['text'][:600]}" for s in snippets
    ) or "(контекст из учебников не найден — пиши по своим знаниям, аккуратно)"
    sources = sorted({s["title"] for s in snippets})

    framing = _ALGO_FRAMING.get(kind, "эталонный ПОШАГОВЫЙ клинический алгоритм")

    sys_prompt = (
        "Ты — профессор нейрохирургии. Дай " + framing + " по теме — строго "
        "по шагам, ПРОНУМЕРОВАННЫМ списком, в правильном порядке. Для каждого "
        "шага кратко укажи, что делается и зачем (ключевое обоснование). "
        "Отдельно отметь критические точки и опасные зоны. Без воды — только "
        "сам алгоритм. Пиши СТРОГО на русском, точными терминами (латиница — "
        "только для терминов и сокращений; без иероглифов). Формат — Markdown: "
        "заголовок, нумерованный список шагов (при необходимости короткие "
        "подпункты), затем раздел ## Опасные моменты и раздел ## Источники со "
        "списком использованных книг."
    )
    user_prompt = (
        f"Тема: {topic}\n"
        + (f"Задание, с которым студент не справился: {question}\n" if question else "")
        + (f"Его (слабый) ответ: {answer}\n" if answer else "")
        + f"\nКонтекст из учебников:\n{context}\n\n"
        + (f"Доступные источники: {', '.join(sources)}" if sources else "")
    )

    client = MiniMaxClient(timeout=180.0)
    try:
        resp = client.chat(
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": user_prompt}],
            temperature=0.2,
        )
        body = extract_text(resp).strip()
    finally:
        client.close()

    if sources and "Источник" not in body:
        body += "\n\n## Источники\n" + "\n".join(f"- {s}" for s in sources)
    return body
