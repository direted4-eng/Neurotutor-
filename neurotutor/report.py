"""Retrospective competency report.

Reads the accumulated `mastery` (per concept × Bloom) and `responses` (every
graded answer) to surface where the resident stands: per-domain coverage and
mastery, weakest reviewed topics, untouched concepts (known gaps), scenario
performance by kind, and a recent trend. This is the retrospective lens on the
loop — what you know, what's shaky, what you haven't touched yet.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .db.store import connect

KIND_RU = {
    "recall": "Припоминание", "case": "Клинические случаи",
    "surgical_steps": "Ход операции", "emergency": "Экстренные алгоритмы",
    "crisis": "Интраоп. кризисы",
}


def build_report() -> dict:
    now = datetime.now(timezone.utc)
    d7 = (now - timedelta(days=7)).isoformat()
    d14 = (now - timedelta(days=14)).isoformat()

    with connect() as conn:
        domains = [dict(r) for r in conn.execute(
            """SELECT d.code, d.title, d.target_mastery,
                      COUNT(DISTINCT c.id) AS concepts,
                      COUNT(DISTINCT CASE WHEN m.last_review IS NOT NULL
                                          THEN c.id END) AS touched,
                      AVG(CASE WHEN m.last_review IS NOT NULL
                               THEN m.mastery END) AS avg_mastery
               FROM domains d
               LEFT JOIN concepts c ON c.domain_id = d.id
               LEFT JOIN mastery  m ON m.concept_id = c.id
               GROUP BY d.id ORDER BY d.code""").fetchall()]

        weak = [dict(r) for r in conn.execute(
            """SELECT c.name, d.code AS domain, m.bloom_level, m.mastery,
                      m.lapses, m.review_count
               FROM mastery m
               JOIN concepts c ON c.id = m.concept_id
               JOIN domains  d ON d.id = c.domain_id
               WHERE m.last_review IS NOT NULL
               ORDER BY m.mastery ASC, m.lapses DESC LIMIT 8""").fetchall()]

        untouched = [dict(r) for r in conn.execute(
            """SELECT c.name, d.code AS domain
               FROM concepts c JOIN domains d ON d.id = c.domain_id
               WHERE NOT EXISTS (
                   SELECT 1 FROM mastery m
                   WHERE m.concept_id = c.id AND m.last_review IS NOT NULL)
               ORDER BY d.code, c.name""").fetchall()]

        kinds = [dict(r) for r in conn.execute(
            """SELECT role AS kind, COUNT(*) AS n, AVG(grade) AS avg_grade
               FROM responses WHERE grade IS NOT NULL
               GROUP BY role ORDER BY n DESC""").fetchall()]

        total = conn.execute(
            "SELECT COUNT(*) FROM responses WHERE grade IS NOT NULL").fetchone()[0]
        recent = conn.execute(
            "SELECT AVG(grade) FROM responses WHERE grade IS NOT NULL "
            "AND created_at >= ?", (d7,)).fetchone()[0]
        prior = conn.execute(
            "SELECT AVG(grade) FROM responses WHERE grade IS NOT NULL "
            "AND created_at >= ? AND created_at < ?", (d14, d7)).fetchone()[0]

    return {"domains": domains, "weak": weak, "untouched": untouched,
            "kinds": kinds, "total_answers": total,
            "recent_avg": recent, "prior_avg": prior}


def _pct(x: float | None) -> str:
    return f"{round((x or 0) * 100)}%" if x is not None else "—"


def format_report_telegram() -> str:
    r = build_report()
    if not r["total_answers"]:
        return ("📊 <b>Карта компетенций</b>\n\nПока нет оценённых ответов. "
                "Ответь на несколько вопросов дня — и здесь появится анализ "
                "пробелов.")

    lines = ["📊 <b>Карта компетенций</b>\n"]

    # trend
    if r["recent_avg"] is not None:
        arrow = ""
        if r["prior_avg"] is not None:
            delta = (r["recent_avg"] - r["prior_avg"]) * 100
            arrow = f" ({'+' if delta >= 0 else ''}{round(delta)} п.п. к прошлой неделе)"
        lines.append(f"Средняя за 7 дней: <b>{_pct(r['recent_avg'])}</b>{arrow}")
    lines.append(f"Всего оценённых ответов: {r['total_answers']}\n")

    # per-domain coverage + mastery
    lines.append("<b>По доменам</b> (охват · уровень):")
    for d in r["domains"]:
        cov = f"{d['touched']}/{d['concepts']}"
        mast = _pct(d["avg_mastery"])
        tgt = _pct(d["target_mastery"])
        flag = "✅" if (d["avg_mastery"] or 0) >= (d["target_mastery"] or 0.85) else "🔻"
        lines.append(f"  {flag} {d['title']}: {cov} концептов · {mast} (цель {tgt})")

    # weakest
    if r["weak"]:
        lines.append("\n<b>Слабые места</b> (низкий mastery):")
        for w in r["weak"][:5]:
            lines.append(f"  🔻 {w['name']} [{w['domain']}/Блум {w['bloom_level']}] "
                         f"— {_pct(w['mastery'])}"
                         + (f", срывов: {w['lapses']}" if w["lapses"] else ""))

    # scenario performance
    if r["kinds"]:
        lines.append("\n<b>По типам заданий</b>:")
        for k in r["kinds"]:
            lines.append(f"  • {KIND_RU.get(k['kind'], k['kind'])}: "
                         f"{_pct(k['avg_grade'])} ({k['n']})")

    # gaps
    if r["untouched"]:
        names = ", ".join(u["name"] for u in r["untouched"][:8])
        more = f" …и ещё {len(r['untouched']) - 8}" if len(r["untouched"]) > 8 else ""
        lines.append(f"\n<b>Ещё не затронуто</b> ({len(r['untouched'])}): {names}{more}")

    return "\n".join(lines)
