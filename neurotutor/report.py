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

# Drill task-types (responses.role = kind). Distinct namespace from the
# conversational tutor roles below — they were being mixed in one section.
KIND_RU = {
    "recall": "Припоминание", "case": "Клинические случаи",
    "surgical_steps": "Ход операции", "emergency": "Экстренные алгоритмы",
    "crisis": "Интраоп. кризисы",
}

# Conversational tutor roles (responses.role = role name from ROLE_BY_MODE).
ROLE_RU = {
    "anatomist": "Анатомия", "clinician": "Клиника", "radiologist": "Радиология",
    "examiner": "Экзамен (OSCE)", "diagnostician": "Диагностика",
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

        # Program-wide progress: each concept is "touched" once any Bloom level
        # has a graded review, and "mastered" once its mean reviewed mastery
        # reaches its domain's target. Untouched concepts have avg_m = NULL, so
        # the `>= tgt` test is false for them (NULL comparison) — as intended.
        program = dict(conn.execute(
            """SELECT COUNT(*) AS total,
                      SUM(touched)  AS touched,
                      SUM(CASE WHEN avg_m >= tgt THEN 1 ELSE 0 END) AS mastered
               FROM (
                   SELECT c.id, d.target_mastery AS tgt,
                          MAX(CASE WHEN m.last_review IS NOT NULL
                                   THEN 1 ELSE 0 END) AS touched,
                          AVG(CASE WHEN m.last_review IS NOT NULL
                                   THEN m.mastery END) AS avg_m
                   FROM concepts c
                   JOIN domains d ON d.id = c.domain_id
                   LEFT JOIN mastery m ON m.concept_id = c.id
                   GROUP BY c.id)""").fetchone())

        total = conn.execute(
            "SELECT COUNT(*) FROM responses WHERE grade IS NOT NULL").fetchone()[0]
        recent = conn.execute(
            "SELECT AVG(grade) FROM responses WHERE grade IS NOT NULL "
            "AND created_at >= ?", (d7,)).fetchone()[0]
        prior = conn.execute(
            "SELECT AVG(grade) FROM responses WHERE grade IS NOT NULL "
            "AND created_at >= ? AND created_at < ?", (d14, d7)).fetchone()[0]

    return {"domains": domains, "weak": weak, "untouched": untouched,
            "kinds": kinds, "total_answers": total, "program": program,
            "recent_avg": recent, "prior_avg": prior}


def _pct(x: float | None) -> str:
    return f"{round((x or 0) * 100)}%" if x is not None else "—"


def _bar(done: int, total: int, width: int = 10) -> str:
    """Ten-cell progress bar; rounds to nearest cell but never shows a full
    bar unless the count is actually complete."""
    frac = (done / total) if total else 0.0
    filled = round(frac * width)
    if filled == width and done < total:
        filled = width - 1
    if filled == 0 and done > 0:
        filled = 1
    return "▰" * filled + "▱" * (width - filled)


def format_report_telegram() -> str:
    r = build_report()
    if not r["total_answers"]:
        return ("📊 <b>Карта компетенций</b>\n\nПока нет оценённых ответов. "
                "Ответь на несколько вопросов дня — и здесь появится анализ "
                "пробелов.")

    lines = ["📊 <b>Карта компетенций</b>\n"]

    # program-wide progress: headline scale of how much of the whole
    # neurosurgery program has been touched vs actually mastered.
    p = r["program"]
    if p and p["total"]:
        tot = p["total"]
        tch = p["touched"] or 0
        mst = p["mastered"] or 0
        lines.append("🎓 <b>Освоение программы</b>")
        lines.append(f"{_bar(tch, tot)}  Затронуто: "
                     f"<b>{round(tch / tot * 100)}%</b> ({tch}/{tot})")
        lines.append(f"{_bar(mst, tot)}  Освоено:   "
                     f"<b>{round(mst / tot * 100)}%</b> ({mst}/{tot})\n")

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

    # performance split by namespace: drill task-types vs conversational roles
    drill_kinds = [k for k in r["kinds"] if k["kind"] in KIND_RU]
    chat_roles = [k for k in r["kinds"] if k["kind"] in ROLE_RU]
    other = [k for k in r["kinds"]
             if k["kind"] not in KIND_RU and k["kind"] not in ROLE_RU]

    if drill_kinds:
        lines.append("\n<b>По типам заданий</b> (дрилл):")
        for k in drill_kinds:
            lines.append(f"  • {KIND_RU[k['kind']]}: "
                         f"{_pct(k['avg_grade'])} ({k['n']})")
    if chat_roles:
        lines.append("\n<b>По ролям в диалоге</b>:")
        for k in chat_roles:
            lines.append(f"  • {ROLE_RU[k['kind']]}: "
                         f"{_pct(k['avg_grade'])} ({k['n']})")
    if other:
        lines.append("\n<b>Прочее</b>:")
        for k in other:
            lines.append(f"  • {k['kind']}: {_pct(k['avg_grade'])} ({k['n']})")

    # gaps
    if r["untouched"]:
        names = ", ".join(u["name"] for u in r["untouched"][:8])
        more = f" …и ещё {len(r['untouched']) - 8}" if len(r["untouched"]) > 8 else ""
        lines.append(f"\n<b>Ещё не затронуто</b> ({len(r['untouched'])}): {names}{more}")

    return "\n".join(lines)
