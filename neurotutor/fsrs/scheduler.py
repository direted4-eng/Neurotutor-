from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

try:
    from fsrs import Card, FSRS, Rating, State
    _HAS_FSRS = True
except Exception:  # pragma: no cover - allow import without lib installed
    _HAS_FSRS = False

from ..db.store import connect


@dataclass
class ReviewOutcome:
    stability: float
    difficulty: float
    next_review: datetime
    mastery: float


def _mastery_from_state(stability: float, difficulty: float, lapses: int) -> float:
    """Heuristic mastery in [0,1] derived from FSRS state.

    Stability tracks retention horizon (days); difficulty 1..10. We map
    log-stability with a difficulty/lapses penalty.
    """
    import math

    s = max(stability, 0.1)
    base = math.log10(s + 1) / math.log10(60 + 1)  # 60d ≈ mastered
    penalty = (difficulty - 1) / 18 + min(lapses, 10) * 0.02
    return max(0.0, min(1.0, base - penalty))


def schedule(
    concept_id: int,
    bloom_level: int,
    rating: int,
    now: datetime | None = None,
) -> ReviewOutcome:
    """Update FSRS state for a (concept, bloom) pair after a review.

    rating: 1=again, 2=hard, 3=good, 4=easy.
    """
    if not _HAS_FSRS:
        raise RuntimeError("fsrs package not installed")

    now = now or datetime.now(timezone.utc)
    fsrs = FSRS()

    with connect() as conn:
        row = conn.execute(
            """SELECT stability, difficulty, last_review, review_count, lapses
               FROM mastery WHERE concept_id=? AND bloom_level=?""",
            (concept_id, bloom_level),
        ).fetchone()

        if row and row["last_review"]:
            card = Card(
                stability=row["stability"] or 0.0,
                difficulty=row["difficulty"] or 5.0,
                state=State.Review,
                last_review=datetime.fromisoformat(row["last_review"]),
                reps=row["review_count"] or 0,
                lapses=row["lapses"] or 0,
            )
        else:
            card = Card()

        card, _log = fsrs.review_card(card, Rating(rating), now)

        mastery = _mastery_from_state(card.stability, card.difficulty, card.lapses)
        conn.execute(
            """INSERT INTO mastery(concept_id, bloom_level, mastery, stability,
                                    difficulty, last_review, next_review,
                                    review_count, lapses)
               VALUES (?,?,?,?,?,?,?,?,?)
               ON CONFLICT(concept_id, bloom_level) DO UPDATE SET
                    mastery=excluded.mastery,
                    stability=excluded.stability,
                    difficulty=excluded.difficulty,
                    last_review=excluded.last_review,
                    next_review=excluded.next_review,
                    review_count=excluded.review_count,
                    lapses=excluded.lapses""",
            (
                concept_id,
                bloom_level,
                mastery,
                card.stability,
                card.difficulty,
                now.isoformat(),
                card.due.isoformat(),
                card.reps,
                card.lapses,
            ),
        )
        conn.commit()

    return ReviewOutcome(
        stability=card.stability,
        difficulty=card.difficulty,
        next_review=card.due,
        mastery=mastery,
    )


def due_today(limit: int = 20) -> list[dict]:
    now = datetime.now(timezone.utc).isoformat()
    with connect() as conn:
        rows = conn.execute(
            """SELECT m.concept_id, m.bloom_level, m.mastery, m.next_review,
                      c.name, c.slug, d.code AS domain
               FROM mastery m
               JOIN concepts c ON c.id = m.concept_id
               JOIN domains  d ON d.id = c.domain_id
               WHERE m.next_review IS NOT NULL AND m.next_review <= ?
               ORDER BY m.next_review ASC LIMIT ?""",
            (now, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def pick_new(limit: int = 2, domain: str | None = None) -> list[dict]:
    """Pick concepts that have never been reviewed at any Bloom level.

    A "new" concept is one whose mastery rows all have last_review IS NULL
    (or has no mastery rows at all). Ordered by concept id for stability.
    """
    sql = """
        SELECT c.id AS concept_id, c.name, c.slug, c.summary, c.sources,
               d.code AS domain, p.name AS parent
        FROM concepts c
        JOIN domains d ON d.id = c.domain_id
        LEFT JOIN concepts p ON p.id = c.parent_id
        WHERE NOT EXISTS (
            SELECT 1 FROM mastery m
            WHERE m.concept_id = c.id AND m.last_review IS NOT NULL
        )
    """
    params: list = []
    if domain:
        sql += " AND d.code = ?"
        params.append(domain)
    sql += " ORDER BY c.id ASC LIMIT ?"
    params.append(limit)
    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]
