"""Export a JSON snapshot of mastery for the Live-maps dashboard.

Why a file snapshot instead of a live API:
    The dashboard backend (kiske-api) runs in its own Docker container and
    serves ``/root/eugene_life/static`` as static files (volume-mounted). By
    writing ``study.json`` into that directory we expose it at
    ``/static/study.json`` with **no** cross-container DB access and **no**
    image rebuild. The two systems stay decoupled, and the last good snapshot
    keeps serving even if NeuroTutor is down — the most robust wiring.

The snapshot is regenerated after every graded answer (see
``agent.drill.answer_pending``) and on a cron backstop, so it tracks the real
mastery map rather than the hard-coded placeholders the front-end shipped with.

Mastery numbers come straight from the canonical ``mastery.mastery`` column
(FSRS-derived, see ``fsrs.scheduler._mastery_from_state``) — we do not invent a
second metric. ``coverage`` (reviewed concepts / total) is added so early
progress is visible while mastery is still ramping from near-zero.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ..db.store import connect

log = logging.getLogger(__name__)

# Where the dashboard backend serves static files from. Overridable so the
# snapshot path isn't hard-wired to one deployment.
DEFAULT_OUTPUT = "/root/eugene_life/static/study.json"


def _output_path() -> Path:
    return Path(os.getenv("NEUROTUTOR_DASHBOARD_JSON", DEFAULT_OUTPUT))


def build_snapshot(now: datetime | None = None) -> dict:
    """Aggregate the mastery map into a dashboard-ready dict (per domain)."""
    now = now or datetime.now(timezone.utc)
    now_iso = now.isoformat()

    domains: list[dict] = []
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT d.id, d.code, d.title, d.target_mastery,
                   COUNT(DISTINCT c.id)                                 AS concepts,
                   COUNT(DISTINCT CASE WHEN m.review_count > 0
                                       THEN c.id END)                   AS reviewed,
                   AVG(m.mastery)                                       AS avg_mastery,
                   SUM(CASE WHEN m.next_review IS NOT NULL
                            AND m.next_review <= ? THEN 1 ELSE 0 END)   AS due
            FROM domains d
            LEFT JOIN concepts c ON c.domain_id = d.id
            LEFT JOIN mastery  m ON m.concept_id = c.id
            GROUP BY d.id
            ORDER BY d.id
            """,
            (now_iso,),
        ).fetchall()

    for r in rows:
        concepts = r["concepts"] or 0
        avg_mastery = r["avg_mastery"] or 0.0
        domains.append({
            "code":        r["code"],
            "title":       r["title"],
            "mastery_pct": round(avg_mastery * 100),
            "target_pct":  round((r["target_mastery"] or 0.0) * 100),
            "concepts":    concepts,
            "reviewed":    r["reviewed"] or 0,
            "due":         r["due"] or 0,
        })

    # Per-concept rows feed the knowledge-map page (/static/knowledge/):
    # it joins this list client-side, so the map stays live with zero extra
    # wiring. 183 concepts ≈ 25 KB — cheap enough to ship in every snapshot.
    concepts: list[dict] = []
    with connect() as conn:
        crows = conn.execute(
            """
            SELECT c.name, c.slug, d.code AS domain,
                   AVG(m.mastery)                                  AS avg_mastery,
                   MAX(m.review_count)                             AS reviews,
                   SUM(CASE WHEN m.next_review IS NOT NULL
                            AND m.next_review <= ? THEN 1 ELSE 0 END) AS due
            FROM concepts c
            JOIN domains d ON d.id = c.domain_id
            LEFT JOIN mastery m ON m.concept_id = c.id
            GROUP BY c.id
            ORDER BY d.id, c.id
            """,
            (now_iso,),
        ).fetchall()
    for r in crows:
        concepts.append({
            "name":        r["name"],
            "slug":        r["slug"],
            "domain":      r["domain"],
            "mastery_pct": round((r["avg_mastery"] or 0.0) * 100),
            "reviewed":    bool(r["reviews"]),
            "due":         r["due"] or 0,
        })

    total_concepts = sum(d["concepts"] for d in domains)
    total_reviewed = sum(d["reviewed"] for d in domains)
    total_due = sum(d["due"] for d in domains)
    # Overall mastery is concept-weighted so big domains aren't drowned out by
    # tiny ones; domains with no concepts contribute nothing.
    weighted = sum(d["mastery_pct"] * d["concepts"] for d in domains)
    overall_pct = round(weighted / total_concepts) if total_concepts else 0

    return {
        "generated_at": now_iso,
        "overall": {
            "mastery_pct": overall_pct,
            "concepts":    total_concepts,
            "reviewed":    total_reviewed,
            "due":         total_due,
        },
        "domains": domains,
        "concepts": concepts,
    }


def write_snapshot(path: Path | None = None) -> Path | None:
    """Build and atomically write the snapshot. Best-effort: never raises.

    Returns the path written, or ``None`` on failure (e.g. the dashboard's
    static dir isn't mounted on this host). Callers treat a miss as harmless —
    the dashboard just keeps serving the previous snapshot.
    """
    path = path or _output_path()
    try:
        data = build_snapshot()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replace so the dashboard never reads a half-written file.
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        log.info("dashboard snapshot written: %s (overall %d%%)",
                 path, data["overall"]["mastery_pct"])
        return path
    except Exception:
        log.exception("dashboard snapshot failed")
        return None


if __name__ == "__main__":
    import sys

    # Quiet by default so the hourly cron backstop doesn't dump 25 KB of JSON
    # into cron.log each run. Pass -v / --print for the full snapshot.
    verbose = any(a in ("-v", "--print") for a in sys.argv[1:])
    out = write_snapshot()
    if out:
        data = build_snapshot()
        print(f"wrote {out} (overall {data['overall']['mastery_pct']}%, "
              f"{data['overall']['concepts']} concepts)")
        if verbose:
            print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print("snapshot failed (see log)")
        sys.exit(1)
