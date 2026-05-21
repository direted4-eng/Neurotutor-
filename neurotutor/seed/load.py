from __future__ import annotations

import json
from pathlib import Path

from ..db.store import connect, init_db

SEED_DIR = Path(__file__).parent


def _slugify(s: str) -> str:
    return s.lower().replace(" ", "_").replace("-", "_")


def seed_concepts() -> int:
    data = json.loads((SEED_DIR / "concepts_anatomy.json").read_text(encoding="utf-8"))
    n = 0
    with connect() as conn:
        for c in data:
            dom = conn.execute(
                "SELECT id FROM domains WHERE code=?", (c["domain"],)
            ).fetchone()
            if not dom:
                continue
            conn.execute(
                """INSERT OR IGNORE INTO concepts(domain_id, name, slug, summary, sources)
                   VALUES (?,?,?,?,?)""",
                (dom["id"], c["name"], c.get("slug") or _slugify(c["name"]),
                 c.get("summary"), json.dumps(c.get("sources", []), ensure_ascii=False)),
            )
            n += 1

        # Initialize mastery rows at Bloom 1..3 for every concept.
        for row in conn.execute("SELECT id FROM concepts").fetchall():
            for lvl in (1, 2, 3):
                conn.execute(
                    """INSERT OR IGNORE INTO mastery(concept_id, bloom_level)
                       VALUES (?,?)""",
                    (row["id"], lvl),
                )
        conn.commit()
    return n


def seed_classifications() -> int:
    data = json.loads((SEED_DIR / "classifications.json").read_text(encoding="utf-8"))
    n = 0
    with connect() as conn:
        for item in data:
            dom = conn.execute(
                "SELECT id FROM domains WHERE code=?", (item["domain"],)
            ).fetchone()
            conn.execute(
                """INSERT OR IGNORE INTO classifications(code, title, domain_id,
                                                          payload, source)
                   VALUES (?,?,?,?,?)""",
                (item["code"], item["title"], dom["id"] if dom else None,
                 json.dumps(item["payload"], ensure_ascii=False), item.get("source")),
            )
            n += 1
        conn.commit()
    return n


def seed_cases() -> int:
    """Load clinical vignettes from cases.json.

    `concept_slugs` in the JSON is resolved to concept ids; unknown slugs are
    silently dropped so the seed survives a partial concepts table.
    """
    path = SEED_DIR / "cases.json"
    if not path.exists():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    n = 0
    with connect() as conn:
        for item in data:
            dom = conn.execute(
                "SELECT id FROM domains WHERE code=?", (item["domain"],)
            ).fetchone()
            slugs = item.get("concept_slugs", []) or []
            concept_ids: list[int] = []
            if slugs:
                placeholders = ",".join("?" * len(slugs))
                rows = conn.execute(
                    f"SELECT id FROM concepts WHERE slug IN ({placeholders})",
                    slugs,
                ).fetchall()
                concept_ids = [r["id"] for r in rows]

            # Deduplicate by title (no unique constraint, but seed should be idempotent).
            existing = conn.execute(
                "SELECT id FROM cases WHERE title=?", (item["title"],)
            ).fetchone()
            if existing:
                continue

            conn.execute(
                """INSERT INTO cases(title, domain_id, difficulty, presentation,
                                      workup, differential, plan, complications,
                                      rubric, concept_ids)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    item["title"],
                    dom["id"] if dom else None,
                    item.get("difficulty", 3),
                    item["presentation"],
                    json.dumps(item.get("workup", {}), ensure_ascii=False),
                    json.dumps(item.get("differential", []), ensure_ascii=False),
                    json.dumps(item.get("plan", {}), ensure_ascii=False),
                    json.dumps(item.get("complications", []), ensure_ascii=False),
                    json.dumps(item.get("rubric", {}), ensure_ascii=False),
                    json.dumps(concept_ids),
                ),
            )
            n += 1
        conn.commit()
    return n


def seed_all() -> dict:
    init_db()
    return {"concepts": seed_concepts(),
            "classifications": seed_classifications(),
            "cases": seed_cases()}
