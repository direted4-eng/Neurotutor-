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


def seed_all() -> dict:
    init_db()
    return {"concepts": seed_concepts(),
            "classifications": seed_classifications()}
