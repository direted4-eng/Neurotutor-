"""Migrate NeuroTutor to the full A–M curriculum.

Idempotent. Run from the repo root:

    python3 -m neurotutor.seed.migrate_curriculum

Steps:
  1. Upsert the canonical domain set (DOMAINS in db.store) — codes, titles, targets.
  2. Remap legacy concepts/cases/classifications off the retired codes
     (pathology, clinical) onto the nozology/clinical domains they belong to,
     preserving every FSRS mastery row (we only repoint domain_id).
  3. Load seed/curriculum.json (~170 concepts) — INSERT OR IGNORE by slug.
  4. Initialise mastery rows at Bloom 1..3 for any concept missing them.
  5. Drop the retired domains once they hold nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..db.store import connect, init_db, DOMAINS

SEED_DIR = Path(__file__).parent

# slug -> new domain code (concepts previously under pathology/clinical)
CONCEPT_REMAP = {
    "subarachnoid_hemorrhage": "vascular",
    "glioblastoma": "oncology",
    "emergency_ликворея_с_менингитом_после_операции": "neurocritical",
    "crisis_повреждение_сонной_артерии_при_транссфеноидальном_доступе": "neurocritical",
}

# case title -> new domain code
CASE_REMAP = {
    "Аневризматическое САК передней соединительной артерии": "vascular",
    "Глиобластома левой височной доли": "oncology",
    "Острая субдуральная гематома после падения": "trauma",
    "АВМ задней теменной области, разрыв": "vascular",
    "Идиопатическая нормотензивная гидроцефалия": "hydrocephalus",
    "Травма шейного отдела C5-C6 с тетраплегией": "spine",
    "Менингиома сфеноидального крыла": "oncology",
    "Пинеальная опухоль с обструктивной гидроцефалией": "oncology",
}

# classification code -> new domain code
CLASSIFICATION_REMAP = {
    "gcs": "trauma",
    "hunt_hess": "vascular",
    "fisher": "vascular",
    "mfisher": "vascular",
    "spetzler_martin": "vascular",
    "who_cns_2021": "oncology",
    "asia": "spine",
    "house_brackmann": "functional",
}

RETIRED_DOMAINS = ("pathology", "clinical")


def _domain_id(conn, code: str) -> int | None:
    row = conn.execute("SELECT id FROM domains WHERE code=?", (code,)).fetchone()
    return row["id"] if row else None


def migrate() -> dict:
    # 1) Canonical domains: init_db seeds via INSERT OR IGNORE, then force-update
    #    titles/targets so renamed §A/§B labels take effect on existing rows.
    init_db()
    stats = {"domains_upserted": 0, "concepts_remapped": 0, "cases_remapped": 0,
             "classifications_remapped": 0, "curriculum_loaded": 0,
             "mastery_initialised": 0, "domains_dropped": 0}

    with connect() as conn:
        for code, title, target in DOMAINS:
            conn.execute(
                """INSERT INTO domains(code, title, target_mastery) VALUES (?,?,?)
                   ON CONFLICT(code) DO UPDATE SET title=excluded.title,
                                                   target_mastery=excluded.target_mastery""",
                (code, title, target),
            )
            stats["domains_upserted"] += 1

        # 2) Remap legacy rows off retired codes.
        for slug, dcode in CONCEPT_REMAP.items():
            did = _domain_id(conn, dcode)
            if did:
                cur = conn.execute(
                    "UPDATE concepts SET domain_id=? WHERE slug=?", (did, slug))
                stats["concepts_remapped"] += cur.rowcount
        for title, dcode in CASE_REMAP.items():
            did = _domain_id(conn, dcode)
            if did:
                cur = conn.execute(
                    "UPDATE cases SET domain_id=? WHERE title=?", (did, title))
                stats["cases_remapped"] += cur.rowcount
        for code, dcode in CLASSIFICATION_REMAP.items():
            did = _domain_id(conn, dcode)
            if did:
                cur = conn.execute(
                    "UPDATE classifications SET domain_id=? WHERE code=?", (did, code))
                stats["classifications_remapped"] += cur.rowcount

        # 3) Load curriculum.json concepts.
        data = json.loads((SEED_DIR / "curriculum.json").read_text(encoding="utf-8"))
        for c in data:
            did = _domain_id(conn, c["domain"])
            if not did:
                continue
            cur = conn.execute(
                """INSERT OR IGNORE INTO concepts(domain_id, name, slug, summary, sources)
                   VALUES (?,?,?,?,?)""",
                (did, c["name"], c["slug"], c.get("summary"),
                 json.dumps(c.get("sources", []), ensure_ascii=False)),
            )
            stats["curriculum_loaded"] += cur.rowcount

        # 4) Mastery rows at Bloom 1..3 for every concept lacking them.
        for row in conn.execute("SELECT id FROM concepts").fetchall():
            for lvl in (1, 2, 3):
                cur = conn.execute(
                    "INSERT OR IGNORE INTO mastery(concept_id, bloom_level) VALUES (?,?)",
                    (row["id"], lvl))
                stats["mastery_initialised"] += cur.rowcount

        # 5) Drop retired domains only if nothing references them anymore.
        for code in RETIRED_DOMAINS:
            did = _domain_id(conn, code)
            if did is None:
                continue
            refs = (
                conn.execute("SELECT COUNT(*) n FROM concepts WHERE domain_id=?", (did,)).fetchone()["n"]
                + conn.execute("SELECT COUNT(*) n FROM cases WHERE domain_id=?", (did,)).fetchone()["n"]
                + conn.execute("SELECT COUNT(*) n FROM classifications WHERE domain_id=?", (did,)).fetchone()["n"]
            )
            if refs == 0:
                conn.execute("DELETE FROM domains WHERE id=?", (did,))
                stats["domains_dropped"] += 1
            else:
                print(f"  ! domain '{code}' still has {refs} refs — kept")

        conn.commit()
    return stats


if __name__ == "__main__":
    s = migrate()
    print("curriculum migration complete:")
    for k, v in s.items():
        print(f"  {k:24} {v}")
