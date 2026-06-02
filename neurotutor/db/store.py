from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ..config import SETTINGS

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _open(db_path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or SETTINGS.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection that auto-closes on context exit.

    `sqlite3.Connection`'s own context manager only commits/rolls back —
    it does not close the connection. With WAL enabled (see schema.sql)
    that leaks .wal/.shm handles until GC. Wrap callers in `with connect()`.
    """
    conn = _open(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path | None = None) -> None:
    conn = _open(db_path)
    try:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        conn.commit()
        _seed_domains(conn)
    finally:
        conn.close()


@contextmanager
def cursor() -> Iterator[sqlite3.Cursor]:
    conn = _open()
    try:
        yield conn.cursor()
        conn.commit()
    finally:
        conn.close()


DOMAINS = [
    ("anatomy", "Нейроанатомия", 0.90),
    ("pathology", "Нейропатология и патофизиология", 0.85),
    ("radiology", "Нейрорадиология", 0.85),
    ("clinical", "Клиника и менеджмент", 0.85),
    ("approaches", "Хирургические доступы", 0.80),
]


def _seed_domains(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    for code, title, target in DOMAINS:
        cur.execute(
            "INSERT OR IGNORE INTO domains(code, title, target_mastery) VALUES (?,?,?)",
            (code, title, target),
        )
    conn.commit()


def load_json_seed(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def slugify(text: str) -> str:
    """Stable slug from a (possibly Cyrillic) topic name. \\w is unicode-aware."""
    s = re.sub(r"[^\w]+", "_", text.strip().lower(), flags=re.UNICODE)
    return s.strip("_")[:120] or "concept"


def get_or_create_concept(name: str, domain_code: str, *,
                          slug: str | None = None,
                          summary: str | None = None) -> int | None:
    """Return the concept id for `slug` (or derived from name), creating it
    under `domain_code` if absent. Returns None if the domain is unknown.

    This lets the concept graph grow beyond the static seed: any topic the
    resident actually engages with (scenario drills, theory notes) becomes a
    tracked concept, so it surfaces in mastery/FSRS and the competency map.
    """
    slug = slug or slugify(name)
    with connect() as conn:
        row = conn.execute("SELECT id FROM concepts WHERE slug=?", (slug,)).fetchone()
        if row:
            return row["id"]
        d = conn.execute("SELECT id FROM domains WHERE code=?",
                         (domain_code,)).fetchone()
        if not d:
            return None
        cur = conn.execute(
            "INSERT INTO concepts(domain_id, name, slug, summary) VALUES (?,?,?,?)",
            (d["id"], name, slug, summary))
        return cur.lastrowid
