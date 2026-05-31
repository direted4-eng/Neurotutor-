#!/usr/bin/env python3
"""Ingest all 13 neurosurgery books into RAG.

Run from /root/neurotutor with venv activated:
    python3 batch_ingest.py [--dry-run] [--only <source>]

Already-ingested sources are skipped automatically (checks rag_chunks table).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("batch_ingest.log"),
    ],
)
log = logging.getLogger(__name__)

BOOKS_DIR = Path("/root/books/neurosurgery")

BOOK_MAP: list[dict] = [
    # (filename, source slug, ref, human title)
    dict(file="greenberg_handbook_neurosurgery.pdf",
         source="greenberg", ref="full",
         title="Greenberg Handbook of Neurosurgery"),
    dict(file="rhoton_cranial_anatomy_surgical_approaches.pdf",
         source="rhoton", ref="cranial",
         title="Rhoton Cranial Anatomy & Surgical Approaches"),
    dict(file="spetzler_color_atlas_microneurosurgery.pdf",
         source="spetzler", ref="full",
         title="Spetzler Color Atlas of Microneurosurgery"),
    dict(file="apuzzo_surgery_human_cerebrum_part1.pdf",
         source="apuzzo_p1", ref="part1",
         title="Apuzzo Surgery of the Human Cerebrum Part 1"),
    dict(file="apuzzo_surgery_human_cerebrum_part3.pdf",
         source="apuzzo_p3", ref="part3",
         title="Apuzzo Surgery of the Human Cerebrum Part 3"),
    dict(file="almefty_meningiomas.pdf",
         source="almefty", ref="meningiomas",
         title="Al-Mefty's Meningiomas"),
    dict(file="color_atlas_cerebral_revascularization.pdf",
         source="revasc_atlas", ref="full",
         title="Color Atlas of Cerebral Revascularization"),
    dict(file="sughrue_glioma_book.pdf",
         source="sughrue_glioma", ref="full",
         title="Sughrue Glioma Surgery"),
    dict(file="brain_anatomy_neuro.pdf",
         source="brain_anatomy", ref="full",
         title="Brain Anatomy for Neurosurgery"),
    dict(file="cerebrovascular_disease.pdf",
         source="cerebrovascular", ref="full",
         title="Cerebrovascular Disease"),
    dict(file="endovascular_stroke.pdf",
         source="endovascular_stroke", ref="full",
         title="Endovascular Treatment of Stroke"),
    dict(file="tsoriev_cervical_intracranial_vessels_anatomy.pdf",
         source="tsoriev", ref="vessels",
         title="Tsoriev: Cervical & Intracranial Vessels Anatomy"),
    dict(file="neurosurgery_board_exam_qa.pdf",
         source="board_qa", ref="full",
         title="Neurosurgery Board Exam Q&A"),
]


def already_ingested(source: str) -> int:
    from neurotutor.db.store import connect
    with connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM rag_chunks WHERE source=?", (source,)
        ).fetchone()
        return row[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="показать что будет проиндексировано, не делать ничего")
    ap.add_argument("--only", metavar="SOURCE",
                    help="проиндексировать только одну книгу по source slug")
    args = ap.parse_args()

    from neurotutor.db.store import init_db
    init_db()

    books = BOOK_MAP
    if args.only:
        books = [b for b in books if b["source"] == args.only]
        if not books:
            log.error("source '%s' не найден в BOOK_MAP", args.only)
            sys.exit(1)

    total_inserted = 0
    for i, book in enumerate(books, 1):
        path = BOOKS_DIR / book["file"]
        source = book["source"]

        if not path.exists():
            log.warning("[%d/%d] ПРОПУСК — файл не найден: %s", i, len(books), path)
            continue

        existing = already_ingested(source)
        if existing:
            log.info("[%d/%d] ПРОПУСК — %s уже есть в БД (%d chunks)",
                     i, len(books), source, existing)
            continue

        size_mb = path.stat().st_size / 1024 / 1024
        log.info("[%d/%d] Начинаю: %s (%.1f МБ)", i, len(books), book["title"], size_mb)

        if args.dry_run:
            log.info("  dry-run: пропускаю")
            continue

        t0 = time.time()
        try:
            from neurotutor.rag.ingest import ingest_pdf
            n = ingest_pdf(path, source=source, ref=book["ref"], title=book["title"])
            elapsed = time.time() - t0
            log.info("  OK: %d chunks за %.0f сек", n, elapsed)
            total_inserted += n
        except Exception as e:
            log.exception("  ОШИБКА при индексации %s: %s", source, e)
            log.info("  Жду 30 сек перед следующей книгой...")
            time.sleep(30)
            continue

        # пауза между книгами чтобы не упереться в RPM
        if i < len(books):
            log.info("  Пауза 5 сек перед следующей книгой...")
            time.sleep(5)

    log.info("Готово. Всего добавлено chunks: %d", total_inserted)


if __name__ == "__main__":
    main()
