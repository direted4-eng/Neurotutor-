from __future__ import annotations

from pathlib import Path

import typer
from rich import print
from rich.table import Table

from .db.store import init_db
from .seed.load import seed_all
from .session import plan_review, turn

app = typer.Typer(help="Neurotutor — нейроанатом-репетитор")


@app.command()
def init() -> None:
    """Создать БД и загрузить seed-концепты + классификации."""
    init_db()
    stats = seed_all()
    print(f"[green]OK[/] seeded: {stats}")


@app.command()
def due() -> None:
    """Показать карточки, по которым подошёл срок повторения."""
    rows = plan_review()
    t = Table("concept", "domain", "bloom", "mastery", "next_review")
    for r in rows:
        t.add_row(r["name"], r["domain"], str(r["bloom_level"]),
                  f"{r['mastery']:.2f}", r["next_review"])
    print(t)


@app.command()
def ask(mode: str = typer.Argument(..., help="diagnostic|review|new|case|osce|imaging"),
        message: str = typer.Argument(...)) -> None:
    """Один turn агента (без интерактивного цикла)."""
    out = turn(mode, message)
    print(f"[bold]{out['role']}[/]: {out['reply']}")
    for step in out["trace"]:
        print(f"  → tool {step['tool']} args={step['args']}")


@app.command()
def ingest(path: Path,
           source: str = typer.Option(..., help="greenberg|youmans|rhoton|article"),
           ref: str = typer.Option("", help="chapter / section / PMID")) -> None:
    """Ingest a PDF into the RAG store."""
    from .rag.ingest import ingest_pdf
    n = ingest_pdf(path, source=source, ref=ref or None)
    print(f"[green]ingested[/] {n} chunks from {path.name}")


@app.command()
def pubmed(query: str, k: int = 5,
           filter: str = typer.Option("", help="e.g. 'review[pt]'")) -> None:
    from .rag.sources import pubmed_search
    for art in pubmed_search(query, max_results=k, filter_=filter or None):
        print(f"[bold]{art['pmid']}[/] {art['title']} ({art['year']})")
        print(f"  {art['journal']}")
        print(f"  {art['url']}")


if __name__ == "__main__":
    app()
