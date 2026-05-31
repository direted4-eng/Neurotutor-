from __future__ import annotations

from pathlib import Path

import typer
from rich import print
from rich.table import Table

from .db.store import init_db
from .seed.load import seed_all
from .session import end as end_session
from .session import plan_new, plan_review, start as start_session, turn

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


@app.command(name="new-concepts")
def new_concepts(limit: int = 5,
                 domain: str = typer.Option("", help="anatomy|pathology|...")) -> None:
    """Показать концепты, которые ещё ни разу не повторялись."""
    rows = plan_new(limit=limit, domain=domain or None)
    if not rows:
        print("[yellow]Новых концептов не осталось — изучены все.[/]")
        return
    t = Table("concept", "domain", "parent", "summary")
    for r in rows:
        t.add_row(r["name"], r["domain"], r["parent"] or "—",
                  (r["summary"] or "")[:60])
    print(t)


@app.command()
def ask(mode: str = typer.Argument(..., help="diagnostic|review|new|case|osce|imaging"),
        message: str = typer.Argument(...),
        persona: str = typer.Option("corvin", help="corvin|lin|plain"),
        no_persist: bool = typer.Option(
            False, "--no-persist",
            help="Не открывать сессию и не писать responses в БД")) -> None:
    """Один turn агента (без интерактивного цикла)."""
    sid = None if no_persist else start_session(mode, notes="cli ask")
    try:
        out = turn(mode, message, persona=persona, session_id=sid)
    finally:
        if sid is not None:
            end_session(sid)
    print(f"[bold]{out['role']} / {out['persona']}[/]: {out['reply']}")
    for step in out["trace"]:
        print(f"  → tool {step['tool']} args={step['args']}")


@app.command()
def drill(user: str = typer.Argument(..., help="telegram user id"),
          n: int = typer.Option(3, help="сколько вопросов сгенерировать")) -> None:
    """Сгенерировать N вопросов дня (pending) для пользователя."""
    from .agent.drill import generate_drill
    qs = generate_drill(user, n=n)
    if not qs:
        print("[yellow]Нет новых вопросов (есть незакрытые или нечего повторять).[/]")
        return
    for q in qs:
        print(f"[bold]{q['concept']}[/] (Блум {q['bloom_level']}): {q['prompt']}")


@app.command()
def report() -> None:
    """Ретроспективная карта компетенций: пробелы, слабые места, тренд."""
    import re
    from .report import format_report_telegram
    print(re.sub(r"<[^>]+>", "", format_report_telegram()))


@app.command()
def personas() -> None:
    """Список доступных персонажей."""
    from .agent.persona import PERSONAS
    t = Table("code", "name", "signature")
    for p in PERSONAS.values():
        t.add_row(p.code, p.name, p.signature or "—")
    print(t)


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
