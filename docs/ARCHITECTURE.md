# Архитектура Neurotutor

## Слои

```
┌─────────────────────────────────────────────────────────┐
│ CLI (typer) — neurotutor/cli.py                         │
├─────────────────────────────────────────────────────────┤
│ Session loop — neurotutor/session.py                    │
│   modes: diagnostic | review | new | case | osce | imaging │
├─────────────────────────────────────────────────────────┤
│ Agent orchestrator (Hermes-style tool loop)             │
│   neurotutor/agent/orchestrator.py                      │
│   ↓ выбирает Role → даёт ей подмножество tools          │
├─────────────────────────────────────────────────────────┤
│ Roles                Tools                              │
│   diagnostician      query_anatomy                      │
│   anatomist          interpret_imaging  (vision)        │
│   clinician          lookup_classification              │
│   radiologist        case_simulator                     │
│   examiner           grade_answer                       │
│                      schedule_fsrs                      │
│                      rag_search                         │
│                      pubmed_search                      │
├─────────────────────────────────────────────────────────┤
│ MiniMax client (text + vision + embeddings)             │
├─────────────────────────────────────────────────────────┤
│ Storage: SQLite                                         │
│   domains, concepts, mastery(concept×bloom),            │
│   classifications, cases, images, sessions, responses,  │
│   rag_chunks                                            │
└─────────────────────────────────────────────────────────┘
```

## Ключевые решения

**Mastery per (concept × bloom_level)** — одна концепция «SAH» имеет до 6 строк mastery. Recall (1) можно довести до 0.95, а Apply (3) держится на 0.4 — и FSRS планирует повторение именно проблемного уровня.

**Stateless agent** — состояние только в SQLite. Перезапуск VPS не теряет прогресс. Агент в одном `run_turn` делает до 6 циклов tool-calling и возвращает финальный ответ.

**Vision по одному изображению** — `interpret_imaging` читает файл, шлёт в MiniMax-VL, бросает буфер. Это ограничение 4 ГБ VPS; не накапливаем картинки в памяти.

**RAG двух уровней** — статика (учебники) и live (PubMed). См. `docs/RAG_SETUP.md`.

**Examiner без RAG** — на OSCE ученик не имеет права заглядывать в книгу, и агент не должен.

## Поток одного хода

1. Пользователь → `session.turn(mode, msg)`
2. По mode выбирается Role (например `clinician` для `case`)
3. `orchestrator.run_turn`: system-prompt роли + tools-схемы только из её списка
4. MiniMax возвращает либо текст (конец), либо `tool_calls`
5. Каждый tool вызывается локально (SQLite/HTTP), результат → обратно в messages
6. До 6 итераций; финальный текст + полный trace возвращаются вызывающему

## Дальнейшие шаги

1. Заполнить `concepts` полным списком (~300 концептов на ординатуру).
2. Собрать 50+ виньеток в `cases` с rubric и concept_ids.
3. Ingest Greenberg → проверить RAG на 20 эталонных вопросах.
4. Добавить BM25-канал (`rag_fts`) и reranking.
5. Подключить кэш PubMed по PMID.
6. Калибровать `_mastery_from_state` под целевые горизонты ординатуры.
