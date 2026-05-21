# Neurotutor

AI-тьютор по нейроанатомии и нейрохирургии: LLM-агент с инструментами + интервальные повторения (FSRS) + RAG по учебникам и PubMed. Mastery отслеживается на парах **концепт × уровень Блума** — знать определение и применять в кейсе считаются отдельно.

> ⚕️ Образовательный инструмент для ординаторов и студентов. Не предназначен для принятия клинических решений.

## Стек

- **Python 3.11+**, CLI на `typer` + `rich`
- **LLM:** MiniMax (text + vision + embeddings) через `httpx`
- **БД:** SQLite (schema в `neurotutor/db/schema.sql`)
- **FSRS:** библиотека `fsrs` для планирования повторений
- **RAG:** локальный векторный поиск + PubMed E-utilities

## Установка

```bash
git clone <repo-url>
cd Neurotutor

python -m venv .venv
# Linux/Mac:
. .venv/bin/activate
# Windows:
. .venv\Scripts\Activate.ps1

pip install -r requirements.txt
cp .env.example .env   # затем отредактируйте — см. ниже
```

## Настройка `.env`

Минимум для запуска:

```
MINIMAX_API_KEY=<ваш ключ>
MINIMAX_GROUP_ID=<group id>
PUBMED_EMAIL=you@example.com    # NCBI требует контактный email
```

Опционально:
- `PUBMED_API_KEY` — поднимает лимит NCBI с 3/с до 10/с
- `NEUROTUTOR_DB`, `NEUROTUTOR_RAG_DIR` — кастомные пути
- `MINIMAX_TEXT_MODEL` / `MINIMAX_VISION_MODEL` / `MINIMAX_EMBED_MODEL`

## Первый запуск

```bash
# 1. Создать БД и загрузить seed-данные (домены, концепты, классификации)
python -m neurotutor.cli init

# 2. Проверить, что персонажи на месте
python -m neurotutor.cli personas

# 3. Один вопрос агенту (без интерактивного цикла)
python -m neurotutor.cli ask diagnostic "Опишите анатомию виллизиева круга"
```

## CLI: команды

| Команда | Что делает |
|---|---|
| `init` | Создать схему БД, загрузить домены, концепты, классификации |
| `due` | Показать карточки, по которым подошёл срок повторения |
| `ask <mode> <message>` | Один turn агента в заданном режиме |
| `personas` | Список персонажей (Корвин, Линь, plain) |
| `ingest <path>` | Проиндексировать PDF учебника в RAG |
| `pubmed <query>` | Поиск по PubMed |

### Режимы (`mode`)

- `diagnostic` — 30 адаптивных вопросов для оценки mastery с нуля
- `review` — повторение карточек по FSRS-расписанию
- `new` — введение 1-2 новых концептов
- `case` — клинический case-based learning по 6 шагам Гарварда
- `osce` — экзаменационная OSCE-станция с таймером
- `imaging` — разбор КТ/МРТ/ангио

### Персонажи

- `corvin` (по умолчанию) — Профессор Корвин, сухой строгий ментор
- `lin` — Доктор Линь, мягкий объясняющий attending
- `plain` — без характера, нейтральный экзаменационный тон

```bash
python -m neurotutor.cli ask case "62 г, внезапная головная боль, ригидность" --persona lin
```

## RAG: индексация учебников

```bash
python -m neurotutor.cli ingest /data/books/greenberg.pdf --source greenberg --ref full
python -m neurotutor.cli ingest /data/books/rhoton.pdf --source rhoton --ref cranial
```

Подробности (чанкинг, reranking, гибридный поиск, эталонные пары для валидации) — в [`docs/RAG_SETUP.md`](docs/RAG_SETUP.md).

## Архитектура

Подробное описание слоёв и потока — в [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

Кратко:

```
CLI → Session(mode) → Orchestrator(role + persona) → MiniMax + tools → SQLite
                                                       │
                                  query_anatomy, interpret_imaging,
                                  lookup_classification, case_simulator,
                                  grade_answer, schedule_fsrs,
                                  rag_search, pubmed_search
```

Агент **stateless** — всё состояние в SQLite. До 6 циклов tool-calling за один turn.

## Структура

```
neurotutor/
├── cli.py              — typer-команды
├── session.py          — выбор роли по mode, цикл turn'ов
├── config.py           — настройки из .env
├── agent/
│   ├── orchestrator.py — Hermes-style tool-calling loop
│   ├── roles.py        — diagnostician / anatomist / clinician / radiologist / examiner
│   ├── persona.py      — Корвин / Линь / plain
│   └── tools.py        — реализации 8 инструментов + JSON-схемы
├── llm/minimax.py      — клиент MiniMax (text + vision + embed)
├── fsrs/scheduler.py   — обёртка над fsrs с расчётом mastery
├── rag/
│   ├── ingest.py       — PDF → chunks → embeddings
│   ├── retriever.py    — векторный поиск
│   └── sources.py      — PubMed E-utilities
├── db/
│   ├── schema.sql      — SQLite схема
│   └── store.py        — connect, init_db
└── seed/
    ├── load.py
    ├── concepts_anatomy.json
    └── classifications.json
```

## Статус

Ранняя стадия. Архитектура зафиксирована, ядро работает, но:
- seed концептов мал (расширяется по мере использования)
- нужен ingest учебников локально, в репо их нет
- тесты в работе

См. issues для текущих задач.
