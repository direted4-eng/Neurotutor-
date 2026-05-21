# RAG: настройка с нуля для нейрохирургического тьютора

Цель — собрать ретривер, который выдаёт **проверяемые** цитаты из учебников (Greenberg, Youmans, Rhoton) и **рецензируемой** медлитературы (PubMed, Cochrane, гайдлайны), а не «мнение модели». Под 4 ГБ VPS, без локальных LLM.

---

## 0. Что мы строим

Два уровня источников:

| Уровень | Содержимое | Хранилище | Когда вызывать |
|---|---|---|---|
| **L1: статичная база** | Greenberg, Youmans, Rhoton, отобранные гайдлайны | SQLite `rag_chunks` + эмбеддинги MiniMax | Любой клинический/анатомический факт |
| **L2: живые источники** | PubMed, Cochrane, NICE, AANS/CNS guidelines | На лету через `pubmed_search` / fetch | Свежие данные, спорные вопросы, верификация |

Промежуточный кэш L2 (выдержки статей по PMID) кладём в те же `rag_chunks` с `source='pubmed'`, чтобы повторные обращения не били по сети.

---

## 1. Подготовка окружения

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # заполнить MINIMAX_API_KEY и PUBMED_EMAIL
python -m neurotutor.cli init
```

`init` создаёт SQLite-схему (см. `neurotutor/db/schema.sql`), грузит домены, концепты-затравку и классификации.

### Опционально: ускорение поиска через sqlite-vss

Брут-форс по 50–100k чанков на 4 ГБ работает за 200–500 мс — приемлемо для интерактива. Если коллекция растёт за 200k:

```bash
pip install sqlite-vss
```

И в `db/store.py` при коннекте:

```python
import sqlite_vss
conn.enable_load_extension(True)
sqlite_vss.load(conn)
conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS vss_chunks USING vss0(embedding(1024))")
```

Затем после каждой вставки в `rag_chunks` — `INSERT INTO vss_chunks(rowid, embedding) VALUES (?, ?)`.

---

## 2. Подготовка PDF учебников

### 2.1 Что подавать на вход

**Greenberg's Handbook** — главный плотный источник. Подавайте **полную книгу одним PDF**, но разбивайте по главам в `ref` (например `ref="SAH"`, `ref="TBI"`). Структура книги тезисная — это идеальный материал для chunk-based recall.

**Youmans & Winn** — пять томов. Подавайте **по главе** с `ref="vol2/ch.34/Aneurysmal SAH"`. Не закидывайте все 5 томов разом — индекс распухнет, релевантность поплывёт.

**Rhoton's Atlas / Cranial Anatomy** — текст глав плюс подписи к рисункам. Сами рисунки храните в `images` с `concept_id`, тогда `interpret_imaging` сможет дёрнуть конкретный срез.

### 2.2 Очистка PDF

`pypdf` извлекает текст быстро, но колонтитулы и переносы строк ломают чанки. Прогоняйте через эту чистку до индексации:

```python
import re
def clean(t: str) -> str:
    t = re.sub(r"-\n", "", t)              # переносы слов
    t = re.sub(r"\n{2,}", "\n\n", t)       # абзацы
    t = re.sub(r"Page \d+ of \d+", "", t)  # колонтитулы — подгоните под свой PDF
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()
```

Если PDF — скан (старые издания Youmans бывают такими), нужна OCR-стадия: `ocrmypdf input.pdf output.pdf --language eng` (один раз, до ingest).

### 2.3 Чанкинг (детали в `rag/ingest.py`)

- **Размер**: 600 токенов целевой, 500–800 допустимо. Для медучебника 800 — потолок: дальше падает точность retrieval.
- **Overlap**: 120 токенов. Меньше — теряются переходы между смежными концептами (например «вазоспазм» через несколько предложений после «SAH»).
- **Граница**: режем по предложениям (`(?<=[.!?])\s+`), не по фиксированному числу символов. Иначе диагноз обрывается посередине.
- **Метаданные** обязательно: `source`, `ref` (глава/раздел/PMID), `title`. Без `ref` цитата бесполезна для верификации.

Команды ingest:

```bash
python -m neurotutor.cli ingest /data/books/greenberg.pdf --source greenberg --ref full
python -m neurotutor.cli ingest /data/books/youmans_vol2_ch34.pdf --source youmans --ref "vol2/ch34/aSAH"
python -m neurotutor.cli ingest /data/books/rhoton_cranial.pdf --source rhoton --ref "cranial"
```

### 2.4 Что НЕ грузить

- Учебники физиологии общего профиля — шум, не нейрохирургия.
- Старые издания (Greenberg <8 ed., Youmans <7 ed.) — устарели по WHO 2021 опухолям и ведению SAH.
- Чужие конспекты, лекции PowerPoint без референсов — нет цитируемости.

---

## 3. Эмбеддинги

MiniMax `embo-01` даёт 1024-мерные векторы. Хранятся как `float32` BLOB → 4 КБ/чанк. 100k чанков ≈ 400 МБ — помещается в 4 ГБ VPS без проблем.

### Бюджет вызовов

Greenberg 11 ed ≈ 1700 страниц × ~300 ток/стр = ~500k токенов = ~830 чанков по 600. Youmans 5 томов ≈ 3000 чанков. Rhoton ≈ 400 чанков. Итого **~4–5k чанков** на индексацию — однократно.

Батч 32 чанка на вызов: `embed()` уже сделан так в `rag/ingest.py`. Не делайте по одному — будет дорого и медленно.

### Перенос/бэкап

```bash
sqlite3 data/neurotutor.sqlite ".backup data/neurotutor.bak"
```

Эмбеддинги пересчитывать не нужно при смене модели MiniMax-чата — только при смене `MINIMAX_EMBED_MODEL`. Тогда перебор только embedding-колонки, без переингеста PDF.

---

## 4. Retrieval

`rag_search(query, source=None, k=5)`:

1. Эмбеддит запрос (1 вызов API).
2. Брут-форс косинус по всем чанкам в SQLite (с фильтром `source`, если задан).
3. Возвращает топ-k с `{id, source, ref, title, text, score}`.

### Reranking (опционально, рекомендую)

Косинус по эмбеддингам отбирает кандидатов широко. Для медицины лучше добавить второй проход на LLM-reranking. Дёшево, заметно повышает precision:

```python
def rerank(query: str, candidates: list[dict], top_n: int = 3) -> list[dict]:
    sys = "Score each passage 0..1 by how directly it answers the query. JSON only."
    user = json.dumps({"query": query,
                       "passages": [{"id": c["id"], "text": c["text"][:500]}
                                    for c in candidates]})
    # один вызов chat, парсинг → пересортировка → top_n
```

Берите k=10 для retrieval, потом rerank до 3 — на 4 ГБ это два API-вызова на запрос, не страшно.

### Гибридный поиск

Чистый векторный поиск проваливается на коротких запросах с редкими терминами («mFisher 3»). Добавьте BM25-канал:

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS rag_fts USING fts5(text, content='rag_chunks', content_rowid='id');
INSERT INTO rag_fts(rowid, text) SELECT id, text FROM rag_chunks;
```

И при поиске объединяйте топ-k от vector и FTS, потом reranker. Это **критично для классификаций и аббревиатур**.

---

## 5. Живые источники: PubMed и далее

`rag/sources.py::pubmed_search` уже подключён к E-utilities. Что важно настроить:

### 5.1 NCBI ключ

Без ключа — 3 запроса/с, с ключом — 10/с. Регистрация на https://www.ncbi.nlm.nih.gov/account/settings/. В `.env`:

```
PUBMED_EMAIL=you@example.com
PUBMED_API_KEY=...
```

### 5.2 Фильтры качества

Передавайте `filter_` для отсечения мусора:

| Цель | Фильтр |
|---|---|
| Систематические обзоры | `systematic review[pt] OR meta-analysis[pt]` |
| Гайдлайны | `guideline[pt] OR practice guideline[pt]` |
| Только последние 5 лет | `("2020/01/01"[PDAT] : "3000"[PDAT])` |
| Высокий уровень доказательности | `randomized controlled trial[pt]` |

Дефолтный шаблон в коде — `pt[]` без фильтра. Для клинических вопросов агент должен сам подмешивать фильтр (через системный промпт роли Clinician). Я уже выписал инструкцию в `roles.py::CLINICIAN`.

### 5.3 Кэширование PubMed

Каждая статья по PMID кладётся в `rag_chunks` с `source='pubmed'`, `ref=PMID`. Тогда последующий `rag_search(..., source='pubmed')` найдёт её локально. Добавьте в `pubmed_search` после выкачивания:

```python
from ..llm.minimax import MiniMaxClient
client = MiniMaxClient()
vecs = client.embed([a["abstract"] for a in articles if a["abstract"]])
# upsert по (source='pubmed', ref=pmid)
```

(Сейчас в коде этого нет — добавьте, когда наберётся стабильный набор тем.)

### 5.4 Какие ещё авторитетные источники подключать

- **Cochrane Library** — через CENTRAL внутри PubMed (`AND "Cochrane Database Syst Rev"[Journal]`).
- **NICE Guidelines** — нет API, но статичные PDF можно ingest как обычные документы с `source='nice'`.
- **AANS/CNS Guidelines** — те же PDF.
- **UpToDate** — закрытый, API нет; не подключайте парсеры — ToS.
- **Radiopaedia** — у них публичный API не для коммерческого использования; для self-study можно дергать страницы вручную и класть в `rag_chunks` с `source='radiopaedia'` + ссылкой.
- **Neurosurgical Atlas (Cohen-Gadol)** — статьи скачиваются в PDF, ingest как rhoton-уровневый источник для доступов.

Не добавляйте: Wikipedia (не peer-reviewed для клиники), Medscape (вторичка), любые блоги. Для тьютора уровня ординатуры это шум.

---

## 6. Интеграция с агентом

Инструменты `rag_search` и `pubmed_search` уже зарегистрированы в `agent/tools.py`. Что важно про их использование ролями:

- **Anatomist** — `rag_search(source='rhoton'|'greenberg')` для верификации анатомических утверждений.
- **Clinician** — `pubmed_search` с фильтром `guideline[pt]` для протоколов; `rag_search(source='greenberg')` для базовых схем.
- **Radiologist** — `rag_search` по радиологическим главам; `interpret_imaging` для самой картинки.
- **Examiner** — RAG не даём! На OSCE ученик не лезет в книгу. Поэтому в `roles.py::EXAMINER` инструмент `rag_search` намеренно не указан.

### Цитирование

Каждый ответ агента, опирающийся на RAG, **должен** включать ссылку: `[Greenberg p.SAH/p.123]`, `[PMID 38xxx]`. Это добавляется в системный промпт ролей (см. правки ниже) и проверяется в `grade_answer` как отдельный критерий — иначе агент скатится в фантазии.

---

## 7. Валидация качества ретривера

Без оценки качество retrieval вы не контролируете. Минимальный набор:

1. Соберите **20–50 эталонных пар** `(вопрос, правильный chunk_id)` вручную: например «Какая степень Hunt-Hess при стопоре и гемипарезе?» → известный чанк Greenberg.
2. Скрипт прогона: для каждого вопроса — `rag_search(q, k=5)`, проверка попадания нужного chunk_id в топ-5.
3. Метрики: **Recall@5** (доля попаданий) и **MRR** (1/rank). Цель — Recall@5 ≥ 0.85, MRR ≥ 0.6.

Если ниже — крутите по приоритету: чанкинг (overlap до 200), reranking, гибрид BM25, и только потом — смена модели эмбеддингов.

---

## 8. Что НЕЛЬЗЯ делать

- Не клеить ответ агента из одного top-1 чанка без верификации — модель должна **синтезировать из ≥2 источников** и явно отметить расхождения.
- Не индексировать PDF с водяными знаками поверх текста (часто бывает у пиратских копий) — извлечение даёт «GREENBERG GREENBERG GREENBERG» в каждом чанке, рушит эмбеддинги.
- Не смешивать `source` в одном чанке — если кусок главы цитирует RCT, всё равно `source='greenberg'`, а PMID цитаты идёт в `ref`.
- Не отдавать пациентам/коллегам без врача — в системном промпте всех ролей должна стоять явная дисклеймер-строчка («educational, not clinical decision»).

---

## 9. Чек-лист готовности к продуктивной работе

- [ ] `.env` заполнен (MiniMax + PubMed)
- [ ] `python -m neurotutor.cli init` отработал
- [ ] Хотя бы Greenberg проингестен (`ingest --source greenberg`)
- [ ] FTS-таблица `rag_fts` создана (гибрид)
- [ ] Reranker подключён в `rag/retriever.py`
- [ ] 20 эталонных пар оценены, Recall@5 ≥ 0.85
- [ ] `pubmed_search('subarachnoid hemorrhage management', filter_='guideline[pt]')` возвращает свежие гайдлайны
- [ ] Ответы агента содержат `[Source: ...]` маркеры
