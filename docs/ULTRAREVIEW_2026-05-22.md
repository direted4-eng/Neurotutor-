# Ultrareview — 2026-05-22

**Branch:** `claude/neural-anatomy-llm-2dKA0` → `main`
**Scope:** 31 files changed, 2747 insertions(+)
**Session:** https://claude.ai/code/session_01UiETc26eLutxBMKcSR6tVr
**Findings:** 11 (1 nit, 10 normal)

## Summary table

| # | ID | Severity | File | Issue |
|---|---|---|---|---|
| 1 | bug_001 | normal | `neurotutor/db/store.py` | SQLite connections leaked: `sqlite3` context manager does not close |
| 2 | bug_002 | normal | `neurotutor/agent/orchestrator.py` | MAX_STEPS fallback omits `persona` key → CLI KeyError |
| 3 | bug_003 | nit | `neurotutor/llm/minimax.py` | `vision_chat` mutates caller's last message via shallow copy |
| 4 | bug_004 | normal | `neurotutor/llm/minimax.py` | Embeddings always typed `db`; queries should use `type='query'` |
| 5 | bug_006 | normal | `neurotutor/llm/minimax.py` | Vision chat hardcodes `image/png` MIME; JPEG/WebP get wrong content-type |
| 6 | bug_007 | normal | `requirements.txt` | Missing `lxml` dependency: PubMed search fails at runtime |
| 7 | bug_008 | nit/normal | `neurotutor/agent/tools.py` | `case_simulator` "interpretation" step has no backing field in schema/seed |
| 8 | merged_bug_009 | normal | `neurotutor/session.py` | `fsrs_rating` collapses across Bloom levels in `_persist_responses` |
| 9 | bug_010 | normal | `neurotutor/llm/minimax.py` | `MINIMAX_GROUP_ID` loaded into config but never sent in API requests |
| 10 | bug_024 | normal | `neurotutor/session.py` | `interactive()` generator yields twice per loop, breaks `.send()` protocol |
| 11 | bug_033 | normal | `neurotutor/agent/tools.py` | `grade_answer` JSON parsing fails on fenced output, silently zeros every grade |

---

## bug_007 — Missing lxml dependency: PubMed search fails at runtime

**Severity:** normal
**File:** `requirements.txt:1-9`

### PR comment
PubMed integration uses BeautifulSoup with the `'xml'` parser (`neurotutor/rag/sources.py:pubmed_search`), which requires the `lxml` backend — bs4 ships no built-in XML tree builder. `requirements.txt` only declares `beautifulsoup4>=4.12` with no `lxml`, so the first call to `pubmed_search` raises `bs4.FeatureNotFound` and the entire PubMed channel (CLI `pubmed` command + `pubmed_search` agent tool used by the Clinician role) is unusable on a fresh install. Fix: add `lxml>=4.9` to `requirements.txt`.

### Details
`neurotutor/rag/sources.py:pubmed_search` parses the PubMed E-utilities response with:

```python
soup = BeautifulSoup(r.text, "xml")
```

The `"xml"` feature in BeautifulSoup is **not** provided by `beautifulsoup4` itself — bs4 only ships `html.parser` as built-in. The XML tree builder lives in `lxml` and is the only XML parser bs4 supports.

After a clean `pip install -r requirements.txt`, the first call to `pubmed_search()` raises:

```
bs4.FeatureNotFound: Couldn't find a tree builder with the features
you requested: xml. Do you need to install a parser library?
```

**Blast radius:** the CLI command `python -m neurotutor.cli pubmed "<query>"` and the `pubmed_search` agent tool given to the **Clinician** role both fail. Orchestrator catches the exception so session does not crash, but the tool returns `{"error": "FeatureNotFound: ..."}` and the cited-source guarantee is broken.

### Fix
```diff
 beautifulsoup4>=4.12
+lxml>=4.9
```

Alternative: switch `pubmed_search` to `xml.etree.ElementTree` (no extra dep, more code churn).

---

## bug_004 — Embeddings always typed 'db'; queries should use type='query'

**Severity:** normal
**File:** `neurotutor/llm/minimax.py:85-96`

### PR comment
MiniMax embo-01 is an asymmetric embedding model where `type='db'` and `type='query'` map texts into different geometries; `embed()` in `neurotutor/llm/minimax.py` hardcodes `'type': 'db'` for every call, so `retriever.search` ends up encoding user queries with the document/passage embedder. The result is silently degraded RAG recall — the pipeline still returns results, just from the wrong subspace, undermining the Recall@5 ≥ 0.85 target in `docs/RAG_SETUP.md`. Fix: add a `type_` parameter to `embed()` (default `'db'`) and pass `type_='query'` from `neurotutor/rag/retriever.py:search`.

### Details
`embed()` unconditionally sends `"type": "db"`:

```python
r = self._client.post(
    "/embeddings",
    json={
        "model": SETTINGS.embed_model,
        "texts": list(texts),
        "type": "db",
    },
)
```

MiniMax `embo-01` is **asymmetric**. Its `type` field distinguishes `"db"` (passage encoder) from `"query"` (query encoder), and those two encoders project text into *different* subspaces. Cosine similarity is meaningful only when one vector is `db`-typed and the other is `query`-typed.

Two callers of `embed()`:
1. `neurotutor/rag/ingest.py` (`ingest_pdf`) — `type="db"` is **correct** at index time.
2. `neurotutor/rag/retriever.py:search` (line 17) — embeds the *user query*. Currently uses `db`, lands in the **passage** subspace, not the query subspace.

Cosine still returns ranked results — the failure is silent quality degradation.

### Fix
```python
# neurotutor/llm/minimax.py
def embed(self, texts: Iterable[str], *, type_: str = "db") -> list[list[float]]:
    r = self._client.post(
        "/embeddings",
        json={"model": SETTINGS.embed_model, "texts": list(texts), "type": type_},
    )
    ...

# neurotutor/rag/retriever.py
qvec = np.asarray(client.embed([query], type_="query")[0], dtype=np.float32)
```

No data migration needed — existing `db`-typed passage vectors are still valid.

---

## bug_033 — grade_answer JSON parsing fails on fenced output, silently zeros every grade

**Severity:** normal
**File:** `neurotutor/agent/tools.py:268-295`

### PR comment
`grade_answer` (neurotutor/agent/tools.py:300-321) asks MiniMax `Верни JSON: {...}` without forbidding markdown, then does an unguarded `json.loads(text)`. Contemporary instruct-tuned LLMs routinely wrap structured output in ```json fences — `json.loads` raises and the fallback silently returns `score=0.0` + `suggested_rating=1` ("again"), which `session._persist_responses` then writes as `grade=0.0` and the agent feeds straight into `schedule_fsrs`, corrupting the mastery model and the diagnostician's 30-question adaptive flow. Strip fenced wrappers before parse, add "без markdown" to the system prompt, or use a JSON/structured-output mode.

### Details
The bug: system prompt says only `Верни JSON: {"score": 0..1, ...}` and then runs `json.loads(text)` on whatever the model returns. The `except json.JSONDecodeError` branch hard-codes a punitive fallback: `{"score": 0.0, "breakdown": {}, "feedback": text, "suggested_rating": 1}`.

Modern instruct-tuned LLMs (including MiniMax-Text-01) strongly tend to wrap structured output in markdown fences (```json ... ```) unless prompt explicitly says otherwise. `temperature=0.0` does not suppress this.

**Silent downstream corruption:**
1. `session._persist_responses` reads `result.get("score")` → inserts `grade=0.0` into `responses`. Analytics poisoned.
2. Agent follows `grade_answer` with `schedule_fsrs` using `suggested_rating=1` (= "again"). FSRS then resets stability and increments lapses. `_mastery_from_state` applies a per-lapse penalty — mastery tanks toward 0.
3. DIAGNOSTICIAN role's "30 adaptive questions" flow degenerates — user told they know nothing.

### Fix
Any of:
- Strip fences before parse: `m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S); text = m.group(1) if m else text`.
- Add to system prompt: `Без markdown, без ```. Только сырой JSON.`
- Request JSON mode: `response_format={"type": "json_object"}`.

Also log a warning in the except branch instead of silently substituting `score=0.0`.

---

## merged_bug_009 — fsrs_rating collapses across bloom levels in _persist_responses

**Severity:** normal
**File:** `neurotutor/session.py:82-90`

### PR comment
**`_persist_responses` collapses `fsrs_rating` across Bloom levels for the same concept.** `fsrs_by_concept` is keyed only by `concept_id`, but the mastery table — and the whole architecture — is keyed on `(concept_id, bloom_level)`. When a single turn calls `schedule_fsrs` for the same concept at two Bloom levels (e.g. diagnostician sweeping Bloom 1 → 3 for SAH), the second rating overwrites the first; every Bloom-level row for that concept then gets tagged with the same last rating via `fsrs_by_concept.get(concept_id)`. Fix: key the dict by `(concept_id, bloom_level)` and look up with the same tuple.

### Details
`neurotutor/session.py::_persist_responses` (82-115) builds:

```python
fsrs_by_concept: dict[int, int] = {}
for s in trace:
    if s["tool"] != "schedule_fsrs":
        continue
    args = s.get("args") or {}
    cid = args.get("concept_id")
    rating = args.get("rating")
    if cid is not None and rating is not None:
        fsrs_by_concept[cid] = rating
```

Dict keyed on `concept_id` alone. But mastery is tracked per **(concept × Bloom level)** — per `docs/ARCHITECTURE.md` and `schema.sql` PRIMARY KEY `(concept_id, bloom_level)`. Walk-through with SAH (id=9):

1. `grade_answer(concept_id=9, bloom_level=1, ...)` recall → score 0.9.
2. `schedule_fsrs(concept_id=9, bloom_level=1, rating=4)` → `fsrs_by_concept[9] = 4`.
3. `grade_answer(concept_id=9, bloom_level=3, ...)` apply → score 0.4.
4. `schedule_fsrs(concept_id=9, bloom_level=3, rating=1)` → `fsrs_by_concept[9] = 1` **(overwrites)**.
5. For the bloom=1 row, `fsrs_by_concept.get(9)` returns **1** — wrong.

`mastery` table itself is correct (FSRS schedule writes directly), but `responses` audit log loses fidelity.

### Fix
```python
fsrs_by_pair: dict[tuple[int, int], int] = {}
for s in trace:
    if s["tool"] != "schedule_fsrs":
        continue
    args = s.get("args") or {}
    cid = args.get("concept_id")
    bl  = args.get("bloom_level")
    rating = args.get("rating")
    if cid is not None and bl is not None and rating is not None:
        fsrs_by_pair[(cid, bl)] = rating
```

Then replace `fsrs_by_concept.get(concept_id)` with `fsrs_by_pair.get((concept_id, bloom))`.

---

## bug_024 — interactive() generator yields twice per loop, breaks .send() protocol

**Severity:** normal
**File:** `neurotutor/session.py:144-155`

### PR comment
`interactive()` yields **twice per loop iteration** ('awaiting' then 'reply'), but the `.send()` protocol only delivers a value to the first yield — the second yield silently consumes the next `.send()` call and discards it. For a natural caller that does `next(gen)` once and then `gen.send(msg)` per turn, every other message is dropped (or, if the caller uses `next(gen)` to advance, the generator reads `user=None` at the awaiting yield and immediately quits via `if user in (None, "/quit"): return`). Fix by collapsing to a single yield per round-trip: `user = (yield reply_or_initial_awaiting)`.

### Details
```python
while True:
    user = (yield {"session_id": sid, "awaiting": "user"})    # yield A
    if user in (None, "/quit"):
        return
    result = turn(mode, user, history=history, session_id=sid)
    history = result["messages"][1:]
    yield {"session_id": sid, "reply": result["reply"],       # yield B
           "trace": result["trace"]}
```

Yield B's resumption value is **not bound to anything** — whatever caller `.send()`s at that point is silently discarded.

| Step | Caller | What happens |
|---|---|---|
| 1 | `next(gen)` | runs to yield A → emits `{awaiting}` |
| 2 | `gen.send("Q1")` | `user = "Q1"`; runs `turn(...)`; yield B → emits reply |
| 3 | `gen.send("Q2")` | value of yield B = `"Q2"` (discarded!); loop iterates; yield A → emits `{awaiting}` — **Q2 lost** |
| 4 | `gen.send("Q3")` | `user = "Q3"`; treats as second question, but caller intended Q2 |

`interactive()` has no live callers yet, so latent bug.

### Fix
```python
reply = {"session_id": sid, "awaiting": "user"}
while True:
    user = (yield reply)
    if user in (None, "/quit"):
        return
    result = turn(mode, user, history=history, session_id=sid)
    history = result["messages"][1:]
    reply = {"session_id": sid, "reply": result["reply"],
             "trace": result["trace"]}
```

---

## bug_002 — MAX_STEPS fallback omits persona key, crashes CLI with KeyError

**Severity:** normal
**File:** `neurotutor/agent/orchestrator.py:72-75`

### PR comment
MAX_STEPS fallback at orchestrator.py:73-75 returns a dict without the 'persona' key, but cli.py:65 unconditionally subscripts `out['persona']` when printing. Any turn that exhausts the 6-step tool-call loop raises KeyError in the CLI handler instead of printing the intended '[max tool-call steps reached]' message. Fix: add `'persona': pers.code` to the fallback dict (mirroring the normal return at lines 48-51).

### Details
`run_turn` has two asymmetric return shapes:
- Normal (48-51): `{role, persona, reply, trace, messages}`
- MAX_STEPS fallback (73-75): `{role, reply, trace, messages}` — `persona` missing

`cli.py:65` uses bracket subscript `out['persona']` (not `.get()`). MAX_STEPS=6 is hit when the model loops on tool-calls. Crashes with KeyError instead of the intended `[max tool-call steps reached]` message.

### Fix
```python
return {"role": role.name, "persona": pers.code,
        "reply": "[max tool-call steps reached]",
        "trace": trace, "messages": messages}
```

---

## bug_001 — SQLite connections leaked: sqlite3 context manager does not close

**Severity:** normal
**File:** `neurotutor/db/store.py:13-17`

### PR comment
`connect()` in `neurotutor/db/store.py:14-18` returns a bare `sqlite3.Connection`, but the codebase uses it as `with connect() as conn:` in ~14 sites (`agent/tools.py`, `fsrs/scheduler.py`, `session.py`, `rag/ingest.py`, `rag/retriever.py`, `seed/load.py`). Python's `sqlite3.Connection` context manager only commits/rolls back the transaction — it does **not** close the connection — so each call holds the connection open until the local `conn` is garbage-collected, which is undefined timing on non-CPython and longer than block-scope even on CPython (WAL mode keeps `.wal`/`.shm` handles alive). Fix: either make `connect()` itself a `@contextmanager` that closes in `finally`, or wrap call sites in `contextlib.closing(connect())`.

### Details
Python `sqlite3` docs explicitly: *"The context manager neither implicitly opens a new transaction nor closes the connection."*

Affected call sites:
- `neurotutor/agent/tools.py` — `query_anatomy`, `lookup_classification`, `case_simulator`
- `neurotutor/fsrs/scheduler.py` — `schedule`, `due_today`, `pick_new`
- `neurotutor/rag/ingest.py` — `ingest_pdf`
- `neurotutor/rag/retriever.py` — `search`
- `neurotutor/seed/load.py` — seed_* functions
- `neurotutor/session.py` — `start`, `end`, `_persist_responses`

Same file's `init_db()` correctly uses `try/finally` with `conn.close()`, and `cursor()` is `@contextmanager` with close in `finally`. Bare `connect()` is the outlier.

WAL mode (`PRAGMA journal_mode = WAL` in `db/schema.sql`) amplifies cost: each open connection holds `.wal`/`.shm` handles alive. Non-CPython runtimes (PyPy, Jython) leak entirely (no refcounting).

### Fix
```python
from contextlib import contextmanager

def _open(db_path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or SETTINGS.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    conn = _open(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
```

Keeps every existing `with connect() as conn:` site working unchanged.

---

## bug_006 — Vision chat hardcodes image/png MIME; JPEG/WebP get wrong content-type

**Severity:** normal
**File:** `neurotutor/llm/minimax.py:71-78`

### PR comment
vision_chat hardcodes the MIME type as image/png in the data URL regardless of the actual image format, and interpret_imaging passes through any image_path without an extension check. JPEG/JPG/WebP files (common for radiology PACS exports and atlas figures like Rhoton/Greenberg) will be sent with a mismatched content-type, which OpenAI-compatible vision endpoints either reject outright or silently misinterpret. Fix is one line: derive the MIME via mimetypes.guess_type(image_path)[0] (with 'image/png' as fallback) and use it in both content-list branches.

### Details
Both branches in `vision_chat`:

```python
{"image_url": {"url": f"data:image/png;base64,{b64}"}}
```

No inference from extension or magic bytes. `interpret_imaging` in `agent/tools.py` forwards any image_path unmodified.

Project is a neurosurgery tutor; radiology sources are overwhelmingly **JPEG**: PACS exports, Rhoton/Greenberg atlas figures, Radiopaedia. Outcome:
- 400 / content-type mismatch → tool fails.
- Silent codec misinterpretation → confidently wrong radiology reading with no error surface.

### Fix
```python
import mimetypes
mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
...
{"image_url": {"url": f"data:{mime};base64,{b64}"}}
```

Optional: validate suffix in `interpret_imaging` and return clean error for unsupported formats (`.tiff`, `.dcm`).

---

## bug_010 — MINIMAX_GROUP_ID loaded into config but never sent in API requests

**Severity:** normal
**File:** `neurotutor/llm/minimax.py:19-33`

### PR comment
`MINIMAX_GROUP_ID` is loaded into `Settings.minimax_group_id` in config.py and listed in README's 'Минимум для запуска' section and `.env.example` as required, but `neurotutor/llm/minimax.py` never references it — the httpx client only sets `Authorization: Bearer` with no `GroupId` in headers, query params, or body. This either silently breaks setup for MiniMax endpoints that require GroupId (notably embeddings, used by RAG ingest/retrieval), or the field is dead and the README misleads first-time users. Fix is either to pass `GroupId=SETTINGS.minimax_group_id` as a query parameter on each request (or at the client level), or remove the field from config, `.env.example`, and README.

### Details
`config.py` loads `MINIMAX_GROUP_ID` into `Settings.minimax_group_id`. README lists it under **Минимум для запуска**. `.env.example` reserves a slot. But `neurotutor/llm/minimax.py` **never references** `SETTINGS.minimax_group_id` — `httpx.Client` only sets `Authorization: Bearer`.

If endpoint requires GroupId (commonly required for embeddings on multi-group accounts), RAG ingest/retrieval fails 401/403. Otherwise field is dead code and docs are stale.

### Fix
Option A — wire it through:
```python
self._params = {"GroupId": SETTINGS.minimax_group_id} if SETTINGS.minimax_group_id else {}
# pass params=self._params on each .post()
```

Option B — drop from `Settings`, `.env.example`, README if Bearer-only suffices.

---

## bug_003 — vision_chat mutates caller's last message in place via shallow copy

**Severity:** nit
**File:** `neurotutor/llm/minimax.py:60-78`

### PR comment
vision_chat (neurotutor/llm/minimax.py:64) mutates the caller's last message dict. `msgs = list(messages)` shallow-copies only the outer list — `msgs[-1]` is the same dict as `messages[-1]`, so reassigning `last["content"]` (str branch) or appending to `last_content` (list branch) leaks back to the caller. Today the sole caller `interpret_imaging` passes a fresh single-element list per call so nothing breaks, but the function's own `list(messages)` signals an isolation intent that is not delivered. One-line fix: build a fresh last dict, e.g. `msgs = list(messages[:-1]) + [dict(messages[-1])]` (and a fresh content list when appending).

### Details
```python
msgs = list(messages)                # outer list copy, dicts shared
last = msgs[-1]                       # same object as messages[-1]
last_content = last.get("content")
if isinstance(last_content, str):
    last["content"] = [ ... ]         # mutates caller's dict
else:
    last_content.append( ... )        # mutates caller's list
```

Only caller `interpret_imaging` passes a fresh list per call, so latent today. Variable name `msgs` and `list(messages)` signal intended isolation that isn't delivered.

### Fix
```python
msgs = list(messages[:-1])
last = dict(messages[-1])
last_content = last.get("content")
if isinstance(last_content, str):
    last["content"] = [
        {"type": "text", "text": last_content},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]
else:
    last["content"] = list(last_content) + [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]
msgs.append(last)
```

---

## bug_008 — case_simulator 'interpretation' step has no backing field in schema/seed

**Severity:** nit
**File:** `neurotutor/agent/tools.py:75-82`

### PR comment
case_simulator's step enum includes 'interpretation', but neither the cases table schema nor seed/cases.json defines an 'interpretation' column/field — the other five enum values (presentation, workup, differential, plan, complications) all do. When the LLM calls case_simulator(case_id=X, step='interpretation'), case.get('interpretation') silently returns None and the tool emits {id, title, interpretation: null}. The CLINICIAN role explicitly advertises 'interpretation' as step 4 of the Harvard CBL flow, so the model will hit this path naturally. Fix by either dropping 'interpretation' from the enum (and letting the LLM derive interpretation from the workup it just received), or adding an 'interpretation' column to cases + seed canonical interpretations.

### Details
Enum (`tools.py:91-94`):
```python
"step": {"type": "string", "enum": [
    "presentation", "differential", "workup",
    "interpretation", "plan", "complications"
]},
```

`db/schema.sql` `cases` table has columns: `presentation`, `workup`, `differential`, `plan`, `complications`, `rubric`, `concept_ids` — no `interpretation`. `seed/cases.json` has no `interpretation` key.

CLINICIAN role prompt (`roles.py:43-44`) explicitly lists 6 Harvard CBL phases with phase 4 = "интерпретация". Model will naturally call `case_simulator(case_id=X, step="interpretation")` and get `{id, title, interpretation: null}`.

### Fix options
- **Drop `interpretation` from enum** + update CLINICIAN prompt so step 4 is the *student's* job (interpret revealed workup). Smallest change.
- **Add `interpretation` column** + seed canonical interpretations for 8 cases. Schema migration + seed expansion.
