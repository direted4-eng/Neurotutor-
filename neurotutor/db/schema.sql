-- Neurotutor SQLite schema.
-- Mastery is tracked on (concept × Bloom level) pairs, not on concept alone:
-- knowing Hunt-Hess (remember) is different from applying it in a case (apply).

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS domains (
    id          INTEGER PRIMARY KEY,
    code        TEXT UNIQUE NOT NULL,           -- anatomy | pathology | radiology | clinical | approaches
    title       TEXT NOT NULL,
    target_mastery REAL NOT NULL DEFAULT 0.85   -- residency target
);

CREATE TABLE IF NOT EXISTS concepts (
    id          INTEGER PRIMARY KEY,
    domain_id   INTEGER NOT NULL REFERENCES domains(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    slug        TEXT UNIQUE NOT NULL,
    parent_id   INTEGER REFERENCES concepts(id) ON DELETE SET NULL,
    summary     TEXT,
    sources     TEXT,                            -- JSON list of refs (Greenberg p.X, Rhoton ch.Y)
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_concepts_domain ON concepts(domain_id);
CREATE INDEX IF NOT EXISTS idx_concepts_parent ON concepts(parent_id);

-- Bloom levels: 1=remember 2=understand 3=apply 4=analyze 5=evaluate 6=create
CREATE TABLE IF NOT EXISTS mastery (
    concept_id      INTEGER NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    bloom_level     INTEGER NOT NULL CHECK (bloom_level BETWEEN 1 AND 6),
    mastery         REAL NOT NULL DEFAULT 0.0,   -- 0..1
    stability       REAL NOT NULL DEFAULT 0.0,   -- FSRS state
    difficulty      REAL NOT NULL DEFAULT 5.0,   -- FSRS state
    last_review     TIMESTAMP,
    next_review     TIMESTAMP,
    review_count    INTEGER NOT NULL DEFAULT 0,
    lapses          INTEGER NOT NULL DEFAULT 0,
    card_json       TEXT,                        -- full FSRS Card state (v6 Card.to_json)
    PRIMARY KEY (concept_id, bloom_level)
);

CREATE INDEX IF NOT EXISTS idx_mastery_due ON mastery(next_review);

CREATE TABLE IF NOT EXISTS classifications (
    id          INTEGER PRIMARY KEY,
    code        TEXT UNIQUE NOT NULL,            -- e.g. who_cns_2021, spetzler_martin
    title       TEXT NOT NULL,
    domain_id   INTEGER REFERENCES domains(id),
    payload     TEXT NOT NULL,                   -- JSON: grades, criteria, scoring
    source      TEXT
);

CREATE TABLE IF NOT EXISTS cases (
    id              INTEGER PRIMARY KEY,
    title           TEXT NOT NULL,
    domain_id       INTEGER REFERENCES domains(id),
    difficulty      INTEGER NOT NULL DEFAULT 3,  -- 1..5
    presentation    TEXT NOT NULL,
    workup          TEXT,                         -- JSON: tests, expected findings
    differential    TEXT,                         -- JSON list
    plan            TEXT,                         -- JSON: management steps
    complications   TEXT,                         -- JSON list
    rubric          TEXT,                         -- JSON: OSCE-style grading
    concept_ids     TEXT                          -- JSON list of concept refs
);

CREATE TABLE IF NOT EXISTS images (
    id              INTEGER PRIMARY KEY,
    case_id         INTEGER REFERENCES cases(id) ON DELETE CASCADE,
    concept_id      INTEGER REFERENCES concepts(id) ON DELETE SET NULL,
    modality        TEXT,                         -- CT | MRI-T1 | MRI-T2 | FLAIR | DWI | angio | atlas
    path            TEXT NOT NULL,                -- local path; loaded one at a time
    annotations     TEXT                          -- JSON: structures with coords
);

CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY,
    started_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at    TIMESTAMP,
    mode        TEXT NOT NULL,                    -- diagnostic | review | new | case | osce
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS responses (
    id              INTEGER PRIMARY KEY,
    session_id      INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    concept_id      INTEGER REFERENCES concepts(id) ON DELETE SET NULL,
    bloom_level     INTEGER NOT NULL,
    case_id         INTEGER REFERENCES cases(id) ON DELETE SET NULL,
    role            TEXT NOT NULL,                -- anatomist | clinician | radiologist | examiner | diagnostician
    prompt          TEXT NOT NULL,
    answer          TEXT,
    grade           REAL,                          -- 0..1
    fsrs_rating     INTEGER,                       -- 1 again | 2 hard | 3 good | 4 easy
    rubric_breakdown TEXT,                          -- JSON
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_responses_session ON responses(session_id);
CREATE INDEX IF NOT EXISTS idx_responses_concept ON responses(concept_id);

-- RAG store. sqlite-vss is loaded if available; otherwise we fall back to
-- a plain BLOB column and brute-force cosine in Python.
CREATE TABLE IF NOT EXISTS rag_chunks (
    id          INTEGER PRIMARY KEY,
    source      TEXT NOT NULL,                    -- greenberg | youmans | rhoton | pubmed | radiopaedia
    ref         TEXT,                              -- chapter/section/PMID
    title       TEXT,
    text        TEXT NOT NULL,
    tokens      INTEGER,
    embedding   BLOB,                              -- float32 vector
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_rag_source ON rag_chunks(source);

-- Questions pushed to the student (e.g. morning drill via Telegram) that are
-- awaiting an answer. One open row per question; answered_at set once graded.
CREATE TABLE IF NOT EXISTS pending_questions (
    id          INTEGER PRIMARY KEY,
    user_id     TEXT NOT NULL,
    concept_id  INTEGER REFERENCES concepts(id) ON DELETE CASCADE,
    bloom_level INTEGER NOT NULL DEFAULT 1,
    prompt      TEXT NOT NULL,
    rubric      TEXT,                              -- JSON: criteria for grade_answer
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    answered_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_pending_user ON pending_questions(user_id, answered_at);
