"""Test bootstrap.

CRITICAL: point the suite at a throwaway temp DB *before* anything imports
``neurotutor.config`` (which freezes ``SETTINGS`` at import). This guarantees
tests never read or mutate the live ``data/neurotutor.sqlite``. We set the env
vars directly (not via ``.env``); ``load_dotenv`` runs with ``override=False``,
so these win.
"""
from __future__ import annotations

import logging
import os
import tempfile

# Keep expected error tracebacks (e.g. the mocked 529 in the degrade test) out
# of the test output — they're asserted on, not failures.
logging.disable(logging.CRITICAL)

_TMP = tempfile.mkdtemp(prefix="neurotutor-test-")

# Force the temp DB / RAG dir / dashboard path — must override whatever the real
# environment or .env would supply, so tests are fully isolated from prod data.
os.environ["NEUROTUTOR_DB"] = os.path.join(_TMP, "test.sqlite")
os.environ["NEUROTUTOR_RAG_DIR"] = os.path.join(_TMP, "rag")
os.environ["NEUROTUTOR_DASHBOARD_JSON"] = os.path.join(_TMP, "study.json")

# MiniMaxClient() refuses to construct without a key; tests mock the client, but
# a placeholder keeps any incidental construction from raising.
os.environ.setdefault("MINIMAX_API_KEY", "test-key")

# Keep Telegram sends inert (no token → send_telegram returns None, no network).
os.environ["TELEGRAM_BOT_TOKEN"] = ""
os.environ.setdefault("TELEGRAM_USER_ID", "424242")
