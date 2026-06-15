#!/usr/bin/env bash
# Run the NeuroTutor test suite in full isolation from production data.
#
# We export a throwaway DB / RAG / dashboard path into the ENVIRONMENT *before*
# Python starts, so neurotutor.config (which freezes SETTINGS at import) can
# never resolve to the live data/neurotutor.sqlite — no matter how the test
# loader orders imports. `-t .` makes discovery import modules as the `tests`
# package, so tests/__init__.py runs too (belt and suspenders).
#
# Usage:  ./run_tests.sh            # whole suite
#         ./run_tests.sh -v         # verbose
#         ./run_tests.sh tests.test_grading   # one module (with -m unittest)
set -euo pipefail
cd "$(dirname "$0")"

TMPDIR_TEST="$(mktemp -d -t neurotutor-tests-XXXXXX)"
trap 'rm -rf "$TMPDIR_TEST"' EXIT

export NEUROTUTOR_DB="$TMPDIR_TEST/test.sqlite"
export NEUROTUTOR_RAG_DIR="$TMPDIR_TEST/rag"
export NEUROTUTOR_DASHBOARD_JSON="$TMPDIR_TEST/study.json"
export MINIMAX_API_KEY="${MINIMAX_API_KEY:-test-key}"
export TELEGRAM_BOT_TOKEN=""          # keep all Telegram sends inert (no network)
export TELEGRAM_USER_ID="424242"

PY=venv/bin/python
[ -x "$PY" ] || PY=python3

if [ "$#" -gt 0 ] && [[ "$1" == tests.* ]]; then
    exec "$PY" -m unittest "$@"
fi
exec "$PY" -m unittest discover -s tests -t . "$@"
