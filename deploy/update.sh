#!/usr/bin/env bash
# Deploy NeuroTutor from git instead of editing the running copy in place.
#
# The audit flagged "deploy by live edit, no tests" as a top reliability risk:
# the grader and router have been fixed repeatedly, with nothing to catch a
# regression. This pulls, installs, runs the test gate, and only then restarts —
# so a red suite blocks the deploy rather than shipping a broken loop.
#
# Usage (from anywhere):  /root/neurotutor/deploy/update.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> git pull"
git pull --ff-only

echo "==> install deps"
venv/bin/pip install -q -r requirements.txt

echo "==> migrate DB (idempotent)"
venv/bin/python -m neurotutor.seed.migrate_curriculum || {
    echo "!! migration failed — aborting deploy"; exit 1; }

echo "==> test gate"
./run_tests.sh || { echo "!! tests failed — NOT restarting the bot"; exit 1; }

echo "==> restart service"
systemctl restart neurotutor-bot

echo "==> install cron"
crontab deploy/neurotutor.cron

echo "OK: deployed $(git rev-parse --short HEAD)"
