#!/usr/bin/env bash
# One-shot: set up venv, install, build index, serve. For a fresh clone.
set -euo pipefail

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f data/index/meta.json ]; then
  echo "[run] building index ..."
  python3 scripts/build_index.py
fi

exec uvicorn app.main:app --port 8000
