#!/usr/bin/env bash
set -euo pipefail

if command -v python3.13 >/dev/null 2>&1; then
  PYTHON=python3.13
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi

"$PYTHON" -m venv venv
./venv/bin/python -m pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  printf '%s\n' "Created .env from .env.example. Please edit .env and set your API keys."
else
  printf '%s\n' ".env already exists. Skipped copying .env.example."
fi

printf '%s\n' "Setup complete. Run: ./venv/bin/python run.py"
