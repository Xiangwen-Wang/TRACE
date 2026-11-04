#!/bin/bash
# Double-click this in Finder (or run ./start.command) to launch the optimizer.
# Finds the project venv automatically and starts run.py.
cd "$(dirname "$0")"

# locate a python: project venv (../.venv or ./.venv) else system python3
PY=""
for cand in "../.venv/bin/python" "./.venv/bin/python" "$(command -v python3)"; do
  if [ -x "$cand" ]; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
  echo "No Python found. Prepare the project venv first (needs torch/openai/rdkit/sentence-transformers)."
  read -r -p "Press Enter to exit..." _; exit 1
fi

echo "Using Python: $PY"
"$PY" run.py
echo ""
read -r -p "Finished. Press Enter to close this window..." _
