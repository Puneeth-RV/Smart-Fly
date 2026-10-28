#!/usr/bin/env bash
# Serve the live fly simulation locally. Nothing leaves this machine.
set -e
cd "$(dirname "$0")"
PORT="${1:-8080}"
if [ ! -f out/scene/net.json ]; then
  echo "Live network not found in out/scene/. Building it..."
  . .venv/bin/activate
  python src/android/export_live.py
  cp out/live/net.json out/live/*.bin out/scene/
fi
python3 out/scene/build.py
echo
echo "  Fly at the Controls -> http://localhost:$PORT"
echo "  (Ctrl-C to stop)"
echo
cd out/scene && exec python3 -m http.server "$PORT"
