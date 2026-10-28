#!/usr/bin/env bash
# Start (or resume) the campaign and the live site in the background.
# Safe to run again: anything already running is left alone.
#
#   scripts/start.sh          then open http://localhost:8090/
#   scripts/start.sh site     the site only, no training
#
# On a laptop, keep it plugged in with the lid open: asleep, nothing runs.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
mkdir -p runs/campaign

# Each process gets its own session, so closing this terminal does not stop it.
launch() {  # launch <log> <command...>
  "$PY" - "$@" <<'EOF'
import subprocess, sys
log, cmd = sys.argv[1], sys.argv[2:]
p = subprocess.Popen(cmd, stdout=open(log, "a"), stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)
print(p.pid)
EOF
}

if [[ "${1:-}" != "site" ]]; then
  if pgrep -f "smartfly.campaign" >/dev/null; then
    echo "campaign: already running"
  else
    pid=$(launch runs/campaign/campaign.log "$PY" -u -m smartfly.campaign)
    if command -v caffeinate >/dev/null; then         # macOS: stay awake while it runs
      launch /dev/null caffeinate -i -s -w "$pid" >/dev/null
    fi
    echo "campaign: started (pid $pid); progress in runs/campaign/"
  fi
fi

if pgrep -f "smartfly.web.server" >/dev/null; then
  echo "site: already running"
else
  launch runs/site.log "$PY" -u -m smartfly.web.server >/dev/null
  echo "site: started -> http://localhost:8090/"
fi
