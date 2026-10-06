#!/usr/bin/env bash
# Start (or resume) the campaign and the public site, detached from this
# terminal and kept awake while the campaign runs. Safe to run again: anything
# already running is left alone.
#
#   ./run_site.sh            then open http://localhost:8090/
#
# To share it, put a tunnel in front of the local port instead of opening one:
#   cloudflared tunnel --url http://localhost:8090
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

# Each process gets its own session, so closing this terminal (or the tool
# that ran it) does not take it down; nohup alone was not enough on macOS.
launch() {  # launch <log> <args...>
  "$PY" - "$@" <<'PYEOF'
import subprocess, sys
log, cmd = sys.argv[1], sys.argv[2:]
p = subprocess.Popen(cmd, stdout=open(log, "a"), stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)
print(p.pid)
PYEOF
}

if pgrep -f "src/nes/campaign.py" >/dev/null; then
  echo "campaign: already running"
else
  pid=$(launch out/campaign.log "$PY" -u src/nes/campaign.py)
  launch /dev/null caffeinate -i -s -w "$pid" >/dev/null
  echo "campaign: started (pid $pid), resuming from out/campaign_state.json"
fi

if pgrep -f "src/nes/site_server.py" >/dev/null; then
  echo "site: already running"
else
  launch out/site_server.log "$PY" -u src/nes/site_server.py --port 8090 >/dev/null
  echo "site: started -> http://localhost:8090/"
fi
echo "Keep the Mac plugged in with the lid open; the campaign pauses when it sleeps."
