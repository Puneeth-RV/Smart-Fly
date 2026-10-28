"""The public site: watch the fly train, and watch it replay levels it has mastered.

Runs two channels (see fly_channel.py) and serves the page in out/site/.
Visitors are read-only. The one thing they can do is ask the theater to play a
level the fly has already mastered; requests are checked against the campaign,
the queue is capped, and each address can ask once every 20 seconds.

Binds to 127.0.0.1 by default: put a tunnel (e.g. cloudflared) in front to share
it, rather than opening a port on the machine.

    python -u src/nes/site_server.py            # http://localhost:8090
"""

from __future__ import annotations

import argparse
import json
import re
import socketserver
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "out"
SITE = OUT / "site"

from nes.fly_channel import Channel, campaign_state  # noqa: E402

LEVEL_RE = re.compile(r"^[1-8]-[1-4]$")
COOLDOWN_S = 20.0


def training_rows(path: Path, limit: int = 400) -> list[dict]:
    """Progress lines from the trainer's log, newest `limit`, for the charts."""
    rows = []
    if not path.exists():
        return rows
    pat = re.compile(r"update +\d+ +frames +([\d.]+)M .*mean x +([\d.]+) .*flags (\d+)/(\d+)")
    for line in path.read_text(errors="ignore").splitlines()[-4000:]:
        mm = pat.search(line)
        if mm:
            rows.append({"frames": float(mm[1]), "mean_x": float(mm[2]),
                         "flags": int(mm[3]), "of": int(mm[4])})
    return rows[-limit:]


class Handler(SimpleHTTPRequestHandler):
    channels: dict[str, Channel] = {}
    train_log: Path = OUT / "campaign_train.log"
    last_request: dict[str, float] = {}
    req_lock = threading.Lock()

    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()

    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code=200):
        self._send(json.dumps(obj).encode(), "application/json", code)

    def _client(self) -> str:
        # Behind a tunnel every request comes from localhost; the tunnel passes
        # the visitor's address along.
        return (self.headers.get("CF-Connecting-IP")
                or self.headers.get("X-Forwarded-For", "").split(",")[0].strip()
                or self.client_address[0])

    def list_directory(self, path):             # no directory listings
        self.send_error(404)
        return None

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path.startswith("/api/state/"):
            ch = self.channels.get(path.rsplit("/", 1)[-1])
            return self._json(ch.snapshot()) if ch else self._json({"error": "no channel"}, 404)
        if path.startswith("/api/frame/"):
            ch = self.channels.get(path.rsplit("/", 1)[-1].removesuffix(".jpg"))
            if not ch or ch.jpeg is None:
                return self._send(b"", "text/plain", 503)
            return self._send(ch.jpeg, "image/jpeg")
        if path == "/api/campaign":
            st = campaign_state() or {}
            return self._json({
                "order": st.get("order", []), "current": st.get("current"),
                "mastered": st.get("mastered", []), "parked": st.get("parked", []),
                "tests": st.get("tests", [])[-40:],
                "level_frames": st.get("level_frames", {}),
                "training": training_rows(self.train_log),
            })
        if path in ("/", ""):
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/theater":
            return self._json({"error": "not found"}, 404)
        try:
            n = min(int(self.headers.get("Content-Length", 0)), 256)
            level = str(json.loads(self.rfile.read(n) or b"{}").get("level", ""))
        except Exception:
            return self._json({"ok": False, "reason": "Bad request."}, 400)
        if not LEVEL_RE.match(level):
            return self._json({"ok": False, "reason": "Unknown level."}, 400)
        who, now = self._client(), time.time()
        with self.req_lock:
            wait = COOLDOWN_S - (now - self.last_request.get(who, 0))
            if wait > 0:
                return self._json({"ok": False, "reason": f"One request every "
                                   f"{COOLDOWN_S:.0f} s; try again in {wait:.0f} s."}, 429)
            self.last_request[who] = now
        return self._json(self.channels["theater"].request(level))


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--ckpt", default=str(OUT / "campaign.pt"))
    ap.add_argument("--train-log", default=str(OUT / "campaign_train.log"))
    args = ap.parse_args()
    ck = Path(args.ckpt)
    Handler.train_log = Path(args.train_log)
    Handler.channels = {"live": Channel("live", ck), "theater": Channel("theater", ck)}
    for ch in Handler.channels.values():
        threading.Thread(target=ch.run, daemon=True, name=ch.name).start()
    handler = lambda *a, **k: Handler(*a, directory=str(SITE), **k)  # noqa: E731
    print(f"\n  The fly plays Mario -> http://localhost:{args.port}/\n", flush=True)
    with Server((args.host, args.port), handler) as srv:
        srv.serve_forever()


if __name__ == "__main__":
    main()
