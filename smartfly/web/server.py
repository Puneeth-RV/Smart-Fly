"""The site server: two channels behind a small read-only API, plus the page in site/.

    GET  /api/state/<live|theater>      what the fly is seeing and doing now
    GET  /api/frame/<live|theater>.jpg  the current game frame
    GET  /api/campaign                  levels, tests and training progress
    POST /api/theater {"level": "1-1"}  ask the theater for a mastered level

Visitors can only watch and request mastered levels: the queue is capped and
each address may ask once every 20 seconds. It binds to 127.0.0.1; to share
it, put a tunnel in front (e.g. `cloudflared tunnel --url http://localhost:8090`)
rather than opening a port.

    python -m smartfly.web.server
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import socketserver
import threading
import time
from http.server import SimpleHTTPRequestHandler
from pathlib import Path

from smartfly.campaign import FLY, LOG
from smartfly.paths import SITE
from smartfly.web.channel import Channel, campaign_state

LEVEL = re.compile(r"^[1-8]-[1-4]$")
COOLDOWN_S = 20.0
PROGRESS = re.compile(r"update +\d+ +frames +([\d.]+)M .*mean x +([\d.]+) .*flags (\d+)/(\d+)")


def training_progress(log: Path, limit: int = 400) -> list[dict]:
    """The trainer's progress lines, newest `limit`, for the charts."""
    if not log.exists():
        return []
    rows = []
    for line in log.read_text(errors="ignore").splitlines()[-4000:]:
        m = PROGRESS.search(line)
        if m:
            rows.append(
                {"frames": float(m[1]), "mean_x": float(m[2]), "flags": int(m[3]), "of": int(m[4])}
            )
    return rows[-limit:]


def campaign_summary() -> dict:
    """Levels, tests and training progress, as the page's campaign view needs them."""
    st = campaign_state()
    return {
        "order": st.get("order", []),
        "current": st.get("current"),
        "mastered": st.get("mastered", []),
        "parked": st.get("parked", []),
        "tests": st.get("tests", [])[-40:],
        "training": training_progress(LOG),
    }


class Handler(SimpleHTTPRequestHandler):
    channels: dict[str, Channel] = {}
    last_request: dict[str, float] = {}
    request_lock = threading.Lock()

    def log_message(self, *args):
        pass

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()

    def list_directory(self, path):  # no directory listings
        self.send_error(404)

    def _send(self, body: bytes, content_type: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(json.dumps(obj).encode(), "application/json", code)

    def _visitor(self) -> str:
        # Behind a tunnel every request arrives from localhost; it passes the visitor on.
        return (
            self.headers.get("CF-Connecting-IP")
            or self.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or self.client_address[0]
        )

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
            return self._json(campaign_summary())
        if path in ("", "/"):
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/theater":
            return self._json({"error": "not found"}, 404)
        try:
            n = min(int(self.headers.get("Content-Length", 0)), 256)
            level = str(json.loads(self.rfile.read(n) or b"{}").get("level", ""))
        except (ValueError, AttributeError):
            return self._json({"ok": False, "reason": "Bad request."}, 400)
        if not LEVEL.match(level):
            return self._json({"ok": False, "reason": "Unknown level."}, 400)
        who, now = self._visitor(), time.time()
        with self.request_lock:
            wait = COOLDOWN_S - (now - self.last_request.get(who, 0))
            if wait > 0:
                return self._json(
                    {
                        "ok": False,
                        "reason": f"One request every {COOLDOWN_S:.0f} s; "
                        f"try again in {wait:.0f} s.",
                    },
                    429,
                )
            self.last_request[who] = now
        return self._json(self.channels["theater"].request(level))


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the live site.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--checkpoint", default=str(FLY), help="the fly to play")
    args = ap.parse_args()
    ckpt = Path(args.checkpoint)
    Handler.channels = {"live": Channel("live", ckpt), "theater": Channel("theater", ckpt)}
    for ch in Handler.channels.values():
        threading.Thread(target=ch.run, daemon=True, name=ch.name).start()

    def handler(*a, **k):
        return Handler(*a, directory=str(SITE), **k)

    print(f"\n  Smart-Fly -> http://localhost:{args.port}/\n", flush=True)
    with Server((args.host, args.port), handler) as srv:
        srv.serve_forever()


if __name__ == "__main__":
    main()
