"""Serve the fly playing Mario, live.

The brain runs here in Python rather than in the page: it is 92,743 neurons and
3.7M synapses and the browser only has to draw what comes back. Nothing is
recorded -- the frame you see is the one the fly is looking at right now, and
the buttons were chosen by the spikes it produced a few milliseconds ago.

Endpoints:
    /frame.jpg   the NES screen, as the fly sees it happen
    /retina.png  the 15x24 contrast retina actually fed to the photoreceptors
    /state       JSON: action, x position, spikes, per-population activity
"""

from __future__ import annotations

import argparse
import io
import json
import socketserver
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nes.mario import ACTIONS, Mario  # noqa: E402
from nes.mario_agent import MarioFly  # noqa: E402

STATE = {"ready": False, "error": None, "action": 0, "action_name": "none",
         "x": 0, "max_x": 0, "best_ever": 0, "spikes": 0, "fps": 0.0,
         "episode": 0, "activity": {}, "retina": "",
         "baseline_random": 579, "baseline_constant": 315}
FRAME = {"jpeg": None}
LOCK = threading.Lock()
STOP = threading.Event()


SPLIT = {}


def _baseline(stage: str) -> float:
    try:
        from nes.mario_agent import stage_baseline
        return float(stage_baseline(f"SuperMarioBros-{stage}-v0"))
    except Exception:
        return 0.0


def _load_split():
    try:
        from nes.train_all import TRAIN, VALID, TEST
        for s_ in TRAIN: SPLIT[s_] = "trained on"
        for s_ in VALID: SPLIT[s_] = "validation"
        for s_ in TEST: SPLIT[s_] = "never seen"
    except Exception:
        pass


def play_loop(policy_path: Path, brain_steps: int, frame_skip: int, quality: int,
              calib_path: Path | None = None, stages: list | None = None):
    try:
        _load_split()
        agent = MarioFly(brain_steps=brain_steps, frame_skip=frame_skip)
        with LOCK:
            STATE["neurons"] = int(agent.n)
            STATE["synapses"] = int(agent.W.nnz)
        # Use the exact calibration the policy was trained with; recalibrating
        # here would hand the readout a slightly different brain.
        if calib_path is not None and calib_path.exists():
            z = np.load(calib_path)
            agent.brain.bias = z["bias"].astype(np.float32)
            agent.brain.v_th = z["v_th"].astype(np.float32)
        else:
            m0 = Mario(); agent.calibrate(m0, iters=30); m0.env.close()
        stages = stages or [f"{w}-{t}" for w in range(1, 9) for t in range(1, 5)]
        si = 0
        mario = Mario(f"SuperMarioBros-{stages[si]}-v0")

        Wr = np.load(policy_path)["Wr"] if policy_path.exists() else None
        if Wr is None:
            print(f"[no policy at {policy_path}; the fly will act on untrained weights]")
            rng = np.random.default_rng(0)
            Wr = rng.normal(0, 0.05, (agent.n_feat + 1, agent.n_actions)).astype(np.float32)

        pools = {"photoreceptors": agent.photo, "optic lobe": agent.grid_idx}
        with LOCK:
            STATE["ready"] = True

        retina = mario.reset()
        agent.brain.reset(); agent.prev_px = None
        n, t0, ep, best = 0, time.time(), 1, 0
        while not STOP.is_set():
            a = agent.act(retina, Wr)
            for _ in range(agent.frame_skip):
                retina, _, done, info = mario.step(a)
                if done:
                    break
            n += 1

            img = Image.fromarray(mario.frame.astype(np.uint8), "RGB")
            img = img.resize((256, 240), Image.NEAREST)
            buf = io.BytesIO(); img.save(buf, "JPEG", quality=quality)
            rbuf = io.BytesIO()
            Image.fromarray((np.clip(retina, 0, 1) * 255).astype(np.uint8), "L") \
                 .resize((15 * 6, 24 * 6), Image.NEAREST).save(rbuf, "PNG")

            tr = agent.brain.trace
            dt = time.time() - t0
            best = max(best, mario.max_x)
            with LOCK:
                FRAME["jpeg"] = buf.getvalue()
                FRAME["retina"] = rbuf.getvalue()
                STATE.update({
                    "stage": stages[si], "split": SPLIT.get(stages[si], ""),
                    "stage_baseline": _baseline(stages[si]),
                    "action": int(a), "action_name": ACTIONS[a],
                    "x": int(info.get("x_pos", 0)), "max_x": int(mario.max_x),
                    "best_ever": int(best), "episode": ep,
                    "spikes": int(agent.brain.spikes.sum()),
                    "fps": round(n / dt, 1) if dt > 0 else 0.0,
                    "activity": {k: round(float((tr[v] > 0).mean()), 4)
                                 for k, v in pools.items()},
                })
            if info.get("flag_get"):
                with LOCK:
                    STATE.setdefault("completed", [])
                    if stages[si] not in STATE["completed"]:
                        STATE["completed"].append(stages[si])
            if mario.done:
                # One attempt per stage, then on to the next: the page walks the
                # whole game rather than replaying the level it learned best.
                ep += 1
                with LOCK:
                    STATE.setdefault("results", {})[stages[si]] = {
                        "x": int(mario.max_x), "base": _baseline(stages[si]),
                        "split": SPLIT.get(stages[si], "")}
                si = (si + 1) % len(stages)
                mario.env.close()
                mario = Mario(f"SuperMarioBros-{stages[si]}-v0")
                retina = mario.reset()
                agent.brain.reset(); agent.prev_px = None
                agent.prev_grid = np.zeros(agent.n_grid, dtype=np.float32)
        mario.env.close()
    except Exception as exc:
        with LOCK:
            STATE["error"] = f"{type(exc).__name__}: {exc}"
        raise


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path.startswith("/state"):
            with LOCK:
                return self._send(json.dumps(STATE).encode(), "application/json")
        if self.path.startswith("/frame.jpg"):
            with LOCK:
                j = FRAME.get("jpeg")
            if j is None:
                self.send_response(503); self.end_headers(); return
            return self._send(j, "image/jpeg")
        if self.path.startswith("/retina.png"):
            with LOCK:
                r = FRAME.get("retina")
            if r is None:
                self.send_response(503); self.end_headers(); return
            return self._send(r, "image/png")
        return super().do_GET()


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--policy", default=str(ROOT / "out" / "mario_final_policy.npz"))
    ap.add_argument("--calibration",
                    default=str(ROOT / "out" / "mario_final_calibration.npz"))
    ap.add_argument("--brain-steps", type=int, default=3)
    ap.add_argument("--frame-skip", type=int, default=6)
    ap.add_argument("--quality", type=int, default=82)
    args = ap.parse_args()

    t = threading.Thread(target=play_loop,
                         args=(Path(args.policy), args.brain_steps,
                               args.frame_skip, args.quality,
                               Path(args.calibration)), daemon=True)
    t.start()
    web = ROOT / "out" / "scene"
    handler = lambda *a, **k: Handler(*a, directory=str(web), **k)  # noqa: E731
    print(f"\n  The fly plays Mario -> http://localhost:{args.port}/mario.html")
    print("  (the brain calibrates first; the page says when it is ready)\n")
    with Server(("", args.port), handler) as srv:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            STOP.set()


if __name__ == "__main__":
    main()
