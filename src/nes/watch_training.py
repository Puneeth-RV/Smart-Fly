"""Watch the fly learn Mario while PPO trains it.

Runs beside the trainer without touching it. The trainer saves a checkpoint after
every update; this loads the newest one, plays a level with it at roughly real
speed, and reloads after every attempt -- so what you watch is always the policy
as it stands right now, and you can see it get better.

It uses the same frozen fly brain as training (same calibration file) and samples
actions from the policy the way training does, so its success rate is
representative of what the training log reports.

Endpoints:  /frame.jpg  /state  and the page itself at /training.html
"""

from __future__ import annotations

import argparse
import io
import json
import re
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
OUT = ROOT / "out"

STATE = {"ready": False, "error": None}
FRAME = {}
LOCK = threading.Lock()


def parse_log(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.startswith("update"):
            continue
        try:
            rows.append({
                "update": int(re.search(r"update +(\d+)", line).group(1)),
                "frames": float(re.search(r"frames +([\d.]+)M", line).group(1)),
                "mean_x": float(re.search(r"mean x +([\d.]+)", line).group(1)),
                "max_x": int(re.search(r"max +(\d+)", line).group(1)),
                "flags": int(re.search(r"flags (\d+)/", line).group(1)),
                "of": int(re.search(r"flags \d+/(\d+)", line).group(1)),
                "total_flags": int(re.search(r"total flags (\d+)", line).group(1)),
            })
        except Exception:
            pass
    return rows


def play_loop(level, tag, log_path, calib, brain_steps, frame_skip, dps, policy="greedy"):
    import torch
    from nes.mario import ACTIONS, Mario
    from nes.mario_agent import MarioFly
    from nes.ppo_fly import Memory, build_net, make_fly

    try:
        ck = OUT / f"{tag}.pt"
        while not ck.exists():
            time.sleep(5)
        targs = torch.load(ck, weights_only=False)["args"]
        k = int(targs.get("stack", 1))
        agent, retina_shape = make_fly(int(targs.get("brain_steps", brain_steps)), frame_skip,
                                       calib, bool(targs.get("v2", False)))
        mem = Memory(agent.n_feat, k)
        net = build_net(agent.n_feat * k + len(ACTIONS) if k > 1 else agent.n_feat,
                        len(ACTIONS))
        nf = agent.n_feat
        # A fixed sample of optic-lobe neurons for the spike raster, ordered by
        # where in the visual field they sit, so the raster reads top-to-bottom
        # as the fly's view rather than as arbitrary neuron ids.
        order = np.argsort(agent.grid_cell, kind="stable")
        pick = order[np.linspace(0, order.size - 1, 600).astype(int)]
        raster_ids = agent.grid_idx[pick]
        mario = Mario(f"SuperMarioBros-{level}-v0", retina=retina_shape)
        attempts, wins, hist = 0, 0, []
        with LOCK:
            STATE["ready"] = True
        while True:
            # Newest weights for every attempt, so the viewer tracks training.
            s = torch.load(ck, weights_only=False)
            net.load_state_dict(s["net"])
            mean, var, _ = s["norm"]
            ck_age = time.time() - ck.stat().st_mtime
            retina = mario.reset()
            agent.brain.reset(); agent.prev_px = None
            agent.prev_grid = np.zeros(agent.n_grid, dtype=np.float32)
            flag, dec, a = False, 0, None
            trace, ret = [], 0.0
            while not mario.done:
                t0 = time.time()
                agent.ext = agent._drive(retina)
                fired = np.zeros(raster_ids.size, dtype=bool)
                n_spk = 0
                for _ in range(agent.brain_steps):
                    sp_ = agent.brain.step(agent.ext)
                    fired |= sp_[raster_ids]
                    n_spk += int(sp_.sum())
                f = agent.features()
                ol_map, ol_motion = f[:agent.n_grid].copy(), f[agent.n_grid:2 * agent.n_grid].copy()
                eye = retina.copy()
                if k > 1:
                    f = mem.reset(f) if a is None else mem.push(f, a)
                x = np.clip((f - mean) / np.sqrt(var + 1e-8), -6, 6)
                xt = torch.from_numpy(x.astype(np.float32))[None].requires_grad_(True)
                logits, value = net(xt)
                probs = torch.softmax(logits, -1)[0].detach().numpy().astype(np.float64)
                probs /= probs.sum()
                # Top choice only by default: that is the version that has to know
                # what to do. Sampling would let lucky dice pass for skill.
                if policy == "greedy":
                    a = int(probs.argmax())
                else:
                    a = int(np.random.choice(len(probs), p=probs))
                # Where it was looking: gradient x input of the chosen button with
                # respect to each optic-lobe patch. The patches are laid out by the
                # fly's own ommatidial coordinates, and the retina is wired with the
                # same layout, so a patch maps back to a region of the screen.
                logits[0, a].backward()
                gx = (xt.grad[0] * xt[0]).abs().detach().numpy()
                ng = agent.n_grid
                sal = sum(gx[j * nf:j * nf + ng] + gx[j * nf + ng:j * nf + 2 * ng]
                          + (gx[j * nf + 2 * ng + 24:j * nf + 3 * ng + 24] if agent.graded else 0)
                          for j in range(k))
                sal = sal / (sal.max() + 1e-9)
                # The same reward training uses: progress right, the clock, death,
                # and the bonus for the flag. Scaled the same way (/10).
                rew, died = 0.0, False
                for _ in range(agent.frame_skip):
                    retina, r_, d, info = mario.step(a)
                    rew += float(np.clip(r_, -25, 15))
                    if info.get("flag_get"):
                        flag = True
                    if d:
                        died = not info.get("flag_get")
                        break
                if flag:
                    rew += 50.0
                ret += rew / 10
                trace.append([round(rew / 10, 2), round(float(value[0]), 2),
                              "flag" if flag else ("death" if died else "")])
                trace = trace[-160:]
                dec += 1
                img = Image.fromarray(mario.frame.astype(np.uint8), "RGB")
                buf = io.BytesIO(); img.save(buf, "JPEG", quality=82)
                with LOCK:
                    FRAME["jpeg"] = buf.getvalue()
                    STATE.update({
                        "x": int(info.get("x_pos", 0)), "max_x": int(mario.max_x),
                        "action": ACTIONS[a], "probs": [round(float(p), 3) for p in probs],
                        "value": round(float(value[0]), 3), "attempt": attempts + 1,
                        "ckpt_frames": float(s.get("frames", 0)) / 1e6,
                        "ckpt_age_s": round(ck_age),
                        "spikes": n_spk,
                        "eye": [round(float(v), 2) for v in eye.ravel()],
                        "eye_shape": list(eye.shape),
                        "ol": [round(float(v), 3) for v in ol_map],
                        "motion": [round(float(v), 3) for v in ol_motion],
                        "raster": np.flatnonzero(fired).tolist(),
                        "raster_n": int(raster_ids.size),
                        "reward_trace": trace, "return": round(ret, 2),
                        "attention": [round(float(v), 3) for v in sal],
                        "grid": [20, 16], "policy": policy,
                    })
                if dec > 900:            # a stuck policy should not hold the screen forever
                    break
                time.sleep(max(0.0, 1.0 / dps - (time.time() - t0)))
            attempts += 1
            wins += int(flag)
            hist.append({"attempt": attempts, "x": int(mario.max_x), "flag": flag,
                         "ckpt_frames": float(s.get("frames", 0)) / 1e6})
            with LOCK:
                STATE.update({"attempts": attempts, "wins": wins,
                              "viewer_history": hist[-60:], "last_flag": flag})
            time.sleep(1.2 if flag else 0.6)   # a beat on the end screen
    except Exception as exc:
        with LOCK:
            STATE["error"] = f"{type(exc).__name__}: {exc}"
        raise


class Handler(SimpleHTTPRequestHandler):
    log_path = None

    def log_message(self, *a):
        pass

    def _send(self, body, ctype):
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
                st = dict(STATE)
            st["training"] = parse_log(Handler.log_path)
            return self._send(json.dumps(st).encode(), "application/json")
        if self.path.startswith("/frame.jpg"):
            with LOCK:
                j = FRAME.get("jpeg")
            if j is None:
                self.send_response(503); self.end_headers(); return
            return self._send(j, "image/jpeg")
        return super().do_GET()


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--level", default="1-1")
    ap.add_argument("--tag", default="ppo_fly_11")
    ap.add_argument("--log", default=str(OUT / "ppo_fly_11b.log"))
    ap.add_argument("--calibration", default=str(OUT / "mario_final_calibration.npz"))
    ap.add_argument("--brain-steps", type=int, default=3)
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--policy", default="greedy", choices=["greedy", "sampled"])
    ap.add_argument("--dps", type=float, default=15.0,
                    help="decisions per second; 15 x 4 frames is real game speed")
    args = ap.parse_args()
    Handler.log_path = Path(args.log)
    threading.Thread(target=play_loop, daemon=True,
                     args=(args.level, args.tag, args.log, args.calibration,
                           args.brain_steps, args.frame_skip, args.dps,
                           args.policy)).start()
    web = ROOT / "out" / "scene"
    handler = lambda *a, **k: Handler(*a, directory=str(web), **k)  # noqa: E731
    print(f"\n  Watch the fly learn -> http://localhost:{args.port}/training.html\n")
    with Server(("", args.port), handler) as srv:
        srv.serve_forever()


if __name__ == "__main__":
    main()
