"""A channel: one fly brain playing Mario continuously, observable from outside.

The public site runs two of these side by side:

  live     the level the campaign is training on now, played with the newest
           checkpoint, reloaded between attempts -- so it improves as you watch
  theater  levels the fly has already mastered, played on request (a queue);
           with nothing queued it replays mastered levels in turn

Either way the fly plays its top choice only, from a random idle start, the
same way the campaign tests it. Every decision publishes the game frame and a
snapshot of what the fly saw and did: its eye, its optic lobe, where it was
attending, the buttons it weighed and the reward that followed.
"""

from __future__ import annotations

import io
import json
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out"
CAMPAIGN = OUT / "campaign_state.json"


def campaign_state() -> dict | None:
    try:
        return json.loads(CAMPAIGN.read_text())
    except Exception:
        return None


class Channel:
    def __init__(self, name: str, ckpt: Path, dps: float = 15.0, max_queue: int = 6):
        self.name, self.ckpt, self.dps = name, ckpt, dps
        self.lock = threading.Lock()
        self.state: dict = {"channel": name, "ready": False, "error": None}
        self.jpeg: bytes | None = None
        self.queue: deque[str] = deque()
        self.max_queue = max_queue
        self.history: deque[dict] = deque(maxlen=40)
        self._rotate = 0

    # ---------------------------------------------------------------- queue
    def request(self, level: str) -> dict:
        """Queue a mastered level. Returns the outcome for the caller."""
        st = campaign_state() or {}
        if level not in st.get("mastered", []):
            return {"ok": False, "reason": "That level has not been mastered yet."}
        with self.lock:
            if level in self.queue or self.state.get("level") == level:
                pos = list(self.queue).index(level) + 1 if level in self.queue else 0
                return {"ok": True, "position": pos, "note": "already queued"}
            if len(self.queue) >= self.max_queue:
                return {"ok": False, "reason": "The queue is full; try again shortly."}
            self.queue.append(level)
            return {"ok": True, "position": len(self.queue)}

    def _next_level(self) -> str | None:
        st = campaign_state() or {}
        if self.name == "live":
            return st.get("current")
        with self.lock:
            if self.queue:
                return self.queue.popleft()
        mastered = st.get("mastered", [])
        if not mastered:
            return None
        self._rotate = (self._rotate + 1) % len(mastered)
        return mastered[self._rotate]

    # ----------------------------------------------------------------- play
    def snapshot(self) -> dict:
        with self.lock:
            s = dict(self.state)
            s["queue"] = list(self.queue)
            s["history"] = list(self.history)
        return s

    def run(self):
        try:
            self._run()
        except Exception as exc:                       # surface, don't die silently
            with self.lock:
                self.state["error"] = f"{type(exc).__name__}: {exc}"
            raise

    def _run(self):
        import torch
        from nes.mario import ACTIONS, Mario
        from nes.ppo_fly import Memory, build_net, make_fly

        while not self.ckpt.exists():
            time.sleep(5)
        targs = torch.load(self.ckpt, weights_only=False)["args"]
        k = int(targs.get("stack", 1))
        agent, shape = make_fly(int(targs.get("brain_steps", 6)), int(targs.get("frame_skip", 4)),
                                targs.get("calibration"), bool(targs.get("v2", False)))
        mem = Memory(agent.n_feat, k)
        net = build_net(agent.n_feat * k + len(ACTIONS) if k > 1 else agent.n_feat, len(ACTIONS))
        nf, ng = agent.n_feat, agent.n_grid
        order = np.argsort(agent.grid_cell, kind="stable")
        raster_ids = agent.grid_idx[order[np.linspace(0, order.size - 1, 480).astype(int)]]
        games: dict[str, Mario] = {}
        rng = np.random.default_rng()
        attempts = 0
        loaded_mtime = None
        with self.lock:
            self.state["ready"] = True

        while True:
            level = self._next_level()
            if level is None:
                with self.lock:
                    self.state.update(level=None, idle=True)
                time.sleep(3)
                continue
            mt = self.ckpt.stat().st_mtime
            if mt != loaded_mtime:
                s = torch.load(self.ckpt, weights_only=False)
                net.load_state_dict(s["net"])
                mean, var, _ = s["norm"]
                ck_frames = float(s.get("frames", 0)) / 1e6
                loaded_mtime = mt
            if level not in games:
                games[level] = Mario(f"SuperMarioBros-{level}-v0", retina=shape)
            m = games[level]
            retina = m.reset()
            for _ in range(int(rng.integers(0, 31))):
                retina, _, d, _ = m.step(0)
                if d:
                    retina = m.reset()
            agent.brain.reset(); agent.prev_px = None
            agent.prev_grid = np.zeros(ng, np.float32)
            a, flag, dec, best, since, ret = None, False, 0, 0, 0, 0.0
            trace: list = []
            attempts += 1
            while not m.done and since < 120 and dec < 1500:
                t0 = time.time()
                eye = retina.copy()
                agent.ext = agent._drive(retina)
                fired = np.zeros(raster_ids.size, bool)
                active = 0
                for _ in range(agent.brain_steps):
                    sp = agent.brain.step(agent.ext)
                    fired |= sp[raster_ids]
                    active += int(sp.sum())
                f = agent.features()
                deep, motion = f[:ng].copy(), f[ng:2 * ng].copy()
                first = f[2 * ng + 24:3 * ng + 24].copy() if agent.graded else None
                obs = mem.reset(f) if a is None else mem.push(f, a)
                x = np.clip((obs - mean) / np.sqrt(var + 1e-8), -6, 6).astype(np.float32)
                xt = torch.from_numpy(x)[None].requires_grad_(True)
                logits, value = net(xt)
                probs = torch.softmax(logits, -1)[0].detach().numpy().astype(np.float64)
                a = int(probs.argmax())
                logits[0, a].backward()
                gx = (xt.grad[0] * xt[0]).abs().detach().numpy()
                sal = sum(gx[j * nf:j * nf + ng] + gx[j * nf + ng:j * nf + 2 * ng]
                          + (gx[j * nf + 2 * ng + 24:j * nf + 3 * ng + 24] if agent.graded else 0)
                          for j in range(k))
                sal = sal / (sal.max() + 1e-9)

                rew, died = 0.0, False
                for _ in range(agent.frame_skip):
                    retina, r_, d, info = m.step(a)
                    rew += float(np.clip(r_, -25, 15))
                    if info.get("flag_get"):
                        flag = True
                    if d:
                        died = not info.get("flag_get")
                        break
                if flag:
                    rew += 50.0
                ret += rew / 10
                dec += 1
                if m.max_x > best:
                    best, since = m.max_x, 0
                else:
                    since += 1
                trace.append([round(rew / 10, 2), round(float(value[0]), 2),
                              "flag" if flag else ("death" if died else "")])
                trace = trace[-160:]

                buf = io.BytesIO()
                Image.fromarray(m.frame.astype(np.uint8), "RGB").save(buf, "JPEG", quality=80)
                r3 = lambda v: [round(float(t), 3) for t in v]  # noqa: E731
                with self.lock:
                    self.jpeg = buf.getvalue()
                    self.state.update({
                        "idle": False, "level": level, "attempt": attempts,
                        "x": int(info.get("x_pos", 0)), "max_x": int(m.max_x),
                        "action": ACTIONS[a], "actions": ACTIONS, "probs": r3(probs),
                        "value": round(float(value[0]), 2), "return": round(ret, 2),
                        "reward_trace": trace, "ckpt_frames": round(ck_frames, 2),
                        "eye": r3(eye.ravel()), "eye_shape": list(eye.shape),
                        "deep": r3(deep), "motion": r3(motion),
                        "first": None if first is None else r3(first),
                        "grid": [20, 16], "attention": r3(sal),
                        "raster": np.flatnonzero(fired).tolist(), "raster_n": int(raster_ids.size),
                        "active": active, "t": time.time(),
                    })
                time.sleep(max(0.0, 1.0 / self.dps - (time.time() - t0)))

            with self.lock:
                self.history.appendleft({"level": level, "x": int(m.max_x), "flag": flag,
                                         "t": time.time(), "ckpt_frames": round(ck_frames, 2)})
                self.state.update(last_result={"level": level, "flag": flag, "x": int(m.max_x)})
            time.sleep(1.5 if flag else 0.8)
