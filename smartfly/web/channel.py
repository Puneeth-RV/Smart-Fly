"""A channel: one copy of the fly playing continuously, observable from outside.

The site runs two:

    live     the level the campaign is training on, with the newest weights,
             reloaded between attempts -- so it improves as you watch
    theater  a level the fly has mastered, 1-1 until someone asks for another;
             it stays on the level asked for last, with requests queued

Both play the top choice only, from random starts, exactly as the campaign
tests. Every decision publishes a snapshot of what the fly saw and did: its
retina, photoreceptors, optic-lobe maps, attention, the buttons it weighed, and
the reward that followed. Frames are published twice as often, for the stream.
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

from smartfly.campaign import STATE
from smartfly.game import ACTIONS, Mario
from smartfly.paths import MODELS
from smartfly.train import FLAG_BONUS, STUCK_LIMIT

DECISIONS_PER_SECOND = 15.0
SHOW_EVERY = 2  # game frames per published frame: 30 a second at 15 decisions
RASTER_NEURONS = 480


def campaign_state() -> dict:
    """The running campaign's state, or the snapshot shipped with the trained fly."""
    for path in (STATE, MODELS / "campaign-state.json"):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            continue
    return {}


def _jpeg(frame: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frame.astype(np.uint8), "RGB").save(buf, "JPEG", quality=80)
    return buf.getvalue()


def _r3(values) -> list[float]:
    return [round(float(v), 3) for v in values]


class Channel:
    def __init__(self, name: str, checkpoint: Path, max_queue: int = 6):
        self.name, self.checkpoint, self.max_queue = name, checkpoint, max_queue
        self.lock = threading.Lock()
        self.state: dict = {"channel": name, "ready": False, "error": None}
        self.jpeg: bytes | None = None
        self.frame_id = 0
        self.fresh = threading.Condition(self.lock)
        self.queue: deque[str] = deque()
        self.history: deque[dict] = deque(maxlen=40)
        self._stay: str | None = None

    # ----------------------------------------------------------------- queue
    def request(self, level: str) -> dict:
        """Queue a mastered level for the theater."""
        if level not in campaign_state().get("mastered", []):
            return {"ok": False, "reason": "That level has not been mastered yet."}
        with self.lock:
            if self.state.get("level") == level:
                return {"ok": True, "position": 0}
            if level in self.queue:
                return {"ok": True, "position": list(self.queue).index(level) + 1}
            if len(self.queue) >= self.max_queue:
                return {"ok": False, "reason": "The queue is full; try again shortly."}
            self.queue.append(level)
            return {"ok": True, "position": len(self.queue)}

    def _next_level(self) -> str | None:
        st = campaign_state()
        if self.name == "live":
            return st.get("current")
        with self.lock:
            if self.queue:
                self._stay = self.queue.popleft()
        mastered = st.get("mastered", [])
        if self._stay not in mastered:
            self._stay = mastered[0] if mastered else None
        return self._stay

    def _publish(self, jpeg: bytes) -> None:
        with self.fresh:
            self._show(jpeg)

    def _show(self, jpeg: bytes) -> None:  # with the lock held
        self.jpeg = jpeg
        self.frame_id += 1
        self.fresh.notify_all()

    def next_frame(self, after: int, timeout: float = 10.0) -> tuple[int, bytes | None]:
        """The newest frame once it is newer than `after` (or the timeout passes)."""
        with self.fresh:
            self.fresh.wait_for(lambda: self.frame_id != after, timeout)
            return self.frame_id, self.jpeg

    def snapshot(self) -> dict:
        with self.lock:
            return self.state | {"queue": list(self.queue), "history": list(self.history)}

    # ------------------------------------------------------------------ play
    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:  # surface it on the page, don't die silently
            with self.lock:
                self.state["error"] = f"{type(exc).__name__}: {exc}"
            raise

    def _run(self) -> None:
        from smartfly.policy import Player

        while not self.checkpoint.exists():
            time.sleep(5)
        player = Player(self.checkpoint)
        fly = player.fly
        lay = fly.layout
        order = np.argsort(fly.map_cell, kind="stable")
        raster = fly.map_neurons[order[np.linspace(0, order.size - 1, RASTER_NEURONS).astype(int)]]
        loaded = self.checkpoint.stat().st_mtime
        games: dict[str, Mario] = {}
        rng = np.random.default_rng()
        attempts = 0
        with self.lock:
            self.state["ready"] = True

        while True:
            level = self._next_level()
            if level is None:
                with self.lock:
                    self.state.update(level=None, idle=True)
                time.sleep(3)
                continue
            if self.checkpoint.stat().st_mtime != loaded:  # newer weights from training
                player = Player(self.checkpoint, fly=fly)
                loaded = self.checkpoint.stat().st_mtime
            if level not in games:
                games[level] = Mario(level)
            game = games[level]
            retina = game.reset()
            for _ in range(int(rng.integers(0, 31))):
                retina, _, done, _ = game.step(0)
                if done:
                    retina = game.reset()
            player.reset()
            attempts += 1
            flag, best, since, total, decisions = False, 0, 0, 0.0, 0
            rewards: deque = deque(maxlen=160)
            info: dict = {}
            while not game.done and since < STUCK_LIMIT and decisions < 1500 and not self.queue:
                t0 = time.time()
                seen = retina.copy()
                d = player.decide(retina, attention=True)
                fired = np.flatnonzero(fly.brain.active[raster]).tolist()
                reward, died = 0.0, False
                for i in range(1, player.frame_skip + 1):
                    retina, r, done, info = game.step(d.action)
                    reward += float(np.clip(r, -25, 15))
                    flag |= bool(info.get("flag_get"))
                    if done:
                        died = not info.get("flag_get")
                        break
                    if i % SHOW_EVERY == 0 and i < player.frame_skip:
                        self._publish(_jpeg(game.frame))
                        due = t0 + i / (player.frame_skip * DECISIONS_PER_SECOND)
                        time.sleep(max(0.0, due - time.time()))
                if flag:
                    reward += FLAG_BONUS
                total += reward / 10
                decisions += 1
                if game.max_x > best:
                    best, since = game.max_x, 0
                else:
                    since += 1
                rewards.append(
                    [
                        round(reward / 10, 2),
                        round(d.value, 2),
                        "flag" if flag else ("death" if died else ""),
                    ]
                )
                jpeg = _jpeg(game.frame)
                f = d.features
                with self.fresh:
                    self._show(jpeg)
                    self.state.update(
                        {
                            "idle": False,
                            "level": level,
                            "attempt": attempts,
                            "x": int(info.get("x_pos", 0)),
                            "max_x": int(game.max_x),
                            "action": ACTIONS[d.action],
                            "actions": ACTIONS,
                            "probs": _r3(d.probs),
                            "value": round(d.value, 2),
                            "return": round(total, 2),
                            "reward_trace": list(rewards),
                            "ckpt_frames": round(player.frames_trained / 1e6, 2),
                            "eye": _r3(seen.ravel()),
                            "eye_shape": list(seen.shape),
                            "photo": _r3(f[lay["eye"]]),
                            "deep": _r3(f[lay["deep"]]),
                            "motion": _r3(f[lay["motion"]]),
                            "first": _r3(f[lay["first"]]),
                            "grid": [20, 16],
                            "attention": _r3(d.attention),
                            "raster": fired,
                            "raster_n": RASTER_NEURONS,
                            "active": int(fly.brain.active.sum()),
                            "t": time.time(),
                        }
                    )
                time.sleep(max(0.0, 1.0 / DECISIONS_PER_SECOND - (time.time() - t0)))
            if self.queue:  # cut short for a request: not a result
                continue
            with self.lock:
                result = {
                    "level": level,
                    "x": int(game.max_x),
                    "flag": flag,
                    "t": time.time(),
                    "ckpt_frames": round(player.frames_trained / 1e6, 2),
                }
                self.history.appendleft(result)
                self.state["last_result"] = result
            time.sleep(1.5 if flag else 0.8)
