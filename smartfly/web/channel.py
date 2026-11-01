"""A channel: one copy of the fly playing continuously, observable from outside.

The site runs two:

    live     the level the campaign is training on, with the newest weights,
             reloaded between attempts -- so it improves as you watch
    theater  levels the fly has mastered, played on request from a queue, and
             in turn when nothing is queued

Both play the top choice only, from random starts, exactly as the campaign
tests. Every decision publishes the frame and a snapshot of what the fly saw and
did: its retina, photoreceptors, optic-lobe maps, attention, the buttons it
weighed, and the reward that followed.
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
from smartfly.game import Mario
from smartfly.paths import MODELS
from smartfly.web.episode import play_attempt, raster_neurons

DECISIONS_PER_SECOND = 15.0


def campaign_state() -> dict:
    """The running campaign's state, or the snapshot shipped with the trained fly."""
    for path in (STATE, MODELS / "campaign-state.json"):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            continue
    return {}


class Channel:
    def __init__(self, name: str, checkpoint: Path, max_queue: int = 6):
        self.name, self.checkpoint, self.max_queue = name, checkpoint, max_queue
        self.lock = threading.Lock()
        self.state: dict = {"channel": name, "ready": False, "error": None}
        self.jpeg: bytes | None = None
        self.queue: deque[str] = deque()
        self.history: deque[dict] = deque(maxlen=40)
        self._rotation = 0

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
                return self.queue.popleft()
        mastered = st.get("mastered", [])
        if not mastered:
            return None
        self._rotation = (self._rotation + 1) % len(mastered)
        return mastered[self._rotation]

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
        raster = raster_neurons(player)
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
            attempts += 1
            rewards: deque = deque(maxlen=160)
            flag, t0 = False, time.time()
            for step in play_attempt(player, game, rng, raster):
                flag = step.flag
                s = step.state
                rewards.append(
                    [s["reward"], s["value"], "flag" if flag else ("death" if step.died else "")]
                )
                buf = io.BytesIO()
                Image.fromarray(step.frame.astype(np.uint8), "RGB").save(buf, "JPEG", quality=80)
                with self.lock:
                    self.jpeg = buf.getvalue()
                    self.state.update(
                        s,
                        idle=False,
                        level=level,
                        attempt=attempts,
                        reward_trace=list(rewards),
                        t=time.time(),
                    )
                time.sleep(max(0.0, 1.0 / DECISIONS_PER_SECOND - (time.time() - t0)))
                t0 = time.time()
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
