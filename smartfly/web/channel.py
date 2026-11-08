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
from smartfly.game import Mario
from smartfly.paths import MODELS
from smartfly.web.episode import play_attempt, raster_neurons

DECISIONS_PER_SECOND = 15.0
SHOW_EVERY = 2  # game frames per published frame: 30 a second at 15 decisions


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
        self._due = 0.0

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

    def _wait_turn(self, frame_s: float) -> None:
        """Hold each frame until its slot, so frames leave evenly spaced."""
        now = time.time()
        self._due = max(self._due + frame_s, now - frame_s)  # after a stall, resume, don't rush
        time.sleep(max(0.0, self._due - now))

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
        raster = raster_neurons(player)
        loaded = self.checkpoint.stat().st_mtime
        games: dict[str, Mario] = {}
        rng = np.random.default_rng()
        attempts = 0
        with self.lock:
            self.state["ready"] = True

        def between(frame: np.ndarray) -> None:
            jpeg = _jpeg(frame)
            self._wait_turn(frame_s)
            self._publish(jpeg)

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
            frame_s = SHOW_EVERY / (player.frame_skip * DECISIONS_PER_SECOND)
            flag = False
            steps = play_attempt(
                player,
                game,
                rng,
                raster,
                show_every=SHOW_EVERY,
                on_frame=between,
                stop=lambda: bool(self.queue),
            )
            for step in steps:
                flag = step.flag
                s = step.state
                rewards.append(
                    [s["reward"], s["value"], "flag" if flag else ("death" if step.died else "")]
                )
                jpeg = _jpeg(step.frame)
                self._wait_turn(frame_s)
                with self.fresh:
                    self._show(jpeg)
                    self.state.update(
                        s,
                        idle=False,
                        level=level,
                        attempt=attempts,
                        reward_trace=list(rewards),
                        t=time.time(),
                    )
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
