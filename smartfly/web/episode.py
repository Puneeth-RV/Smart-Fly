"""One attempt at a level, decision by decision, with everything the page shows.

Shared by the live channels and the recorder, so a replay is exactly what the
live site would have shown.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from smartfly.game import ACTIONS, Mario
from smartfly.policy import Player
from smartfly.train import FLAG_BONUS, STUCK_LIMIT

RASTER_NEURONS = 480
MAX_DECISIONS = 1500


def raster_neurons(player: Player) -> np.ndarray:
    """A fixed sample of optic-lobe neurons, ordered top of the view to bottom."""
    fly = player.fly
    order = np.argsort(fly.map_cell, kind="stable")
    return fly.map_neurons[order[np.linspace(0, order.size - 1, RASTER_NEURONS).astype(int)]]


@dataclass
class Step:
    frame: np.ndarray  # the game screen after the buttons were pressed
    state: dict  # what the page shows for this decision
    flag: bool
    died: bool


def _r3(values) -> list[float]:
    return [round(float(v), 3) for v in values]


def play_attempt(
    player: Player,
    game: Mario,
    rng: np.random.Generator,
    raster: np.ndarray,
    condition: str = "normal",
    start_jitter: int = 30,
    show_every: int = 0,
    on_frame: Callable[[np.ndarray], None] | None = None,
    stop: Callable[[], bool] = lambda: False,
) -> Iterator[Step]:
    """Play one attempt from a random start, top choice only, as the campaign tests.

    `condition` is what the eye is shown: the game ("normal"), a blank screen
    ("blind") or the same pixels in random positions ("scrambled"). `on_frame`
    sees every `show_every`-th game frame within a decision; `stop` ends it early.
    """
    fly, lay = player.fly, player.fly.layout
    retina = game.reset()
    for _ in range(int(rng.integers(0, start_jitter + 1))):
        retina, _, done, _ = game.step(0)
        if done:
            retina = game.reset()
    player.reset()
    flag, best, since, total, decisions = False, 0, 0, 0.0, 0
    while not game.done and since < STUCK_LIMIT and decisions < MAX_DECISIONS and not stop():
        seen = retina
        if condition == "blind":
            seen = np.zeros_like(retina)
        elif condition == "scrambled":
            seen = rng.permutation(retina.ravel()).reshape(retina.shape)
        d = player.decide(seen, attention=True)
        fired = np.flatnonzero(fly.brain.active[raster]).tolist()
        reward, died, info = 0.0, False, {}
        for i in range(1, player.frame_skip + 1):
            retina, r, done, info = game.step(d.action)
            reward += float(np.clip(r, -25, 15))
            flag |= bool(info.get("flag_get"))
            if done:
                died = not info.get("flag_get")
                break
            if on_frame and i % show_every == 0 and i < player.frame_skip:
                on_frame(game.frame)
        if flag:
            reward += FLAG_BONUS
        total += reward / 10
        decisions += 1
        if game.max_x > best:
            best, since = game.max_x, 0
        else:
            since += 1
        f = d.features
        state = {
            "x": int(info.get("x_pos", 0)),
            "max_x": int(game.max_x),
            "action": ACTIONS[d.action],
            "actions": ACTIONS,
            "probs": _r3(d.probs),
            "value": round(d.value, 2),
            "reward": round(reward / 10, 2),
            "return": round(total, 2),
            "ckpt_frames": round(player.frames_trained / 1e6, 2),
            "eye": _r3(seen.ravel()),
            "eye_shape": list(seen.shape),
            "photo": _r3(f[lay["eye"]]),
            "deep": _r3(f[lay["deep"]]),
            "grid": [20, 16],
            "attention": _r3(d.attention),
            "raster": fired,
            "raster_n": RASTER_NEURONS,
            "active": int(fly.brain.active.sum()),
        }
        yield Step(game.frame, state, flag, died)
