"""Super Mario Bros on the NES emulator, and the frame-to-retina reduction.

The emulator (gym-super-mario-bros) runs headless at several hundred frames a
second, restarts with a function call and reports Mario's position every frame.
Each frame is reduced to the fly's retina: 15 rows x 24 columns of local
contrast covering the playfield.
"""

from __future__ import annotations

import warnings

import numpy as np

from smartfly import compat

RETINA = (15, 24)  # rows, columns: a wide eye for a wide screen
PLAYFIELD = slice(48, 224)  # frame rows below the score bar, above the floor's edge
ACTIONS = ["none", "right", "run-jump", "run", "run-jump-b", "jump", "left"]
LEVELS = [f"{w}-{s}" for w in range(1, 9) for s in range(1, 5)]


def retina(frame: np.ndarray, shape: tuple[int, int] = RETINA) -> np.ndarray:
    """Reduce a 240x256 RGB frame to the retina, values in [0, 1].

    Each cell is the standard deviation of brightness within its block, not the
    mean: Mario's world is flat colour with hard edges, so local contrast marks
    pipes, blocks, gaps and enemies far better than brightness does.
    """
    rows, cols = shape
    g = frame.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    g = g[PLAYFIELD, :] / 255.0
    bh, bw = g.shape[0] // rows, g.shape[1] // cols
    out = g[: bh * rows, : bw * cols].reshape(rows, bh, cols, bw).std(axis=(1, 3))
    peak = out.max()
    return out / peak if peak > 1e-6 else out


class Mario:
    """One level of the game, observed through the retina.

    `max_x` is the furthest Mario has got. The emulator's `x_pos` is two RAM
    bytes that briefly read as garbage (up to 0xFFFF) during some transitions,
    and taken at face value one such frame once scored a run at 81x random, so
    a reading that jumps further than Mario can move is ignored.

    `warped` turns true when Mario jumps forward through a pipe, as at the end
    of 1-2.
    """

    def __init__(self, level: str = "1-1", retina_shape: tuple[int, int] = RETINA):
        compat.check()
        warnings.filterwarnings("ignore", module="gym")
        import gym_super_mario_bros
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        from nes_py.wrappers import JoypadSpace

        self.level = level
        self.retina_shape = retina_shape
        self.env = JoypadSpace(
            gym_super_mario_bros.make(
                f"SuperMarioBros-{level}-v0", apply_api_compatibility=True, render_mode="rgb_array"
            ),
            SIMPLE_MOVEMENT,
        )
        self.reset()

    @property
    def ram(self) -> np.ndarray:
        env = self.env
        while not hasattr(env, "ram"):
            env = env.env
        return env.ram

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset()
        self.frame = np.asarray(obs)
        self.done = False
        self.max_x = 0
        self.warped = False
        self._last_x: int | None = None
        return self.observe()

    def observe(self) -> np.ndarray:
        return retina(self.frame, self.retina_shape)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        obs, reward, term, trunc, info = self.env.step(int(action))
        self.frame = np.asarray(obs)
        self.done = bool(term or trunc)
        x = int(info.get("x_pos", 0))
        if x > 5000:
            pass  # garbage bytes
        elif self._last_x is None or not 0 <= x - self._last_x <= 24:
            # First reading, a pipe warp or a respawn: re-anchor without counting it.
            if self._last_x is not None and x > self._last_x:
                self.warped = True
            self._last_x = x
            if self.max_x == 0:
                self.max_x = x
        else:
            self.max_x = max(self.max_x, x)
            self._last_x = x
        return self.observe(), float(reward), self.done, info

    def close(self) -> None:
        self.env.close()
