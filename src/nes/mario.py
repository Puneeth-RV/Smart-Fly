"""Super Mario Bros on a real NES emulator, wired to the connectome.

Why this and not the phone game: the emulator is a library, not a device. It
runs headless at ~574 fps, never shows an advert, restarts with a function call,
and reports `x_pos` continuously -- so partial competence is visible instead of
everything looking identical to failure. Driving Subway Surfers over adb gave
none of that.

The brain is the same one throughout: real wiring, transmitter-signed weights,
per-neuron resting drive fitted on real stimuli. Only the eyes change -- a NES
frame is reduced to the same 15x24 contrast retina the network was built around.

Three numpy-2 fixes are applied on import. nes-py and gym-super-mario-bros both
predate numpy 2, where uint8 arithmetic no longer silently widens: the ROM header
overflowed on `uint8 * 1024`, the game-state helpers overflowed on `byte * 0x100`,
and old gym refers to the removed np.bool8. They are patched in place rather than
worked around, because every one of them is a hard crash on import.
"""

from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")
for _old, _new in (("bool8", np.bool_), ("float_", np.float64), ("int_", np.int64)):
    if not hasattr(np, _old):
        setattr(np, _old, _new)

import gym_super_mario_bros  # noqa: E402
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT  # noqa: E402
from nes_py.wrappers import JoypadSpace  # noqa: E402

RETINA_W, RETINA_H = 15, 24
ACTIONS = ["none", "right", "run-jump", "run", "run-jump-b", "jump", "left"]


class Mario:
    """The game, plus the reduction from NES frame to the brain's retina."""

    RANDOM_STAGES = "SuperMarioBrosRandomStages-v0"

    def __init__(self, world: str = "SuperMarioBros-1-1-v0", seed: int = 0,
                 retina: tuple[int, int] = (RETINA_H, RETINA_W)):
        # (rows, cols). The default 24x15 is the upright phone shape inherited
        # from the runner; Mario's screen is wide, and (15, 24) gives the
        # direction that matters -- what is coming from the right -- 60% more
        # resolution for the same 360 cells.
        self.rh, self.rw = retina
        self.env = JoypadSpace(
            gym_super_mario_bros.make(world, apply_api_compatibility=True,
                                      render_mode="rgb_array"),
            SIMPLE_MOVEMENT,
        )
        self.n_actions = len(SIMPLE_MOVEMENT)
        self.frame = None
        self.info = {}
        self.reset()

    def reset(self):
        obs, info = self.env.reset()
        self.stage = (info.get("world"), info.get("stage")) if info else None
        self.frame = np.asarray(obs)
        self.info = info or {}
        self.done = False
        self.max_x = 0
        self._last_x = None          # anchored on the first real reading
        return self.observe()

    def step(self, action: int):
        obs, reward, term, trunc, info = self.env.step(int(action))
        self.frame = np.asarray(obs)
        self.info = info
        self.done = bool(term or trunc)
        # x_pos is two RAM bytes, and during some transitions (seen on 4-4) they
        # read as garbage up to 0xFFFF. Taken at face value one such frame
        # scored a run at 81x random. Mario cannot move more than a few pixels
        # a frame, so an implausible jump is treated as a glitch, not progress.
        x = int(info.get("x_pos", 0))
        if x > 5000:
            pass                        # garbage bytes: no level is this long
        elif self._last_x is None or not (0 <= x - self._last_x <= 24):
            # First reading, a pipe warp or a respawn: re-anchor here without
            # counting the jump. An earlier version only scored small steps
            # from 0, so the level's own starting offset locked it out and
            # every run scored zero.
            self._last_x = x
            self.max_x = max(self.max_x, x) if self.max_x == 0 else self.max_x
        else:
            self.max_x = max(self.max_x, x)
            self._last_x = x
        return self.observe(), float(reward), self.done, info

    # ------------------------------------------------------------------
    def observe(self) -> np.ndarray:
        """Reduce the 240x256 NES frame to the 15x24 contrast retina.

        The brain was wired against a retina of this shape, so the frame has to
        be brought to it rather than the other way round. The playfield is
        cropped above the HUD and below the ground line, then each retina cell
        takes the standard deviation of its block rather than the mean: Mario's
        world is flat colour with hard edges, so local contrast marks pipes,
        blocks, gaps and enemies far better than brightness does.
        """
        g = self.frame.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        g /= 255.0
        g = g[48:224, :]                       # drop the score HUD, keep the ground
        h, w = g.shape
        bh, bw = h // self.rh, w // self.rw
        blocks = g[: bh * self.rh, : bw * self.rw].reshape(self.rh, bh, self.rw, bw)
        out = blocks.std(axis=(1, 3))
        m = out.max()
        return (out / m) if m > 1e-6 else out


def random_baseline(episodes: int = 30, max_frames: int = 1500, seed: int = 0) -> dict:
    """What pressing buttons at random achieves, as the number to beat."""
    rng = np.random.default_rng(seed)
    m = Mario()
    xs = []
    for _ in range(episodes):
        m.reset()
        for _ in range(max_frames):
            m.step(int(rng.integers(m.n_actions)))
            if m.done:
                break
        xs.append(m.max_x)
    m.env.close()
    return {"episodes": episodes, "mean_x": float(np.mean(xs)),
            "median_x": float(np.median(xs)), "max_x": int(np.max(xs))}


def hold_right_baseline(episodes: int = 10, max_frames: int = 1500) -> dict:
    """Mario rewards holding right almost more than it rewards thinking.

    Worth measuring explicitly: any agent that cannot beat 'always press right'
    has not demonstrated anything, and that is a much harder bar than random.
    """
    m = Mario()
    xs = []
    for _ in range(episodes):
        m.reset()
        for _ in range(max_frames):
            m.step(1)                           # 'right'
            if m.done:
                break
        xs.append(m.max_x)
    m.env.close()
    return {"episodes": episodes, "mean_x": float(np.mean(xs)),
            "median_x": float(np.median(xs)), "max_x": int(np.max(xs))}


if __name__ == "__main__":
    import time
    m = Mario()
    r = m.observe()
    print(f"retina {r.shape}, range {r.min():.2f}-{r.max():.2f}")
    print("\nwhat the fly sees at the start of 1-1:")
    for row in r:
        print("  " + "".join("#" if v > .55 else ("+" if v > .18 else ".") for v in row))
    t0 = time.perf_counter()
    for _ in range(600):
        m.step(1)
        if m.done:
            m.reset()
    print(f"\n{600/(time.perf_counter()-t0):,.0f} fps with retina reduction")
    m.env.close()
    print("\nrandom      :", random_baseline(episodes=20))
    print("hold right  :", hold_right_baseline(episodes=8))
