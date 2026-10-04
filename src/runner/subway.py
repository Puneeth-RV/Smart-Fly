"""A headless, fast Subway-Surfers-style endless runner.

This is the training ground for the connectome agent. The real game on an
emulator runs at ~10fps with ~100ms touch latency, which is far too slow to
collect the 1e5-1e6 transitions that learning needs, so the brain is trained
here and transferred afterwards.

Fidelity choices that matter for transfer:

  * Three lanes, obstacles scrolling toward a player pinned at z=0.
  * The same five actions the real game accepts: none / left / right / up / down,
    where up==jump and down==roll.
  * Speed ramps with distance, as it does in the real game.
  * Obstacle taxonomy matches what the player must actually discriminate:
    a train must be dodged laterally, a low barrier jumped, a high barrier
    rolled under. Choosing the wrong evasion is fatal, so the agent cannot
    succeed with a single reflex.
  * Levels are generated guaranteed-solvable: every depth has at least one
    passable lane. An unwinnable frame would poison the reward signal.

The observation is a small grayscale "retina" rather than the underlying state
vector, because the whole point is to drive photoreceptors. Real game frames get
reduced to this same shape and encoding, so the brain sees one representation
across both environments.
"""

from __future__ import annotations

import numpy as np

# Obstacle encoding. 0 is free space.
EMPTY = 0
TRAIN = 1  # blocks the lane outright; must be avoided by changing lane
LOW = 2  # jump over it
HIGH = 3  # roll under it
COIN = 4  # optional reward, never lethal

# Actions, in the order the real game's swipes map to.
A_NONE, A_LEFT, A_RIGHT, A_JUMP, A_ROLL = range(5)
N_ACTIONS = 5

N_LANES = 3

# Retina dimensions. Deliberately tiny: the fly's visual system is ~800
# ommatidia per eye, and a coarse field keeps the LIF simulation tractable.
RETINA_H = 24
RETINA_W = 15  # divisible by N_LANES so each lane owns 5 columns


class SubwayEnv:
    """Endless runner with a retina observation.

    Parameters
    ----------
    depth:
        How many rows of track are held in front of the player. Also the
        vertical extent of what the retina can see.
    jump_frames, roll_frames:
        Duration of the evasive states. While airborne the player clears LOW;
        while rolling it clears HIGH. Neither helps against TRAIN.
    base_speed, max_speed, speed_ramp:
        Rows advanced per step, growing with distance travelled.
    coin_rate:
        Probability of seeding a coin in a free cell.
    seed:
        RNG seed, for reproducible level sequences.
    """

    def __init__(
        self,
        depth: int = 24,
        jump_frames: int = 6,
        roll_frames: int = 6,
        base_speed: float = 0.34,
        max_speed: float = 1.0,
        speed_ramp: float = 4000.0,
        coin_rate: float = 0.06,
        crash_penalty: float = -8.0,
        coin_reward: float = 0.5,
        progress_reward: float = 0.02,
        lane_change_frames: int = 2,
        seed: int | None = None,
    ) -> None:
        self.depth = depth
        self.jump_frames = jump_frames
        self.roll_frames = roll_frames
        self.base_speed = base_speed
        self.max_speed = max_speed
        self.speed_ramp = speed_ramp
        self.coin_rate = coin_rate
        self.crash_penalty = crash_penalty
        self.coin_reward = coin_reward
        self.progress_reward = progress_reward
        self.lane_change_frames = lane_change_frames
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------
    # level generation
    # ------------------------------------------------------------------
    def _blank_track(self) -> np.ndarray:
        return np.zeros((self.depth, N_LANES), dtype=np.int8)

    def _spawn_row(self) -> np.ndarray:
        """Generate one row of track that always leaves an escape.

        At least one lane is left free of lethal obstacles. TRAIN is the only
        obstacle that cannot be answered in place, so rows are built by first
        choosing a guaranteed-safe lane and only then filling the others.
        """
        row = np.zeros(N_LANES, dtype=np.int8)
        safe_lane = int(self.rng.integers(N_LANES))

        # Difficulty grows with distance: more obstacles, more trains.
        difficulty = min(1.0, self.distance / 3000.0)
        fill_p = 0.25 + 0.45 * difficulty

        for lane in range(N_LANES):
            if lane == safe_lane:
                continue
            if self.rng.random() < fill_p:
                # Trains become relatively more common as difficulty rises.
                weights = np.array([0.34 + 0.3 * difficulty, 0.33, 0.33])
                weights /= weights.sum()
                row[lane] = int(self.rng.choice([TRAIN, LOW, HIGH], p=weights))

        if self.rng.random() < self.coin_rate:
            free = np.flatnonzero(row == EMPTY)
            if free.size:
                row[int(self.rng.choice(free))] = COIN
        return row

    # ------------------------------------------------------------------
    # gym-style API
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self.track = self._blank_track()
        self.lane = 1
        self.target_lane = 1
        self.lane_timer = 0
        self.air = 0  # frames of jump remaining
        self.low = 0  # frames of roll remaining
        self.distance = 0.0
        self.scroll = 0.0
        self.score = 0
        self.coins = 0
        self.steps = 0
        self.done = False
        # Give the player a few empty rows of runway before the first obstacle.
        for d in range(self.depth):
            self.track[d] = self._spawn_row() if d > 6 else 0
        return self.observe()

    @property
    def speed(self) -> float:
        ramp = min(1.0, self.distance / self.speed_ramp)
        return self.base_speed + (self.max_speed - self.base_speed) * ramp

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset()")

        reward = 0.0
        self.steps += 1

        # --- apply action -------------------------------------------------
        # Lane changes are only accepted when not mid-change, mirroring the
        # real game's input debouncing.
        if action == A_LEFT and self.lane_timer == 0 and self.target_lane > 0:
            self.target_lane -= 1
            self.lane_timer = self.lane_change_frames
        elif action == A_RIGHT and self.lane_timer == 0 and self.target_lane < N_LANES - 1:
            self.target_lane += 1
            self.lane_timer = self.lane_change_frames
        elif action == A_JUMP and self.air == 0 and self.low == 0:
            self.air = self.jump_frames
        elif action == A_ROLL and self.low == 0 and self.air == 0:
            self.low = self.roll_frames

        # --- advance evasive-state timers ---------------------------------
        if self.lane_timer > 0:
            self.lane_timer -= 1
            if self.lane_timer == 0:
                self.lane = self.target_lane
        if self.air > 0:
            self.air -= 1
        if self.low > 0:
            self.low -= 1

        # --- scroll the world --------------------------------------------
        self.scroll += self.speed
        rows_advanced = int(self.scroll)
        self.scroll -= rows_advanced

        for _ in range(rows_advanced):
            arriving = self.track[0].copy()
            self.track[:-1] = self.track[1:]
            self.track[-1] = self._spawn_row()
            self.distance += 1.0
            reward += self.progress_reward

            # Collision is resolved against the lane the player occupies. While
            # mid-change the player is treated as straddling both lanes, so a
            # change begun too late still clips the obstacle.
            occupied = {self.lane}
            if self.lane_timer > 0:
                occupied.add(self.target_lane)

            for lane in occupied:
                cell = arriving[lane]
                if cell == COIN:
                    self.coins += 1
                    reward += self.coin_reward
                elif cell == TRAIN:
                    reward += self.crash_penalty
                    self.done = True
                elif cell == LOW and self.air == 0:
                    reward += self.crash_penalty
                    self.done = True
                elif cell == HIGH and self.low == 0:
                    reward += self.crash_penalty
                    self.done = True
            if self.done:
                break

        self.score = int(self.distance) + 10 * self.coins
        info = {
            "distance": self.distance,
            "coins": self.coins,
            "score": self.score,
            "speed": self.speed,
            "lane": self.lane,
        }
        return self.observe(), reward, self.done, info

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------
    def observe(self) -> np.ndarray:
        """Render the world as a small grayscale retina in [0, 1].

        Layout is a straight-on view: each lane is a vertical band, nearer
        track is lower in the image. Obstacle classes get distinct intensities
        so a downstream visual system can learn to discriminate them, and the
        player's own state is drawn into the bottom rows -- flies do receive
        proprioceptive and efference-copy signals, so this is not cheating.
        """
        img = np.zeros((RETINA_H, RETINA_W), dtype=np.float32)
        band = RETINA_W // N_LANES
        view_h = RETINA_H - 3

        # Projective depth mapping. A one-row-per-depth rendering gives a near
        # obstacle a footprint of ~1% of the retina, which is far too small a
        # perturbation to survive propagation through the brain -- measurably so:
        # the motor representation was identical with and without it. Under
        # perspective, screen extent falls off as 1/distance, so the imminent
        # obstacle that actually matters dominates the visual field while distant
        # track compresses toward the horizon. This is both what the real game
        # looks like and what makes the signal large enough to act on.
        k = 3.0
        edge = [int(round(view_h * (1.0 - k / (k + d)))) for d in range(self.depth + 1)]
        for d in range(min(self.depth, view_h)):
            r1 = view_h - edge[d]  # nearer boundary (lower on screen)
            r0 = view_h - edge[d + 1]  # farther boundary
            if r1 <= r0:
                r0 = max(0, r1 - 1)
            for lane in range(N_LANES):
                cell = self.track[d, lane]
                if cell == EMPTY:
                    continue
                intensity = {TRAIN: 1.0, LOW: 0.66, HIGH: 0.40, COIN: 0.18}[int(cell)]
                x0 = lane * band
                img[max(0, r0) : max(0, r1), x0 : x0 + band] = intensity

        # Self-state block: which lane we're in, and whether airborne/rolling.
        band_x0 = self.lane * band
        img[view_h, band_x0 : band_x0 + band] = 1.0
        if self.lane_timer > 0:
            t_x0 = self.target_lane * band
            img[view_h, t_x0 : t_x0 + band] = 0.5
        if self.air > 0:
            img[view_h + 1, :] = self.air / self.jump_frames
        if self.low > 0:
            img[view_h + 2, :] = self.low / self.roll_frames
        return img

    # ------------------------------------------------------------------
    # debugging
    # ------------------------------------------------------------------
    def render_ascii(self) -> str:
        """Text view of the track, nearest row last. For eyeballing behaviour."""
        glyph = {EMPTY: " . ", TRAIN: "[T]", LOW: "_L_", HIGH: "‾H‾", COIN: " o "}
        lines = []
        for d in range(min(self.depth, 18) - 1, -1, -1):
            lines.append("".join(glyph[int(c)] for c in self.track[d]))
        state = "air" if self.air else ("roll" if self.low else "run")
        marker = ["   "] * N_LANES
        marker[self.lane] = " ^ "
        lines.append("".join(marker))
        lines.append(f"dist={int(self.distance)} coins={self.coins} {state} v={self.speed:.2f}")
        return "\n".join(lines)


def random_rollout(episodes: int = 200, seed: int = 0) -> dict:
    """Baseline statistics for a uniformly random policy.

    Any learned agent has to beat this to be worth anything, so it is the
    reference point every training run is measured against.
    """
    env = SubwayEnv(seed=seed)
    rng = np.random.default_rng(seed + 1)
    dists, rewards = [], []
    for _ in range(episodes):
        env.reset()
        total = 0.0
        while not env.done:
            _, r, _, info = env.step(int(rng.integers(N_ACTIONS)))
            total += r
        dists.append(info["distance"])
        rewards.append(total)
    return {
        "episodes": episodes,
        "mean_distance": float(np.mean(dists)),
        "median_distance": float(np.median(dists)),
        "max_distance": float(np.max(dists)),
        "mean_return": float(np.mean(rewards)),
    }


if __name__ == "__main__":
    import time

    env = SubwayEnv(seed=7)
    print(env.render_ascii())
    print()

    # Throughput matters: the whole clone-then-transfer plan rests on this
    # being orders of magnitude faster than the emulator.
    env.reset()
    n = 200_000
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    steps = 0
    while steps < n:
        if env.done:
            env.reset()
        env.step(int(rng.integers(N_ACTIONS)))
        steps += 1
    dt = time.perf_counter() - t0
    print(f"throughput: {steps / dt:,.0f} steps/s ({dt:.2f}s for {steps:,})")
    print()
    print("random policy baseline:", random_rollout(episodes=300))
