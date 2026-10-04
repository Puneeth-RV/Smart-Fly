"""The connectome playing Super Mario Bros, with the readout evolved on distance.

The brain is fixed: real wiring, transmitter-signed weights, per-neuron resting
drive fitted against Mario's own frames rather than some other stimulus. Only the
map from neural activity to controller buttons is learned.

It is learned by evolution rather than gradients because Mario has no oracle --
there is no "correct action" to imitate, the way there was for the runner. An
evolution strategy needs only the score, and the score here is how far right
Mario got, which is exactly what we want to maximise.

The bar is honest and set in advance:
    every constant action  <= 315   (they all stall at the first pipe)
    random buttons            579   (its jitter clears the pipe)
so anything below ~580 is not evidence of anything.

Readout is taken from the photoreceptors and first-order optic lobe. That is
where the task signal still measurably exists; deeper populations were shown to
be at chance.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from brain.lif import ConnectomeBrain, LIFConfig, homeostatic_bias, load_connectome  # noqa: E402
from nes.mario import ACTIONS, Mario  # noqa: E402

OUT = ROOT / "out"


class MarioFly:
    """Connectome + retina + linear readout over early visual activity."""

    def __init__(self, brain_steps: int = 4, frame_skip: int = 4, seed: int = 0):
        W, raw, pops, meta = load_connectome(with_raw=True)
        photo = np.asarray(pops["photoreceptor_R16"], dtype=np.int64)
        ol = np.asarray(pops["optic_lobe"], dtype=np.int64)
        keep = np.unique(np.concatenate([photo, ol]))
        pos = -np.ones(W.shape[0], dtype=np.int64)
        pos[keep] = np.arange(keep.size)

        self.W = W[np.ix_(keep, keep)].tocsr()
        self.brain = ConnectomeBrain(
            self.W,
            LIFConfig(gain=1.19, noise=0.0, threshold_spread=0.35,
                      refractory=0, seed=seed),
            normalize=False,          # already normalised in the full matrix
        )
        self.photo = pos[photo]
        self.photo = self.photo[self.photo >= 0]
        direct = np.flatnonzero(np.asarray(abs(raw[photo]).sum(axis=0)).ravel() > 0)
        ol1 = np.intersect1d(direct, ol)
        self.ol1 = pos[ol1]; self.ol1 = self.ol1[self.ol1 >= 0]
        self.brain_steps = brain_steps
        self.frame_skip = frame_skip
        self.n = self.brain.n

        # Retinotopy: reuse the per-pixel assignment used everywhere else.
        from train.agent import AgentConfig, FlyAgent
        full = FlyAgent(W, raw, pops, AgentConfig(seed=seed),
                        LIFConfig(gain=1.19, seed=seed), meta=meta)
        self.photo_pixel = full.photo_pixel[: self.photo.size]
        self.cfg = full.cfg
        self.prev_px = None
        self.ext = np.zeros(self.n, dtype=np.float32)

        # Feature pooling: the readout sees pooled activity, not 92k neurons.
        self.n_feat = 160
        self.pools = [np.array_split(self.photo, 32),
                      np.array_split(self.ol1, 128)]
        self.n_actions = len(ACTIONS)

    # ------------------------------------------------------------------
    def calibrate(self, mario: Mario, iters: int = 30):
        """Fit each neuron's resting drive against Mario's own frames.

        Calibrating on a different stimulus distribution is the sort of mistake
        that silently costs you the signal, so the frames come from the game.
        """
        frames = []
        mario.reset()
        rng = np.random.default_rng(0)
        for _ in range(240):
            r, _, done, _ = mario.step(int(rng.integers(mario.n_actions)))
            frames.append(r)
            if done:
                mario.reset()
        i = {"k": 0}

        def nxt():
            f = frames[i["k"] % len(frames)]
            i["k"] += 1
            return self._drive(f)
        info = homeostatic_bias(self.brain, nxt, target_hz=6.0, iters=iters,
                                steps_per_frame=self.brain_steps)
        return info

    def _drive(self, retina: np.ndarray) -> np.ndarray:
        flat = retina.ravel()
        px = flat[self.photo_pixel]
        c = self.cfg
        prev = self.prev_px if self.prev_px is not None else px
        motion = prev - px
        self.prev_px = px.copy()
        e = np.zeros(self.n, dtype=np.float32)
        e[self.photo] = c.input_gain * (
            c.sustained_weight * (px.mean() - px) + c.motion_gain * motion)
        return e

    def features(self) -> np.ndarray:
        tr = self.brain.trace
        f = np.empty(self.n_feat, dtype=np.float32)
        k = 0
        for group in self.pools:
            for idx in group:
                f[k] = tr[idx].mean() if idx.size else 0.0
                k += 1
        return f

    def act(self, retina: np.ndarray, Wr: np.ndarray) -> int:
        self.ext = self._drive(retina)
        for _ in range(self.brain_steps):
            self.brain.step(self.ext)
        x = self.features()
        return int((x @ Wr[:-1] + Wr[-1]).argmax())

    # ------------------------------------------------------------------
    def episode(self, mario: Mario, Wr: np.ndarray, max_frames: int = 1600) -> int:
        retina = mario.reset()
        self.brain.reset()
        self.prev_px = None
        f = 0
        while f < max_frames:
            a = self.act(retina, Wr)
            for _ in range(self.frame_skip):     # hold the button, as a player would
                retina, _, done, _ = mario.step(a)
                f += 1
                if done:
                    break
            if mario.done:
                break
        return mario.max_x


def evolve(agent: MarioFly, mario: Mario, generations: int, pop: int,
           sigma: float, lr: float, eps_per: int, seed: int = 0):
    """Elitist (1+lambda) evolution strategy on the readout weights.

    The weighted-sum update used by OpenAI-style ES averages the whole
    population, which on a rugged landscape walks away from good solutions: a
    run of it found a candidate worth 1506 and then drifted back to 297. Level
    1-1 is deterministic, so a weight set that scores well scores well every
    time and there is no reason to average it away. This keeps the best thing
    found so far and only ever replaces it with something strictly better, so
    the curve cannot go backwards.

    Step size adapts: shrink after a generation that fails to improve, grow a
    little after one that does, which concentrates the search once it is onto
    something.
    """
    rng = np.random.default_rng(seed)
    d = agent.n_feat + 1

    def score(W):
        return float(np.mean([agent.episode(mario, W) for _ in range(eps_per)]))

    best_W = rng.normal(0, 0.05, (d, agent.n_actions)).astype(np.float32)
    best = score(best_W)
    hist = [{"gen": 0, "best": best, "sigma": sigma}]
    print(f"  gen  0  best x = {best:6.0f}")
    stale = 0
    for g in range(1, generations + 1):
        t0 = time.time()
        gen_best, gen_best_W = -1.0, None
        for _ in range(pop):
            cand = best_W + sigma * rng.normal(0, 1, (d, agent.n_actions)).astype(np.float32)
            s_ = score(cand)
            if s_ > gen_best:
                gen_best, gen_best_W = s_, cand
        improved = gen_best > best
        if improved:
            best, best_W = gen_best, gen_best_W
            sigma = min(sigma * 1.15, 0.4)
            stale = 0
        else:
            stale += 1
            sigma = max(sigma * 0.82, 0.02)
        hist.append({"gen": g, "best": best, "gen_best": gen_best,
                     "sigma": float(sigma), "improved": bool(improved)})
        print(f"  gen {g:2d}  best x = {best:6.0f}   this gen {gen_best:6.0f}"
              f"   sigma {sigma:.3f}   ({time.time()-t0:.0f}s)"
              + ("  <-- new best" if improved else ""))
    return best_W, hist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=0.12)
    ap.add_argument("--lr", type=float, default=0.06)
    ap.add_argument("--eps-per", type=int, default=1,
                    help="level 1-1 is deterministic; >1 only costs time")
    ap.add_argument("--brain-steps", type=int, default=4)
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--tag", default="mario")
    args = ap.parse_args()

    print("building brain ...")
    agent = MarioFly(brain_steps=args.brain_steps, frame_skip=args.frame_skip)
    print(f"  subnetwork {agent.n:,} neurons, {agent.W.nnz:,} synapses")
    print(f"  readout: {agent.n_feat} pooled features -> {agent.n_actions} buttons")
    mario = Mario()
    print("calibrating on Mario's own frames ...")
    info = agent.calibrate(mario)
    print(f"  {info['final']['mean_hz']:.1f} Hz, silent {info['final']['silent_frac']:.0%}")

    print("\nbaselines:  constant actions <= 315   random 579")
    print("evolving the readout on distance travelled:")
    Wr, hist = evolve(agent, mario, args.generations, args.pop,
                      args.sigma, args.lr, args.eps_per)

    final = [agent.episode(mario, Wr) for _ in range(10)]
    print(f"\nfinal: mean x {np.mean(final):.0f}, best {max(final)}, over 10 episodes")
    print(f"       random 579 | best constant 315")
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / f"{args.tag}_policy.npz", Wr=Wr)
    (OUT / f"{args.tag}.json").write_text(json.dumps(
        {"history": hist, "final_mean": float(np.mean(final)),
         "final_best": int(max(final)), "random_baseline": 579,
         "best_constant": 315}, indent=2))
    print(f"wrote {OUT/f'{args.tag}_policy.npz'}")
    mario.env.close()


if __name__ == "__main__":
    main()
