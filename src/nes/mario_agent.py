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

        # Feature pooling, retinotopic rather than arbitrary.
        #
        # The first version pooled 92,743 neurons into 160 bins chosen by index
        # order, which discards *where* activity is -- and where is exactly what
        # you need to time a jump. The optic lobe carries ommatidial coordinates,
        # so its cells are pooled onto a spatial grid instead: the readout sees a
        # coarse map of the visual field rather than a bag of averages.
        #
        # Each cell also contributes its change since the previous frame. A
        # single frame cannot tell an approaching pipe from a receding one, and
        # the fly's optic lobe is built for motion above all else.
        GX, GY = 20, 16
        h1 = np.asarray(meta.get("hex1"), dtype=np.float32)
        h2 = np.asarray(meta.get("hex2"), dtype=np.float32)
        ol_global = ol
        hh1, hh2 = h1[ol_global], h2[ol_global]
        ok = np.isfinite(hh1) & np.isfinite(hh2)
        cells = [[] for _ in range(GX * GY)]
        if ok.sum() > 100:
            def norm(v):
                lo, hi = np.percentile(v[ok], 2), np.percentile(v[ok], 98)
                return np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
            u, w = norm(hh1), norm(hh2)
            gx = np.clip((np.nan_to_num(u, nan=.5) * (GX - 1)).round(), 0, GX - 1).astype(int)
            gy = np.clip((np.nan_to_num(w, nan=.5) * (GY - 1)).round(), 0, GY - 1).astype(int)
            loc = pos[ol_global]
            for k in range(len(ol_global)):
                if loc[k] >= 0:
                    cells[gy[k] * GX + gx[k]].append(loc[k])
        # Flatten the grid into one index array plus a cell id per entry, so
        # pooling is a single bincount rather than 320 separate mean() calls --
        # at this resolution that difference is 17.5s per episode against 1.5s.
        self.grid = [np.asarray(c, dtype=np.int64) for c in cells]
        gi, gc = [], []
        for ci, c in enumerate(self.grid):
            if c.size:
                gi.append(c); gc.append(np.full(c.size, ci, dtype=np.int64))
        self.grid_idx = np.concatenate(gi) if gi else np.zeros(0, np.int64)
        self.grid_cell = np.concatenate(gc) if gc else np.zeros(0, np.int64)
        self.grid_count = np.maximum(
            np.bincount(self.grid_cell, minlength=GX * GY), 1).astype(np.float32)
        pi, pc = [], []
        for bi, b in enumerate(np.array_split(self.photo, 24)):
            if b.size:
                pi.append(b); pc.append(np.full(b.size, bi, dtype=np.int64))
        self.photo_idx = np.concatenate(pi); self.photo_cell = np.concatenate(pc)
        self.photo_count = np.maximum(np.bincount(self.photo_cell, minlength=24),
                                      1).astype(np.float32)
        self.photo_bins = np.array_split(self.photo, 24)
        self.n_grid = GX * GY
        self.n_feat = self.n_grid * 2 + len(self.photo_bins)   # map + motion + photo
        self.prev_grid = np.zeros(self.n_grid, dtype=np.float32)
        used = sum(1 for c in self.grid if c.size)
        print(f"  retinotopic pooling: {GX}x{GY} grid, {used}/{self.n_grid} cells populated, "
              f"{self.n_feat} features")
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
        g = (np.bincount(self.grid_cell, weights=tr[self.grid_idx],
                         minlength=self.n_grid) / self.grid_count).astype(np.float32)
        d = g - self.prev_grid
        self.prev_grid = g
        p = (np.bincount(self.photo_cell, weights=tr[self.photo_idx],
                         minlength=24) / self.photo_count).astype(np.float32)
        return np.concatenate([g, d, p])

    def act(self, retina: np.ndarray, Wr: np.ndarray) -> int:
        self.ext = self._drive(retina)
        for _ in range(self.brain_steps):
            self.brain.step(self.ext)
        x = self.features()
        return int((x @ Wr[:-1] + Wr[-1]).argmax())

    # ------------------------------------------------------------------
    def episode(self, mario: Mario, Wr: np.ndarray, max_frames: int = 3000) -> int:
        retina = mario.reset()
        self.brain.reset()
        self.prev_px = None
        self.prev_grid = np.zeros(self.n_grid, dtype=np.float32)
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
        self.last_timeout = (f >= max_frames) and not mario.done
        return mario.max_x


ALL_STAGES = [f"SuperMarioBros-{w}-{t}-v0" for w in range(1, 9) for t in range(1, 5)]
HELD_OUT = ["SuperMarioBros-5-1-v0", "SuperMarioBros-6-1-v0",
            "SuperMarioBros-7-1-v0", "SuperMarioBros-2-3-v0",
            "SuperMarioBros-8-1-v0"]
TRAIN_POOL = [s for s in ALL_STAGES if s not in HELD_OUT]


_BASELINES = None


def stage_baseline(level: str) -> float:
    """Random-play mean distance on a stage, measured once and cached.

    Stages differ wildly in how far random play gets you -- 2-2 averages 1223,
    1-3 averages 318 -- so raw distance compared across different stages mostly
    measures which stage was drawn. Scoring relative to each stage's own random
    baseline makes 1.0 mean "as good as mashing buttons" everywhere.
    """
    global _BASELINES
    if _BASELINES is None:
        f = OUT / "stage_baselines.json"
        _BASELINES = json.loads(f.read_text()) if f.exists() else {}
    b = _BASELINES.get(level, {}).get("mean")
    return float(b) if b and b > 0 else 400.0


def evaluate(agent: MarioFly, levels, W, max_frames=3000) -> float:
    """Mean of distance / random-baseline over a set of stages.

    1.0 means random-level play, 2.0 twice as far as random. A raw-distance
    version of this scored a candidate drawn onto easy stages above a better one
    drawn onto hard ones, which turned selection into a lottery: 27 of 30
    generations "improved" while the step size inflated to its ceiling.

    Scoring on several stages is the whole point: optimising a single
    deterministic level rewards memorising a button sequence, and the agent that
    results does nothing anywhere else. A fixed set keeps the comparison between
    candidates fair -- a different draw each time would mostly measure luck.
    """
    rs = []
    for lv in levels:
        m = Mario(lv)
        x = agent.episode(m, W, max_frames=max_frames)
        m.env.close()
        rs.append(x / stage_baseline(lv))
    return float(np.mean(rs))


def evolve(agent: MarioFly, mario: Mario, generations: int, pop: int,
           sigma: float, lr: float, eps_per: int, seed: int = 0,
           stage_pool=None, stages_per_gen: int = 3, ckpt: Path | None = None,
           start_W=None, start_best: float = -1.0):
    """Elitist (1+lambda) ES, scored on stages redrawn every generation.

    Two things matter here beyond the usual. The stages are resampled each
    generation from the whole game, so a weight set cannot survive by memorising
    one level -- but they are held fixed *within* a generation, so candidates are
    compared on equal ground rather than on who drew the easier draw. And because
    the incumbent's score was measured on a different draw, it is re-scored on
    the new one before being compared, otherwise the elite would be defended by a
    number that no longer means anything.

    Progress is written to disk every generation: this is a run measured in hours
    and it should not have to start over.
    """
    rng = np.random.default_rng(seed)
    d = agent.n_feat + 1
    pool = stage_pool or TRAIN_POOL

    best_W = start_W if start_W is not None else \
        rng.normal(0, 0.05, (d, agent.n_actions)).astype(np.float32)
    best = start_best
    hist = []
    stale = 0
    for g in range(1, generations + 1):
        t0 = time.time()
        draw = [pool[i] for i in rng.choice(len(pool), stages_per_gen, replace=False)]
        # Re-score the incumbent on this generation's stages, so the comparison
        # is like for like.
        best = evaluate(agent, draw, best_W)
        # A candidate must beat the incumbent by a clear margin. Best-of-ten on
        # one small draw nearly always edges out the incumbent by luck alone --
        # 19 of 21 generations "improved" without any real trend -- and every
        # such false improvement also inflated the step size.
        gen_best, gen_best_W = best * 1.05, None
        for _ in range(pop):
            cand = best_W + sigma * rng.normal(0, 1, (d, agent.n_actions)).astype(np.float32)
            s_ = evaluate(agent, draw, cand)
            if s_ > gen_best:
                gen_best, gen_best_W = s_, cand
        improved = gen_best_W is not None
        if improved:
            best, best_W = gen_best, gen_best_W
            sigma = min(sigma * 1.12, 0.45); stale = 0
        else:
            stale += 1
            sigma = max(sigma * 0.88, 0.03)
            if stale >= 6:          # fully converged: re-open the search
                sigma = 0.25; stale = 0
        names = ",".join(x.split("-")[1] + "-" + x.split("-")[2] for x in draw)
        print(f"  gen {g:3d}  {best:5.2f}x random on [{names}]   sigma {sigma:.3f}"
              f"   ({time.time()-t0:.0f}s)" + ("  <-- improved" if improved else ""))
        hist.append({"gen": g, "score": float(best), "stages": draw,
                     "sigma": float(sigma), "improved": bool(improved)})
        if ckpt is not None:
            np.savez_compressed(ckpt, Wr=best_W, best=best, gen=g)
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
    ap.add_argument("--stages-per-gen", type=int, default=3)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--levels", default="1-1",
                    help="comma-separated stages to train on, e.g. 1-1,2-1,3-1,4-1. "
                         "Training on several is what stops it memorising one.")
    args = ap.parse_args()

    print("building brain ...")
    agent = MarioFly(brain_steps=args.brain_steps, frame_skip=args.frame_skip)
    print(f"  subnetwork {agent.n:,} neurons, {agent.W.nnz:,} synapses")
    print(f"  readout: {agent.n_feat} pooled features -> {agent.n_actions} buttons")
    if args.levels == "all":
        stages = TRAIN_POOL
        print(f"  training pool: {len(stages)} stages (all but {len(HELD_OUT)} held out)")
        print(f"  held out for evaluation: "
              + ", ".join(h.split('-')[1] + '-' + h.split('-')[2] for h in HELD_OUT))
    else:
        stages = [f"SuperMarioBros-{v.strip()}-v0" for v in args.levels.split(",")]
        print(f"  training across {len(stages)} stages: {args.levels}")
    mario = Mario(stages[0])
    print("calibrating on Mario's own frames ...")
    info = agent.calibrate(mario)
    print(f"  {info['final']['mean_hz']:.1f} Hz, silent {info['final']['silent_frac']:.0%}")

    print("\nbaselines:  constant actions <= 315   random 579")
    print("evolving the readout on distance travelled:")
    ckpt = OUT / f"{args.tag}_policy.npz"
    start_W, start_best = None, -1.0
    if args.resume and ckpt.exists():
        z = np.load(ckpt); start_W = z["Wr"]; start_best = float(z["best"])
        print(f"  resuming from {ckpt.name} at x={start_best:.0f} (gen {int(z['gen'])})")
    Wr, hist = evolve(agent, mario, args.generations, args.pop, args.sigma,
                      args.lr, args.eps_per, stage_pool=stages,
                      stages_per_gen=args.stages_per_gen, ckpt=ckpt,
                      start_W=start_W, start_best=start_best)

    held_out = HELD_OUT
    final = [agent.episode(Mario(lv), Wr) for lv in stages[:6]]
    unseen = [agent.episode(Mario(lv), Wr) for lv in held_out]
    print(f"\ntrained stages : {[int(v) for v in final]}  mean {np.mean(final):.0f}")
    print(f"never seen     : {[int(v) for v in unseen]}  mean {np.mean(unseen):.0f}")
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
