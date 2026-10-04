"""Train the connectome's Mario readout across the whole game, in parallel.

Why this replaces the elitist search
------------------------------------
Keeping the single best candidate was right on one deterministic level: a good
weight set stays good, so there is nothing to average away. Across random stages
it is wrong. Scores swing from 0.9x to 2.9x random depending on the draw, and a
tiny weight change can send Mario down an entirely different path, so the best
of ten candidates nearly always beats the incumbent by chance -- 30 of 38
generations "improved" while the average stayed flat and the step size sat at
its ceiling. When the signal is noisy you average over it rather than select on
it.

So this is a standard evolution strategy:
  * antithetic pairs (theta + s*e, theta - s*e), which cancels much of the noise
  * fitness by rank, so one freak run cannot dominate the update
  * every candidate scored on the same stages within a generation
  * an Adam step on the estimated gradient

Data split
----------
32 stages split three ways: 22 to train on, 5 to choose the checkpoint with, and
5 never touched until the very end. Choosing the checkpoint on the final test
stages would quietly make them part of training.

Scores are distance divided by that stage's random-play average (measured, 8
runs per stage), so 1.0 means "as good as mashing buttons" on any level.

Games are independent, so they run across worker processes. Calibration happens
once in the parent and is handed to every worker, so all of them run exactly the
same brain.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "out"

ALL = [f"{w}-{t}" for w in range(1, 9) for t in range(1, 5)]
TEST = ["5-1", "6-1", "7-1", "2-3", "8-1"]          # untouched until the end
VALID = ["1-3", "3-2", "4-1", "6-2", "7-3"]         # checkpoint selection only
TRAIN = [s for s in ALL if s not in TEST and s not in VALID]

_AGENT = None


def _init_worker(calib_path: str, brain_steps: int, frame_skip: int):
    global _AGENT
    import warnings
    warnings.filterwarnings("ignore")
    from nes.mario_agent import MarioFly
    _AGENT = MarioFly(brain_steps=brain_steps, frame_skip=frame_skip)
    z = np.load(calib_path)
    _AGENT.brain.bias = z["bias"].astype(np.float32)
    _AGENT.brain.v_th = z["v_th"].astype(np.float32)


def _play(args):
    """One game: (weights, stage) -> distance / random baseline."""
    W, stage = args
    from nes.mario import Mario
    from nes.mario_agent import stage_baseline
    lv = f"SuperMarioBros-{stage}-v0"
    m = Mario(lv)
    try:
        x = _AGENT.episode(m, W)
    finally:
        m.env.close()
    return x / stage_baseline(lv)


def score_many(pool, Ws, stages):
    jobs = [(W, s) for W in Ws for s in stages]
    r = np.asarray(pool.map(_play, jobs, chunksize=1), dtype=np.float64)
    return r.reshape(len(Ws), len(stages)).mean(axis=1)


def centred_ranks(x):
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--stages", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=0.08)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--brain-steps", type=int, default=3)
    ap.add_argument("--frame-skip", type=int, default=6)
    ap.add_argument("--init", default=str(OUT / "marioall_policy.npz"))
    ap.add_argument("--tag", default="mario_final")
    ap.add_argument("--validate-every", type=int, default=8)
    args = ap.parse_args()

    from nes.mario import Mario
    from nes.mario_agent import MarioFly

    print(f"stages: {len(TRAIN)} train, {len(VALID)} validation, {len(TEST)} test")
    print("  test (never trained or selected on): " + ", ".join(TEST))

    # Calibrate once, then every worker loads the identical brain.
    agent = MarioFly(brain_steps=args.brain_steps, frame_skip=args.frame_skip)
    m = Mario("SuperMarioBros-1-1-v0")
    info = agent.calibrate(m, iters=30); m.env.close()
    calib = OUT / f"{args.tag}_calibration.npz"
    np.savez_compressed(calib, bias=agent.brain.bias, v_th=agent.brain.v_th)
    print(f"calibrated: {info['final']['mean_hz']:.1f} Hz, "
          f"silent {info['final']['silent_frac']:.0%}")

    d, A = agent.n_feat + 1, agent.n_actions
    rng = np.random.default_rng(0)
    if Path(args.init).exists():
        theta = np.load(args.init)["Wr"].astype(np.float64)
        if theta.shape != (d, A):
            theta = rng.normal(0, 0.05, (d, A))
        else:
            print(f"starting from {Path(args.init).name}")
    else:
        theta = rng.normal(0, 0.05, (d, A))

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(args.workers, initializer=_init_worker,
                    initargs=(str(calib), args.brain_steps, args.frame_skip))
    print(f"{args.workers} parallel workers\n")

    m_, v_ = np.zeros_like(theta), np.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_val, hist = -1.0, []
    t_end = time.time() + args.hours * 3600
    g = 0
    try:
        while time.time() < t_end:
            g += 1
            t0 = time.time()
            draw = list(rng.choice(TRAIN, args.stages, replace=False))
            E = rng.normal(0, 1, (args.pairs, d, A))
            Ws = [(theta + args.sigma * e).astype(np.float32) for e in E] + \
                 [(theta - args.sigma * e).astype(np.float32) for e in E]
            f = score_many(pool, Ws, draw)
            fp, fn = f[:args.pairs], f[args.pairs:]
            rk = centred_ranks(f)
            grad = np.tensordot(rk[:args.pairs] - rk[args.pairs:], E, axes=(0, 0)) \
                / (args.pairs * args.sigma)
            grad -= 0.002 * theta                           # mild weight decay
            m_ = b1 * m_ + (1 - b1) * grad
            v_ = b2 * v_ + (1 - b2) * grad ** 2
            mh, vh = m_ / (1 - b1 ** g), v_ / (1 - b2 ** g)
            theta = theta + args.lr * mh / (np.sqrt(vh) + eps)

            rec = {"gen": g, "pop_mean": float(f.mean()), "pop_max": float(f.max()),
                   "stages": draw, "secs": time.time() - t0}
            line = (f"gen {g:4d}  pop mean {f.mean():5.2f}x  max {f.max():5.2f}x  "
                    f"on [{','.join(draw)}]  ({rec['secs']:.0f}s)")

            if g % args.validate_every == 0:
                val = float(score_many(pool, [theta.astype(np.float32)], VALID)[0])
                rec["valid"] = val
                line += f"   validation {val:5.2f}x"
                if val > best_val:
                    best_val = val
                    np.savez_compressed(OUT / f"{args.tag}_policy.npz",
                                        Wr=theta.astype(np.float32), valid=val, gen=g)
                    line += "  <-- best, saved"
            np.savez_compressed(OUT / f"{args.tag}_latest.npz",
                                Wr=theta.astype(np.float32), gen=g)
            hist.append(rec)
            print(line, flush=True)
            (OUT / f"{args.tag}.json").write_text(json.dumps(
                {"history": hist, "best_valid": best_val}, indent=1))
    except KeyboardInterrupt:
        print("\ninterrupted -- evaluating what we have")

    # Final, honest evaluation of the checkpoint chosen on validation.
    pf = OUT / f"{args.tag}_policy.npz"
    final_W = np.load(pf)["Wr"] if pf.exists() else theta.astype(np.float32)
    per = {}
    for split, stages in (("train", TRAIN), ("validation", VALID), ("test", TEST)):
        jobs = [(final_W, s) for s in stages]
        r = pool.map(_play, jobs, chunksize=1)
        per[split] = dict(zip(stages, [float(x) for x in r]))
        print(f"{split:>10}: mean {np.mean(r):.2f}x random  "
              + "  ".join(f"{s}:{x:.2f}" for s, x in zip(stages, r)), flush=True)
    pool.close(); pool.join()
    (OUT / f"{args.tag}_eval.json").write_text(json.dumps(
        {"per_stage": per, "best_valid": best_val, "generations": g}, indent=1))
    print(f"\nFINAL test (never seen): {np.mean(list(per['test'].values())):.2f}x random")


if __name__ == "__main__":
    main()
