"""Evaluate trained agents properly, with confidence intervals.

The training loop evaluates on 12 episodes so it can do so often, which is far
too noisy to compare conditions: this task's episode lengths are heavy-tailed, so
12-episode means swing by several units on nothing. Any claim that one agent beats
another has to survive a real sample and a test, which is what this does.

Conditions are run on the *same* episode seeds, so differences are not the luck
of the level draw. Reported per condition: mean distance with a bootstrap 95%
interval, and a permutation test against the random-policy floor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from brain.lif import LIFConfig, calibrate_gain, load_connectome  # noqa: E402
from gym.subway import N_ACTIONS, SubwayEnv  # noqa: E402
from train.agent import AgentConfig, FlyAgent  # noqa: E402

OUT = ROOT / "out"


def bootstrap_ci(x: np.ndarray, iters: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(iters, x.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def permutation_p(a: np.ndarray, b: np.ndarray, iters: int = 20_000, seed: int = 0) -> float:
    """Two-sided p for mean(a) != mean(b), by label shuffling."""
    rng = np.random.default_rng(seed)
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    n = a.size
    count = 0
    for _ in range(iters):
        rng.shuffle(pool)
        if abs(pool[:n].mean() - pool[n:].mean()) >= obs:
            count += 1
    return (count + 1) / (iters + 1)


def run_agent(fly: FlyAgent | None, seeds: list[int], rng_seed: int = 0) -> np.ndarray:
    """Distances over a fixed set of episode seeds. fly=None means random policy."""
    rng = np.random.default_rng(rng_seed)
    out = []
    for s in seeds:
        env = SubwayEnv(seed=s)
        obs = env.reset()
        if fly is not None:
            fly.reset_state()
        frames = 0
        info = {"distance": 0.0}
        while not env.done and frames < 600:
            if fly is None:
                a = int(rng.integers(N_ACTIONS))
            else:
                a, _ = fly.perceive_and_act(obs, 0.0)
            obs, _, _, info = env.step(a)
            frames += 1
        out.append(info["distance"])
    return np.asarray(out, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run2")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seeds = list(range(500_000, 500_000 + args.episodes))
    W, raw, pops, meta = load_connectome(with_raw=True)
    gain, _ = calibrate_gain(W, target_hz=6.0, steps=150)

    results: dict[str, np.ndarray] = {}
    print(f"evaluating on {args.episodes} shared episode seeds\n")

    print("  random policy ...")
    results["random"] = run_agent(None, seeds, rng_seed=args.seed)

    acfg = AgentConfig(seed=args.seed)
    lif = LIFConfig(gain=gain, seed=args.seed)

    print("  untrained brain (fixed wiring, no learning) ...")
    results["untrained"] = run_agent(FlyAgent(W, raw, pops, acfg, lif, meta=meta), seeds)

    for kind in ("mb", "readout"):
        ck = OUT / f"{args.tag}_{kind}.pkl"
        if not ck.exists():
            print(f"  [skip {kind}: no checkpoint at {ck.name}]")
            continue
        print(f"  {kind} (trained) ...")
        fly = FlyAgent(W, raw, pops, acfg, lif, meta=meta)
        fly.load(ck)
        results[kind] = run_agent(fly, seeds)

    base = results["random"]
    print(f"\n{'condition':12s} {'mean':>7s} {'95% CI':>16s} {'median':>7s} "
          f"{'max':>6s} {'vs random':>10s} {'p':>8s}")
    print("-" * 74)
    summary = {}
    for name, x in results.items():
        lo, hi = bootstrap_ci(x, seed=args.seed)
        p = permutation_p(x, base, seed=args.seed) if name != "random" else float("nan")
        print(f"{name:12s} {x.mean():7.1f} [{lo:6.1f},{hi:6.1f}] {np.median(x):7.1f} "
              f"{x.max():6.0f} {x.mean() / base.mean():9.2f}x {p:8.4f}")
        summary[name] = {"mean": float(x.mean()), "ci": [lo, hi],
                         "median": float(np.median(x)), "max": float(x.max()),
                         "ratio_vs_random": float(x.mean() / base.mean()),
                         "p_vs_random": None if name == "random" else float(p)}

    # The comparison the project is actually about.
    if "mb" in results and "untrained" in results:
        p = permutation_p(results["mb"], results["untrained"], seed=args.seed)
        print(f"\nmb vs untrained brain (did the mushroom body add anything?): p={p:.4f}")
        summary["mb_vs_untrained_p"] = float(p)
    if "mb" in results and "readout" in results:
        p = permutation_p(results["mb"], results["readout"], seed=args.seed)
        print(f"mb vs trained readout (fly's rule vs generic decoder):        p={p:.4f}")
        summary["mb_vs_readout_p"] = float(p)

    path = OUT / f"{args.tag}_eval.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
