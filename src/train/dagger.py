"""Train the readout on the states the fly itself visits, not the oracle's.

A decoder fitted to oracle-driven states reached 38.7% per-action recall and
still played at the random baseline. The gap is distribution shift: the oracle
never visits the situations a mediocre policy gets itself into, so the decoder
has never seen them, and in a game where one wrong move is fatal those are
exactly the states that matter.

DAgger closes that loop. Roll out the current policy, ask the oracle what it
*should* have done in each state the policy actually reached, add those to the
training set, refit, repeat. The data distribution converges on the policy's own,
which is the one it has to perform well on.

Everything else stays as measured: no uniform tonic drive, per-neuron homeostatic
bias, 48 timesteps per frame with no refractory period, and the readout taken
from the photoreceptors and first-order optic lobe -- the stages that still carry
the signal, rather than the mushroom body, which real flies do not use for fast
visuomotor control anyway.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from brain.lif import LIFConfig, homeostatic_bias, load_connectome  # noqa: E402
from gym.subway import N_ACTIONS, SubwayEnv  # noqa: E402
from train.agent import AgentConfig, FlyAgent  # noqa: E402
from train.probe import oracle, softmax_fit  # noqa: E402

OUT = ROOT / "out"


def build(steps: int = 48, target_hz: float = 6.0, seed: int = 0):
    W, raw, pops, meta = load_connectome(with_raw=True)
    fly = FlyAgent(
        W, raw, pops,
        AgentConfig(seed=seed, tonic_drive=0.0, input_gain=2.0, motion_gain=3.0,
                    sustained_weight=0.25, brain_steps_per_frame=steps),
        LIFConfig(gain=1.19, noise=0.0, threshold_spread=0.35, refractory=0, seed=seed),
        meta=meta,
    )

    def src():
        env = SubwayEnv(seed=777)
        env.reset()
        st = {}
        def nxt():
            if env.done:
                env.reset()
            a = oracle(env)
            o, _, _, _ = env.step(a)
            px = o.ravel()[fly.photo_pixel]
            c = fly.cfg
            prev = st.get("p", px)
            st["p"] = px
            e = np.zeros(fly.brain.n, np.float32)
            e[fly.photo] = c.input_gain * (
                c.sustained_weight * (px.mean() - px) + c.motion_gain * (prev - px))
            return e
        return nxt

    info = homeostatic_bias(fly.brain, src(), target_hz=target_hz, iters=35,
                            steps_per_frame=steps)
    direct = np.flatnonzero(np.asarray(abs(raw[fly.photo]).sum(axis=0)).ravel() > 0)
    ol = np.intersect1d(direct, np.asarray(pops["optic_lobe"], dtype=np.int64))
    read = np.concatenate([fly.photo, ol])
    return fly, read, info


def rollout(fly, read, Wt, mu, sd, seeds, eps=0.0, collect=True, max_frames=600):
    """Run the policy; optionally record (state, oracle label) for every frame."""
    rng = np.random.default_rng(0)
    X, Y, dists = [], [], []
    for s in seeds:
        env = SubwayEnv(seed=s)
        obs = env.reset()
        fly.reset_state()
        f = 0
        info = {"distance": 0}
        while not env.done and f < max_frames:
            fly.perceive_and_act(obs, 0.0)
            v = fly.brain.trace[read]
            if collect:
                X.append(v.copy())
                Y.append(oracle(env))        # what the oracle would have done here
            if Wt is None:
                a = int(rng.integers(N_ACTIONS))
            else:
                x = np.append((v - mu) / sd, 1.0).astype(np.float32)
                a = int((x @ Wt).argmax())
                if eps and rng.random() < eps:
                    a = int(rng.integers(N_ACTIONS))
            obs, _, _, info = env.step(a)
            f += 1
        dists.append(info["distance"])
    return (np.asarray(X, np.float32) if X else None,
            np.asarray(Y) if Y else None, np.asarray(dists, float))


def fit(X, Y, epochs=700):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = np.hstack([(X - mu) / sd, np.ones((len(X), 1), np.float32)])
    return softmax_fit(Xn, Y, epochs=epochs, lr=0.5), mu, sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--episodes-per-iter", type=int, default=45)
    ap.add_argument("--eval-episodes", type=int, default=60)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--tag", default="dagger")
    args = ap.parse_args()

    print("building brain (no tonic drive, per-neuron homeostatic bias) ...")
    fly, read, info = build(steps=args.steps)
    print(f"  readout: {read.size} neurons (photoreceptors + first-order optic lobe)")
    print(f"  calibration: {info['final']['mean_hz']:.1f} Hz, "
          f"silent {info['final']['silent_frac']:.0%}")

    # Round 0: the oracle's own distribution, as before.
    print("\nround 0 — oracle demonstrations")
    X, Y, _ = rollout(fly, read, None, None, None,
                      seeds=range(20_000, 20_000 + args.episodes_per_iter),
                      collect=True)
    # Oracle actions drive round 0, so relabel by rolling the oracle itself.
    Xo, Yo = [], []
    for s in range(20_000, 20_000 + args.episodes_per_iter):
        env = SubwayEnv(seed=s)
        obs = env.reset()
        fly.reset_state()
        f = 0
        while not env.done and f < 600:
            a = oracle(env)
            fly.perceive_and_act(obs, 0.0)
            Xo.append(fly.brain.trace[read].copy())
            Yo.append(a)
            obs, _, _, _ = env.step(a)
            f += 1
    X, Y = np.asarray(Xo, np.float32), np.asarray(Yo)
    Wt, mu, sd = fit(X, Y)

    evals = range(81_000, 81_000 + args.eval_episodes)
    _, _, d = rollout(fly, read, Wt, mu, sd, evals, collect=False)
    curve = [{"round": 0, "frames": int(len(X)), "mean": float(d.mean()),
              "median": float(np.median(d)), "max": float(d.max())}]
    print(f"  {len(X):6d} frames | closed-loop distance {d.mean():5.1f} "
          f"(median {np.median(d):.0f}, max {d.max():.0f})")

    for it in range(1, args.iters + 1):
        # Collect the states this policy actually reaches, labelled by the oracle.
        seeds = range(30_000 + it * 1000, 30_000 + it * 1000 + args.episodes_per_iter)
        Xi, Yi, _ = rollout(fly, read, Wt, mu, sd, seeds, eps=0.05, collect=True)
        X = np.concatenate([X, Xi])
        Y = np.concatenate([Y, Yi])
        Wt, mu, sd = fit(X, Y)
        _, _, d = rollout(fly, read, Wt, mu, sd, evals, collect=False)
        curve.append({"round": it, "frames": int(len(X)), "mean": float(d.mean()),
                      "median": float(np.median(d)), "max": float(d.max())})
        print(f"round {it} — +{len(Xi)} own-distribution frames "
              f"({len(X)} total) | closed-loop distance {d.mean():5.1f} "
              f"(median {np.median(d):.0f}, max {d.max():.0f})")

    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / f"{args.tag}_policy.npz", Wt=Wt, mu=mu, sd=sd, read=read)
    (OUT / f"{args.tag}.json").write_text(json.dumps(
        {"curve": curve, "calibration": info["final"], "readout_size": int(read.size)},
        indent=2))
    print(f"\nrandom 13.7 | oracle 68.9")
    print(f"wrote {OUT/f'{args.tag}_policy.npz'}")


if __name__ == "__main__":
    main()
