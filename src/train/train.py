"""Train the connectome agent on the runner, against two reference points.

Three agents share the identical brain and the identical fixed body mapping, and
differ only in what is allowed to change:

  random    no brain at all; uniform action choice. The floor.
  mb        plasticity confined to KC->MBON synapses, gated by dopamine through
            the connectome's own DAN compartments. The biological condition.
  readout   a linear softmax policy trained by REINFORCE on the same brain
            state. Synapses the fly cannot change, so this is not a model of
            fly learning -- it is the control that says how much signal the
            brain state carries at all.

Reporting `mb` alone would be uninterpretable: if it learns, nothing says
whether the mushroom body did the work or the brain's fixed dynamics were
already enough; if it fails, nothing says whether the rule is wrong or the
representation is unusable. The controls make the result legible either way.
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

from brain.lif import LIFConfig, calibrate_gain, load_connectome, sensorimotor_subgraph  # noqa: E402
from gym.subway import N_ACTIONS, SubwayEnv, random_rollout  # noqa: E402
from train.agent import AgentConfig, FlyAgent  # noqa: E402

OUT = ROOT / "out"


class ReadoutAgent:
    """Linear softmax policy over brain state, trained with REINFORCE.

    Deliberately simple: the point is to measure how linearly decodable the
    brain's state is for this task, not to build the strongest possible agent.
    """

    def __init__(self, fly: FlyAgent, lr: float = 0.05, seed: int = 0) -> None:
        self.fly = fly
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        # Features: pooled descending activity, MBON activity, and a coarse
        # summary of Kenyon-cell population activity.
        self.d = N_ACTIONS + fly.mbon.size + 8
        self.Wp = np.zeros((self.d, N_ACTIONS), dtype=np.float32)
        self.baseline = 0.0

    def features(self) -> np.ndarray:
        b = self.fly.brain
        pool = np.array([b.trace[p].mean() if p.size else 0.0
                         for p in self.fly.action_pools], dtype=np.float32)
        mbon = b.trace[self.fly.mbon]
        kc = b.trace[self.fly.kc]
        chunks = np.array_split(kc, 8)
        kcsum = np.array([c.mean() if c.size else 0.0 for c in chunks], dtype=np.float32)
        return np.concatenate([pool, mbon, kcsum])

    def act(self, obs: np.ndarray, explore: float) -> tuple[int, tuple]:
        cfg = self.fly.cfg
        flat = obs.ravel()
        self.fly._ext[:] = cfg.tonic_drive
        px = flat[self.fly.photo_pixel]
        self.fly._ext[self.fly.photo] = cfg.tonic_drive + cfg.input_gain * (px.mean() - px)
        for _ in range(cfg.brain_steps_per_frame):
            self.fly.brain.step(self.fly._ext)
        x = self.features()
        logits = x @ self.Wp
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        if self.rng.random() < explore:
            a = int(self.rng.integers(N_ACTIONS))
        else:
            a = int(self.rng.choice(N_ACTIONS, p=p))
        return a, (x, p, a)

    def update(self, traj: list[tuple], total_return: float) -> None:
        adv = total_return - self.baseline
        self.baseline += 0.05 * (total_return - self.baseline)
        g = np.zeros_like(self.Wp)
        for x, p, a in traj:
            onehot = np.zeros(N_ACTIONS, dtype=np.float32)
            onehot[a] = 1.0
            g += np.outer(x, onehot - p)
        n = max(len(traj), 1)
        self.Wp += self.lr * adv * g / n
        np.clip(self.Wp, -10, 10, out=self.Wp)


def build_brain(max_hops: int, target_hz: float, subgraph: bool) -> tuple:
    W, raw, pops, meta = load_connectome(with_raw=True)
    print(f"full connectome: {W.shape[0]:,} neurons, {W.nnz:,} signed edges, "
          f"{raw.nnz:,} structural edges")

    if subgraph:
        sources = np.asarray(pops["photoreceptor_R16"], dtype=np.int64)
        sinks = np.asarray(pops["descending"] + pops["motor"], dtype=np.int64)
        keep = sensorimotor_subgraph(W, sources, sinks, max_hops=max_hops)
        print(f"sensorimotor subgraph ({max_hops} hops): {keep.size:,} neurons "
              f"({keep.size / W.shape[0]:.1%} of the brain)")
        W = W[np.ix_(keep, keep)].tocsr()
        raw = raw[np.ix_(keep, keep)].tocsr()
        remap = -np.ones(int(keep.max()) + 1, dtype=np.int64)
        remap[keep] = np.arange(keep.size)
        pops = {k: sorted(int(remap[i]) for i in v if i < remap.size and remap[i] >= 0)
                for k, v in pops.items()}
        print(f"  {W.nnz:,} edges retained")
        print("  populations after reduction:")
        for k in ("photoreceptor_R16", "kenyon_cell", "mbon", "dan_pam", "dan_ppl1",
                  "descending", "motor"):
            print(f"    {k:22s} {len(pops[k]):>7,}")

    gain, info = calibrate_gain(W, target_hz=target_hz, steps=150)
    print(f"calibrated gain={gain:.4g} -> {info['chosen']['mean_hz']:.2f} Hz, "
          f"synchrony {info['chosen']['synchrony']:.2f}, "
          f"{info['chosen']['active_frac']:.1%} of neurons active")
    return W, raw, pops, gain, info, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-episodes", type=int, default=12)
    ap.add_argument("--max-hops", type=int, default=5)
    ap.add_argument("--target-hz", type=float, default=6.0)
    ap.add_argument("--brain-steps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--full-brain", action="store_true",
                    help="skip the sensorimotor reduction and use all 160k neurons")
    ap.add_argument("--agents", default="mb,readout")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    W, raw, pops, gain, caldata, meta = build_brain(args.max_hops, args.target_hz,
                                                    not args.full_brain)

    base = random_rollout(episodes=200, seed=args.seed)
    print(f"\nrandom-policy floor: mean distance {base['mean_distance']:.1f}, "
          f"mean return {base['mean_return']:.2f}")

    results = {"config": vars(args), "random_baseline": base,
               "calibration": caldata["chosen"], "curves": {}}
    wanted = [a.strip() for a in args.agents.split(",") if a.strip()]

    for kind in wanted:
        print(f"\n{'='*66}\ntraining: {kind}\n{'='*66}")
        acfg = AgentConfig(brain_steps_per_frame=args.brain_steps, lr=args.lr, seed=args.seed)
        lif = LIFConfig(gain=gain, seed=args.seed)
        fly = FlyAgent(W, raw, pops, acfg, lif, meta=meta)
        if kind == "mb":
            print(f"plastic KC->MBON synapses: {int(fly.kc_mbon_mask.sum()):,}")
        readout = ReadoutAgent(fly, lr=0.05, seed=args.seed) if kind == "readout" else None

        env = SubwayEnv(seed=args.seed + 1000)
        curve = []
        t0 = time.perf_counter()
        for ep in range(1, args.episodes + 1):
            explore = max(0.05, 0.35 * (1 - ep / args.episodes))
            obs = env.reset()
            fly.reset_state()
            traj, total, frames = [], 0.0, 0
            while not env.done and frames < 600:
                if readout is not None:
                    a, rec = readout.act(obs, explore)
                    traj.append(rec)
                else:
                    a, _ = fly.perceive_and_act(obs, explore)
                obs, r, _, info = env.step(a)
                if readout is None:
                    fly.reinforce(r)
                total += r
                frames += 1
            if readout is not None:
                readout.update(traj, total)

            if ep % args.eval_every == 0 or ep == 1:
                ev = []
                eenv = SubwayEnv(seed=99_000 + ep)
                for k in range(args.eval_episodes):
                    o = eenv.reset()
                    fly.reset_state()
                    f = 0
                    while not eenv.done and f < 600:
                        if readout is not None:
                            a, _ = readout.act(o, 0.0)
                        else:
                            a, _ = fly.perceive_and_act(o, 0.0)
                        o, _, _, ei = eenv.step(a)
                        f += 1
                    ev.append(ei["distance"])
                rec = {"episode": ep, "eval_distance": float(np.mean(ev)),
                       "eval_max": float(np.max(ev)), "train_return": total,
                       "explore": explore, "elapsed_s": time.perf_counter() - t0}
                if kind == "mb":
                    rec["weights"] = fly.weight_change()
                curve.append(rec)
                extra = ""
                if kind == "mb":
                    wc = rec["weights"]
                    extra = f" | dw={wc.get('mean_abs_change',0):.4f} " \
                            f"depressed={wc.get('frac_depressed',0):.1%}"
                print(f"  ep {ep:4d}  eval_dist {rec['eval_distance']:6.1f} "
                      f"(max {rec['eval_max']:5.0f})  eps={explore:.2f}  "
                      f"{rec['elapsed_s']:5.0f}s{extra}")
        results["curves"][kind] = curve
        ckpt = OUT / f"{args.tag}_{kind}.pkl"
        fly.save(ckpt)
        print(f"  saved checkpoint {ckpt}")

    path = OUT / f"{args.tag}.json"
    path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {path}")
    print(f"\n{'agent':10s} {'final eval distance':>20s}  vs random")
    for kind, curve in results["curves"].items():
        if curve:
            f = curve[-1]["eval_distance"]
            print(f"{kind:10s} {f:20.1f}  {f / base['mean_distance']:.2f}x")


if __name__ == "__main__":
    main()
