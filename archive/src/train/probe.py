"""Is the failure in the learning rule, or in the representation?

A flat learning curve has two very different explanations and they demand
different fixes:

  * the brain's state does encode what action is needed, but the mushroom-body
    rule cannot find it (a credit-assignment problem), or
  * the brain's state does not encode it at all, in which case no learning rule
    of any kind would help and the interface or the operating point is at fault.

This separates them. An oracle with full access to the game state supplies the
correct action; a linear decoder is trained on (brain state -> oracle action)
pairs; and the decoder is then run in closed loop. That gives three numbers worth
having:

  oracle distance        the task ceiling, and a check that the task is winnable
  decode accuracy        how linearly available the right action is, vs 20% chance
  closed-loop distance   whether that availability actually suffices to play

If the decoder plays well, the representation is sound and the learning rule is
the problem. If it cannot, no amount of tuning the plasticity will matter.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from brain.lif import LIFConfig, calibrate_gain, load_connectome  # noqa: E402
from runner.subway import (A_JUMP, A_LEFT, A_NONE, A_RIGHT, A_ROLL, HIGH, LOW,  # noqa: E402
                        N_ACTIONS, N_LANES, SubwayEnv, TRAIN)
from train.agent import AgentConfig, FlyAgent  # noqa: E402


def oracle(env: SubwayEnv) -> int:
    """The correct action, computed from the game state the agent cannot see.

    Reacts to the nearest threat in the current lane: a train has to be dodged
    laterally, a low barrier jumped, a high one rolled under. Lateral moves are
    only taken toward a lane that is itself clear for the next few rows, since
    swerving into a second obstacle is no better than hitting the first.
    """
    lane = env.lane
    # How far ahead a threat still needs acting on, given speed and the time an
    # evasion takes.
    horizon = max(2, int(round(env.speed * max(env.jump_frames, env.roll_frames))))

    threat_d, threat = None, None
    for d in range(min(horizon + 1, env.depth)):
        c = int(env.track[d, lane])
        if c in (TRAIN, LOW, HIGH):
            threat_d, threat = d, c
            break
    if threat is None:
        return A_NONE

    if threat == LOW:
        return A_JUMP
    if threat == HIGH:
        return A_ROLL

    # TRAIN: find a lane that is clear over the span we would occupy it.
    span = min(horizon + 2, env.depth)
    for cand in (lane - 1, lane + 1):
        if not (0 <= cand < N_LANES):
            continue
        if all(int(env.track[d, cand]) not in (TRAIN, LOW, HIGH) for d in range(span)):
            return A_LEFT if cand < lane else A_RIGHT
    # No clean lane; take the one whose first obstacle is furthest away.
    best, best_d = None, -1
    for cand in (lane - 1, lane + 1):
        if not (0 <= cand < N_LANES):
            continue
        dd = next((d for d in range(env.depth)
                   if int(env.track[d, cand]) in (TRAIN, LOW, HIGH)), env.depth)
        if dd > best_d:
            best, best_d = cand, dd
    if best is None:
        return A_NONE
    return A_LEFT if best < lane else A_RIGHT


def features(fly: FlyAgent, pops: dict) -> np.ndarray:
    """Brain state the decoder is allowed to see: the motor-adjacent populations."""
    tr = fly.brain.trace
    pool = np.array([tr[p].mean() if p.size else 0.0 for p in fly.action_pools],
                    dtype=np.float32)
    mbon = tr[fly.mbon]
    kc = np.array([c.mean() for c in np.array_split(tr[fly.kc], 32)], dtype=np.float32)
    dn = np.array([c.mean() for c in np.array_split(tr[fly.dn], 32)], dtype=np.float32)
    mot = np.array([c.mean() for c in np.array_split(tr[np.asarray(pops["motor"])], 16)],
                   dtype=np.float32)
    return np.concatenate([pool, mbon, kc, dn, mot, [1.0]])


def softmax_fit(X: np.ndarray, y: np.ndarray, epochs: int = 300, lr: float = 0.5,
                l2: float = 1e-4, seed: int = 0) -> np.ndarray:
    """Plain multinomial logistic regression by full-batch gradient descent."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    Wt = rng.normal(0, 0.01, (d, N_ACTIONS)).astype(np.float32)
    Y = np.zeros((n, N_ACTIONS), dtype=np.float32)
    Y[np.arange(n), y] = 1.0
    # Class weights: the oracle emits A_NONE most of the time, and an unweighted
    # fit would simply predict it always and look deceptively accurate.
    cnt = np.bincount(y, minlength=N_ACTIONS).astype(np.float32)
    wts = (cnt.sum() / np.maximum(cnt, 1)) ** 0.5
    sw = wts[y][:, None]
    for _ in range(epochs):
        Z = X @ Wt
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        G = X.T @ ((P - Y) * sw) / n + l2 * Wt
        Wt -= lr * G
    return Wt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect-frames", type=int, default=6000)
    ap.add_argument("--eval-episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # 1. The ceiling: how well can the task be played at all?
    print("oracle (full state access) --")
    ds = []
    for ep in range(60):
        env = SubwayEnv(seed=9000 + ep)
        env.reset()
        f = 0
        info = {"distance": 0}
        while not env.done and f < 2000:
            _, _, _, info = env.step(oracle(env))
            f += 1
        ds.append(info["distance"])
    print(f"  mean distance {np.mean(ds):.1f}  median {np.median(ds):.0f}  "
          f"max {np.max(ds):.0f}")
    ceiling = float(np.mean(ds))

    # 2. Collect brain states under oracle control.
    W, raw, pops, meta = load_connectome(with_raw=True)
    gain, _ = calibrate_gain(W, target_hz=6.0, steps=150)
    fly = FlyAgent(W, raw, pops, AgentConfig(seed=args.seed),
                   LIFConfig(gain=gain, seed=args.seed))

    print(f"\ncollecting {args.collect_frames} frames under oracle control ...")
    X, y = [], []
    ep = 0
    while len(X) < args.collect_frames:
        env = SubwayEnv(seed=20_000 + ep)
        obs = env.reset()
        fly.reset_state()
        f = 0
        while not env.done and f < 400 and len(X) < args.collect_frames:
            a_star = oracle(env)
            fly.perceive_and_act(obs, 0.0)  # run the brain on this frame
            X.append(features(fly, pops))
            y.append(a_star)
            obs, _, _, _ = env.step(a_star)  # teacher forcing
            f += 1
        ep += 1
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    print(f"  {X.shape[0]} frames, {X.shape[1]} features")
    print(f"  oracle action mix: {np.bincount(y, minlength=N_ACTIONS)}")

    # 3. Can a linear decoder recover the oracle's action from brain state?
    ntr = int(0.8 * len(Xn))
    Wt = softmax_fit(Xn[:ntr], y[:ntr], seed=args.seed)
    pred = (Xn[ntr:] @ Wt).argmax(1)
    acc = float((pred == y[ntr:]).mean())
    majority = float((y[ntr:] == np.bincount(y[:ntr]).argmax()).mean())
    print(f"\ndecode accuracy (held-out): {acc:.1%}")
    print(f"  chance {1 / N_ACTIONS:.1%} | majority-class {majority:.1%}")
    # Per-class recall matters more than overall accuracy here: the rare
    # evasive actions are the ones that keep the agent alive.
    print("  per-action recall:")
    for a, nm in enumerate(["none", "left", "right", "jump", "roll"]):
        m = y[ntr:] == a
        if m.sum():
            print(f"    {nm:6s} {(pred[m] == a).mean():6.1%}  (n={int(m.sum())})")

    # 4. Does that decoder actually play?
    print(f"\nclosed-loop with the decoded policy ({args.eval_episodes} episodes) ...")
    ds = []
    for e in range(args.eval_episodes):
        env = SubwayEnv(seed=70_000 + e)
        obs = env.reset()
        fly.reset_state()
        f = 0
        info = {"distance": 0}
        while not env.done and f < 600:
            fly.perceive_and_act(obs, 0.0)
            x = (features(fly, pops) - mu) / sd
            obs, _, _, info = env.step(int((x @ Wt).argmax()))
            f += 1
        ds.append(info["distance"])
    print(f"  mean distance {np.mean(ds):.1f}  median {np.median(ds):.0f}  "
          f"max {np.max(ds):.0f}")

    print("\n--- verdict ---")
    print(f"task ceiling (oracle):        {ceiling:6.1f}")
    print(f"random policy floor:          {13.7:6.1f}")
    print(f"linear decode of brain state: {np.mean(ds):6.1f}")
    if np.mean(ds) > 25:
        print("=> the representation carries the needed signal; the learning rule"
              "\n   is what is failing.")
    else:
        print("=> even a supervised linear decoder cannot play from this brain"
              "\n   state, so the limit is the representation or the interface,"
              "\n   not the plasticity rule.")


if __name__ == "__main__":
    main()
