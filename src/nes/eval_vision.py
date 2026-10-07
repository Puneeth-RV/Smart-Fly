"""Is the trained fly using its eyes? Top-choice play with sight intact, removed, scrambled.

Every condition uses the same checkpoint, plays the policy's top choice only (no
sampling), and starts each episode after a different number of idle frames so
the runs are not all identical replays of one deterministic game.

  sight      the eye sees the game
  blind      the eye sees a blank screen (the brain still runs; memory of its
             own last button still works)
  scrambled  the eye sees the same pixels, shuffled into random positions every
             frame -- same brightness, same amount of change, no layout
  random     buttons pressed at random, no brain (the bar to clear)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "out"


def run(cond, ck_path, episodes, seed, level="1-1", quiet=False):
    import torch
    from nes.mario import Mario
    from nes.ppo_fly import STUCK_LIMIT, Memory, build_net, make_fly
    s = torch.load(ck_path, weights_only=False)
    a_ = s["args"]
    agent, shape = make_fly(int(a_["brain_steps"]), int(a_["frame_skip"]),
                            a_["calibration"], bool(a_.get("v2")))
    k = int(a_.get("stack", 1))
    net = build_net(agent.n_feat * k + 7 if k > 1 else agent.n_feat, 7)
    net.load_state_dict(s["net"]); net.eval()
    mean, var, _ = s["norm"]
    mem = Memory(agent.n_feat, k)
    m = Mario(f"SuperMarioBros-{level}-v0", retina=shape)
    rng = np.random.default_rng(seed)
    out = []
    for ep in range(episodes):
        r = m.reset()
        for _ in range(int(rng.integers(0, 40))):          # vary the start
            r, _, _, _ = m.step(0)
        agent.brain.reset(); agent.prev_px = None
        agent.prev_grid = np.zeros(agent.n_grid, np.float32)
        a, best, since, flag = None, 0, 0, False
        while not m.done and since < STUCK_LIMIT:
            if cond == "random":
                a = int(rng.integers(7))
            else:
                eye = r
                if cond == "blind":
                    eye = np.zeros_like(r)
                elif cond == "scrambled":
                    eye = rng.permutation(r.ravel()).reshape(r.shape)
                agent.ext = agent._drive(eye)
                for _ in range(agent.brain_steps):
                    agent.brain.step(agent.ext)
                f = agent.features()
                f = mem.reset(f) if a is None else mem.push(f, a)
                x = np.clip((f - mean) / np.sqrt(var + 1e-8), -6, 6).astype(np.float32)
                with torch.no_grad():
                    a = int(net(torch.from_numpy(x)[None])[0].argmax())
            for _ in range(agent.frame_skip):
                r, _, d, info = m.step(a)
                flag |= bool(info.get("flag_get"))
                if d:
                    break
            if m.max_x > best:
                best, since = m.max_x, 0
            else:
                since += 1
        out.append({"x": int(m.max_x), "flag": flag})
        if not quiet:
            print(f"  {cond:9s} ep {ep + 1}: x {m.max_x:5d}{'  FLAG' if flag else ''}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(OUT / "ppo_fly_v2.pt"))
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--cond", default="sight,blind,scrambled,random")
    ap.add_argument("--level", default="1-1")
    args = ap.parse_args()
    res = {}
    for c in args.cond.split(","):
        res[c] = run(c, args.ckpt, args.episodes, seed=7, level=args.level)
    print("\ncondition   mean x   flags")
    for c, v in res.items():
        print(f"{c:10s} {np.mean([e['x'] for e in v]):7.0f}   {sum(e['flag'] for e in v)}/{len(v)}")
    (OUT / "eval_vision.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
