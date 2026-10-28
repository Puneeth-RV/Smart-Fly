"""Watch the connectome agent play, with the brain's own state alongside.

Renders three things per frame: the track, the retina the brain is actually
seeing, and the activity of each population in the sensorimotor chain. The point
is to be able to tell *why* the agent did something -- whether the obstacle
reached the retina at all, whether the mushroom body registered it, and which
action pool won -- rather than only watching it succeed or crash.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from brain.lif import LIFConfig, calibrate_gain, load_connectome  # noqa: E402
from runner.subway import SubwayEnv  # noqa: E402
from train.agent import AgentConfig, FlyAgent  # noqa: E402

ACTIONS = ["·", "◀", "▶", "▲", "▼"]
ACTION_NAMES = ["none", "left", "right", "jump", "roll"]
CLEAR = "\033[2J\033[H"


def bar(frac: float, width: int = 18) -> str:
    n = int(round(np.clip(frac, 0, 1) * width))
    return "█" * n + "·" * (width - n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="a .pkl saved by train.py")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true", help="no animation, just results")
    args = ap.parse_args()

    W, raw, pops, meta = load_connectome(with_raw=True)
    gain, _ = calibrate_gain(W, target_hz=6.0, steps=150)
    fly = FlyAgent(W, raw, pops, AgentConfig(seed=args.seed),
                   LIFConfig(gain=gain, seed=args.seed), meta=meta)
    tag = "untrained"
    if args.checkpoint:
        fly.load(Path(args.checkpoint))
        tag = Path(args.checkpoint).stem

    stages = ["photoreceptor_R16", "optic_lobe", "visual_projection",
              "central_brain", "kenyon_cell", "mbon", "descending", "motor"]
    idx = {k: np.asarray(pops[k], dtype=np.int64) for k in stages}

    dists = []
    for ep in range(args.episodes):
        obs = env_reset = None
        env = SubwayEnv(seed=1234 + ep)
        obs = env.reset()
        fly.reset_state()
        frames = 0
        while not env.done and frames < 600:
            action, info = fly.perceive_and_act(obs, 0.0)
            obs, r, done, gi = env.step(action)
            frames += 1
            if not args.quiet:
                tr = fly.brain.trace
                lines = [f"episode {ep + 1}/{args.episodes}   agent: {tag}", ""]
                track = env.render_ascii().split("\n")
                retina = ["".join("#" if v > 0.9 else ("+" if v > 0.05 else " ")
                                  for v in row) for row in obs]
                lines.append(f"{'track':<22}   {'retina (what the brain sees)':<17}")
                for i in range(max(len(track), len(retina))):
                    t = track[i] if i < len(track) else ""
                    rr = retina[i] if i < len(retina) else ""
                    lines.append(f"{t:<22}   |{rr}|")
                lines.append("")
                for k in stages:
                    a = float((tr[idx[k]] > 0).mean())
                    lines.append(f"  {k:20s} {bar(a)} {a:5.1%}")
                lines.append("")
                d = info["drive"]
                d = d - d.min()
                d = d / (d.max() + 1e-9)
                lines.append("  action drive:")
                for i, nm in enumerate(ACTION_NAMES):
                    mark = "  <=" if i == action else ""
                    lines.append(f"    {nm:6s} {bar(float(d[i]))} {info['drive'][i]:7.3f}{mark}")
                print(CLEAR + "\n".join(lines), flush=True)
                time.sleep(1.0 / args.fps)
        dists.append(gi["distance"])
        if args.quiet:
            print(f"  episode {ep + 1}: distance {gi['distance']:.0f} coins {gi['coins']}")

    print(f"\n{tag}: mean distance {np.mean(dists):.1f} over {args.episodes} episodes "
          f"(max {np.max(dists):.0f})")


if __name__ == "__main__":
    main()
