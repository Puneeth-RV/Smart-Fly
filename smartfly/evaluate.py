"""Is the fly using its eyes? Top-choice play with sight intact, removed or scrambled.

Every condition uses the same trained fly, plays its top choice only (no
sampling), and starts each attempt after a random number of idle frames, so
attempts are not replays of one deterministic game.

    normal     the eye sees the game
    blind      the eye sees a blank screen; the brain still runs, and the
               policy still remembers its own last button
    scrambled  the eye sees the same pixels shuffled into random positions:
               same brightness, same amount of change, no layout
    random     buttons pressed at random, no fly at all

    python -m smartfly.evaluate models/fly-1-1.pt --level 1-1 --attempts 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from smartfly.game import ACTIONS, Mario
from smartfly.policy import Player
from smartfly.train import STUCK_LIMIT

CONDITIONS = ("normal", "blind", "scrambled", "random")


def play(
    player: Player | None,
    level: str,
    attempts: int,
    condition: str = "normal",
    seed: int = 7,
    verbose: bool = False,
) -> list[dict]:
    """Play `attempts` attempts; returns the distance and whether it reached the flag."""
    rng = np.random.default_rng(seed)
    game = Mario(level)
    results = []
    for i in range(attempts):
        retina = game.reset()
        for _ in range(int(rng.integers(0, 40))):
            retina, _, _, _ = game.step(0)
        if player:
            player.reset()
        best, since, flag = 0, 0, False
        while not game.done and since < STUCK_LIMIT:
            if condition == "random" or player is None:
                action = int(rng.integers(len(ACTIONS)))
            else:
                seen = retina
                if condition == "blind":
                    seen = np.zeros_like(retina)
                elif condition == "scrambled":
                    seen = rng.permutation(retina.ravel()).reshape(retina.shape)
                action = player.decide(seen).action
            for _ in range(player.frame_skip if player else 4):
                retina, _, done, info = game.step(action)
                flag |= bool(info.get("flag_get"))
                if done:
                    break
            if game.max_x > best:
                best, since = game.max_x, 0
            else:
                since += 1
        results.append({"x": int(game.max_x), "flag": flag})
        if verbose:
            print(
                f"  {condition:9s} attempt {i + 1}: distance {game.max_x:5d}"
                f"{'  reached the flag' if flag else ''}",
                flush=True,
            )
    game.close()
    return results


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Top-choice play with sight intact, removed or scrambled."
    )
    ap.add_argument("checkpoint")
    ap.add_argument("--level", default="1-1")
    ap.add_argument("--attempts", type=int, default=10)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--save", help="write the results as JSON here")
    args = ap.parse_args()
    player = Player(args.checkpoint)
    results = {
        c: play(player, args.level, args.attempts, c, verbose=True)
        for c in args.conditions.split(",")
    }
    print(f"\n{'condition':10s} {'flags':>7s} {'mean distance':>14s}")
    for c, rs in results.items():
        print(
            f"{c:10s} {sum(r['flag'] for r in rs):>3d}/{len(rs):<3d} "
            f"{np.mean([r['x'] for r in rs]):14.0f}"
        )
    if args.save:
        summary = {"checkpoint": str(args.checkpoint), "level": args.level, "results": results}
        Path(args.save).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
