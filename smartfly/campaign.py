"""The campaign: the fly works through all 32 levels, unattended.

Loop until every level is passed:

1. Train the current level for a chunk (1.5M frames, ~40 minutes on a laptop).
   60% of attempts are on the current level; levels already seen share the
   rest, so they are not forgotten.
2. Test it: 10 attempts, top choice only, each from a random start.
3. 8 of 10 reach the end -> mastered, move to the next level.
   Budget spent (25M frames) and still flat -> set it aside, revisit later.
   Budget spent but getting close (a recent test at 3/10+, or distance up
   15% across the last three) -> another 7.5M frames, up to 50M.
4. A level set aside comes back with practice starts: half its attempts begin a
   few seconds before the furthest point reached (see `smartfly.train`).

All state lives in runs/campaign/state.json and the fly in runs/campaign/fly.pt,
so it can be stopped at any point and resumes where it was. The fly that passed
each level is kept as runs/campaign/mastered-<level>.pt, since fly.pt moves on.

    python -m smartfly.campaign
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time

import numpy as np

from smartfly.game import LEVELS
from smartfly.paths import CAMPAIGN, MODELS

STATE = CAMPAIGN / "state.json"
FLY = CAMPAIGN / "fly.pt"
LOG = CAMPAIGN / "train.log"
PRACTICE = 0.5  # share of attempts on a revisited level that start near where it failed


def load_state(seed_from: str | None) -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    st = {
        "order": LEVELS,
        "current": LEVELS[0],
        "mastered": [],
        "parked": [],
        "level_frames": {},
        "level_total": {},
        "budget": {},
        "tests": [],
        "started": time.time(),
    }
    if seed_from and not FLY.exists():
        shutil.copy(seed_from, FLY)  # start from a fly that already plays
        st["seeded_from"] = seed_from
    save(st)
    return st


def save(st: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(STATE)


def frames_played() -> int:
    import torch

    return int(torch.load(FLY, weights_only=False).get("frames", 0)) if FLY.exists() else 0


def level_mix(st: dict) -> str:
    cur = st["current"]
    seen = [lv for lv in st["mastered"] + st["parked"] if lv != cur]
    if not seen:
        return f"{cur}:1"
    w = 0.4 / len(seen)
    return ",".join([f"{cur}:0.6"] + [f"{lv}:{w:.4f}" for lv in seen])


def train_chunk(st: dict, chunk: float) -> None:
    """Train in a child process, so a crash in one chunk cannot take the campaign down."""
    target = frames_played() + chunk
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "smartfly.train",
        "--out",
        str(CAMPAIGN),
        "--levels",
        level_mix(st),
        "--frames",
        str(target),
        "--ent-start",
        "0.007",
        "--ent-end",
        "0.007",
    ]
    if st["current"] in st["parked"]:  # a revisit: start some attempts near where it failed
        cmd += ["--practice", f"{st['current']}:{PRACTICE}"]
    if FLY.exists():
        cmd.append("--resume")
    with open(LOG, "a") as log:
        log.write(
            f"\n=== {time.strftime('%Y-%m-%d %H:%M')} {st['current']}  "
            f"mix {level_mix(st)}  to {target / 1e6:.1f}M frames ===\n"
        )
        log.flush()
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)


def test(level: str, attempts: int) -> tuple[int, float]:
    from smartfly.evaluate import play
    from smartfly.policy import Player

    results = play(Player(FLY), level, attempts, seed=int(time.time()) % 100000)
    return sum(r["flag"] for r in results), float(np.mean([r["x"] for r in results]))


def next_level(st: dict) -> str | None:
    for lv in st["order"]:
        if lv not in st["mastered"] and lv not in st["parked"] and lv != st["current"]:
            return lv
    return st["parked"][0] if st["parked"] else None  # all tried: revisit, oldest first


def budget(st: dict, level: str, args: argparse.Namespace) -> float:
    return st.get("budget", {}).get(level, args.budget)


def getting_close(st: dict, level: str, args: argparse.Namespace) -> bool:
    """A recent test came near passing, or distance is still rising."""
    recent = [t for t in st["tests"] if t["level"] == level][-3:]
    if not recent:
        return False
    if max(t["flags"] for t in recent) >= args.extend_if:
        return True
    return len(recent) == 3 and recent[-1]["mean_x"] > recent[0]["mean_x"] * 1.15


def run(args: argparse.Namespace) -> None:
    st = load_state(args.seed_from)
    print(f"campaign: current {st['current']}, mastered {st['mastered']}", flush=True)
    while st["current"]:
        cur = st["current"]
        before = frames_played()
        train_chunk(st, args.chunk)
        spent = frames_played() - before
        st["level_frames"][cur] = st["level_frames"].get(cur, 0) + spent
        st.setdefault("level_total", {})[cur] = st["level_total"].get(cur, 0) + spent

        flags, mean_x = test(cur, args.attempts)
        st["tests"].append(
            {
                "t": time.time(),
                "level": cur,
                "flags": flags,
                "of": args.attempts,
                "mean_x": round(mean_x),
                "frames": frames_played(),
            }
        )
        print(
            f"{time.strftime('%H:%M')}  {cur}: {flags}/{args.attempts} flags, mean x "
            f"{mean_x:.0f}, {st['level_frames'][cur] / 1e6:.1f}M frames on it",
            flush=True,
        )

        if flags >= args.pass_flags:
            if cur in st["parked"]:
                st["parked"].remove(cur)
            st["mastered"].append(cur)
            shutil.copy(FLY, CAMPAIGN / f"mastered-{cur}.pt")
            print(f"  MASTERED {cur}", flush=True)
            st["current"] = next_level(st)
        elif st["level_frames"][cur] >= budget(st, cur, args):
            room = budget(st, cur, args) + args.extend <= args.hard_cap
            if getting_close(st, cur, args) and room:
                st.setdefault("budget", {})[cur] = budget(st, cur, args) + args.extend
                print(f"  EXTENDED {cur} to {st['budget'][cur] / 1e6:.1f}M frames", flush=True)
            else:
                if cur in st["parked"]:
                    st["parked"].remove(cur)
                st["parked"].append(cur)
                st["level_frames"][cur] = 0
                st.get("budget", {}).pop(cur, None)
                print(f"  PARKED {cur} (will return)", flush=True)
                st["current"] = next_level(st)
        save(st)
    print("campaign complete: every level passed", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Work through all 32 levels, unattended.")
    ap.add_argument("--chunk", type=float, default=1.5e6, help="frames of training between tests")
    ap.add_argument("--attempts", type=int, default=10, help="attempts per test")
    ap.add_argument("--pass-flags", type=int, default=8, help="flags needed to pass a test")
    ap.add_argument("--budget", type=float, default=25e6, help="first budget per level")
    ap.add_argument("--extend", type=float, default=7.5e6, help="extra budget when getting close")
    ap.add_argument(
        "--extend-if",
        type=int,
        default=3,
        help="'close' means a recent test reached this many flags",
    )
    ap.add_argument("--hard-cap", type=float, default=50e6, help="most frames per level per visit")
    ap.add_argument(
        "--seed-from",
        default=str(MODELS / "fly-1-1.pt"),
        help="a trained fly to start from, if the campaign is new",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
