"""Campaign: the fly works through Super Mario Bros level by level, unattended.

Loop, forever (or until every level is passed):
  1. train the current level in chunks, with earlier levels mixed in so they
     are not forgotten
  2. after each chunk, test it: top-choice play only, random starts, on the
     current level
  3. passes the bar -> mastered, move to the next level
     too long without passing -> park it and move on; parked levels come back
     once the rest of the game has been tried

Every chunk resumes from the same checkpoint, and all progress lives in
out/campaign.json, so the whole thing can be stopped and restarted at any time
and picks up where it was. Run it detached:

    python -u src/nes/campaign.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "out"
# Not campaign.json: the trainer writes out/<tag>.json (its episode log), and
# with tag "campaign" it overwrote this file every update.
STATE = OUT / "campaign_state.json"
TAG = "campaign"

ORDER = [f"{w}-{s}" for w in range(1, 9) for s in range(1, 5)]


def load_state(args):
    if STATE.exists():
        return json.loads(STATE.read_text())
    st = {"order": ORDER, "current": ORDER[0], "mastered": [], "parked": [],
          "level_frames": {}, "tests": [], "started": time.time()}
    # Start from the fly that already learned 1-1 rather than from nothing.
    if args.seed_from and Path(args.seed_from).exists() and not (OUT / f"{TAG}.pt").exists():
        shutil.copy(args.seed_from, OUT / f"{TAG}.pt")
        st["seeded_from"] = args.seed_from
    save(st)
    return st


def save(st):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(STATE)


def total_frames():
    import torch
    ck = OUT / f"{TAG}.pt"
    if not ck.exists():
        return 0
    return int(torch.load(ck, weights_only=False).get("frames", 0))


def mix(st):
    """Current level gets 60% of attempts; levels already seen share the rest."""
    cur = st["current"]
    seen = [lv for lv in st["mastered"] + st["parked"] if lv != cur]
    if not seen:
        return f"{cur}:1"
    w = 0.4 / len(seen)
    return ",".join([f"{cur}:0.6"] + [f"{lv}:{w:.4f}" for lv in seen])


def train_chunk(st, args):
    target = total_frames() + args.chunk
    cmd = [sys.executable, "-u", str(ROOT / "src/nes/ppo_fly.py"), "--v2",
           "--brain-steps", "6", "--rollout", "512", "--tag", TAG,
           "--levels", mix(st), "--frames", str(target),
           "--start-jitter", "30", "--ent-start", "0.007", "--ent-end", "0.007"]
    if (OUT / f"{TAG}.pt").exists():
        cmd.append("--resume")
    with open(OUT / "campaign_train.log", "a") as log:
        log.write(f"\n=== {time.strftime('%H:%M')} {st['current']}  mix {mix(st)}  "
                  f"to {target / 1e6:.1f}M frames ===\n")
        log.flush()
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)


def test(level, episodes):
    from nes.eval_vision import run
    res = run("sight", str(OUT / f"{TAG}.pt"), episodes, seed=int(time.time()) % 100000,
              level=level, quiet=True)
    return sum(e["flag"] for e in res), float(np.mean([e["x"] for e in res]))


def next_level(st):
    for lv in st["order"]:
        if lv not in st["mastered"] and lv not in st["parked"] and lv != st["current"]:
            return lv
    # Everything tried once: return to parked levels, oldest first.
    return st["parked"][0] if st["parked"] else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=float, default=1.5e6, help="frames between tests")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--pass-flags", type=int, default=8, help="flags out of --episodes to pass")
    ap.add_argument("--max-level-frames", type=float, default=25e6,
                    help="park a level after this much training without passing")
    ap.add_argument("--seed-from", default=str(OUT / "ppo_fly_v2.pt"))
    args = ap.parse_args()
    st = load_state(args)
    print(f"campaign: current {st['current']}, mastered {st['mastered']}", flush=True)
    while st["current"]:
        cur = st["current"]
        f0 = total_frames()
        train_chunk(st, args)
        st["level_frames"][cur] = st["level_frames"].get(cur, 0) + total_frames() - f0
        flags, mean_x = test(cur, args.episodes)
        rec = {"t": time.time(), "level": cur, "flags": flags, "of": args.episodes,
               "mean_x": round(mean_x), "frames": total_frames()}
        st["tests"].append(rec)
        print(f"{time.strftime('%H:%M')}  {cur}: {flags}/{args.episodes} flags, "
              f"mean x {mean_x:.0f}, {st['level_frames'][cur] / 1e6:.1f}M frames on it",
              flush=True)
        if flags >= args.pass_flags:
            if cur in st["parked"]:
                st["parked"].remove(cur)
            st["mastered"].append(cur)
            print(f"  MASTERED {cur}", flush=True)
            st["current"] = next_level(st)
        elif st["level_frames"][cur] >= args.max_level_frames:
            if cur not in st["parked"]:
                st["parked"].append(cur)
            else:                              # rotate to the back of the queue
                st["parked"].remove(cur); st["parked"].append(cur)
            st["level_frames"][cur] = 0
            print(f"  PARKED {cur} (will return)", flush=True)
            st["current"] = next_level(st)
        save(st)
    print("campaign complete: every level passed", flush=True)


if __name__ == "__main__":
    main()
