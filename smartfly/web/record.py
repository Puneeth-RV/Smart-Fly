"""Record the fly playing, for the replay site (GitHub Pages, no server).

For every mastered level: one run with sight that reaches the end (it keeps
playing attempts until one does, and says how many it took), one blind and one
scrambled. For the level being learned: one practice attempt with the newest
weights. Each run is a JPEG per decision plus everything the page draws.

    python -m smartfly.web.record                  # -> runs/pages, ready to publish
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from smartfly.campaign import FLY
from smartfly.game import ACTIONS, Mario
from smartfly.paths import MODELS, RUNS, SITE
from smartfly.policy import Player
from smartfly.web.episode import play_attempt, raster_neurons
from smartfly.web.server import campaign_summary

MODE = '<meta name="smartfly-mode" content="replay">'


def _bytes(values) -> str:
    """Values in [0, 1] as base64 bytes: a quarter of the size of JSON numbers."""
    q = np.clip(np.round(np.asarray(values, np.float32) * 255), 0, 255).astype(np.uint8)
    return base64.b64encode(q.tobytes()).decode()


def record(
    player: Player,
    level: str,
    condition: str,
    out: Path,
    rng: np.random.Generator,
    until_flag: int = 0,
) -> dict:
    """Play one attempt (or up to `until_flag` until one reaches the end) into `out`."""
    raster = raster_neurons(player)
    game = Mario(level)
    for attempt in range(1, max(until_flag, 1) + 1):
        steps, frames = [], []
        for step in play_attempt(player, game, rng, raster, condition):
            s = step.state
            frames.append(step.frame.astype(np.uint8))
            steps.append(
                {
                    "x": s["x"],
                    "max_x": s["max_x"],
                    "a": s["actions"].index(s["action"]),
                    "p": s["probs"],
                    "v": s["value"],
                    "r": s["reward"],
                    "ret": s["return"],
                    "act": s["active"],
                    "eye": _bytes(s["eye"]),
                    "photo": _bytes(s["photo"]),
                    "deep": _bytes(s["deep"]),
                    "att": _bytes(s["attention"]),
                    "ras": s["raster"],
                    "ev": "flag" if step.flag else ("death" if step.died else ""),
                }
            )
            if step.flag:
                break
        flag = bool(steps and steps[-1]["ev"] == "flag")
        print(f"  {level} {condition:9s} attempt {attempt}: distance {game.max_x}", flush=True)
        if flag or attempt >= until_flag:
            break
    game.close()

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(out / f"{i:04d}.jpg", quality=72)
    meta = {
        "id": out.name,
        "level": level,
        "condition": condition,
        "flag": flag,
        "x": int(game.max_x),
        "attempts": attempt,
        "decisions": len(steps),
        "ckpt_frames": round(player.frames_trained / 1e6, 2),
        "recorded": time.time(),
    }
    run = meta | {
        "actions": ACTIONS,
        "eye_shape": list(game.retina_shape),
        "grid": [20, 16],
        "raster_n": len(raster),
        "steps": steps,
    }
    (out / "run.json").write_text(json.dumps(run, separators=(",", ":")))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Record the fly for the replay site.")
    ap.add_argument("--checkpoint", default=str(MODELS / "fly-campaign.pt"), help="mastered levels")
    ap.add_argument("--practice", default=str(FLY), help="the fly in training, for practice")
    ap.add_argument("--out", default=str(RUNS / "pages"))
    ap.add_argument("--tries", type=int, default=10, help="attempts to get a run that finishes")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(SITE, out)
    page = out / "index.html"
    page.write_text(page.read_text().replace("<head>", "<head>\n" + MODE, 1))

    replays = out / "replays"
    camp = campaign_summary()
    rng = np.random.default_rng(args.seed)
    runs = []
    player = Player(args.checkpoint)
    for level in camp["mastered"]:
        runs.append(record(player, level, "normal", replays / f"{level}-normal", rng, args.tries))
        for condition in ("blind", "scrambled"):
            runs.append(record(player, level, condition, replays / f"{level}-{condition}", rng))
    if camp["current"] and Path(args.practice).exists():
        practice = Player(args.practice, fly=player.fly)
        meta = record(practice, camp["current"], "normal", replays / "practice", rng)
        runs.append(meta | {"practice": True})

    (replays / "index.json").write_text(json.dumps({"runs": runs, "campaign": camp}))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"wrote {out}: {len(runs)} runs, {size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
