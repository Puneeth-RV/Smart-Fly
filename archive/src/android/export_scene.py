"""Record a live session and export it for the 3D viewer.

Produces, from a real run against the real game:
  out/scene/frames.jpg   a sprite sheet of the phone screen over time
  out/scene/scene.json   per-frame action, retina, and population activity

Everything in scene.json comes from the actual run -- the retina is what the
brain was fed, the activity levels are measured from the spiking network, and
the action is the swipe that was really dispatched to the device. The viewer
replays it rather than animating a plausible-looking fiction.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from android.bridge import ACTION_NAMES, AndroidGame, DeviceConfig  # noqa: E402
from android.fastcapture import FastCapture  # noqa: E402
from brain.lif import LIFConfig, calibrate_gain, load_connectome  # noqa: E402
from train.agent import AgentConfig, FlyAgent  # noqa: E402

FW, FH = 132, 288  # per-frame size in the sprite sheet
COLS = 12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--every", type=int, default=4, help="keep 1 of N control steps")
    ap.add_argument("--serial", default=None)
    ap.add_argument("--checkpoint", default=str(ROOT / "out" / "run4_mb.pkl"))
    ap.add_argument("--out", default=str(ROOT / "out" / "scene"))
    args = ap.parse_args()

    devs = AndroidGame.devices()
    if not devs:
        print("no device"); sys.exit(1)
    dev = AndroidGame(DeviceConfig(serial=args.serial or devs[0]))

    W, raw, pops, meta = load_connectome(with_raw=True)
    gain, _ = calibrate_gain(W, target_hz=6.0, steps=150)
    fly = FlyAgent(W, raw, pops, AgentConfig(seed=0), LIFConfig(gain=gain, seed=0), meta=meta)
    ck = Path(args.checkpoint)
    if ck.exists():
        fly.load(ck)

    stages = ["photoreceptor_R16", "optic_lobe", "visual_projection", "central_brain",
              "kenyon_cell", "mbon", "descending", "motor"]
    idx = {k: np.asarray(pops[k], dtype=np.int64) for k in stages}

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    shots, recs = [], []
    stream = FastCapture(dev.cfg.serial, dev.w, dev.h)
    stream.grab()
    print(f"capturing {args.frames} frames ...")
    t0 = time.time()
    step = 0
    while len(shots) < args.frames:
        gray = stream.grab()
        retina = dev.frame_to_retina(gray)
        action, info = fly.perceive_and_act(retina, 0.0)
        dev.act(action)
        step += 1
        if step % args.every:
            continue
        img = Image.fromarray((np.clip(gray, 0, 1) * 255).astype(np.uint8), mode="L")
        shots.append(img.resize((FW, FH), Image.BILINEAR).convert("RGB"))
        tr = fly.brain.trace
        d = info["relative"]
        recs.append({
            "a": int(action),
            "act": {k: round(float((tr[idx[k]] > 0).mean()), 4) for k in stages},
            "drive": [round(float(v), 4) for v in d],
            # Retina quantised to 0-9; it only ever drives a heat map.
            "ret": "".join(str(int(min(9, max(0, round(v * 9))))) for v in retina.ravel()),
        })
    stream.close()
    dt = time.time() - t0
    print(f"{step} control steps in {dt:.1f}s = {step/dt:.1f} fps; kept {len(shots)}")

    rows = (len(shots) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * FW, rows * FH), (0, 0, 0))
    for i, im in enumerate(shots):
        sheet.paste(im, ((i % COLS) * FW, (i // COLS) * FH))
    sheet.save(outdir / "frames.jpg", quality=72, optimize=True)

    (outdir / "scene.json").write_text(json.dumps({
        "fw": FW, "fh": FH, "cols": COLS, "n": len(shots),
        "retina_w": 15, "retina_h": 24,
        "actions": ACTION_NAMES,
        "stages": stages,
        "neurons": {k: int(idx[k].size) for k in stages},
        "fps_measured": round(step / dt, 1),
        "frames": recs,
    }))
    sz = (outdir / "frames.jpg").stat().st_size / 1e6
    print(f"wrote {outdir}/frames.jpg ({sz:.2f} MB) and scene.json")


if __name__ == "__main__":
    main()
