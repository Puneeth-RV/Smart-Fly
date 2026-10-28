"""Put the connectome on the real Subway Surfers, running on an emulator.

Usage:
    python src/android/run_live.py --calibrate      # check what the brain sees
    python src/android/run_live.py --play           # let the fly play
    python src/android/run_live.py --play --record out/fly_plays.gif

The brain is the same 166,197-neuron MaleCNS network used in simulation, reading
the emulator's framebuffer through the same 15x24 contrast retina and issuing
swipes through adb.

Expect it to play badly. Measured in simulation, the task-relevant signal is
59.7% decodable at the retina and 15.7% by the Kenyon cells, so the motor output
carries little information about what is on screen. That is a real property of
this modelling approach rather than a bug in the plumbing, and it is what every
"connectome plays a video game" demo leaves unmeasured. What is genuine here is
the pathway: real wiring, real transmitter signs, real mushroom-body synapses,
real descending neurons driving the touches.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from android.bridge import (ACTION_NAMES, AdbError, AndroidGame,  # noqa: E402
                            DeviceConfig, LaneGeometry)
from android.fastcapture import FastCapture  # noqa: E402
from brain.lif import LIFConfig, calibrate_gain, load_connectome  # noqa: E402
from train.agent import AgentConfig, FlyAgent  # noqa: E402


def build_fly(checkpoint: Path | None, seed: int = 0) -> FlyAgent:
    W, raw, pops, meta = load_connectome(with_raw=True)
    gain, _ = calibrate_gain(W, target_hz=6.0, steps=150)
    fly = FlyAgent(W, raw, pops, AgentConfig(seed=seed),
                   LIFConfig(gain=gain, seed=seed), meta=meta)
    if checkpoint and Path(checkpoint).exists():
        fly.load(Path(checkpoint))
        print(f"loaded {checkpoint}")
    print(f"brain: {fly.brain.n:,} neurons | retinotopy: {fly.retinotopy}")
    return fly


def show(dev: AndroidGame, r: np.ndarray) -> None:
    print("\n".join("".join("#" if v > 0.6 else ("+" if v > 0.2 else ".") for v in row)
                    for row in r))


def calibrate(dev: AndroidGame) -> None:
    """Print the raw frame's coarse layout next to the derived retina.

    The lane geometry is the one hand-tuned part of the whole pipeline, and a
    bad crop makes a working brain look broken, so it is worth eyeballing before
    trusting anything downstream.
    """
    gray = dev.grab()
    print(f"frame {gray.shape[1]}x{gray.shape[0]}  "
          f"mean {gray.mean():.3f}  min {gray.min():.3f}  max {gray.max():.3f}")
    h, w = gray.shape
    g = dev.cfg.geometry
    print(f"\ncrop: x {g.left:.2f}-{g.right:.2f}  y {g.top:.2f}-{g.bottom:.2f}  "
          f"obstacle threshold {g.obstacle_threshold:.2f}")
    # Coarse view of the whole screen, to see where the playfield actually is.
    small = gray[:: max(1, h // 28), :: max(1, w // 40)]
    print("\nwhole screen (coarse):")
    for row in small:
        print("".join("#" if v > g.obstacle_threshold else ("+" if v > 0.25 else ".")
                      for v in row))
    print("\nderived retina (what the brain sees):")
    show(dev, dev.frame_to_retina(gray))
    print("\nIf obstacles do not appear as # in the retina, adjust LaneGeometry "
          "in src/android/bridge.py.")


def play(dev: AndroidGame, fly: FlyAgent, seconds: float, record: Path | None,
         verbose: bool = True, stream: FastCapture | None = None) -> None:
    frames = []
    t0 = time.time()
    n = 0
    acted = np.zeros(5, dtype=int)
    while time.time() - t0 < seconds:
        gray = stream.grab() if stream is not None else dev.grab()
        retina = dev.frame_to_retina(gray)
        action, info = fly.perceive_and_act(retina, 0.0)
        dev.act(action)
        acted[action] += 1
        n += 1
        if record is not None:
            frames.append((gray[:: max(1, gray.shape[0] // 240)][:, :: max(1, gray.shape[1] // 135)],
                           retina.copy(), action))
        if verbose and n % 5 == 0:
            tr = fly.brain.trace
            print(f"\rframe {n:4d} | {time.time() - t0:5.1f}s | "
                  f"action {ACTION_NAMES[action]:5s} | "
                  f"KC {float((tr[fly.kc] > 0).mean()):5.1%} "
                  f"DN {float((tr[fly.dn] > 0).mean()):5.1%}", end="", flush=True)
    dt = time.time() - t0
    print(f"\n\n{n} frames in {dt:.1f}s = {n / dt:.1f} fps")
    print("actions taken: " + ", ".join(f"{ACTION_NAMES[i]} {acted[i]}" for i in range(5)))

    if record is not None and frames:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib import animation
        except ImportError:
            print("matplotlib unavailable; skipping recording")
            return
        fig, axes = plt.subplots(1, 2, figsize=(7, 5))
        im0 = axes[0].imshow(frames[0][0], cmap="gray", vmin=0, vmax=1)
        im1 = axes[1].imshow(frames[0][1], cmap="magma", vmin=0, vmax=1)
        axes[0].set_title("Subway Surfers"); axes[1].set_title("what the fly sees")
        for a in axes:
            a.set_xticks([]); a.set_yticks([])
        txt = fig.suptitle("")

        def upd(i):
            im0.set_data(frames[i][0]); im1.set_data(frames[i][1])
            txt.set_text(f"frame {i + 1}/{len(frames)} — {ACTION_NAMES[frames[i][2]]}")
            return im0, im1, txt

        ani = animation.FuncAnimation(fig, upd, frames=len(frames), interval=120, blit=False)
        record.parent.mkdir(parents=True, exist_ok=True)
        ani.save(record, writer=animation.PillowWriter(fps=8))
        print(f"wrote {record}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--record", type=Path, default=None)
    ap.add_argument("--checkpoint", default=str(ROOT / "out" / "run4_mb.pkl"))
    ap.add_argument("--serial", default=None)
    ap.add_argument("--slow-capture", action="store_true",
                    help="use per-frame screencap instead of the video stream")
    args = ap.parse_args()

    try:
        devs = AndroidGame.devices()
    except AdbError as e:
        print(f"adb problem: {e}")
        sys.exit(1)
    if not devs:
        print("No emulator running. Start one first (see README).")
        sys.exit(1)
    dev = AndroidGame(DeviceConfig(serial=args.serial or devs[0], geometry=LaneGeometry()))
    print(f"device {dev.cfg.serial}  screen {dev.w}x{dev.h}")

    if args.calibrate or not args.play:
        print("\nbenchmark:", dev.benchmark(n=8))
        calibrate(dev)
        if not args.play:
            return
    fly = build_fly(Path(args.checkpoint))
    stream = None
    if not args.slow_capture:
        try:
            stream = FastCapture(dev.cfg.serial, dev.w, dev.h)
            stream.grab()
            print("using streaming capture (screenrecord -> ffmpeg)")
        except Exception as e:
            print(f"streaming capture unavailable ({e}); falling back to screencap")
    try:
        play(dev, fly, args.seconds, args.record, stream=stream)
    finally:
        if stream is not None:
            stream.close()


if __name__ == "__main__":
    main()
