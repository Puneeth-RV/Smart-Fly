"""Drive the real Subway Surfers on an Android emulator with the trained brain.

This is the transfer target. The brain is trained in the clone because the
emulator cannot supply anywhere near enough experience -- measured below, adb
screen capture runs at roughly 5-15 fps with 60-150 ms of input latency, against
the clone's 83,000 steps/s. Training here directly would take weeks of wall
clock to reach the sample counts that take minutes in simulation.

The honest difficulty is the sim-to-real gap, and it lives entirely in
`frame_to_retina`. The brain never sees pixels from the game; it sees the same
15x24 contrast retina it was trained on. So the real frame has to be reduced to
that representation, which requires knowing where the three lanes are on screen
and what counts as an obstacle. That mapping is geometric and calibrated once per
device resolution. It is the weakest link in the whole pipeline: everything
upstream is measured, this part is a hand-built approximation, and a
misconfigured lane geometry will make a perfectly good brain look broken.

Nothing here downloads or installs the game. Getting Subway Surfers onto the
emulator is left to the user, through the Play Store on a Google-APIs system
image, for the obvious reason that pulling a commercial APK off a mirror site is
neither legal nor safe.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RETINA_H, RETINA_W = 24, 15
N_LANES = 3

# Swipe gestures, in the order the agent's action space uses.
A_NONE, A_LEFT, A_RIGHT, A_JUMP, A_ROLL = range(5)
ACTION_NAMES = ["none", "left", "right", "jump", "roll"]


class AdbError(RuntimeError):
    pass


def _run(args: list[str], timeout: float = 20.0) -> bytes:
    p = subprocess.run(args, capture_output=True, timeout=timeout)
    if p.returncode != 0:
        raise AdbError(f"{' '.join(args)} failed: {p.stderr.decode(errors='replace')[:300]}")
    return p.stdout


@dataclass
class LaneGeometry:
    """Where the playfield is on screen, in fractions of width and height.

    Fractions rather than pixels so one calibration survives a resolution change.
    `top`/`bottom` bound the stretch of track the agent looks at: too tall and
    distant track dominates, too short and obstacles appear with no time to react.
    """

    left: float = 0.25
    right: float = 0.75
    top: float = 0.34
    bottom: float = 0.92
    # How far above its own depth band's median a cell must sit to count as an
    # obstacle, in units of that band's spread.
    #
    # An absolute brightness threshold does not work on this game. Measured on a
    # live frame: distant trains sit at 0.70, but the rails are 0.42-0.48 and the
    # track bed is 0.19-0.29, and all three shift with depth, so any fixed cut
    # either labels the rails as obstacles or misses near ones entirely. Judging
    # each depth band against its own median adapts to that gradient and keeps
    # the permanent track furniture out of the retina.
    obstacle_k: float = 1.2
    obstacle_threshold: float = 0.55  # retained for the absolute-mode fallback


@dataclass
class DeviceConfig:
    serial: str | None = None
    geometry: LaneGeometry = field(default_factory=LaneGeometry)
    swipe_ms: int = 60
    adb: str = "adb"


class AndroidGame:
    """Screen capture and touch injection over adb."""

    def __init__(self, cfg: DeviceConfig | None = None) -> None:
        self.cfg = cfg or DeviceConfig()
        if shutil.which(self.cfg.adb) is None:
            raise AdbError(
                "adb not found. Install the Android platform tools first "
                "(brew install --cask android-platform-tools)."
            )
        self.w, self.h = self._screen_size()
        self._latency: list[float] = []
        self._pending: list[subprocess.Popen] = []

    # ------------------------------------------------------------------
    def _adb(self, *args: str) -> bytes:
        base = [self.cfg.adb]
        if self.cfg.serial:
            base += ["-s", self.cfg.serial]
        return _run(base + list(args))

    def _screen_size(self) -> tuple[int, int]:
        out = self._adb("shell", "wm", "size").decode()
        # "Physical size: 1080x1920"
        for tok in out.split():
            if "x" in tok and tok.replace("x", "").isdigit():
                w, h = tok.split("x")
                return int(w), int(h)
        raise AdbError(f"could not parse screen size from: {out!r}")

    @staticmethod
    def devices(adb: str = "adb") -> list[str]:
        out = _run([adb, "devices"]).decode().splitlines()[1:]
        return [l.split()[0] for l in out if l.strip() and "device" in l.split()[-1]]

    # ------------------------------------------------------------------
    def grab(self) -> np.ndarray:
        """Capture one frame as a float grayscale array in [0, 1].

        Uses `screencap` in raw form rather than PNG: decoding PNG per frame
        costs more than the capture itself, and at these frame rates that
        matters.
        """
        raw = self._adb("exec-out", "screencap")
        if len(raw) < 16:
            raise AdbError("screencap returned no data")
        w = int.from_bytes(raw[0:4], "little")
        h = int.from_bytes(raw[4:8], "little")
        fmt = int.from_bytes(raw[8:12], "little")
        # Newer Android adds a colorspace word before the pixel data.
        for header in (16, 12):
            expected = w * h * 4
            if len(raw) - header == expected:
                buf = np.frombuffer(raw[header : header + expected], dtype=np.uint8)
                img = buf.reshape(h, w, 4)
                break
        else:
            raise AdbError(
                f"unexpected screencap payload: {w}x{h} fmt={fmt} "
                f"bytes={len(raw)} (expected {w * h * 4} + 12 or 16)"
            )
        rgb = img[:, :, :3].astype(np.float32) / 255.0
        return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    # ------------------------------------------------------------------
    def frame_to_retina(self, gray: np.ndarray) -> np.ndarray:
        """Reduce a game frame to the retina the brain was trained on.

        The playfield is cropped, split into three lane columns, and resampled
        onto the retina grid with the same projective row spacing the clone uses,
        so a near obstacle occupies the same share of the visual field in both.
        """
        g = self.cfg.geometry
        h, w = gray.shape
        x0, x1 = int(w * g.left), int(w * g.right)
        y0, y1 = int(h * g.top), int(h * g.bottom)
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            raise AdbError("lane geometry produced an empty crop; recalibrate")

        band = RETINA_W // N_LANES
        view_h = RETINA_H - 3
        out = np.zeros((RETINA_H, RETINA_W), dtype=np.float32)

        # Same 1/(k+d) row mapping as the simulator, so near track occupies the
        # same number of retina rows in both environments.
        k = 3.0
        depth = view_h
        edges = [int(round(view_h * (1.0 - k / (k + d)))) for d in range(depth + 1)]
        ch = crop.shape[0]
        for d in range(depth):
            r1, r0 = view_h - edges[d], view_h - edges[d + 1]
            if r1 <= r0:
                r0 = max(0, r1 - 1)
            # Row band in the crop, nearest track at the bottom.
            c1 = int(ch * (1.0 - edges[d] / max(view_h, 1)))
            c0 = int(ch * (1.0 - edges[d + 1] / max(view_h, 1)))
            c0, c1 = min(c0, c1), max(c0, c1)
            strip = crop[max(c0, 0) : max(c1, c0 + 1), :]
            if strip.size == 0:
                continue
            # Adaptive, per-band: an obstacle is what stands out against the
            # track at this depth, not against a fixed brightness.
            med = float(np.median(strip))
            mad = float(np.median(np.abs(strip - med))) + 1e-3
            cut = med + g.obstacle_k * mad * 1.4826
            lw = strip.shape[1] // N_LANES
            for lane in range(N_LANES):
                cell = strip[:, lane * lw : (lane + 1) * lw]
                occupancy = float((cell > cut).mean()) if cell.size else 0.0
                out[max(r0, 0) : max(r1, 0), lane * band : (lane + 1) * band] = occupancy
        return out

    # ------------------------------------------------------------------
    def act(self, action: int) -> None:
        """Inject the swipe corresponding to an action."""
        if action == A_NONE:
            return
        cx, cy = self.w // 2, int(self.h * 0.6)
        d = int(min(self.w, self.h) * 0.18)
        moves = {
            A_LEFT: (cx + d, cy, cx - d, cy),
            A_RIGHT: (cx - d, cy, cx + d, cy),
            A_JUMP: (cx, cy + d, cx, cy - d),
            A_ROLL: (cx, cy - d, cx, cy + d),
        }
        x1, y1, x2, y2 = moves[action]
        # Fire and forget. `adb shell input swipe` blocks for ~100ms waiting on
        # the shell to exit, which at streaming frame rates becomes the loop's
        # bottleneck -- the swipe itself has already been dispatched to the
        # device long before that. Reaping happens opportunistically.
        base = [self.cfg.adb]
        if self.cfg.serial:
            base += ["-s", self.cfg.serial]
        proc = subprocess.Popen(
            base + ["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2),
                    str(self.cfg.swipe_ms)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._pending.append(proc)
        self._pending = [q for q in self._pending if q.poll() is None][-12:]

    # ------------------------------------------------------------------
    def benchmark(self, n: int = 20) -> dict:
        """Measure what the emulator can actually deliver.

        Worth running before believing any transfer result: if capture is slower
        than the reaction time the task needs, the agent is being asked to play
        blindfolded and its failure says nothing about the brain.
        """
        t0 = time.perf_counter()
        for _ in range(n):
            self.grab()
        cap = (time.perf_counter() - t0) / n
        t0 = time.perf_counter()
        for _ in range(5):
            self.act(A_NONE)
            self._adb("shell", "true")
        rt = (time.perf_counter() - t0) / 5
        return {
            "capture_s": cap,
            "capture_fps": 1.0 / cap if cap else float("inf"),
            "adb_roundtrip_s": rt,
            "screen": f"{self.w}x{self.h}",
        }


def preview(dev: AndroidGame) -> str:
    """ASCII view of what the brain would see, for calibrating lane geometry."""
    r = dev.frame_to_retina(dev.grab())
    rows = ["".join("#" if v > 0.6 else ("+" if v > 0.2 else ".") for v in row) for row in r]
    return "\n".join(rows)


if __name__ == "__main__":
    import sys

    try:
        devs = AndroidGame.devices()
    except AdbError as e:
        print(f"adb unavailable: {e}")
        sys.exit(1)
    if not devs:
        print("no adb devices. Start an emulator first.")
        sys.exit(1)
    print(f"devices: {devs}")
    dev = AndroidGame(DeviceConfig(serial=devs[0]))
    print("benchmark:", dev.benchmark(n=10))
    print("\nwhat the brain sees (calibrate LaneGeometry until obstacles show as #):")
    print(preview(dev))
