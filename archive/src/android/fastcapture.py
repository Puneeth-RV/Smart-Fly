"""Continuous frame capture from the emulator, instead of one screenshot at a time.

`adb exec-out screencap` costs a full round trip and a full-frame transfer per
call, which measured at **2.1 fps** driving the connectome against a game running
at 60. The fly was reacting to obstacles half a second stale, so most of its
failure on the real game was latency rather than anything about the brain.

This instead runs `screenrecord` on the device, which encodes H.264 on the
device's own encoder and streams it out over adb, and decodes with ffmpeg into
raw grayscale frames on a reader thread. The agent always reads the most recent
decoded frame, so it never queues up behind stale ones.

`screenrecord` caps at 180 seconds per invocation, so the stream is restarted
automatically when it ends.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time

import numpy as np


class FastCapture:
    """Streaming grayscale frames from an Android device.

    Parameters
    ----------
    serial: device serial, e.g. "emulator-5554".
    width, height: native screen size, used to size the decode.
    scale: downscale factor applied by ffmpeg. The retina is 15x24, so decoding
        at full 1080x2400 wastes most of the work; a quarter-size frame carries
        far more detail than the retina can use.
    fps: capture rate requested of ffmpeg.
    """

    def __init__(self, serial: str, width: int, height: int, scale: int = 4,
                 fps: int = 30, adb: str = "adb") -> None:
        if shutil.which(adb) is None:
            raise RuntimeError("adb not found")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found (brew install ffmpeg)")
        self.serial = serial
        self.adb = adb
        self.w = width // scale
        self.h = height // scale
        self.fps = fps
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frames_seen = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    def _spawn(self) -> tuple[subprocess.Popen, subprocess.Popen]:
        rec = subprocess.Popen(
            [self.adb, "-s", self.serial, "exec-out",
             "screenrecord", "--output-format=h264", "--time-limit", "180",
             "--bit-rate", "8000000", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        ff = subprocess.Popen(
            ["ffmpeg", "-loglevel", "quiet", "-f", "h264", "-i", "pipe:0",
             "-f", "rawvideo", "-pix_fmt", "gray",
             "-vf", f"scale={self.w}:{self.h},fps={self.fps}", "pipe:1"],
            stdin=rec.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        rec.stdout.close()  # ffmpeg owns it now
        return rec, ff

    def _run(self) -> None:
        nbytes = self.w * self.h
        while not self._stop.is_set():
            rec, ff = self._spawn()
            try:
                while not self._stop.is_set():
                    buf = ff.stdout.read(nbytes)
                    if not buf or len(buf) < nbytes:
                        break  # stream ended; screenrecord hit its time limit
                    img = np.frombuffer(buf, dtype=np.uint8).reshape(self.h, self.w)
                    with self._lock:
                        self._frame = img.astype(np.float32) / 255.0
                        self._frames_seen += 1
            finally:
                for p in (ff, rec):
                    try:
                        p.kill()
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    def grab(self, timeout: float = 10.0) -> np.ndarray:
        """Most recent decoded frame. Blocks only until the first one arrives."""
        t0 = time.time()
        while True:
            with self._lock:
                if self._frame is not None:
                    return self._frame
            if time.time() - t0 > timeout:
                raise RuntimeError("no frames decoded; is the device streaming?")
            time.sleep(0.02)

    @property
    def frames_seen(self) -> int:
        with self._lock:
            return self._frames_seen

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def benchmark(serial: str, width: int, height: int, seconds: float = 6.0) -> dict:
    """Measure the delivered frame rate, against screencap's ~2 fps."""
    with FastCapture(serial, width, height) as cap:
        cap.grab()  # wait for the stream to start
        t0, n0 = time.time(), cap.frames_seen
        time.sleep(seconds)
        dt, dn = time.time() - t0, cap.frames_seen - n0
        f = cap.grab()
        return {"decoded_fps": dn / dt, "frame_shape": f.shape,
                "mean": float(f.mean())}


if __name__ == "__main__":
    import sys
    serial = sys.argv[1] if len(sys.argv) > 1 else "emulator-5554"
    out = subprocess.run(["adb", "-s", serial, "shell", "wm", "size"],
                         capture_output=True).stdout.decode()
    wh = next(t for t in out.split() if "x" in t and t.replace("x", "").isdigit())
    w, h = (int(v) for v in wh.split("x"))
    print(f"device {serial} {w}x{h}")
    print(benchmark(serial, w, h))
