"""Serve the fly playing real Subway Surfers, live.

    emulator (real game, 60fps)
        -> screenrecord/H.264 -> ffmpeg -> frames
        -> 166,197-neuron brain  (here, in Python, not in the browser)
        -> swipe over adb
        -> browser sees the frame and the brain state as they happen

Nothing is recorded and nothing is replayed. The page shows the current frame
and the current spike counts; close it and the fly keeps playing, open it again
and you rejoin wherever it has got to.

Why the brain runs here rather than in the page: the browser build has to ship
its weights and stay inside a JavaScript performance budget, so it runs the
92,743-neuron visual system. Here there is no such limit -- the full brain runs,
and the page only has to draw what comes back.

Two endpoints do the work:
    /stream.mjpg   multipart JPEG, the game as the fly is seeing it
    /state         JSON, polled: action, per-population activity, score

MJPEG rather than a WebSocket because an <img> consumes it natively, it draws
straight to a canvas on the same origin, and it adds no dependency.
"""

from __future__ import annotations

import argparse
import json
import socketserver
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from android.bridge import ACTION_NAMES, AndroidGame, DeviceConfig  # noqa: E402
from android.fastcapture import FastCapture  # noqa: E402
from brain.lif import LIFConfig, homeostatic_bias, load_connectome  # noqa: E402
from runner.subway import SubwayEnv  # noqa: E402
from train.agent import AgentConfig, FlyAgent  # noqa: E402
from train.probe import oracle  # noqa: E402

STATE = {"action": 0, "action_name": "none", "spikes": 0, "frames": 0,
         "fps": 0.0, "activity": {}, "ready": False, "error": None}
FRAME = {"jpeg": None}
LOCK = threading.Lock()
STOP = threading.Event()

STAGES = ["photoreceptor_R16", "optic_lobe", "visual_projection", "central_brain",
          "kenyon_cell", "mbon", "descending", "motor"]


def brain_loop(serial: str, policy_path: Path, steps: int, jpeg_quality: int) -> None:
    try:
        dev = AndroidGame(DeviceConfig(serial=serial))
        W, raw, pops, meta = load_connectome(with_raw=True)
        fly = FlyAgent(
            W, raw, pops,
            AgentConfig(seed=0, tonic_drive=0.0, input_gain=2.0, motion_gain=3.0,
                        sustained_weight=0.25, brain_steps_per_frame=steps),
            LIFConfig(gain=1.19, noise=0.0, threshold_spread=0.35, refractory=0, seed=0),
            meta=meta,
        )

        # Calibrate the resting drive on simulated stimuli, which is where the
        # readout was fitted; the real game is then fed through the same retina.
        def src():
            env = SubwayEnv(seed=777)
            env.reset()
            st = {}
            def nxt():
                if env.done:
                    env.reset()
                a = oracle(env)
                o, _, _, _ = env.step(a)
                px = o.ravel()[fly.photo_pixel]
                c = fly.cfg
                prev = st.get("p", px)
                st["p"] = px
                e = np.zeros(fly.brain.n, np.float32)
                e[fly.photo] = c.input_gain * (
                    c.sustained_weight * (px.mean() - px) + c.motion_gain * (prev - px))
                return e
            return nxt
        homeostatic_bias(fly.brain, src(), target_hz=6.0, iters=35, steps_per_frame=steps)

        pol = None
        if policy_path.exists():
            d = np.load(policy_path)
            pol = (d["read"].astype(np.int64), d["Wt"], d["mu"], d["sd"])

        idx = {k: np.asarray(pops[k], dtype=np.int64) for k in STAGES}
        stream = FastCapture(dev.cfg.serial, dev.w, dev.h)
        stream.grab()

        with LOCK:
            STATE["ready"] = True

        # Watchdog for the menu. A crash drops the real game back to a menu --
        # often behind a seasonal popup -- and the brain will happily keep
        # swiping at it forever, which looks exactly like the page being stuck.
        # Gameplay scrolls, so consecutive frames differ; a menu is nearly
        # static. When the picture stops moving, work through the taps that get
        # a run started again.
        # Watchdog. A crash drops the real game back to a menu, often behind a
        # popup, and the brain will keep swiping at it forever -- which looks
        # exactly like the page being stuck.
        #
        # Dismissal is done with the back key, not by tapping coordinates.
        # Blind taps in a free-to-play game land on purchase buttons and ad
        # banners; one opened an ad that punched out to the browser. Back closes
        # dialogs and ads and cannot open anything. Only once back has stopped
        # helping does it tap, and then only the bottom strip where "tap to
        # play" lives.
        prev_frame = None
        still_since = None
        nudges = 0

        n, t0 = 0, time.time()
        while not STOP.is_set():
            gray = stream.grab()

            if prev_frame is not None and prev_frame.shape == gray.shape:
                motion = float(np.abs(gray - prev_frame).mean())
                now = time.time()
                if motion < 0.004:
                    if still_since is None:
                        still_since = now
                        nudges = 0
                    elif now - still_since > 1.8:
                        # Never the back key: in this game back *pauses a run*,
                        # so the watchdog would stop the very thing it is meant
                        # to keep going. Only two taps are used, both on the
                        # bottom strip -- centre is "tap to play" on the main
                        # menu, right is "resume" on the pause screen -- and
                        # nothing else is ever touched.
                        fx = 0.50 if nudges % 2 == 0 else 0.75
                        dev._adb("shell", "input", "tap",
                                 str(int(dev.w * fx)), str(int(dev.h * 0.945)))
                        nudges += 1
                        still_since = now + 0.8
                        with LOCK:
                            STATE["nudges"] = STATE.get("nudges", 0) + 1
                else:
                    still_since = None
                with LOCK:
                    STATE["motion"] = round(motion, 5)
                    STATE["in_run"] = still_since is None
            prev_frame = gray
            retina = dev.frame_to_retina(gray)
            fly.perceive_and_act(retina, 0.0)        # drives the brain
            tr = fly.brain.trace
            if pol is not None:
                read, Wt, mu, sd = pol
                x = np.append((tr[read] - mu) / sd, 1.0).astype(np.float32)
                action = int((x @ Wt).argmax())
            else:
                action = int(np.argmax([tr[p].mean() if p.size else 0
                                        for p in fly.action_pools]))
            dev.act(action)

            n += 1
            if n % 2 == 0:            # encode every other frame; plenty for the eye
                img = Image.fromarray((np.clip(gray, 0, 1) * 255).astype(np.uint8), "L")
                img = img.resize((270, 585), Image.BILINEAR).convert("RGB")
                import io
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=jpeg_quality)
                with LOCK:
                    FRAME["jpeg"] = buf.getvalue()
            dt = time.time() - t0
            with LOCK:
                STATE["action"] = action
                STATE["action_name"] = ACTION_NAMES[action]
                STATE["spikes"] = int(fly.brain.spikes.sum())
                STATE["frames"] = n
                STATE["fps"] = round(n / dt, 1) if dt > 0 else 0.0
                STATE["activity"] = {k: round(float((tr[idx[k]] > 0).mean()), 4)
                                     for k in STAGES}
        stream.close()
    except Exception as exc:  # surfaced to the page rather than dying silently
        with LOCK:
            STATE["error"] = f"{type(exc).__name__}: {exc}"
        raise


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path.startswith("/state"):
            with LOCK:
                body = json.dumps(STATE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/frame.jpg"):
            # Single JPEG, polled by the page. MJPEG via multipart is more
            # elegant but Safari will not decode it in an <img>, and silently
            # shows nothing -- which looks exactly like the backend being dead.
            with LOCK:
                jpg = FRAME["jpeg"]
            if jpg is None:
                self.send_response(503); self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(jpg)))
            self.end_headers()
            try:
                self.wfile.write(jpg)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        if self.path.startswith("/stream.mjpg"):
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                last = None
                while not STOP.is_set():
                    with LOCK:
                        jpg = FRAME["jpeg"]
                    if jpg is not None and jpg is not last:
                        last = jpg
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: " + str(len(jpg)).encode()
                                         + b"\r\n\r\n" + jpg + b"\r\n")
                    else:
                        time.sleep(0.01)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        return super().do_GET()


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--serial", default=None)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--quality", type=int, default=72)
    ap.add_argument("--policy", default=str(ROOT / "out" / "dagger_policy.npz"))
    args = ap.parse_args()

    devs = AndroidGame.devices()
    if not devs:
        print("No emulator running. Start one first:")
        print("  emulator -avd flyphone -gpu host &")
        sys.exit(1)
    serial = args.serial or devs[0]
    print(f"device {serial}")

    t = threading.Thread(target=brain_loop,
                         args=(serial, Path(args.policy), args.steps, args.quality),
                         daemon=True)
    t.start()

    web = ROOT / "out" / "scene"
    handler = lambda *a, **k: Handler(*a, directory=str(web), **k)  # noqa: E731
    print(f"\n  Fly plays Subway Surfers, live -> http://localhost:{args.port}/live.html")
    print("  (brain is warming up; the page will say when it is ready)\n")
    with Server(("", args.port), handler) as srv:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            STOP.set()


if __name__ == "__main__":
    main()
