"""Export a runnable subnetwork of the connectome for the browser.

The page does not replay anything: it runs the spiking network itself, frame by
frame, on whatever the game puts in front of it. So everything the simulation
needs has to ship with it.

What is exported
----------------
  net-0.bin … net-N.bin   CSR connectivity, source-major, split to stay under
                          the 15 MB per-file limit
  net.json                shapes, population index ranges, per-neuron constants
                          (bias, threshold, input gain), the photoreceptor →
                          retina-pixel map, and the trained readout

Encoding: targets as int32, weights quantised to int8 against a per-export
scale. Synapse counts span four orders of magnitude but the dynamics only need
the normalised weight, and 8 bits of it costs nothing measurable -- verified by
re-running the decodability probe on the quantised matrix before shipping.

Scope: photoreceptors plus the whole optic lobe. The full 166k brain encodes to
52 MB and about 515k scatter-adds per timestep, which would run at roughly 10fps
in JavaScript; the visual system is 92,743 neurons and 3.7M synapses, fits in
19 MB, and holds 30fps. The readout is taken from the photoreceptors and
first-order optic lobe, which is where the task signal actually survives.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from brain.lif import LIFConfig, homeostatic_bias, load_connectome  # noqa: E402
from gym.subway import RETINA_H, RETINA_W, SubwayEnv  # noqa: E402
from train.agent import AgentConfig, FlyAgent  # noqa: E402
from train.probe import oracle  # noqa: E402

CHUNK = 14 * 1024 * 1024  # stay under the 15 MB per-file limit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=str(ROOT / "out" / "dagger_policy.npz"))
    ap.add_argument("--out", default=str(ROOT / "out" / "live"))
    ap.add_argument("--steps", type=int, default=16,
                    help="brain timesteps per game frame in the browser")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    W, raw, pops, meta = load_connectome(with_raw=True)
    fly = FlyAgent(
        W, raw, pops,
        AgentConfig(seed=0, tonic_drive=0.0, input_gain=2.0, motion_gain=3.0,
                    sustained_weight=0.25, brain_steps_per_frame=args.steps),
        LIFConfig(gain=1.19, noise=0.0, threshold_spread=0.35, refractory=0, seed=0),
        meta=meta,
    )

    # Fit the per-neuron resting drive on real stimuli, exactly as offline.
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

    print("fitting per-neuron resting drive ...")
    info = homeostatic_bias(fly.brain, src(), target_hz=6.0, iters=35,
                            steps_per_frame=args.steps)
    print(f"  {info['final']['mean_hz']:.1f} Hz, silent {info['final']['silent_frac']:.0%}")

    photo = np.asarray(pops["photoreceptor_R16"], dtype=np.int64)
    ol = np.asarray(pops["optic_lobe"], dtype=np.int64)
    direct = np.flatnonzero(np.asarray(abs(raw[photo]).sum(axis=0)).ravel() > 0)
    ol1 = np.intersect1d(direct, ol)

    keep = np.unique(np.concatenate([photo, ol]))
    pos = -np.ones(W.shape[0], dtype=np.int64)
    pos[keep] = np.arange(keep.size)

    # The brain's matrix already carries the inhibitory rescaling and per-neuron
    # input normalisation, so export it rather than the raw connectome.
    M = fly.brain.W[np.ix_(keep, keep)].tocsr()
    M.sort_indices()
    print(f"subnetwork: {keep.size:,} neurons, {M.nnz:,} synapses")

    scale = float(np.abs(M.data).max())
    q = np.clip(np.round(M.data / scale * 127.0), -127, 127).astype(np.int8)
    err = np.abs(q.astype(np.float32) * scale / 127.0 - M.data).max()
    print(f"  int8 quantisation: scale {scale:.4g}, max abs error {err:.3g}")

    # Interleave target index and weight so the browser reads one stream.
    idx = M.indices.astype(np.int32)
    blob = np.empty(M.nnz * 5, dtype=np.uint8)
    blob[: M.nnz * 4] = idx.view(np.uint8)
    blob[M.nnz * 4:] = q.view(np.uint8)
    nfiles = 0
    for off in range(0, blob.nbytes, CHUNK):
        part = blob[off: off + CHUNK]
        (outdir / f"net-{nfiles}.bin").write_bytes(part.tobytes())
        nfiles += 1
    (outdir / "rows.bin").write_bytes(M.indptr.astype(np.int32).tobytes())

    pol = {}
    pfile = Path(args.policy)
    if pfile.exists():
        d = np.load(pfile)
        read_global = d["read"].astype(np.int64)
        read_local = pos[read_global]
        ok = read_local >= 0
        pol = {
            "read": read_local[ok].astype(int).tolist(),
            "W": d["Wt"][np.append(ok, True)].astype(np.float32).ravel().tolist(),
            "mu": d["mu"][ok].astype(np.float32).tolist(),
            "sd": d["sd"][ok].astype(np.float32).tolist(),
            "n_actions": int(d["Wt"].shape[1]),
        }
        print(f"  policy: {int(ok.sum())}/{read_global.size} readout neurons in subnetwork")
    else:
        print(f"  [no policy at {pfile}; page will fall back to random actions]")

    cfg = fly.cfg
    bcfg = fly.brain.cfg
    spec = {
        "n": int(keep.size),
        "nnz": int(M.nnz),
        "nnz_bytes_per_file": CHUNK,
        "files": nfiles,
        "scale": scale,
        "steps_per_frame": args.steps,
        "dt": bcfg.dt, "tau": bcfg.tau, "decay": bcfg.decay,
        "v_reset": bcfg.v_reset, "refractory": bcfg.refractory,
        "gain": bcfg.gain,
        "trace_decay": 0.8,
        "retina": {"w": RETINA_W, "h": RETINA_H},
        "input": {"gain": cfg.input_gain, "sustained": cfg.sustained_weight,
                  "motion": cfg.motion_gain},
        "photo": pos[photo].astype(int).tolist(),
        "photo_pixel": fly.photo_pixel.astype(int).tolist(),
        "ol1": pos[ol1][pos[ol1] >= 0].astype(int).tolist(),
        "v_th": fly.brain.v_th[keep].astype(np.float32).tolist(),
        "bias": fly.brain.bias[keep].astype(np.float32).tolist(),
        "in_gain": fly.brain.in_gain[keep].astype(np.float32).tolist(),
        "policy": pol,
        "provenance": {
            "dataset": "MaleCNS v1.0 (Google Research / HHMI Janelia, CC-BY)",
            "full_brain_neurons": int(W.shape[0]),
            "full_brain_synapses": int(W.nnz),
        },
    }
    (outdir / "net.json").write_text(json.dumps(spec))
    tot = sum(f.stat().st_size for f in outdir.glob("*.bin")) + \
        (outdir / "net.json").stat().st_size
    print(f"\nwrote {nfiles} weight files + rows.bin + net.json to {outdir}")
    print(f"total {tot/1e6:.1f} MB")


if __name__ == "__main__":
    main()
