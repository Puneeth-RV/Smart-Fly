"""Export a connectome that two flies can each run, for pursuit.

Every published connectome demo so far is one brain against a world -- a game,
a drone, a physics body. This exports a brain two individuals can run at once so
they can react to each other: one fly's eyes see the other fly, and vice versa.

Readout is bilateral, not learned. A fly turns toward something by comparing its
two eyes: more activity on the right, turn right. That is how the optomotor
response and visual pursuit actually work, and it needs no trained decoder --
which matters, because the thing measured to fail in this network is deep
fine-grained decoding, and this is neither deep nor fine-grained.

The population read is the visual projection neurons: the LC cells are the optic
lobe's object-detection output and the real substrate of pursuit in the animal.

Scope: photoreceptors + optic lobe + visual projection. Both flies share one copy
of the weights in the browser and keep their own membrane state, so a second fly
costs state, not another 25 MB.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from brain.lif import LIFConfig, homeostatic_bias, load_connectome  # noqa: E402
from runner.subway import SubwayEnv  # noqa: E402
from train.agent import AgentConfig, FlyAgent  # noqa: E402
from train.probe import oracle  # noqa: E402

RAW = ROOT / "data" / "raw"
ANNOT = RAW / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
CHUNK = 14 * 1024 * 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "out" / "pursuit"))
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    W, raw, pops, meta = load_connectome(with_raw=True)
    fly = FlyAgent(
        W, raw, pops,
        AgentConfig(seed=0, tonic_drive=0.0, input_gain=2.0, motion_gain=3.0,
                    sustained_weight=0.25, brain_steps_per_frame=args.steps),
        LIFConfig(gain=1.19, noise=0.0, threshold_spread=0.35, refractory=0, seed=0),
        meta=meta,
    )

    def src():
        env = SubwayEnv(seed=777); env.reset(); st = {}
        def nxt():
            if env.done:
                env.reset()
            a = oracle(env)
            o, _, _, _ = env.step(a)
            px = o.ravel()[fly.photo_pixel]; c = fly.cfg
            prev = st.get("p", px); st["p"] = px
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
    vp = np.asarray(pops["visual_projection"], dtype=np.int64)
    keep = np.unique(np.concatenate([photo, ol, vp]))
    pos = -np.ones(W.shape[0], dtype=np.int64)
    pos[keep] = np.arange(keep.size)

    M = fly.brain.W[np.ix_(keep, keep)].tocsr(); M.sort_indices()
    print(f"subnetwork: {keep.size:,} neurons, {M.nnz:,} synapses")

    # Hemisphere split, from the release's own soma-side annotation.
    z = np.load(ROOT / "data" / "processed" / "connectome.npz")
    bodies = z["bodies"]
    ann = pd.read_feather(ANNOT, columns=["bodyId", "somaSide", "type",
                                          "fruDsx", "dimorphism"]).set_index("bodyId")
    side_of, male_of = {}, {}
    for b, row in ann.iterrows():
        side_of[int(b)] = row.somaSide
        male_of[int(b)] = (isinstance(row.dimorphism, str) and "male" in row.dimorphism) \
            or (isinstance(row.fruDsx, str) and row.fruDsx.startswith("fru"))

    def split(idx_global):
        L, R = [], []
        for g in idx_global:
            s = side_of.get(int(bodies[g]))
            loc = int(pos[g])
            if loc < 0:
                continue
            if s == "L":
                L.append(loc)
            elif s == "R":
                R.append(loc)
        return L, R

    # Photoreceptors carry almost no soma-side annotation (4 left, 9 right), so
    # each one's eye is inferred from the hemisphere of the optic-lobe cells it
    # actually projects to -- the same connectivity trick used for retinotopy.
    side_code = np.zeros(W.shape[0], dtype=np.int8)      # -1 left, +1 right
    for g in ol:
        s_ = side_of.get(int(bodies[g]))
        side_code[g] = -1 if s_ == "L" else (1 if s_ == "R" else 0)
    Wp = raw[photo]
    photo_eye = np.zeros(photo.size, dtype=np.int8)
    for i in range(photo.size):
        a0, a1 = Wp.indptr[i], Wp.indptr[i + 1]
        tg, wt = Wp.indices[a0:a1], Wp.data[a0:a1]
        sc = side_code[tg]
        m = sc != 0
        photo_eye[i] = 1 if (m.any() and float((sc[m] * wt[m]).sum()) > 0) else -1
    print(f"  photoreceptor eye from targets: L {int((photo_eye<0).sum()):,}  "
          f"R {int((photo_eye>0).sum()):,}")

    # Per-eye retinotopy. The single merged retina used elsewhere is wrong for
    # two eyes: a left-eye photoreceptor would land on a right-half pixel and
    # then be masked out, so the map and the mask contradicted each other and
    # the bearing signal vanished. Each eye is laid out over its own half of the
    # field instead, using the ommatidial coordinates recovered earlier.
    RW_, RH_ = 15, 24
    h1 = np.asarray(meta.get("hex1"), dtype=np.float32)
    h2 = np.asarray(meta.get("hex2"), dtype=np.float32)
    have_h = np.isfinite(h1) & np.isfinite(h2)
    Wp_all = raw[photo]
    ppos = np.full((photo.size, 2), np.nan, dtype=np.float32)
    for i in range(photo.size):
        a0, a1 = Wp_all.indptr[i], Wp_all.indptr[i + 1]
        tg, wt = Wp_all.indices[a0:a1], Wp_all.data[a0:a1]
        m = have_h[tg]
        if m.any():
            ppos[i, 0] = np.average(h1[tg[m]], weights=wt[m])
            ppos[i, 1] = np.average(h2[tg[m]], weights=wt[m])
    photo_pixel_eye = np.zeros(photo.size, dtype=np.int64)
    half = RW_ // 2
    for eye, cols in ((-1, (0, half)), (1, (RW_ - half, RW_))):
        sel = np.flatnonzero(photo_eye == eye)
        P = ppos[sel]
        ok = np.isfinite(P[:, 0])
        if ok.sum() < 8:
            photo_pixel_eye[sel] = (np.arange(sel.size) % (RW_ * RH_))
            continue
        u = np.zeros(sel.size); v = np.zeros(sel.size)
        for k in (0, 1):
            col = P[ok, k]
            lo, hi = np.percentile(col, 2), np.percentile(col, 98)
            t = np.clip((P[:, k] - lo) / max(hi - lo, 1e-6), 0, 1)
            if k == 0: u = t
            else: v = t
        u = np.nan_to_num(u, nan=0.5); v = np.nan_to_num(v, nan=0.5)
        c0, c1 = cols
        cx = np.clip(c0 + u * (c1 - c0 - 1), c0, c1 - 1).round()
        ry = np.clip(v * (RH_ - 1), 0, RH_ - 1).round()
        photo_pixel_eye[sel] = (ry * RW_ + cx).astype(np.int64)
    print(f"  per-eye retinotopy: left covers cols 0-{half-1}, "
          f"right covers {RW_-half}-{RW_-1}")
    for eye, nm in ((-1, "left"), (1, "right")):
        sel = photo_eye == eye
        print(f"    {nm:5s} eye: {int(sel.sum()):>5,} photoreceptors over "
              f"{len(np.unique(photo_pixel_eye[sel])):>4} pixels")

    vpL, vpR = split(vp)
    olL, olR = split(ol)
    phL, phR = split(photo)
    print(f"  visual projection  L {len(vpL):,}  R {len(vpR):,}")
    print(f"  optic lobe         L {len(olL):,}  R {len(olR):,}")
    print(f"  photoreceptors     L {len(phL):,}  R {len(phR):,}")
    male_in = sum(1 for g in keep if male_of.get(int(bodies[g]), False))
    print(f"  male-specific / fruitless neurons in subnetwork: {male_in:,}")

    scale = float(np.abs(M.data).max())
    q = np.clip(np.round(M.data / scale * 127.0), -127, 127).astype(np.int8)
    idx = M.indices.astype(np.int32)
    blob = np.empty(M.nnz * 5, dtype=np.uint8)
    blob[: M.nnz * 4] = idx.view(np.uint8)
    blob[M.nnz * 4:] = q.view(np.uint8)
    n_files = 0
    for off in range(0, blob.nbytes, CHUNK):
        (out / f"net-{n_files}.bin").write_bytes(blob[off: off + CHUNK].tobytes())
        n_files += 1
    M.indptr.astype(np.int32).tofile(out / "rows.bin")

    bcfg = fly.brain.cfg
    spec = {
        "n": int(keep.size), "nnz": int(M.nnz), "files": n_files, "scale": scale,
        "steps_per_frame": args.steps,
        "dt": bcfg.dt, "decay": bcfg.decay, "gain": bcfg.gain,
        "refractory": bcfg.refractory, "trace_decay": 0.8,
        "retina": {"w": 15, "h": 24},
        "input": {"gain": fly.cfg.input_gain, "sustained": fly.cfg.sustained_weight,
                  "motion": fly.cfg.motion_gain},
        "photo": pos[photo].astype(int).tolist(),
        "photo_pixel": fly.photo_pixel.astype(int).tolist(),
        "photo_eye": photo_eye.astype(int).tolist(),
        "photo_pixel_eye": photo_pixel_eye.astype(int).tolist(),
        "vp_L": vpL, "vp_R": vpR,
        "ol_L": olL[:12000], "ol_R": olR[:12000],   # for the activity readout only
        "v_th": fly.brain.v_th[keep].astype(np.float32).tolist(),
        "bias": fly.brain.bias[keep].astype(np.float32).tolist(),
        "in_gain": fly.brain.in_gain[keep].astype(np.float32).tolist(),
        "male_specific": int(male_in),
        "provenance": {"dataset": "MaleCNS v1.0 (Google Research / HHMI Janelia, CC-BY)",
                       "full_brain_neurons": int(W.shape[0]),
                       "full_brain_synapses": int(W.nnz)},
    }
    (out / "net.json").write_text(json.dumps(spec))
    tot = sum(f.stat().st_size for f in out.glob("*")) / 1e6
    print(f"\nwrote {n_files} weight files + rows.bin + net.json -> {out}  ({tot:.1f} MB)")


if __name__ == "__main__":
    main()
