"""Can the fly tell things apart? Read objects back out of its visual system.

Plays 1-1 with right-leaning random buttons, and at every decision labels the
frame from the game's own memory:

  goomba   an enemy 8-96 px ahead of Mario on screen
  wall     a solid tile (pipe, block, step) 1-3 tiles ahead at Mario's height
  gap      no ground 1-4 tiles ahead

Then fits a linear readout for each label from four inputs and scores it on
episodes it was not fitted on (balanced accuracy, 50% = chance):

  fly       optic-lobe features the policy actually gets (664)
  retina    the 15x24 contrast image the photoreceptors receive (360)
  wide      the same 360 cells laid out 24 wide x 15 tall, no brain
  colour    a 24x15 colour thumbnail, no brain (1080)

If the fly is far below the retina, the brain is losing what the eye had. If
the retina is far below the wide or colour images, the eye itself is the limit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "out"

ACTION_P = np.array([.03, .25, .20, .25, .15, .10, .02])


def tile(ram, col, row):
    if not 0 <= row < 13:
        return 0
    page = (col // 16) % 2
    return int(ram[0x500 + page * 208 + row * 16 + col % 16])


def labels(ram, x_level):
    mx = int(ram[0x3AD])                   # Mario's x on screen
    my = int(ram[0xCE])                    # Mario's y on screen (top of sprite)
    goomba = any(ram[0x0F + i] and 8 <= int(ram[0x3AE + i]) - mx <= 96
                 for i in range(5))
    c = (x_level + 8) // 16
    rows = {(my - 32) // 16, (my - 32 + 15) // 16}
    wall = any(tile(ram, c + k, r) for k in (1, 2, 3) for r in rows)
    gap = any(tile(ram, c + k, 12) == 0 for k in (1, 2, 3, 4))
    return [goomba, wall, gap]


def wide(frame, colour=False):
    f = frame.astype(np.float32)[48:224] / 255.0
    H, W = 15, 24
    bh, bw = f.shape[0] // H, f.shape[1] // W
    b = f[:bh * H, :bw * W].reshape(H, bh, W, bw, 3)
    if colour:
        return b.mean(axis=(1, 3)).ravel()
    g = b @ np.array([0.299, 0.587, 0.114], np.float32)
    out = g.std(axis=(1, 3))
    return (out / max(out.max(), 1e-6)).ravel()


def collect(n_dec, calib, seed=0, brain_steps=3, sustained=None, graded=False, gain=0.9, rest=0.5,
            v2=False):
    from nes.mario import Mario
    from nes.mario_agent import MarioFly
    retina_shape = (15, 24) if v2 else (24, 15)
    agent = MarioFly(brain_steps=brain_steps, frame_skip=4, graded=v2,
                     retina=retina_shape if v2 else None)
    if sustained is not None:
        agent.cfg.sustained_weight = sustained
    if v2:
        pass
    elif graded:
        from brain.lif import GradedBrain
        agent.brain = GradedBrain(agent.brain.W, gain=gain)
        agent.brain.bias[:] = rest      # resting depolarisation, so inhibition can signal
        photo = agent.photo

        def drive(retina):
            px = retina.ravel()[agent.photo_pixel]
            prev = agent.prev_px if agent.prev_px is not None else px
            agent.prev_px = px.copy()
            e = np.zeros(agent.n, np.float32)
            e[photo] = 0.2 + 0.8 * px + 1.5 * (px - prev)
            return e
        agent._drive = drive
    else:
        z = np.load(calib)
        agent.brain.bias = z["bias"].astype(np.float32)
        agent.brain.v_th = z["v_th"].astype(np.float32)
    m = Mario("SuperMarioBros-1-1-v0", retina=retina_shape)
    rng = np.random.default_rng(seed)
    X = {"fly": [], "neurons": [], "typed": [], "retina": [], "wide": [], "colour": [],
         "photo": [], "first": []}
    first = np.random.default_rng(1).choice(agent.ol1, min(3000, agent.ol1.size), replace=False)
    neurons, typed_cell, typed_n = readouts(agent)
    active = np.zeros(agent.n, np.int64)
    Y, ep_id, ep = [], [], 0
    retina = m.reset(); agent.brain.reset(); agent.prev_px = None
    agent.prev_grid = np.zeros(agent.n_grid, np.float32)
    x_level = 40
    nes = m.env
    while not hasattr(nes, "ram"):
        nes = nes.env
    while len(Y) < n_dec:
        agent.ext = agent._drive(retina)
        for _ in range(agent.brain_steps):
            agent.brain.step(agent.ext)
        ram = nes.ram
        if len(Y) % 1 == 0:
            tr = agent.brain.trace
            active += agent.brain.spikes
            X["fly"].append(agent.features()); X["retina"].append(retina.ravel().copy())
            X["neurons"].append(tr[neurons].copy())
            X["photo"].append(tr[agent.photo].copy())
            X["first"].append(tr[first].copy())
            X["typed"].append(np.bincount(typed_cell, weights=tr[typed_cell >= 0] if False else None,
                                          minlength=0) if False else
                              np.bincount(typed_cell[typed_cell >= 0],
                                          weights=tr[np.flatnonzero(typed_cell >= 0)],
                                          minlength=typed_n) / np.maximum(np.bincount(
                                              typed_cell[typed_cell >= 0], minlength=typed_n), 1))
            X["wide"].append(wide(m.frame)); X["colour"].append(wide(m.frame, True))
            Y.append(labels(ram, x_level)); ep_id.append(ep)
        a = int(rng.choice(7, p=ACTION_P))
        for _ in range(4):
            retina, _, d, info = m.step(a)
            x_level = int(info.get("x_pos", x_level))
            if d:
                break
        if m.done:
            ep += 1
            retina = m.reset(); agent.brain.reset(); agent.prev_px = None
            agent.prev_grid = np.zeros(agent.n_grid, np.float32)
        if len(Y) % 500 == 0:
            print(f"  {len(Y)} decisions, {ep} episodes", flush=True)
    ol = np.flatnonzero(typed_cell >= 0)
    print(f"  optic-lobe neurons that spiked at all on sampled steps: "
          f"{(active[ol] > 0).mean():.0%}")
    return {k: np.array(v, np.float32) for k, v in X.items()}, np.array(Y), np.array(ep_id)


def readouts(agent, n_neurons=3000, n_types=30, gx=6, gy=4):
    """Two richer reads of the same optic lobe.

    neurons: individual cells (a random 3,000 with visual-field coordinates), no
             averaging at all -- the ceiling for what a readout could get.
    typed:   the 30 most numerous cell types, each pooled on a coarse 6x4 map.
             The policy's grid averages every cell type in a patch together,
             and ON and OFF cells responding to the same edge cancel out.
    """
    import pandas as pd
    W, pops, meta = __import__("brain.lif", fromlist=["x"]).load_connectome()
    z = np.load(ROOT / "data/processed/connectome.npz")
    bodies = z["bodies"]
    photo = np.asarray(pops["photoreceptor_R16"], np.int64)
    ol = np.asarray(pops["optic_lobe"], np.int64)
    keep = np.unique(np.concatenate([photo, ol]))
    a = pd.read_feather(ROOT / "data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather",
                        columns=["bodyId", "type"])
    tmap = dict(zip(a.bodyId.to_numpy(), a["type"].fillna("").to_numpy()))
    types = np.array([tmap.get(int(b), "") for b in bodies[keep]])
    h1, h2 = meta["hex1"][keep], meta["hex2"][keep]
    has = np.isfinite(h1) & np.isfinite(h2) & np.isin(keep, ol)
    rng = np.random.default_rng(0)
    neurons = rng.choice(np.flatnonzero(has), min(n_neurons, int(has.sum())), replace=False)
    names, counts = np.unique(types[has & (types != "")], return_counts=True)
    top = names[np.argsort(-counts)[:n_types]]
    print(f"  top cell types: {', '.join(top[:12])} ...")
    def q(v, n):
        lo, hi = np.nanpercentile(v[has], 2), np.nanpercentile(v[has], 98)
        return np.clip(((v - lo) / (hi - lo) * n).astype(int), 0, n - 1)
    cx, cy = q(np.nan_to_num(h1), gx), q(np.nan_to_num(h2), gy)
    tix = {t: i for i, t in enumerate(top)}
    cell = -np.ones(keep.size, np.int64)
    for i in np.flatnonzero(has):
        t = tix.get(types[i])
        if t is not None:
            cell[i] = (t * gy + cy[i]) * gx + cx[i]
    return neurons, cell, n_types * gx * gy


def probe(X, y, train, test):
    mu, sd = X[train].mean(0), X[train].std(0) + 1e-6
    Xs = (X - mu) / sd
    Xs = np.hstack([Xs, np.ones((len(Xs), 1), np.float32)])
    t = np.where(y, 1.0, -1.0)
    # class-balanced ridge, lambda picked on a slice of the training episodes
    w_pos = 0.5 / max(y[train].mean(), 1e-3); w_neg = 0.5 / max(1 - y[train].mean(), 1e-3)
    sw = np.where(y, w_pos, w_neg)

    def fit(idx, lam):
        A = Xs[idx] * sw[idx, None]
        return np.linalg.solve(Xs[idx].T @ A + lam * np.eye(Xs.shape[1]), A.T @ t[idx])

    def bacc(w, idx):
        p = Xs[idx] @ w > 0
        return 0.5 * ((p & y[idx]).sum() / max(y[idx].sum(), 1)
                      + (~p & ~y[idx]).sum() / max((~y[idx]).sum(), 1))
    cut = int(len(train) * 0.75)
    tr, va = train[:cut], train[cut:]
    lam = max([1, 10, 100, 1000, 10000], key=lambda l: bacc(fit(tr, l), va))
    return bacc(fit(train, lam), test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=int, default=4000)
    ap.add_argument("--brain-steps", type=int, default=3)
    ap.add_argument("--sustained", type=float, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--graded", action="store_true")
    ap.add_argument("--v2", action="store_true", help="wide eye + graded brain, as trained")
    ap.add_argument("--gain", type=float, default=0.9)
    ap.add_argument("--rest", type=float, default=0.5)
    ap.add_argument("--calibration", default=str(OUT / "mario_final_calibration.npz"))
    args = ap.parse_args()
    X, Y, ep = collect(args.decisions, args.calibration, brain_steps=args.brain_steps,
                       sustained=args.sustained, graded=args.graded, gain=args.gain, rest=args.rest, v2=args.v2)
    order = np.arange(len(Y))
    split = ep[int(len(ep) * 0.7)]
    train, test = order[ep < split], order[ep >= split]
    names = ["goomba", "wall", "gap"]
    res = {"n": len(Y), "rate": {n: float(Y[:, i].mean()) for i, n in enumerate(names)}}
    print(f"\n{len(Y)} decisions; label rates "
          + ", ".join(f"{n} {Y[:, i].mean():.0%}" for i, n in enumerate(names)))
    print(f"{'':8s}" + "".join(f"{n:>9s}" for n in names))
    for k in ["photo", "first", "fly", "typed", "neurons", "retina"]:
        row = [probe(X[k], Y[:, i], train, test) for i in range(3)]
        res[k] = dict(zip(names, row))
        print(f"{k:8s}" + "".join(f"{v:9.0%}" for v in row), flush=True)
    (OUT / f"probe_vision{args.tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
