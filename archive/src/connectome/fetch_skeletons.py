"""Fetch real neuron morphologies and pack them for the browser.

Cell-body points give you the shape of the brain but nothing of what a neuron
looks like. The release ships a traced skeleton for every neuron -- the actual
branching arbor, thousands of points each -- and rendering those is the
difference between a dot cloud and something worth looking at.

A full skeleton is ~2,000 points, so everything is downsampled on the way out:
branch structure is preserved (every branch point and every tip is kept) and
only the long straight runs between them are thinned. That holds the shape at a
fraction of the size.

Selection is by cell type rather than at random, so the result is anatomically
legible: the compass ring, the mushroom body, the optic lobe's repeating
columns, the long descending tracts into the nerve cord.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
ANNOT = RAW / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
BASE = ("https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/"
        "skeletons-malecns/skeletons-swc")
CACHE = ROOT / "data" / "skeletons"

# What to draw, and how many of each. Chosen for anatomy that reads at a glance.
SELECTION = [
    ("EPG",              "type_prefix", "EPG",        50,  "compass ring"),
    ("Delta7",           "type_prefix", "Delta7",     42,  "ring inhibition"),
    ("PEN",              "type_prefix", "PEN",        42,  "rotates the bump"),
    ("MBON",             "class",       "MBON",       97,  "mushroom body output"),
    ("DAN",              "class",       "DAN",        60,  "dopamine"),
    ("KC",               "class",       "Kenyon_Cell",120, "mushroom body"),
    ("LC",               "type_prefix", "LC",        140,  "visual projection"),
    ("descending",       "superclass",  "descending_neuron", 150, "brain to nerve cord"),
    ("motor",            "superclass",  "vnc_motor",  90,  "to the muscles"),
    ("Mi1",              "type_prefix", "Mi1",       110,  "optic lobe columns"),
    ("T4",               "type_prefix", "T4",        120,  "motion detectors"),
    ("T5",               "type_prefix", "T5",        110,  "motion detectors"),
    ("L1",               "type_prefix", "L1",         90,  "lamina"),
    ("photoreceptor",    "type_prefix", "R1-R6",      80,  "the eye"),
]


def pick(a: pd.DataFrame, kind: str, key: str, n: int, seed: int = 0) -> list[int]:
    t = a.type.fillna("")
    if kind == "type_prefix":
        m = t.str.startswith(key)
    elif kind == "class":
        m = a["class"].fillna("") == key
    else:
        m = a.superclass.fillna("") == key
    ids = a.loc[m, "bodyId"].to_numpy()
    if len(ids) > n:
        ids = np.random.default_rng(seed).choice(ids, n, replace=False)
    return [int(v) for v in ids]


def fetch(body: int) -> bytes | None:
    f = CACHE / f"{body}.swc"
    if f.exists():
        return f.read_bytes()
    try:
        with urlopen(f"{BASE}/{body}.swc", timeout=40) as r:
            data = r.read()
        f.write_bytes(data)
        return data
    except Exception:
        return None


def parse_swc(data: bytes):
    """Return (points Nx3, parent index per point) with ids remapped to 0..N-1."""
    xyz, parent, idmap = [], [], {}
    for line in data.decode("utf8", "replace").splitlines():
        if not line or line[0] == "#":
            continue
        p = line.split()
        if len(p) < 7:
            continue
        nid = int(p[0])
        idmap[nid] = len(xyz)
        xyz.append((float(p[2]), float(p[3]), float(p[4])))
        parent.append(int(p[6]))
    if not xyz:
        return None, None
    P = np.asarray(xyz, dtype=np.float32)
    par = np.array([idmap.get(q, -1) for q in parent], dtype=np.int32)
    return P, par


def simplify(P: np.ndarray, par: np.ndarray, keep_every: int):
    """Thin the straight runs, keep every branch point and tip.

    Edges are emitted as index pairs, so the viewer draws the arbor as line
    segments without needing the tree structure.
    """
    n = len(P)
    nchild = np.zeros(n, dtype=np.int32)
    for q in par:
        if q >= 0:
            nchild[q] += 1
    important = (nchild != 1)                 # branch points, tips, roots
    keep = important.copy()
    keep[::keep_every] = True

    # Walk each kept node up to its nearest kept ancestor and emit one edge.
    edges = []
    for i in range(n):
        if not keep[i]:
            continue
        q = par[i]
        while q >= 0 and not keep[q]:
            q = par[q]
        if q >= 0:
            edges.append((i, q))
    idx = np.flatnonzero(keep)
    remap = -np.ones(n, dtype=np.int64)
    remap[idx] = np.arange(len(idx))
    if not edges:
        return P[idx], np.zeros((0, 2), np.int32)
    E = remap[np.asarray(edges, dtype=np.int64)]
    return P[idx], E.astype(np.int32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-every", type=int, default=6)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "out" / "morph"))
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    a = pd.read_feather(ANNOT, columns=["bodyId", "type", "class", "superclass"])
    groups, wanted = [], []
    for name, kind, key, n, desc in SELECTION:
        ids = pick(a, kind, key, n)
        groups.append({"name": name, "desc": desc, "ids": ids})
        wanted += ids
        print(f"  {name:<14} {len(ids):>4}  {desc}")
    wanted = list(dict.fromkeys(wanted))
    print(f"\nfetching {len(wanted):,} skeletons (cached in {CACHE.relative_to(ROOT)}) ...")

    done = {"n": 0, "bytes": 0}
    lock = threading.Lock()
    def job(b):
        d = fetch(b)
        with lock:
            done["n"] += 1
            done["bytes"] += len(d) if d else 0
            if done["n"] % 150 == 0:
                print(f"    {done['n']:>5}/{len(wanted)}  {done['bytes']/1e6:.0f} MB")
        return b, d
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(job, wanted))

    skel = {b: d for b, d in results if d}
    print(f"  got {len(skel):,} of {len(wanted):,}")

    allP, allE, meta, voff = [], [], [], 0
    gi_of = {}
    for gi, g in enumerate(groups):
        for b in g["ids"]:
            gi_of.setdefault(b, gi)
    for b, d in skel.items():
        P, par = parse_swc(d)
        if P is None or len(P) < 8:
            continue
        P2, E = simplify(P, par, args.keep_every)
        if len(E) == 0:
            continue
        allP.append(P2); allE.append(E + voff)
        meta.append({"body": b, "group": gi_of.get(b, 0),
                     "v0": voff, "nv": int(len(P2))})
        voff += len(P2)

    P = np.concatenate(allP); E = np.concatenate(allE)
    # Normalise into the same box the atlas uses, y flipped so it stands upright.
    mn, mx = P.min(0), P.max(0)
    ctr = (mn + mx) / 2
    s = 2.6 / float((mx - mn).max())
    Q = ((P - ctr) * s).astype(np.float32)
    Q[:, 1] *= -1.0

    Q.tofile(out / "verts.bin")
    E.astype(np.uint32).tofile(out / "edges.bin")
    (out / "morph.json").write_text(json.dumps({
        "neurons": len(meta), "vertices": int(len(Q)), "edges": int(len(E)),
        "groups": [{"name": g["name"], "desc": g["desc"]} for g in groups],
        "meta": meta,
        "extent_um": [float(v) for v in (mx - mn) / 1000],
    }))
    tot = sum(f.stat().st_size for f in out.glob("*")) / 1e6
    print(f"\n{len(meta):,} neurons | {len(Q):,} points | {len(E):,} segments | {tot:.1f} MB")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
