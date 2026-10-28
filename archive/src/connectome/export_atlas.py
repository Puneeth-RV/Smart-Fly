"""Export every neuron's real anatomical position, for rendering the brain itself.

139,662 of the 166,700 identified neurons carry a measured soma location in the
release. Rendered at those coordinates you get the actual shape of a fly's
central nervous system -- optic lobes, central brain, ventral nerve cord -- and
once the simulation is running you can watch activity travel through it.

The point is that nothing here is drawn. Positions are measured, the wiring is
measured, and the spikes come from the simulation; the viewer only colours
points.

Output (out/atlas/):
    atlas.bin     float32 xyz per neuron, normalised into a unit-ish box
    atlas.json    index ranges per population, bounds, and the neuron ids
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

from brain.lif import load_connectome  # noqa: E402

RAW = ROOT / "data" / "raw"
ANNOT = RAW / "body-annotations-male-cns-v1.0-minconf-0.5.feather"

# Groups the viewer colours separately. Ordered roughly along the sensory ->
# motor path, so a colour ramp over this order reads as depth through the brain.
GROUPS = [
    ("photoreceptor", lambda a: a.type.fillna("").str.startswith("R1-R6")),
    ("optic_lobe", lambda a: a.superclass == "ol_intrinsic"),
    ("visual_projection", lambda a: a.superclass == "visual_projection"),
    ("central_brain", lambda a: a.superclass == "cb_intrinsic"),
    ("kenyon_cell", lambda a: a["class"] == "Kenyon_Cell"),
    ("mbon", lambda a: a["class"] == "MBON"),
    ("dan", lambda a: a["class"] == "DAN"),
    ("central_complex", lambda a: a["class"] == "CX"),
    ("descending", lambda a: a.superclass == "descending_neuron"),
    ("ascending", lambda a: a.superclass == "ascending_neuron"),
    ("vnc", lambda a: a.superclass == "vnc_intrinsic"),
    ("motor", lambda a: a.superclass.isin(["vnc_motor", "cb_motor"])),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "out" / "atlas"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    W, raw, pops, meta = load_connectome(with_raw=True)
    z = np.load(ROOT / "data" / "processed" / "connectome.npz")
    bodies = z["bodies"]
    print(f"connectome: {len(bodies):,} neurons in the graph")

    a = pd.read_feather(ANNOT, columns=["bodyId", "type", "class", "superclass",
                                        "somaLocation"])
    a = a[a.bodyId.isin(set(bodies.tolist()))]
    have = a.somaLocation.notna()
    print(f"annotated and in graph: {len(a):,}, with soma coords: {int(have.sum()):,}")

    # Position by bodyId, then reorder to the connectome's own index so a
    # simulation state vector can be dropped straight onto the points.
    loc = {}
    for b, v in zip(a.loc[have, "bodyId"].to_numpy(), a.loc[have, "somaLocation"].to_numpy()):
        loc[int(b)] = v
    idx = [i for i, b in enumerate(bodies) if int(b) in loc]
    idx = np.asarray(idx, dtype=np.int64)
    P = np.array([loc[int(bodies[i])] for i in idx], dtype=np.float64)
    print(f"rendering {len(idx):,} neurons")

    # Centre and scale to a convenient size, preserving aspect so the anatomy
    # is not distorted.
    mn, mx = P.min(0), P.max(0)
    ctr = (mn + mx) / 2
    scale = 2.6 / float((mx - mn).max())
    Q = ((P - ctr) * scale).astype(np.float32)
    # Release coordinates are x right, y down, z front; flip y so the animal is
    # upright in a y-up renderer.
    Q[:, 1] *= -1.0
    print(f"extent (microns): {((mx-mn)/1000).round(1)}  ->  box {np.abs(Q).max(0).round(2)}")

    ann = a.set_index("bodyId")
    sub = ann.loc[[int(bodies[i]) for i in idx]].reset_index()
    group_id = np.full(len(sub), len(GROUPS), dtype=np.int32)   # default = other
    for gi, (name, sel) in enumerate(GROUPS):
        m = sel(sub).to_numpy()
        group_id[m & (group_id == len(GROUPS))] = gi
    counts = {GROUPS[i][0]: int((group_id == i).sum()) for i in range(len(GROUPS))}
    counts["other"] = int((group_id == len(GROUPS)).sum())
    print("\npopulations rendered:")
    for k, v in counts.items():
        print(f"  {k:<20} {v:>7,}")

    Q.tofile(out / "atlas.bin")
    group_id.astype(np.int32).tofile(out / "groups.bin")
    idx.astype(np.int32).tofile(out / "index.bin")
    (out / "atlas.json").write_text(json.dumps({
        "n": int(len(idx)),
        "groups": [g[0] for g in GROUPS] + ["other"],
        "counts": counts,
        "extent_um": [float(v) for v in (mx - mn) / 1000],
        "full_brain_neurons": int(len(bodies)),
        "full_brain_synapses": int(W.nnz),
    }))
    tot = sum(f.stat().st_size for f in out.glob("*")) / 1e6
    print(f"\nwrote {out}  ({tot:.1f} MB)")


if __name__ == "__main__":
    main()
