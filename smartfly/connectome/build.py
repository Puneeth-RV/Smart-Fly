"""Turn the raw MaleCNS release into one signed sparse connectivity matrix.

Inputs (CC-BY, fetched by `python -m smartfly.connectome.download`):
  * connectome-weights-*.feather      segment-to-segment synapse counts
  * body-annotations-*.feather        cell class / type / superclass per bodyId
  * body-neurotransmitters-*.feather  predicted transmitter per bodyId

Output, in data/processed/:
  * connectome.npz    CSR matrix W, W[i, j] = signed strength of i -> j; an
                      unsigned copy; bodyIds; ommatidial (hex) coordinates
  * populations.json  matrix indices of named cell populations
  * meta.json         counts and the sign convention

    python -m smartfly.connectome.build      # ~2 min, ~6 GB RAM

Sign convention
---------------
The release predicts a transmitter per neuron, not a sign, so signs are assigned
presynaptically from the transmitter identity:

    acetylcholine -> +1   the main excitatory transmitter in the fly CNS
    GABA          -> -1   inhibitory
    glutamate     -> -1   inhibitory in Drosophila via GluCl-alpha, unlike
                          vertebrates; this is the standard modelling choice
                          but it is a genuine simplification, since a minority
                          of fly glutamatergic synapses are excitatory
    dopamine, serotonin, octopamine -> 0 for fast transmission

The aminergic neurons get zero fast weight because they act as neuromodulators.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import scipy.sparse as sp

from smartfly.paths import PROCESSED as PROC
from smartfly.paths import RAW

WEIGHTS_F = RAW / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
ANNOT_F = RAW / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NT_F = RAW / "body-neurotransmitters-male-cns-v1.0.feather"

# Fast-transmission sign per predicted transmitter. See module docstring.
NT_SIGN = {
    "acetylcholine": 1.0,
    "ach": 1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "glut": -1.0,
    "dopamine": 0.0,
    "serotonin": 0.0,
    "octopamine": 0.0,
    "tyramine": 0.0,
    "histamine": -1.0,  # photoreceptor transmitter; hyperpolarises LMCs
    "unknown": 0.0,
}

# Minimum synapse count for an edge to be kept. The release is thresholded at
# confidence 0.5 but still contains a long tail of 1-2 synapse connections that
# are dominated by reconstruction noise; 3 is the threshold used in most
# published connectome analyses.
MIN_WEIGHT = 3


def _first_present(df: pd.DataFrame, *candidates: str) -> str:
    """Return the first column that exists, so schema drift is loud not silent."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"none of {candidates} in columns: {list(df.columns)[:40]}")


def load_edges() -> pd.DataFrame:
    """Load the edge list, thresholding inside Arrow to keep memory bounded.

    The raw table is ~152M rows; materialising that as a pandas frame of int64
    columns costs several GB before any filtering. Arrow's compute kernels do
    the threshold first, which drops the great majority of rows (most segment
    pairs share only one or two synapses) before anything is converted.
    """
    import pyarrow.compute as pc
    import pyarrow.feather as feather

    print(f"reading {WEIGHTS_F.name} ...")
    t = feather.read_table(WEIGHTS_F)
    names = t.column_names
    pre = next(c for c in ("bodyId_pre", "bodyid_pre", "body_pre", "pre") if c in names)
    post = next(c for c in ("bodyId_post", "bodyid_post", "body_post", "post") if c in names)
    wt = next(c for c in ("weight", "count", "syn_count", "n") if c in names)
    n0 = t.num_rows

    keep = pc.and_(
        pc.greater_equal(t[wt], MIN_WEIGHT),
        pc.not_equal(t[pre], t[post]),  # autapses are reconstruction artefacts
    )
    t = t.filter(keep).select([pre, post, wt])
    print(
        f"  {n0:,} edges -> {t.num_rows:,} after weight>={MIN_WEIGHT} "
        f"and autapse removal ({t.num_rows / n0:.1%} retained)"
    )

    # This is a *segment*-to-segment table: it contains every reconstructed
    # fragment, not just neurons, which is why the raw body count runs into the
    # millions. Restrict both endpoints to identified neurons -- the annotated
    # bodies carrying a superclass, which number 166,700 and are exactly the
    # neurons the release reports.
    neurons = identified_neurons()
    arr = pa.array(neurons)
    t = t.filter(pc.and_(pc.is_in(t[pre], value_set=arr), pc.is_in(t[post], value_set=arr)))
    print(f"  -> {t.num_rows:,} edges between identified neurons")

    w = t.to_pandas()
    del t
    return w.rename(columns={pre: "pre", post: "post", wt: "weight"})


def identified_neurons() -> np.ndarray:
    """Body IDs of real, identified neurons.

    A superclass assignment is what distinguishes a curated neuron from an
    unproofread fragment in the annotation table.
    """
    a = pd.read_feather(ANNOT_F, columns=["bodyId", "superclass"])
    ids = a.loc[a.superclass.notna(), "bodyId"].astype("int64").unique()
    ids.sort()
    print(f"  {len(ids):,} identified neurons (annotated with a superclass)")
    return ids


def load_signs() -> pd.DataFrame:
    print(f"reading {NT_F.name} ...")
    nt = pd.read_feather(NT_F)
    body = _first_present(nt, "body", "bodyId", "bodyid")
    # Prefer the consensus call, fall back to the per-celltype then per-body one.
    cols = [c for c in ("consensus_nt", "celltype_predicted_nt", "predicted_nt") if c in nt.columns]
    nt = nt[[body] + cols].rename(columns={body: "body"})
    nt = nt.drop_duplicates(subset="body", keep="first")
    call = nt[cols[0]]
    for c in cols[1:]:
        call = call.fillna(nt[c])
    nt["nt"] = call.fillna("unknown").str.strip().str.lower()
    nt["sign"] = nt["nt"].map(NT_SIGN)
    unmapped = nt.loc[nt.sign.isna(), "nt"].value_counts()
    if len(unmapped):
        print(f"  unmapped transmitters (treated as modulatory): {dict(unmapped.head(10))}")
    nt["sign"] = nt["sign"].fillna(0.0)
    print(f"  {len(nt):,} bodies with a transmitter call")
    print(f"  sign split: {dict(nt.sign.value_counts())}")
    return nt[["body", "nt", "sign"]]


def population_masks(annot: pd.DataFrame, index: dict[int, int], n: int) -> dict[str, list[int]]:
    """Matrix indices of named cell populations.

    Selection is by the release's own annotations, so these are real identified
    cell populations rather than anything hand-drawn.
    """
    a = annot.copy()
    a["idx"] = a.bodyId.map(index)
    a = a.dropna(subset=["idx"])
    a["idx"] = a["idx"].astype(int)

    def by(mask) -> list[int]:
        return sorted(a.loc[mask, "idx"].unique().tolist())

    t = a["type"].fillna("")
    cls = a["class"].fillna("")
    sup = a["superclass"].fillna("")

    pops = {
        # Vision. R1-R6 are the motion/luminance channel the fly's eye uses
        # here; R7/R8 are chromatic and kept separate.
        "photoreceptor_R16": by(t.str.startswith("R1-R6")),
        "photoreceptor_R7R8": by(t.str.startswith("R7") | t.str.startswith("R8")),
        "optic_lobe": by(sup == "ol_intrinsic"),
        "visual_projection": by(sup == "visual_projection"),
        # Mushroom body: the fly's associative learning centre.
        "kenyon_cell": by(cls == "Kenyon_Cell"),
        "mbon": by(cls == "MBON"),
        # Reinforcement channels. PAM signals reward, PPL1 signals punishment.
        "dan_pam": by((cls == "DAN") & t.str.startswith("PAM")),
        "dan_ppl1": by((cls == "DAN") & t.str.startswith("PPL1")),
        "dan_other": by((cls == "DAN") & ~t.str.startswith("PAM") & ~t.str.startswith("PPL1")),
        # Output path: brain -> VNC -> muscle.
        "descending": by(sup == "descending_neuron"),
        "motor": by(sup.isin(["vnc_motor", "cb_motor"])),
        # Context, for reference and for optional ablations.
        "central_brain": by(sup == "cb_intrinsic"),
        "vnc_intrinsic": by(sup == "vnc_intrinsic"),
    }
    print("\npopulations:")
    for k, v in pops.items():
        print(f"  {k:24s} {len(v):>7,}")
    return pops


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)

    edges = load_edges()
    signs = load_signs()
    print(f"reading {ANNOT_F.name} ...")
    annot = pd.read_feather(ANNOT_F)

    # Index space is every body that actually participates in a surviving edge.
    bodies = np.union1d(edges.pre.to_numpy(), edges.post.to_numpy())
    index = {int(b): i for i, b in enumerate(bodies)}
    n = len(bodies)
    print(f"\n{n:,} neurons in the connectivity graph")

    # Ommatidial coordinates, for reconstructing retinotopy. Only the columnar
    # optic-lobe cells carry them (23,720 cells over 892 columns); the
    # photoreceptors themselves do not, so their retinal positions have to be
    # inferred from the columns they project into.
    hx = annot.dropna(subset=["assignedOlHex1", "assignedOlHex2"])
    hex_by_body = dict(
        zip(hx.bodyId.to_numpy(), zip(hx.assignedOlHex1.to_numpy(), hx.assignedOlHex2.to_numpy()))
    )
    hex1 = np.full(len(bodies), np.nan, dtype=np.float32)
    hex2 = np.full(len(bodies), np.nan, dtype=np.float32)
    for i, b in enumerate(bodies):
        v = hex_by_body.get(int(b))
        if v is not None:
            hex1[i], hex2[i] = v
    print(f"  {int(np.isfinite(hex1).sum()):,} neurons carry ommatidial coordinates")

    sign_by_body = dict(zip(signs.body.to_numpy(), signs.sign.to_numpy()))
    presign = np.array([sign_by_body.get(int(b), 0.0) for b in bodies], dtype=np.float32)
    missing = int((~np.isin(bodies, signs.body.to_numpy())).sum())
    print(f"  {missing:,} neurons lack a transmitter call -> sign 0")

    rows = edges.pre.map(index).to_numpy(dtype=np.int64)
    cols = edges.post.map(index).to_numpy(dtype=np.int64)
    # Signed by the presynaptic neuron's transmitter.
    vals = edges.weight.to_numpy(dtype=np.float32) * presign[rows]

    W = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
    W.eliminate_zeros()

    # The signed matrix zeroes the aminergic neurons, which also erases their
    # wiring, so keep an unsigned structural matrix too: signed W for dynamics,
    # raw W for anatomy (e.g. which cells a photoreceptor contacts).
    raw = sp.csr_matrix(
        (edges.weight.to_numpy(np.float32), (rows, cols)), shape=(n, n), dtype=np.float32
    )
    raw.eliminate_zeros()
    print(f"  raw (unsigned, all transmitters): nnz={raw.nnz:,}")
    print(
        f"  W: {W.shape} nnz={W.nnz:,} "
        f"(exc {int((W.data > 0).sum()):,} / inh {int((W.data < 0).sum()):,})"
    )

    pops = population_masks(annot, index, n)

    np.savez_compressed(
        PROC / "connectome.npz",
        data=W.data,
        indices=W.indices,
        indptr=W.indptr,
        shape=np.array(W.shape),
        raw_data=raw.data,
        raw_indices=raw.indices,
        raw_indptr=raw.indptr,
        bodies=bodies,
        presign=presign,
        hex1=hex1,
        hex2=hex2,
    )
    (PROC / "populations.json").write_text(json.dumps(pops))
    meta = {
        "n_neurons": int(n),
        "n_edges": int(W.nnz),
        "min_weight": MIN_WEIGHT,
        "nt_sign_convention": {k: v for k, v in NT_SIGN.items()},
        "populations": {k: len(v) for k, v in pops.items()},
    }
    (PROC / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {PROC / 'connectome.npz'}")
    print(f"wrote {PROC / 'populations.json'}")
    print(f"wrote {PROC / 'meta.json'}")


if __name__ == "__main__":
    main()
