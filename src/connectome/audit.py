"""Check the architecture's assumptions against the real wiring.

The agent design assumes a specific chain exists in the connectome:

    photoreceptors -> optic lobe -> central brain -> Kenyon cells
                                                  -> MBONs -> ... -> descending
                                                  -> motor neurons

and that the dopaminergic populations innervate MBONs in a compartment
structure. None of that is guaranteed by having downloaded the data, and if any
link is missing or vanishingly thin, the learning design is built on sand. This
script measures each link before any training is attempted, so the design either
earns its assumptions or gets corrected.

Reported per link: whether a direct connection exists, how many neurons on each
side participate, total synapse count, and the shortest path length if there is
no direct edge.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from brain.lif import load_connectome  # noqa: E402

# The chain the agent design depends on.
LINKS = [
    ("photoreceptor_R16", "optic_lobe"),
    ("optic_lobe", "visual_projection"),
    ("visual_projection", "central_brain"),
    ("central_brain", "kenyon_cell"),
    ("kenyon_cell", "mbon"),
    ("mbon", "central_brain"),
    ("central_brain", "descending"),
    ("descending", "motor"),
    ("descending", "vnc_intrinsic"),
    ("vnc_intrinsic", "motor"),
    # Reinforcement machinery.
    ("dan_pam", "mbon"),
    ("dan_ppl1", "mbon"),
    ("dan_pam", "kenyon_cell"),
    # The one the design is least sure of: can MBONs actually influence output?
    ("mbon", "descending"),
    ("mbon", "motor"),
]


def hop_distance(A: sp.csr_matrix, src: np.ndarray, dst: np.ndarray, max_hops: int = 8) -> int:
    """Fewest hops from any neuron in src to any in dst, or -1 if unreachable."""
    reached = np.zeros(A.shape[0], dtype=bool)
    reached[src] = True
    dstmask = np.zeros(A.shape[0], dtype=bool)
    dstmask[dst] = True
    if (reached & dstmask).any():
        return 0
    frontier = reached.copy()
    for h in range(1, max_hops + 1):
        nxt = (A.T @ frontier.astype(np.int8)) > 0
        new = nxt & ~reached
        if not new.any():
            return -1
        if (new & dstmask).any():
            return h
        reached |= new
        frontier = new
    return -1


def main() -> None:
    # Anatomy must be read from the unsigned matrix: the signed one zeroes the
    # aminergic neurons, which would make the dopaminergic projections look
    # absent when they are simply not fast-transmitting.
    W, raw, pops, meta = load_connectome(with_raw=True)
    n = W.shape[0]
    A = (raw != 0).astype(np.int8)
    absW = abs(raw)
    print(f"connectome: {n:,} neurons, {W.nnz:,} edges\n")

    print(f"{'link':44s} {'direct':>8s} {'syn':>12s} {'pre/post used':>16s} {'hops':>5s}")
    print("-" * 92)
    audit = []
    for a, b in LINKS:
        src = np.asarray(pops[a], dtype=np.int64)
        dst = np.asarray(pops[b], dtype=np.int64)
        if src.size == 0 or dst.size == 0:
            print(f"{a+' -> '+b:44s} {'EMPTY':>8s}")
            continue
        block = absW[np.ix_(src, dst)]
        syn = float(block.sum())
        npre = int((np.asarray(block.sum(axis=1)).ravel() > 0).sum())
        npost = int((np.asarray(block.sum(axis=0)).ravel() > 0).sum())
        direct = block.nnz > 0
        hops = 0 if direct else hop_distance(A, src, dst)
        used = f"{npre}/{src.size}, {npost}/{dst.size}"
        print(f"{a+' -> '+b:44s} {'yes' if direct else 'NO':>8s} {syn:12,.0f} "
              f"{used:>16s} {hops:>5d}")
        audit.append({"pre": a, "post": b, "direct": bool(direct), "synapses": syn,
                      "n_pre_used": npre, "n_post_used": npost,
                      "pre_size": int(src.size), "post_size": int(dst.size),
                      "hops": int(hops)})

    # Compartment structure: does each MBON get a distinct dopaminergic input
    # pattern? If every MBON received identical dopamine there would be no
    # compartments and the learning rule could not differentiate behaviours.
    print("\ndopaminergic compartment structure")
    mbon = np.asarray(pops["mbon"], dtype=np.int64)
    dans = np.asarray(pops["dan_pam"] + pops["dan_ppl1"], dtype=np.int64)
    D = np.asarray(absW[np.ix_(dans, mbon)].todense(), dtype=np.float32)
    innervated = int((D.sum(axis=0) > 0).sum())
    print(f"  MBONs receiving any DAN input: {innervated}/{mbon.size}")
    if innervated:
        cols = D[:, D.sum(axis=0) > 0]
        norm = cols / (np.linalg.norm(cols, axis=0, keepdims=True) + 1e-9)
        sim = norm.T @ norm
        off = sim[~np.eye(sim.shape[0], dtype=bool)]
        print(f"  mean pairwise cosine similarity of DAN input patterns: {off.mean():.3f}")
        print(f"  (low means distinct compartments; near 1.0 would mean no "
              f"compartment structure)")

    # How much of the brain actually sits between eye and muscle.
    print("\nsensorimotor reach")
    src = np.asarray(pops["photoreceptor_R16"], dtype=np.int64)
    dst = np.asarray(pops["descending"] + pops["motor"], dtype=np.int64)
    for hops in (3, 4, 5, 6):
        from brain.lif import sensorimotor_subgraph
        keep = sensorimotor_subgraph(W, src, dst, max_hops=hops)
        sub = W[np.ix_(keep, keep)]
        kc = len(set(keep.tolist()) & set(pops["kenyon_cell"]))
        mb = len(set(keep.tolist()) & set(pops["mbon"]))
        print(f"  {hops} hops: {keep.size:>7,} neurons  {sub.nnz:>12,} edges  "
              f"KC {kc:>5,}  MBON {mb:>3}")

    out = ROOT / "out" / "pathway_audit.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(audit, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
