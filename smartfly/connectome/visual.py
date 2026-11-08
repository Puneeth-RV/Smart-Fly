"""The part of the connectome the fly uses: its eye and optic lobe.

The full matrix is 166,197 neurons and 84 MB. The fly needs only the R1-R6
photoreceptors and the optic lobe -- 92,743 neurons and 3.69M connections -- plus
the ommatidial coordinates that say where each optic-lobe column looks. This
module cuts that out into data/visual_system.npz, so playing and training never
touch the full connectome.

    python -m smartfly.connectome.visual
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from smartfly.paths import PROCESSED, VISUAL_SYSTEM


@dataclass
class VisualSystem:
    """Photoreceptors and optic lobe, indexed locally 0..n-1.

    W            signed synapse strengths among these neurons (pre x post, CSR)
    photo        local indices of the R1-R6 photoreceptors
    optic_lobe   local indices of the optic-lobe neurons
    photo_out    each photoreceptor's synapses onto *any* neuron in the brain
                 (photoreceptor x whole-brain, CSR, unsigned counts)
    hex1, hex2   ommatidial coordinates of every neuron in the whole brain
                 (NaN where a neuron has none), indexed like photo_out's columns
    whole_index  for each local neuron, its index in the whole-brain matrix
    """

    W: sp.csr_matrix
    photo: np.ndarray
    optic_lobe: np.ndarray
    photo_out: sp.csr_matrix
    hex1: np.ndarray
    hex2: np.ndarray
    whole_index: np.ndarray

    @property
    def n(self) -> int:
        return self.W.shape[0]


def extract() -> None:
    """Build data/visual_system.npz from data/processed/connectome.npz."""
    z = np.load(PROCESSED / "connectome.npz")
    shape = tuple(int(x) for x in z["shape"])
    W = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=shape)
    raw = sp.csr_matrix((z["raw_data"], z["raw_indices"], z["raw_indptr"]), shape=shape)
    pops = json.loads((PROCESSED / "populations.json").read_text())

    photo = np.asarray(pops["photoreceptor_R16"], np.int64)
    ol = np.asarray(pops["optic_lobe"], np.int64)
    keep = np.unique(np.concatenate([photo, ol]))
    local = -np.ones(shape[0], np.int64)
    local[keep] = np.arange(keep.size)
    sub = W[np.ix_(keep, keep)].tocsr()
    out = raw[photo].tocsr()

    VISUAL_SYSTEM.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        VISUAL_SYSTEM,
        W_data=sub.data,
        W_indices=sub.indices,
        W_indptr=sub.indptr,
        W_shape=np.array(sub.shape),
        photo=local[photo],
        optic_lobe=local[ol],
        out_data=out.data,
        out_indices=out.indices,
        out_indptr=out.indptr,
        out_shape=np.array(out.shape),
        hex1=z["hex1"],
        hex2=z["hex2"],
        whole_index=keep,
    )
    print(
        f"wrote {VISUAL_SYSTEM}: {keep.size:,} neurons ({photo.size:,} photoreceptors), "
        f"{sub.nnz:,} connections, {int(abs(sub).sum()):,} synapses"
    )


def load() -> VisualSystem:
    if not VISUAL_SYSTEM.exists():
        raise FileNotFoundError(
            f"{VISUAL_SYSTEM} is missing. Build it from the connectome:\n"
            "  python -m smartfly.connectome.download\n"
            "  python -m smartfly.connectome.build\n"
            "  python -m smartfly.connectome.visual"
        )
    z = np.load(VISUAL_SYSTEM)
    W = sp.csr_matrix((z["W_data"], z["W_indices"], z["W_indptr"]), shape=tuple(z["W_shape"]))
    out = sp.csr_matrix(
        (z["out_data"], z["out_indices"], z["out_indptr"]), shape=tuple(z["out_shape"])
    )
    return VisualSystem(
        W=W,
        photo=z["photo"],
        optic_lobe=z["optic_lobe"],
        photo_out=out,
        hex1=z["hex1"],
        hex2=z["hex2"],
        whole_index=z["whole_index"],
    )


if __name__ == "__main__":
    extract()
