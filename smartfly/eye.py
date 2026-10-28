"""Where each photoreceptor looks on the screen.

The connectome gives every columnar optic-lobe cell an ommatidial coordinate
(hex1, hex2): which facet of the compound eye its column serves. Photoreceptors
carry no coordinate of their own, so each is placed where the cells it
synapses onto sit. Two things had to be right for the fly to see the screen:

* hex1 and hex2 are axes of a hexagonal lattice, 60 degrees apart. Treated as x
  and y they put the eye on a slanted band, and 40% of the screen -- most of the
  area ahead of and above Mario -- reached no photoreceptor at all.
  `HexToScreen` decorrelates the axes and spreads each along its distribution.
* Positions inferred from synapse targets cluster. Placing them directly still
  left holes, so `photoreceptor_pixels` tiles them evenly: every retina pixel
  gets 9-11 photoreceptors, and neighbours stay neighbours.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class HexToScreen:
    """Map ommatidial (hex1, hex2) coordinates to screen coordinates in [0, 1]^2.

    Fitted once on the optic-lobe columns: their principal axes become screen
    axes (the eye's long axis runs across), and each axis is spread by its own
    distribution so the eye's field fills the screen. Everything placed on the
    screen -- photoreceptors and the optic-lobe maps -- goes through the same
    transform, so they stay in register.
    """

    def __init__(self, hex1: np.ndarray, hex2: np.ndarray):
        P = np.stack([hex1, hex2], 1).astype(np.float64)
        self.mean = P.mean(0)
        _, _, vt = np.linalg.svd(P - self.mean, full_matrices=False)
        self.axes = vt  # long axis, then short axis
        Q = (P - self.mean) @ self.axes.T
        self.sorted = [np.sort(Q[:, k]) for k in (0, 1)]

    def __call__(self, hex1: np.ndarray, hex2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(across, down) in [0, 1]. Missing coordinates are treated as 0."""
        P = np.stack([np.nan_to_num(hex1), np.nan_to_num(hex2)], 1).astype(np.float64)
        Q = (P - self.mean) @ self.axes.T
        u, v = (
            np.interp(Q[:, k], self.sorted[k], np.linspace(0, 1, self.sorted[k].size))
            for k in (0, 1)
        )
        return u.astype(np.float32), v.astype(np.float32)


def photoreceptor_pixels(
    photo_out: sp.csr_matrix,
    hex1: np.ndarray,
    hex2: np.ndarray,
    hexmap: HexToScreen,
    rows: int,
    cols: int,
) -> np.ndarray:
    """The retina pixel (row * cols + col) each photoreceptor responds to.

    `photo_out` holds each photoreceptor's synapses; its position is the
    synapse-weighted mean coordinate of the targets that have one. Placed
    photoreceptors are then tiled evenly: sorted across the eye's long axis
    into `cols` strips, and each strip sorted top to bottom into `rows` cells.
    The few that cannot be placed are spread across the field.
    """
    have = np.isfinite(hex1) & np.isfinite(hex2)
    n = photo_out.shape[0]
    pos = np.full((n, 2), np.nan, np.float32)
    for i in range(n):
        lo, hi = photo_out.indptr[i], photo_out.indptr[i + 1]
        targets, weights = photo_out.indices[lo:hi], np.abs(photo_out.data[lo:hi])
        m = have[targets]
        if m.any():
            pos[i] = (
                np.average(hex1[targets[m]], weights=weights[m]),
                np.average(hex2[targets[m]], weights=weights[m]),
            )
    placed = np.isfinite(pos[:, 0])
    u, v = hexmap(pos[:, 0], pos[:, 1])
    idx = np.flatnonzero(placed)
    pixel = np.zeros(n, np.int64)
    for c, strip in enumerate(np.array_split(idx[np.argsort(u[idx], kind="stable")], cols)):
        for r, cell in enumerate(np.array_split(strip[np.argsort(v[strip], kind="stable")], rows)):
            pixel[cell] = r * cols + c
    pixel[~placed] = np.arange(int((~placed).sum())) % (rows * cols)
    return pixel
