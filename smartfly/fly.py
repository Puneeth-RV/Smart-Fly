"""The fly: eye, visual system, and the features a policy reads from it.

Each decision, `see(retina)` drives the photoreceptors with the retina image,
runs the visual system for a few steps, and returns one feature vector:

    deep    320   optic-lobe activity, averaged on a 20 x 16 map of the visual field
    motion  320   change in `deep` since the previous decision
    photo    24   photoreceptor activity in 24 groups
    first   320   the first layer behind the photoreceptors, on the same map
    eye     360   photoreceptor activity laid out as the 15 x 24 retina

1,344 numbers in all. The order and grouping are what the trained policies were
trained on, so they are kept exactly -- including two that turned out to carry
little: `photo` groups photoreceptors by index rather than position, and
`first` is silent, because photoreceptor output is inhibitory and pins the
cells directly behind them at zero.
"""

from __future__ import annotations

import numpy as np

from smartfly.brain import GradedBrain
from smartfly.connectome.visual import VisualSystem
from smartfly.connectome.visual import load as load_visual_system
from smartfly.eye import HexToScreen, photoreceptor_pixels
from smartfly.game import RETINA

MAP = (20, 16)  # columns, rows of the optic-lobe maps
PHOTO_GROUPS = 24


def _pool(index: np.ndarray, cell: np.ndarray, n_cells: int):
    """Precompute a mean-by-cell: values[index] averaged into `cell` bins."""
    count = np.maximum(np.bincount(cell, minlength=n_cells), 1).astype(np.float32)

    def mean(values: np.ndarray) -> np.ndarray:
        return (np.bincount(cell, weights=values[index], minlength=n_cells) / count).astype(
            np.float32
        )

    return mean


class Fly:
    """A fruit fly's eye and optic lobe, ready to look at Mario.

    `steps` is how many simulation steps the visual system runs per decision.
    """

    def __init__(self, steps: int = 6, vs: VisualSystem | None = None):
        vs = vs or load_visual_system()
        self.steps = steps
        self.brain = GradedBrain(vs.W)
        self.n = vs.n
        self.photo = vs.photo

        # Screen geometry, fitted on the optic-lobe columns.
        ol_whole = vs.whole_index[vs.optic_lobe]
        has_hex = np.isfinite(vs.hex1) & np.isfinite(vs.hex2)
        columns = np.zeros(vs.hex1.size, bool)
        columns[ol_whole] = True
        columns &= has_hex
        self.hexmap = HexToScreen(vs.hex1[columns], vs.hex2[columns])

        # The eye: which retina pixel drives each photoreceptor.
        rows, cols = RETINA
        self.pixel = photoreceptor_pixels(vs.photo_out, vs.hex1, vs.hex2, self.hexmap, rows, cols)
        self.n_pixels = rows * cols

        # Optic-lobe maps: each optic-lobe neuron belongs to one map cell.
        gx, gy = MAP
        u, v = self.hexmap(vs.hex1[ol_whole], vs.hex2[ol_whole])
        cx = np.clip((np.nan_to_num(u, nan=0.5) * (gx - 1)).round(), 0, gx - 1).astype(int)
        cy = np.clip((np.nan_to_num(v, nan=0.5) * (gy - 1)).round(), 0, gy - 1).astype(int)
        cell = cy * gx + cx
        order = np.argsort(cell, kind="stable")
        self.map_neurons = vs.optic_lobe[order]  # optic-lobe neurons, by map cell
        self.map_cell = cell[order]
        self.n_cells = gx * gy

        # The first layer: optic-lobe neurons a photoreceptor synapses onto.
        contacted = np.flatnonzero(np.asarray(abs(vs.photo_out).sum(axis=0)).ravel() > 0)
        cell_of = -np.ones(self.n, np.int64)
        cell_of[vs.optic_lobe] = cell
        first = np.searchsorted(vs.whole_index, np.intersect1d(contacted, ol_whole))
        first = first[cell_of[first] >= 0]
        self.first_neurons = first

        groups = np.repeat(
            np.arange(PHOTO_GROUPS), [g.size for g in np.array_split(self.photo, PHOTO_GROUPS)]
        )
        self._deep = _pool(self.map_neurons, self.map_cell, self.n_cells)
        self._photo = _pool(self.photo, groups, PHOTO_GROUPS)
        self._first = _pool(first, cell_of[first], self.n_cells)
        self._eye = _pool(self.photo, self.pixel, self.n_pixels)

        self.n_features = 3 * self.n_cells + PHOTO_GROUPS + self.n_pixels
        self.reset()

    def reset(self) -> None:
        """Start of an attempt: a resting brain, and no previous frame."""
        self.brain.reset()
        self._prev_px: np.ndarray | None = None
        self._prev_deep = np.zeros(self.n_cells, np.float32)

    def drive(self, retina: np.ndarray) -> np.ndarray:
        """Input current to every neuron: photoreceptors respond to light and change."""
        px = retina.ravel()[self.pixel]
        prev = self._prev_px if self._prev_px is not None else px
        self._prev_px = px.copy()
        current = np.zeros(self.n, np.float32)
        current[self.photo] = 0.2 + 0.8 * px + 1.5 * (px - prev)
        return current

    def see(self, retina: np.ndarray) -> np.ndarray:
        """Show the fly one retina image; returns its feature vector."""
        current = self.drive(retina)
        for _ in range(self.steps):
            self.brain.step(current)
        return self.features()

    def features(self) -> np.ndarray:
        r = self.brain.rate
        deep = self._deep(r)
        motion = deep - self._prev_deep
        self._prev_deep = deep
        return np.concatenate([deep, motion, self._photo(r), self._first(r), self._eye(r)])

    # Slices of the feature vector, for display and analysis.
    @property
    def layout(self) -> dict[str, slice]:
        c, p = self.n_cells, PHOTO_GROUPS
        return {
            "deep": slice(0, c),
            "motion": slice(c, 2 * c),
            "photo": slice(2 * c, 2 * c + p),
            "first": slice(2 * c + p, 3 * c + p),
            "eye": slice(3 * c + p, self.n_features),
        }
