"""Graded (non-spiking) neurons on the connectome's wiring.

Photoreceptors and most of the early fly optic lobe signal with continuous
voltage rather than action potentials. Simulating them as spiking neurons was
measured to destroy the image: a linear readout told "wall ahead" from the
photoreceptors 81% of the time, one synapse in 70%, deeper ~60% (50% = chance),
and more simulated time per frame did not help. Graded units, as used by
connectome-constrained models of the fly visual system (Lappalainen et al.,
Nature 2024), keep it.

Each neuron's output is its membrane value clipped to [0, 1]. Every neuron rests
at 0.5 so that inhibitory input -- which is what photoreceptors send -- can move
it; resting at 0 left the whole optic lobe silent.

The connectome gives synapse counts and a sign per transmitter, not
conductances, so two constants are chosen rather than measured: inhibitory
synapses count 4x (an inhibition-dominated regime, as in real circuits) and the
overall coupling is 0.5.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

INHIBITION = 4.0  # weight multiplier for inhibitory synapses
COUPLING = 0.5  # overall synaptic gain
REST = 0.5  # resting drive of every neuron
TAU = 3.0  # membrane time constant, in simulation steps


class GradedBrain:
    """Leaky graded units: v += (W^T r + rest + input - v) / tau, r = clip(v, 0, 1)."""

    def __init__(self, W: sp.csr_matrix):
        W = W.astype(np.float32).copy()
        W.data[W.data < 0] *= INHIBITION
        self.Wt = W.T.tocsr().astype(np.float32)  # post x pre, for one matvec per step
        self.n = W.shape[0]
        self.alpha = float(1.0 - np.exp(-1.0 / TAU))
        self.rest = np.full(self.n, REST, np.float32)
        self.reset()

    def reset(self) -> None:
        self.v = np.zeros(self.n, np.float32)
        self.rate = np.zeros(self.n, np.float32)

    def step(self, external: np.ndarray) -> np.ndarray:
        """Advance one step with the given external input; returns the rates."""
        drive = COUPLING * self.Wt.dot(self.rate) + self.rest + external
        self.v += self.alpha * (drive - self.v)
        np.clip(self.v, 0.0, 1.0, out=self.rate)
        return self.rate

    @property
    def active(self) -> np.ndarray:
        """Neurons strongly active right now (rate > 0.5), for display."""
        return self.rate > 0.5
