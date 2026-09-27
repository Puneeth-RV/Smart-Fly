"""Leaky integrate-and-fire simulation over the MaleCNS connectivity graph.

Implementation note on performance
----------------------------------
The obvious formulation is a sparse matrix-vector product W^T @ spikes every
timestep, which costs O(E) with E ~ 4e7 regardless of how many neurons actually
fired. Measured on this machine: 6.9 ms/step on CPU, 13.1 ms/step on MPS (which
cannot do sparse CSR at all and has to fall back to a scatter-add over every
edge).

But spikes are sparse -- a few percent of neurons per step, as in the real
animal -- and a silent neuron contributes nothing. So connectivity is stored
presynaptically (CSR row i = the outgoing edges of neuron i) and each step sums
only the rows of neurons that actually fired. Cost becomes proportional to
activity rather than to network size, which is what makes millions of training
steps feasible on a laptop.

Dynamics are deliberately plain: current-based LIF, no conductance dynamics, no
dendritic structure, no spike-timing detail. The connectome constrains *who*
talks to *whom* and with what sign and strength; it says nothing about
biophysics, so inventing detail there would be false precision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"


@dataclass
class LIFConfig:
    """Parameters of the neuron model and of the global synaptic scaling.

    `gain` is the one genuinely free parameter that matters. The connectome
    gives synapse *counts*, not conductances, so an overall scale factor has to
    be chosen. It is calibrated by `calibrate_gain` to put the network in a
    sparse, non-saturating regime; too low and the brain is silent, too high and
    it seizes into synchronous population bursts. Neither extreme can compute.
    """

    tau: float = 20.0  # membrane time constant, ms
    dt: float = 1.0  # timestep, ms
    v_rest: float = 0.0
    v_threshold: float = 1.0
    v_reset: float = 0.0
    refractory: int = 2  # steps of enforced silence after a spike
    gain: float = 0.01  # global synaptic scale applied to normalised input
    # Inhibitory weights are scaled separately. A network whose excitation and
    # inhibition are not balanced has only two behaviours -- silence or
    # saturation -- and neither computes. Values above 1 make the network
    # inhibition-dominated, which is the regime real brains sit in and the only
    # one with a usable dynamic range.
    g_inh: float = 4.0
    # Per-step membrane noise. Measured to actively destroy the visual signal:
    # at 0.02 it is comparable in size to the retinal drive surviving input
    # normalisation, and switching it off raised the linear decodability of the
    # correct action from Kenyon cells from 16.5% to 28.2%. Off by default.
    noise: float = 0.0
    # Spread of spike thresholds across neurons, as a fraction of v_threshold.
    # With identical thresholds the network is a switch rather than a circuit:
    # a uniform tonic drive of 0.05 integrates to 1.02 against a threshold of
    # 1.00, so every neuron crosses at once (98.5% active, no dynamic range) and
    # a hair less leaves the whole network silent (2%). There is no regime in
    # between, and a saturated network cannot carry information. Real neurons
    # differ substantially in excitability; giving them a distribution of
    # thresholds spreads the population across its operating range so that a
    # visual input changes *which* neurons fire instead of toggling all of them.
    threshold_spread: float = 0.35
    seed: int | None = None

    @property
    def decay(self) -> float:
        return float(np.exp(-self.dt / self.tau))


class ConnectomeBrain:
    """A spiking network whose wiring is the fly connectome.

    Parameters
    ----------
    W_pre : csr_matrix
        Signed connectivity, row = presynaptic neuron, col = postsynaptic.
        Values are signed synapse counts.
    cfg : LIFConfig
    """

    def __init__(
        self,
        W_pre: sp.csr_matrix,
        cfg: LIFConfig | None = None,
        normalize: bool = True,
    ) -> None:
        self.cfg = cfg or LIFConfig()
        if not sp.isspmatrix_csr(W_pre):
            W_pre = W_pre.tocsr()
        W = W_pre.astype(np.float32).copy()

        # Apply the inhibitory rescaling before normalisation so that balance is
        # a property of the weights themselves.
        if self.cfg.g_inh != 1.0:
            neg = W.data < 0
            W.data[neg] *= self.cfg.g_inh

        if normalize:
            # Divide each neuron's incoming weights by the total absolute drive
            # it can receive. Synapse counts span four orders of magnitude and
            # carry no units, so without this the spike threshold is arbitrary:
            # a neuron with 10,000 incoming synapses and one with 10 cannot
            # share a threshold. After normalisation, "all my inputs fired at
            # once" is an input of ~1 for every neuron, which makes the
            # threshold interpretable and the dynamics stable.
            incoming = np.asarray(abs(W).sum(axis=0)).ravel()
            incoming[incoming == 0] = 1.0
            W = W @ sp.diags(1.0 / incoming, dtype=np.float32)
            W = W.tocsr()

        self.W = W
        self.n = W.shape[0]
        self.rng = np.random.default_rng(self.cfg.seed)
        # Heterogeneous excitability (see LIFConfig.threshold_spread).
        if self.cfg.threshold_spread > 0:
            th = self.cfg.v_threshold * (
                1.0 + self.cfg.threshold_spread
                * np.random.default_rng(
                    0 if self.cfg.seed is None else self.cfg.seed + 7
                ).standard_normal(self.n)
            )
            self.v_th = np.maximum(th, 0.15 * self.cfg.v_threshold).astype(np.float32)
        else:
            self.v_th = np.full(self.n, self.cfg.v_threshold, dtype=np.float32)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.v = np.full(self.n, self.cfg.v_rest, dtype=np.float32)
        self.refrac = np.zeros(self.n, dtype=np.int16)
        self.spikes = np.zeros(self.n, dtype=bool)
        self.t = 0
        # Exponentially-decayed spike trace, used as the state a readout or a
        # plasticity rule can see. Instantaneous spikes are too sparse in time.
        self.trace = np.zeros(self.n, dtype=np.float32)

    # ------------------------------------------------------------------
    def synaptic_input(self, spikes: np.ndarray) -> np.ndarray:
        """Sum the outgoing weights of the neurons that fired.

        Only the CSR rows of active neurons are touched, so the cost tracks the
        firing rate rather than the edge count.
        """
        active = np.flatnonzero(spikes)
        if active.size == 0:
            return np.zeros(self.n, dtype=np.float32)
        # Slicing rows of a CSR matrix is cheap; the sum over those rows is the
        # total current each postsynaptic neuron receives.
        return np.asarray(self.W[active].sum(axis=0)).ravel()

    def step(self, ext_current: np.ndarray | None = None) -> np.ndarray:
        """Advance one timestep. Returns the boolean spike vector."""
        cfg = self.cfg
        I = self.synaptic_input(self.spikes) * cfg.gain
        if ext_current is not None:
            I = I + ext_current
        if cfg.noise > 0:
            I = I + self.rng.normal(0.0, cfg.noise, self.n).astype(np.float32)

        # Leaky integration, with refractory neurons held at reset.
        self.v = cfg.v_rest + (self.v - cfg.v_rest) * cfg.decay + I
        free = self.refrac <= 0
        self.v[~free] = cfg.v_reset

        self.spikes = free & (self.v >= self.v_th)
        self.v[self.spikes] = cfg.v_reset
        self.refrac = np.maximum(self.refrac - 1, 0)
        self.refrac[self.spikes] = cfg.refractory

        self.trace *= 0.8
        self.trace[self.spikes] += 1.0
        self.t += 1
        return self.spikes

    # ------------------------------------------------------------------
    def run(self, steps: int, ext_current: np.ndarray | None = None) -> np.ndarray:
        """Run for `steps` and return per-neuron spike counts."""
        counts = np.zeros(self.n, dtype=np.int32)
        for _ in range(steps):
            counts += self.step(ext_current)
        return counts

    def firing_rate_hz(self, counts: np.ndarray, steps: int) -> np.ndarray:
        return counts / (steps * self.cfg.dt / 1000.0)


# ----------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------
def load_connectome(with_raw: bool = False):
    """Load the processed connectome as presynaptic-major CSR matrices.

    Returns (W_signed, pops, meta), or (W_signed, W_raw, pops, meta) when
    `with_raw` is set. W_signed drives the dynamics and has the aminergic
    neurons zeroed; W_raw preserves every edge's synapse count regardless of
    transmitter, and is what anatomical queries such as the dopaminergic
    compartment structure must use.
    """
    z = np.load(PROC / "connectome.npz")
    shape = tuple(int(x) for x in z["shape"])
    W = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=shape)
    pops = json.loads((PROC / "populations.json").read_text())
    meta = json.loads((PROC / "meta.json").read_text())
    meta["hex1"] = z["hex1"] if "hex1" in z.files else None
    meta["hex2"] = z["hex2"] if "hex2" in z.files else None
    if not with_raw:
        return W, pops, meta
    raw = sp.csr_matrix((z["raw_data"], z["raw_indices"], z["raw_indptr"]), shape=shape)
    return W, raw, pops, meta


def sensorimotor_subgraph(
    W: sp.csr_matrix,
    sources: np.ndarray,
    sinks: np.ndarray,
    max_hops: int = 6,
) -> np.ndarray:
    """Neurons that can carry a signal from `sources` to `sinks` within max_hops.

    Defined as the intersection of what is reachable forward from the sensory
    populations and what can reach the motor populations backwards. This is a
    principled reduction rather than a hand-picked circuit: it keeps exactly the
    neurons that could causally participate in the eye-to-muscle path at this
    path length, and drops the rest.
    """
    A = (W != 0).astype(np.int8)
    At = A.T.tocsr()

    def expand(seed: np.ndarray, M: sp.csr_matrix) -> np.ndarray:
        reached = np.zeros(W.shape[0], dtype=bool)
        reached[seed] = True
        frontier = reached.copy()
        for _ in range(max_hops):
            nxt = (M.T @ frontier.astype(np.int8)) > 0
            new = nxt & ~reached
            if not new.any():
                break
            reached |= new
            frontier = new
        return reached

    fwd = expand(sources, A)
    bwd = expand(sinks, At)
    keep = fwd & bwd
    keep[sources] = True
    keep[sinks] = True
    return np.flatnonzero(keep)


def gain_sweep(
    W: sp.csr_matrix,
    gains: np.ndarray | None = None,
    steps: int = 200,
    drive: float = 0.05,
    g_inh: float = 4.0,
    seed: int = 0,
    normalize: bool = True,
) -> list[dict]:
    """Measure network activity across a log-spaced range of synaptic gains.

    Returned per gain: mean rate, the fraction of neurons that fired at all, and
    a synchrony index. The last one matters -- a network can show a plausible
    mean rate while actually bursting in lockstep, which looks healthy in the
    average and is useless for computation. Synchrony is the coefficient of
    variation of the population spike count over time; near 0 is asynchronous,
    above ~1 means population-wide bursting.
    """
    if gains is None:
        gains = np.logspace(-3, 1.0, 14)
    n = W.shape[0]
    out = []
    for g in gains:
        cfg = LIFConfig(gain=float(g), g_inh=g_inh, seed=seed)
        b = ConnectomeBrain(W, cfg, normalize=normalize)
        ext = np.full(n, drive, dtype=np.float32) if drive else None
        b.spikes = np.random.default_rng(seed).random(n) < 0.01
        per_step = np.empty(steps, dtype=np.int64)
        counts = np.zeros(n, dtype=np.int32)
        for i in range(steps):
            s = b.step(ext)
            per_step[i] = s.sum()
            counts += s
        rate = float(b.firing_rate_hz(counts, steps).mean())
        # Ignore the first 20% as transient.
        tail = per_step[steps // 5 :]
        sync = float(tail.std() / (tail.mean() + 1e-9))
        out.append({
            "gain": float(g),
            "mean_hz": rate,
            "active_frac": float((counts > 0).mean()),
            "synchrony": sync,
        })
    return out


def calibrate_gain(
    W: sp.csr_matrix,
    target_hz: float = 5.0,
    max_synchrony: float = 0.6,
    **kw,
) -> tuple[float, dict]:
    """Pick the gain giving a rate nearest `target_hz` in an asynchronous regime.

    Connectome-derived networks have no intrinsic scale -- synapse counts are
    not conductances -- so choosing this scale is a required modelling step, not
    a fudge. Candidates that are synchronously bursting are rejected outright
    even if their mean rate looks right.
    """
    sweep = gain_sweep(W, **kw)
    ok = [s for s in sweep if s["synchrony"] <= max_synchrony and s["mean_hz"] > 0.1]
    pool = ok or sweep
    best = min(pool, key=lambda s: abs(np.log(max(s["mean_hz"], 1e-3)) - np.log(target_hz)))
    return best["gain"], {"chosen": best, "sweep": sweep, "asynchronous_candidates": len(ok)}


if __name__ == "__main__":
    import time

    # Synthetic graph with connectome-like statistics, so the machinery can be
    # exercised before the real weights finish downloading.
    print("synthetic benchmark (connectome-scale)")
    n, deg = 160_000, 250
    rng = np.random.default_rng(0)
    rows = np.repeat(np.arange(n), deg)
    cols = rng.integers(0, n, n * deg)
    vals = (rng.random(n * deg) * 10).astype(np.float32)
    vals[rng.random(n * deg) < 0.3] *= -1  # ~30% inhibitory
    W = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
    print(f"  n={n:,} nnz={W.nnz:,}")

    g, info = calibrate_gain(W, target_hz=5.0, steps=120)
    print(f"  {'gain':>9} {'mean_hz':>9} {'active':>8} {'sync':>7}")
    for s in info["sweep"]:
        mark = " <-" if s["gain"] == g else ""
        print(f"  {s['gain']:9.3g} {s['mean_hz']:9.2f} {s['active_frac']:8.2%} "
              f"{s['synchrony']:7.2f}{mark}")
    print(f"  chosen gain={g:.3g}  ({info['asynchronous_candidates']} asynchronous candidates)")

    b = ConnectomeBrain(W, LIFConfig(gain=g, seed=1))
    b.spikes = rng.random(n) < 0.01
    b.step()
    t0 = time.perf_counter()
    nsteps = 200
    counts = b.run(nsteps, np.full(n, 0.05, dtype=np.float32))
    dt = time.perf_counter() - t0
    r = b.firing_rate_hz(counts, nsteps)
    print(f"\n  {nsteps} steps in {dt:.2f}s -> {nsteps/dt:.0f} brain-steps/s")
    print(f"  mean rate {r.mean():.2f} Hz | active neurons {(counts>0).sum():,}/{n:,}")
