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
        self._Wt = None   # lazily built transpose, for the dense path
        self.n = W.shape[0]
        self.rng = np.random.default_rng(self.cfg.seed)
        # Per-neuron scaling of incoming drive. Starts at 1 and is fitted by
        # `homeostatic_scaling`; this is the variable that can actually rescue a
        # neuron receiving too little input, which lowering a threshold cannot.
        self.in_gain = np.ones(self.n, dtype=np.float32)
        # Per-neuron resting drive, fitted by `homeostatic_bias`. This is the
        # intrinsic depolarisation a neuron sits at before synaptic input
        # arrives, and it is what makes a sign-inverting synapse work at all.
        self.bias = np.zeros(self.n, dtype=np.float32)
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
    # Fraction of neurons above which the transpose matvec beats row slicing.
    _DENSE_SWITCH = 0.02

    def synaptic_input(self, spikes: np.ndarray) -> np.ndarray:
        """Total synaptic current each neuron receives this step.

        Two strategies, because neither wins everywhere. Slicing the CSR rows of
        the active neurons is ideal when firing is sparse, but `W[active]`
        *builds a new sparse matrix* on every call, and at the 30-50% activity
        this network actually runs at that rebuild dominates everything -- it
        turned a calibration that should take seconds into eighteen minutes.
        Above a couple of percent active, one matvec against the precomputed
        transpose is far cheaper and runs entirely in scipy's C path.
        """
        active = np.flatnonzero(spikes)
        if active.size == 0:
            return np.zeros(self.n, dtype=np.float32)
        if active.size > self._DENSE_SWITCH * self.n:
            if self._Wt is None:
                self._Wt = self.W.T.tocsr()
            return self._Wt.dot(spikes.astype(np.float32))
        return np.asarray(self.W[active].sum(axis=0)).ravel()

    def step(self, ext_current: np.ndarray | None = None) -> np.ndarray:
        """Advance one timestep. Returns the boolean spike vector."""
        cfg = self.cfg
        I = self.synaptic_input(self.spikes) * cfg.gain * self.in_gain + self.bias
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


# ----------------------------------------------------------------------
# homeostatic calibration
# ----------------------------------------------------------------------
def homeostatic_bias(
    brain: "ConnectomeBrain",
    ext_frames,
    target_hz: float = 6.0,
    iters: int = 45,
    steps_per_frame: int = 8,
    lr: float = 0.12,
    lo: float = 0.0,
    hi: float = 2.5,
    seed: int = 0,
) -> dict:
    """Fit each neuron's own resting drive, replacing the uniform tonic pedestal.

    The fly's first visual synapse is sign-inverting: photoreceptors are
    histaminergic and *inhibitory*, and the lamina cells they contact are driven
    by the removal of that inhibition, not by its arrival. A neuron whose input
    is purely inhibitory can therefore never fire from that input alone -- with
    the tonic drive removed the optic lobe sat at exactly 0% active while the
    photoreceptors fired at 84%, and amplifying the incoming weights 4000x only
    deepened the inhibition. The real cell solves this with intrinsic
    depolarisation that the histamine opposes.

    So every neuron gets its own resting drive, fitted until it fires near
    `target_hz` on real stimuli. This is what the uniform tonic drive was
    standing in for, and the difference is the whole point: a single pedestal
    applied to all 166k neurons is uncorrelated with the stimulus and dominates
    each neuron's input variance, whereas a fitted per-neuron bias places each
    cell at the edge of its own threshold, so it is the stimulus that decides
    which ones cross.
    """
    n = brain.n
    target = max(target_hz, 1e-3)
    hist = []
    for it in range(iters):
        brain.reset()
        counts = np.zeros(n, dtype=np.int32)
        nsteps = 0
        for _ in range(6):
            ext = ext_frames()
            for _ in range(steps_per_frame):
                counts += brain.step(ext)
                nsteps += 1
        rate = counts / (nsteps * brain.cfg.dt / 1000.0)
        # Additive update: bias is a current, so it moves in current units
        # rather than multiplicatively (a bias of 0 cannot be scaled up).
        err = (target - rate) / target
        brain.bias = np.clip(brain.bias + lr * 0.05 * err, lo, hi).astype(np.float32)
        hist.append({"iter": it, "mean_hz": float(rate.mean()),
                     "median_hz": float(np.median(rate)),
                     "silent_frac": float((rate == 0).mean())})
    return {"history": hist, "final": hist[-1],
            "bias": {"min": float(brain.bias.min()),
                     "median": float(np.median(brain.bias)),
                     "max": float(brain.bias.max())}}


def homeostatic_scaling(
    brain: "ConnectomeBrain",
    ext_frames,
    target_hz: float = 6.0,
    iters: int = 40,
    steps_per_frame: int = 8,
    lr: float = 0.5,
    lo: float = 0.02,
    hi: float = 4000.0,
    seed: int = 0,
) -> dict:
    """Let each neuron scale its own incoming drive to hit a target rate.

    Threshold calibration alone cannot fix this network. With inputs normalised
    so that "all my inputs fired" equals 1, a deep neuron with three of three
    thousand inputs active receives about 0.001 of drive; no threshold low
    enough to spike on that is also high enough to be selective, and in practice
    the network simply stayed silent (99% of neurons, 0.5 Hz) however far the
    thresholds were pushed down. Excitability cannot manufacture drive that is
    not arriving.

    Synaptic scaling can. Each neuron multiplies its own incoming weights by a
    fitted factor, so a weakly-driven neuron amplifies what little it receives
    until its rate reaches target, and a saturated one turns itself down. The
    cascade then carries the stimulus through all five stages without an
    external pedestal, which is what the uniform tonic drive was standing in for.

    Like threshold calibration this is homeostatic plasticity, and synaptic
    scaling is the better-documented of the two: cortical neurons really do
    multiplicatively rescale their inputs to defend a target firing rate. It is
    fitted once against real stimulus statistics and then frozen.
    """
    n = brain.n
    target = max(target_hz, 1e-3)
    hist = []
    for it in range(iters):
        brain.reset()
        counts = np.zeros(n, dtype=np.int32)
        nsteps = 0
        for _ in range(6):
            ext = ext_frames()
            for _ in range(steps_per_frame):
                counts += brain.step(ext)
                nsteps += 1
        rate = counts / (nsteps * brain.cfg.dt / 1000.0)

        # Multiplicative, in log-rate. Silent neurons get a fixed upward push,
        # since log(0) gives no gradient to follow.
        err = np.log(target / (rate + 0.05))
        step = np.exp(lr * err)
        step[rate == 0] = np.exp(lr * 1.5)
        brain.in_gain = np.clip(brain.in_gain * step, lo, hi).astype(np.float32)

        hist.append({"iter": it, "mean_hz": float(rate.mean()),
                     "median_hz": float(np.median(rate)),
                     "silent_frac": float((rate == 0).mean())})
    return {"history": hist, "final": hist[-1],
            "in_gain": {"min": float(brain.in_gain.min()),
                        "median": float(np.median(brain.in_gain)),
                        "max": float(brain.in_gain.max())}}


def homeostatic_thresholds(
    brain: "ConnectomeBrain",
    ext_frames,
    target_hz: float = 6.0,
    iters: int = 30,
    steps_per_frame: int = 8,
    lr: float = 0.35,
    floor: float = 0.02,
    ceil: float = 60.0,
    seed: int = 0,
) -> dict:
    """Give every neuron its own threshold, fitted to the input it actually gets.

    A single global gain cannot keep a five-stage cascade in range. Whatever
    value is chosen, some stages saturate while others fall silent, and the only
    way to get the deep stages firing at all was a uniform tonic drive applied to
    all 166k neurons -- a pedestal uncorrelated with the stimulus, which then
    dominated each neuron's input variance and left the visual signal to be
    thresholded away. Measured: 59.7% of the task decodable at the retina,
    15.7% by the Kenyon cells.

    So instead of one gain plus a pedestal, each neuron gets an excitability
    fitted to its own input distribution: run the network on real stimuli,
    measure firing rates, and move each threshold multiplicatively until the
    rate sits near `target_hz`. Neurons that never fire become more excitable;
    runaway neurons become less. Every stage then operates inside its dynamic
    range on visual drive alone, with no pedestal to swamp it.

    This is intrinsic homeostatic plasticity, which real neurons do -- it is the
    mechanism that keeps firing rates stable as synaptic drive changes. It is
    fitted once, offline, and then frozen.

    Parameters
    ----------
    ext_frames : callable
        Returns a fresh external-current vector each call: the visual drive for
        one frame. Calibrating against real stimulus statistics rather than
        noise is the point -- thresholds fitted to the wrong input distribution
        are no better than a global gain.
    """
    rng = np.random.default_rng(seed)
    n = brain.n
    target = max(target_hz, 1e-3)
    hist = []

    for it in range(iters):
        brain.reset()
        counts = np.zeros(n, dtype=np.int32)
        nsteps = 0
        for _ in range(6):                      # a few frames per iteration
            ext = ext_frames()
            for _ in range(steps_per_frame):
                counts += brain.step(ext)
                nsteps += 1
        rate = counts / (nsteps * brain.cfg.dt / 1000.0)

        # Multiplicative update in log-rate: a neuron firing twice as fast as
        # target has its threshold raised by the same factor one firing half as
        # fast has it lowered. Silent neurons are pulled down by a fixed step,
        # since log(0) carries no gradient.
        err = np.log((rate + 0.05) / target)
        step = np.exp(lr * err)
        step[rate == 0] = np.exp(-lr * 1.2)
        brain.v_th = np.clip(brain.v_th * step, floor, ceil).astype(np.float32)

        hist.append({
            "iter": it,
            "mean_hz": float(rate.mean()),
            "median_hz": float(np.median(rate)),
            "silent_frac": float((rate == 0).mean()),
            "active_frac": float((rate > 0).mean()),
        })

    return {"history": hist, "final": hist[-1],
            "v_th": {"min": float(brain.v_th.min()), "max": float(brain.v_th.max()),
                     "median": float(np.median(brain.v_th))}}
