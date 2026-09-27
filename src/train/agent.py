"""Couple the connectome to the game, and let the mushroom body learn.

Interface
---------
Vision in: each R1-R6 photoreceptor is assigned a retina pixel (tiling the
15x24 field), and injects current proportional to that pixel's intensity.
R1-R6 is the fly's motion/luminance channel, which is the right one for a
runner task; the chromatic R7/R8 cells are left undriven.

Motor out: descending neurons are partitioned into five fixed pools, one per
action. The partition is random but seeded and never learned -- it is the
"body", the arbitrary-but-fixed wiring from brain to muscle that any animal is
born with. Learning happens inside the brain, not in this mapping. Action is the
argmax of pooled descending-neuron activity.

Learning
--------
The connectome has no synaptic weights to train, so plasticity has to be added.
It is added only where the fly actually has it: at Kenyon-cell -> MBON synapses,
gated by dopamine. The rule is three-factor,

    dw_ij = -lr * eligibility_ij * dopamine_j

where eligibility is a decaying trace of KC(i) activity coincident with MBON(j)
being read out, and dopamine_j is the reinforcement delivered to MBON j's
compartment. Depression is the sign of the real effect: coincidence of Kenyon
cell activity with dopamine release depresses the KC->MBON synapse, which is why
the rule is negative.

Critically, which DANs innervate which MBON is not assumed -- it is read out of
the connectome's own DAN->MBON connectivity, so the compartment structure is the
fly's, not ours. PAM cells carry reward, PPL1 cells carry punishment.

A trained linear readout on the same network is provided as a control. The
comparison is the actual result worth reporting: does the fly's own restricted,
biologically-sited learning rule beat a generic learned decoder on identical
wiring?
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from brain.lif import ConnectomeBrain, LIFConfig  # noqa: E402
from gym.subway import N_ACTIONS, RETINA_H, RETINA_W, SubwayEnv  # noqa: E402


@dataclass
class AgentConfig:
    # Eight steps per frame: the retina-to-motor path is five synaptic stages
    # deep, so fewer steps than that and the visual signal cannot physically
    # reach the descending neurons before the frame is over.
    brain_steps_per_frame: int = 8
    input_gain: float = 0.8  # current injected per unit retina contrast
    # Fly photoreceptors are histaminergic and *inhibitory*, and light
    # hyperpolarises them: brightness reduces transmitter release, so downstream
    # cells are driven by disinhibition. The retina therefore has to be inverted
    # on its way in -- a dark pixel excites its photoreceptor, a bright one
    # silences it. Driving them the intuitive way round inhibits the entire
    # brain and nothing downstream ever fires.
    photoreceptor_baseline: float = 1.0
    # Spontaneous background drive. Only ~2.8k of 89k optic lobe neurons receive
    # direct photoreceptor input, so with per-neuron input normalisation a
    # retinal signal alone cannot reach threshold. Real brains are not silent at
    # rest; this tonic term puts the network near threshold so signals propagate.
    # Calibrated empirically. The usable window is narrow: at 0.03 the network
    # falls silent, at 0.08 it saturates and the motor representations of
    # different scenes collapse together (cos 0.04 -> 0.30). 0.05 is sparse
    # enough to discriminate and dense enough to propagate.
    tonic_drive: float = 0.05
    # Balance between sustained and transient (motion) drive.
    #
    # The optic lobe's core computation is elementary motion detection: T4 and
    # T5 correlate luminance changes between neighbouring columns over time.
    # Feeding it static per-frame intensity presents a signal that circuitry is
    # not wired to process, and measurement bore that out -- the correct action
    # was 61.8% decodable from the retina and at chance everywhere downstream.
    #
    # Fly photoreceptors are also strongly high-pass in their own right: they
    # adapt within milliseconds and encode change far more than steady level.
    # So the drive is split into a weak sustained term and a dominant transient
    # one computed from successive frames, which is what the downstream
    # circuitry is built to correlate.
    sustained_weight: float = 0.25
    motion_gain: float = 3.0
    lr: float = 0.02  # KC->MBON learning rate
    eligibility_decay: float = 0.85
    dopamine_decay: float = 0.7
    reward_scale: float = 0.5
    weight_floor: float = 0.0  # KC->MBON weights are non-negative (cholinergic)
    weight_ceiling: float = 2.0
    explore: float = 0.15  # epsilon, annealed during training
    # Competitive normalisation of the motor drive. Each action pool has its own
    # constant offset, inherited from how strongly the anatomy happens to couple
    # MBONs to that pool -- measured, one action won 28 of 39 frames while
    # another was never chosen at all, independent of what was on screen. The
    # agent must respond to *changes* in drive, not to fixed anatomical bias, so
    # each action's drive is measured against its own slow baseline. Real
    # premotor circuits do something equivalent through lateral inhibition and
    # adaptation; without it the policy is degenerate whatever the synapses learn.
    drive_baseline_rate: float = 0.02
    seed: int = 0


class FlyAgent:
    """A connectome brain wired to the runner, with a plastic mushroom body."""

    def __init__(
        self,
        W: sp.csr_matrix,
        raw: sp.csr_matrix,
        pops: dict[str, list[int]],
        cfg: AgentConfig | None = None,
        lif: LIFConfig | None = None,
        meta: dict | None = None,
    ) -> None:
        self.cfg = cfg or AgentConfig()
        self.pops = pops
        self.rng = np.random.default_rng(self.cfg.seed)
        self.brain = ConnectomeBrain(W, lif or LIFConfig(seed=self.cfg.seed))
        self.n = self.brain.n

        self.photo = np.asarray(pops["photoreceptor_R16"], dtype=np.int64)
        self.kc = np.asarray(pops["kenyon_cell"], dtype=np.int64)
        self.mbon = np.asarray(pops["mbon"], dtype=np.int64)
        self.pam = np.asarray(pops["dan_pam"], dtype=np.int64)
        self.ppl1 = np.asarray(pops["dan_ppl1"], dtype=np.int64)
        self.dn = np.asarray(pops["descending"], dtype=np.int64)

        self._build_retina_map(raw, meta)
        self._build_action_pools(raw)
        # Anatomy from the unsigned matrix: the signed one zeroes the aminergic
        # neurons, so the dopaminergic compartment map is only visible in raw.
        self._build_mb_plasticity(raw)

        self.reset_state()

    # ------------------------------------------------------------------
    # interfaces
    # ------------------------------------------------------------------
    def _build_retina_map(self, raw: sp.csr_matrix, meta: dict | None = None) -> None:
        """Map retina pixels onto photoreceptors *retinotopically*.

        The first version of this assigned photoreceptor i to pixel i % n_pixels,
        which is a permutation of the visual field. That quietly destroys the
        spatial structure the optic lobe is built to process: neighbouring
        ommatidia project into neighbouring columns, and the columnar circuitry
        assumes that adjacency. Scrambling it leaves the correct action linearly
        decodable from the retina at 61.8% and at chance everywhere downstream --
        the brain cannot compute on input whose geometry has been shuffled.

        Photoreceptors carry no ommatidial coordinates in the release, but the
        columnar cells they drive (L1, L2, Mi1, Tm1 and friends, 23,720 cells
        over 892 columns) do. So each photoreceptor's retinal position is taken
        as the mean hex coordinate of the columns it actually projects into, and
        the retina is then sampled at those positions. The layout comes from the
        connectome rather than from any assumption of ours.
        """
        n_px = RETINA_H * RETINA_W
        n = self.brain.n
        h1 = h2 = None
        if meta is not None:
            h1, h2 = meta.get("hex1"), meta.get("hex2")

        pos = np.full((self.photo.size, 2), np.nan, dtype=np.float32)
        if h1 is not None and h2 is not None:
            h1 = np.asarray(h1, dtype=np.float32)
            h2 = np.asarray(h2, dtype=np.float32)
            have = np.isfinite(h1) & np.isfinite(h2)
            Wp = raw[self.photo]  # outgoing edges of each photoreceptor
            for i in range(self.photo.size):
                lo, hi = Wp.indptr[i], Wp.indptr[i + 1]
                tgt = Wp.indices[lo:hi]
                wt = Wp.data[lo:hi]
                m = have[tgt]
                if m.any():
                    w = wt[m]
                    pos[i, 0] = np.average(h1[tgt[m]], weights=w)
                    pos[i, 1] = np.average(h2[tgt[m]], weights=w)
        placed = int(np.isfinite(pos[:, 0]).sum())

        if placed >= 0.5 * self.photo.size:
            # Normalise the recovered eye coordinates onto the retina grid.
            good = np.isfinite(pos[:, 0])
            p = pos.copy()
            for k in (0, 1):
                col = p[good, k]
                lo, hi = np.percentile(col, 1), np.percentile(col, 99)
                p[:, k] = np.clip((p[:, k] - lo) / max(hi - lo, 1e-6), 0, 1)
            col_ix = np.clip(np.nan_to_num(p[:, 0]) * (RETINA_W - 1), 0, RETINA_W - 1)
            row_ix = np.clip(np.nan_to_num(p[:, 1]) * (RETINA_H - 1), 0, RETINA_H - 1)
            pix = (row_ix.round() * RETINA_W + col_ix.round()).astype(np.int64)
            # Photoreceptors whose position could not be recovered are spread
            # over the field rather than all landing on one pixel.
            bad = ~good
            pix[bad] = np.arange(int(bad.sum())) % n_px
            self.photo_pixel = pix
            self.retinotopy = "connectome-derived"
        else:
            self.photo_pixel = np.arange(self.photo.size) % n_px
            self.retinotopy = "scrambled (no hex coordinates available)"
        self.n_photo_placed = placed
        self._ext = np.zeros(n, dtype=np.float32)

    def _build_action_pools(self, raw: sp.csr_matrix) -> None:
        """Group descending neurons into action pools by their MBON input signature.

        A random partition looks harmless and is fatal. Averaged over ~260
        randomly chosen descending neurons, every MBON ends up with nearly the
        same influence on every pool -- measured at 0.906 mean cosine similarity
        between action columns -- so the mushroom body physically cannot prefer
        one action over another no matter what its synapses do. Learning then has
        nothing to act through, which is precisely what the first training run
        showed: plasticity happening, behaviour unchanged.

        So pools are formed by clustering descending neurons on the profile of
        mushroom-body input they receive. Neurons that listen to the same MBONs
        end up driving the same action, which gives each action a distinct
        signature for the MB to modulate. This also matches how descending
        neurons are actually organised: functional classes driving different
        behaviours receive different central input, rather than being
        interchangeable.
        """
        from scipy.cluster.vq import kmeans2

        A = (raw != 0).astype(np.float32)
        direct = np.asarray(np.abs(raw[np.ix_(self.mbon, self.dn)]).todense(), np.float32)
        # Two-hop as well, since only ~106 of 1314 descending neurons receive
        # direct MBON contact; without it most of the population is unassignable.
        two_hop = (A[self.mbon] @ A)[:, self.dn]
        infl = direct + 0.25 * np.asarray(two_hop.todense(), np.float32)

        prof = infl.T  # (n_dn, n_mbon)
        norms = np.linalg.norm(prof, axis=1, keepdims=True)
        prof = prof / np.maximum(norms, 1e-9)

        has_input = (norms.ravel() > 0)
        labels = np.zeros(self.dn.size, dtype=int)
        if has_input.sum() >= N_ACTIONS:
            _, lab = kmeans2(prof[has_input], N_ACTIONS, minit="++", seed=self.cfg.seed)
            labels[has_input] = lab
        # Neurons with no mushroom-body input carry no signature to cluster on;
        # spread them evenly so pool sizes stay comparable.
        idle = np.flatnonzero(~has_input)
        labels[idle] = np.arange(idle.size) % N_ACTIONS

        self.action_pools = [self.dn[labels == a] for a in range(N_ACTIONS)]
        self.pool_sizes = [int(p.size) for p in self.action_pools]

    def _build_mb_plasticity(self, W: sp.csr_matrix) -> None:
        """Set up the plastic KC->MBON weight block and the DAN compartment map.

        The plastic block is initialised from the real KC->MBON connectivity, so
        only synapses the fly actually has can change. Compartment assignment --
        which dopaminergic cells modulate which MBON -- is read from the
        connectome's DAN->MBON edges.
        """
        # Plastic block, KC x MBON, initialised from real synapse counts.
        block = W[np.ix_(self.kc, self.mbon)]
        block = np.asarray(block.todense(), dtype=np.float32)
        self.kc_mbon_mask = (block != 0).astype(np.float32)
        raw = np.abs(block)
        scale = raw.max() if raw.max() > 0 else 1.0
        self.kc_mbon_w = (raw / scale).astype(np.float32)
        self.kc_mbon_w0 = self.kc_mbon_w.copy()

        # Dopaminergic compartments, straight from the connectome.
        pam_block = np.asarray(np.abs(W[np.ix_(self.pam, self.mbon)]).todense(), np.float32)
        ppl1_block = np.asarray(np.abs(W[np.ix_(self.ppl1, self.mbon)]).todense(), np.float32)
        self.pam_to_mbon = pam_block.sum(axis=0)
        self.ppl1_to_mbon = ppl1_block.sum(axis=0)
        # Normalise so reward and punishment have comparable reach.
        for arr in (self.pam_to_mbon, self.ppl1_to_mbon):
            m = arr.max()
            if m > 0:
                arr /= m

        # MBON -> action coupling, via whatever path the connectome provides.
        # Rather than assume MBONs reach descending neurons directly (they
        # mostly do not), the influence of each MBON on each action pool is
        # measured as its two-hop connection strength into that pool.
        A = (W != 0).astype(np.float32)
        two_hop = (A[self.mbon] @ A).tocsr()
        infl = np.zeros((self.mbon.size, N_ACTIONS), dtype=np.float32)
        for a, pool in enumerate(self.action_pools):
            direct = np.asarray(np.abs(W[np.ix_(self.mbon, pool)]).sum(axis=1)).ravel()
            indirect = np.asarray(two_hop[:, pool].sum(axis=1)).ravel()
            infl[:, a] = direct + 0.25 * indirect
        col = infl.sum(axis=0, keepdims=True)
        col[col == 0] = 1.0
        self.mbon_action = infl / col

    # ------------------------------------------------------------------
    def reset_state(self) -> None:
        self.brain.reset()
        self.eligibility = np.zeros_like(self.kc_mbon_w)
        self.prev_px = None  # previous frame's per-photoreceptor sample
        self.dopamine = np.zeros(self.mbon.size, dtype=np.float32)
        self.last_mbon = np.zeros(self.mbon.size, dtype=np.float32)
        # Baseline persists across episodes: it is a property of the circuit's
        # coupling, not of any one run.
        if not hasattr(self, "drive_baseline"):
            self.drive_baseline = np.zeros(N_ACTIONS, dtype=np.float32)

    # ------------------------------------------------------------------
    def perceive_and_act(self, obs: np.ndarray, explore: float | None = None) -> tuple[int, dict]:
        """Run the brain on one game frame and choose an action."""
        cfg = self.cfg
        eps = cfg.explore if explore is None else explore

        # Drive photoreceptors from the retina, inverted (see AgentConfig).
        flat = obs.ravel()
        self._ext[:] = cfg.tonic_drive
        # Contrast coding, not absolute luminance. Fly photoreceptors adapt to
        # the local mean, and without that adaptation they simply saturate here
        # and the retina stops modulating anything downstream -- an active but
        # blind brain. Subtracting the frame mean keeps them in a graded regime
        # around the tonic operating point, and preserves the inverted sign.
        px = flat[self.photo_pixel]
        # Transient component: what changed since the last frame. Sign is
        # inverted along with the sustained term, since photoreceptors are
        # histaminergic and light reduces their release.
        if self.prev_px is None:
            motion = np.zeros_like(px)
        else:
            motion = self.prev_px - px
        self.prev_px = px.copy()
        self._ext[self.photo] = (
            cfg.tonic_drive
            + cfg.input_gain * cfg.sustained_weight * (px.mean() - px)
            + cfg.input_gain * cfg.motion_gain * motion
        )

        for _ in range(cfg.brain_steps_per_frame):
            self.brain.step(self._ext)

        trace = self.brain.trace
        mbon_act = trace[self.mbon]
        kc_act = trace[self.kc]

        # Action drive has two sources: the mushroom body's learned vote, routed
        # through its measured influence on each pool, plus whatever the rest of
        # the brain is already delivering to the descending neurons.
        # The MBON's effective output is its own spiking scaled by the learned
        # KC drive it currently receives, so plasticity actually changes the vote.
        kc_drive = self.kc_mbon_w.T @ kc_act
        mb_vote = (mbon_act + kc_drive) @ self.mbon_action
        pool_act = np.array([trace[p].mean() if p.size else 0.0 for p in self.action_pools],
                            dtype=np.float32)
        drive = pool_act + mb_vote

        # Compare each action against its own slow baseline, so selection is
        # driven by deviations rather than by fixed anatomical offsets.
        self.drive_baseline += cfg.drive_baseline_rate * (drive - self.drive_baseline)
        relative = drive - self.drive_baseline

        if self.rng.random() < eps:
            action = int(self.rng.integers(N_ACTIONS))
        else:
            action = int(np.argmax(relative))

        # Eligibility: KC activity paired with the MBONs that are currently
        # driving the chosen action. This is what makes credit assignment
        # possible without touching synapses the fly cannot change.
        credit = self.mbon_action[:, action]
        self.eligibility *= cfg.eligibility_decay
        self.eligibility += np.outer(kc_act, credit * (mbon_act > 0))

        self.last_mbon = mbon_act
        info = {
            "drive": drive,
            "relative": relative,
            "kc_active": int((kc_act > 0).sum()),
            "mbon_active": int((mbon_act > 0).sum()),
            "mb_vote": mb_vote,
        }
        return action, info

    # ------------------------------------------------------------------
    def reinforce(self, reward: float) -> None:
        """Deliver reinforcement through the dopaminergic populations.

        Positive reward recruits PAM, negative recruits PPL1, each reaching the
        MBON compartments the connectome says they innervate. Plasticity is then
        applied only to KC->MBON synapses that exist, and is depressing, as in
        the real mushroom body.
        """
        cfg = self.cfg
        r = float(np.clip(reward * cfg.reward_scale, -1.0, 1.0))
        da = (self.pam_to_mbon * r) if r > 0 else (self.ppl1_to_mbon * r)
        self.dopamine = self.dopamine * cfg.dopamine_decay + da

        # Sign of the update. This is the one place the model knowingly departs
        # from the measured biology, so it is worth being explicit.
        #
        # In the real mushroom body the rule is *depressing*: coincidence of
        # Kenyon cell activity with dopamine weakens the KC->MBON synapse. That
        # yields correct behaviour in the animal because MBON output is
        # avoidance-valenced -- depressing the pathway releases approach. Here,
        # MBONs are mapped onto actions by measured anatomical influence, which
        # carries no valence, so that indirection is not available: applying a
        # purely depressing rule makes reward suppress the action that earned it
        # and punishment reinforce it, which is exactly inverted.
        #
        # So the update is signed to make the mapping appetitive: reward
        # potentiates the responsible KC->MBON synapses, punishment depresses
        # them. The mechanism, the site, and the compartment routing are all
        # still the fly's; only this sign convention is ours.
        dw = cfg.lr * self.eligibility * self.dopamine[None, :]
        self.kc_mbon_w += dw * self.kc_mbon_mask
        np.clip(self.kc_mbon_w, cfg.weight_floor, cfg.weight_ceiling, out=self.kc_mbon_w)

    # ------------------------------------------------------------------
    def weight_change(self) -> dict:
        """How far the mushroom body has moved from its initial state."""
        live = self.kc_mbon_mask > 0
        if not live.any():
            return {"n_plastic": 0}
        d = (self.kc_mbon_w - self.kc_mbon_w0)[live]
        return {
            "n_plastic": int(live.sum()),
            "mean_abs_change": float(np.abs(d).mean()),
            "frac_depressed": float((d < -1e-6).mean()),
            "frac_potentiated": float((d > 1e-6).mean()),
        }


    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------
    def save(self, path) -> None:
        """Persist what learning changed, plus the fixed mappings it depends on.

        Only the plastic block is state; the connectome itself is rebuilt from
        source. The action pools are stored because although the clustering that
        produces them is seeded, reproducing it exactly is a needless risk.
        """
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "kc_mbon_w": self.kc_mbon_w,
                "kc_mbon_w0": self.kc_mbon_w0,
                "kc_mbon_mask": self.kc_mbon_mask,
                "mbon_action": self.mbon_action,
                "action_pools": [p.tolist() for p in self.action_pools],
                "photo_pixel": self.photo_pixel,
                "drive_baseline": getattr(self, "drive_baseline", None),
                "cfg": self.cfg,
            }, f)

    def load(self, path) -> None:
        import pickle
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.kc_mbon_w = d["kc_mbon_w"]
        self.kc_mbon_w0 = d["kc_mbon_w0"]
        self.kc_mbon_mask = d["kc_mbon_mask"]
        self.mbon_action = d["mbon_action"]
        self.action_pools = [np.asarray(q, dtype=np.int64) for q in d["action_pools"]]
        self.photo_pixel = d["photo_pixel"]
        if d.get("drive_baseline") is not None:
            self.drive_baseline = d["drive_baseline"]
        self.cfg = d["cfg"]


# ----------------------------------------------------------------------
def run_episode(agent: FlyAgent, env: SubwayEnv, explore: float, learn: bool = True,
                max_frames: int = 600) -> dict:
    obs = env.reset()
    agent.reset_state()
    total = 0.0
    frames = 0
    while not env.done and frames < max_frames:
        action, _ = agent.perceive_and_act(obs, explore)
        obs, reward, done, info = env.step(action)
        if learn:
            agent.reinforce(reward)
        total += reward
        frames += 1
    return {"return": total, "distance": info["distance"], "coins": info["coins"],
            "frames": frames}
