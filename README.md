# SmartFly

Teaching the *Drosophila* connectome to play an endless runner, using the fly's
own learning circuit.

Built on the **MaleCNS v1.0** connectome released 3 September 2026 by Google
Research and HHMI Janelia — the complete male *Drosophila melanogaster* central
nervous system, brain + optic lobes + ventral nerve cord, published in *Cell*
and licensed CC-BY.

## The idea, and why it isn't the usual demo

Within days of the release, people had wired the connectome to Doom, Mario 64,
Minecraft and a crypto trading bot. Every one of those is a *fixed* mapping:
pixels drive photoreceptors, activity propagates through the frozen wiring, motor
neurons press buttons. Nothing learns. The Doom repo says as much in its own
README — "learned survival has not been demonstrated."

Nothing can learn, because **a connectome has no synaptic weights.** It records
topology, synapse counts and predicted transmitters. There is no functional
strength to adjust and no learning rule anywhere in the data.

But the fly has a learning circuit, and it is fully wired in this release. The
mushroom body is a genuine associative-learning centre: Kenyon cells synapse onto
MBONs, and dopaminergic neurons deliver reinforcement that gates plasticity at
exactly those synapses. So plasticity is added **only where the animal actually
has it**, and the reinforcement is routed through the connectome's own
dopaminergic compartments:

```
retina → R1-R6 photoreceptors → optic lobe → visual projection → central brain
       → Kenyon cells → MBONs → descending neurons → VNC motor neurons → action
                          ↑
              PAM (reward) / PPL1 (punishment) dopaminergic neurons
```

## What's here

| path | what it does |
|---|---|
| `src/connectome/build.py` | raw release → signed sparse connectivity matrix |
| `src/connectome/audit.py` | verifies every assumed pathway actually exists |
| `src/brain/lif.py` | leaky integrate-and-fire simulation over the connectome |
| `src/gym/subway.py` | fast headless 3-lane runner (training environment) |
| `src/gym/play.py` | watch the agent play with brain state alongside |
| `src/train/agent.py` | sensorimotor interface + mushroom-body plasticity |
| `src/train/train.py` | training loop, with controls |
| `src/train/report.py` | learning curves and summary |
| `src/android/bridge.py` | transfer to real Subway Surfers over adb |

## Setup

```bash
python3.12 -m venv .venv && ./.venv/bin/pip install pyarrow pandas numpy scipy torch matplotlib

# ~1.15 GB of the 23 GB release; the other four files are for spatial analysis
B=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
for f in connectome-weights-male-cns-v1.0-minconf-0.5.feather \
         body-annotations-male-cns-v1.0-minconf-0.5.feather \
         body-neurotransmitters-male-cns-v1.0.feather; do
  curl -L -o data/raw/$f $B/$f
done

./.venv/bin/python src/connectome/build.py   # → 166,197 neurons, 10.2M edges
./.venv/bin/python src/connectome/audit.py   # verify the pathways
./.venv/bin/python -u src/train/train.py --full-brain --episodes 600
```

## The connectome as processed

- **166,197 neurons**, **10,237,254 edges** (weight ≥ 3, autapses removed)
- 61% excitatory / 39% inhibitory by predicted transmitter
- only **7** neurons lack a transmitter call

Two traps in the raw data, both worth knowing about:

**The flat connectome file is segment-to-segment, not neuron-to-neuron.** Taken
at face value it yields 11.4 million "neurons" — every unproofread fragment.
Restricting to annotated bodies carrying a `superclass` gives exactly 166,700,
which is the neuron count the release reports.

**Signing the matrix by transmitter deletes the dopaminergic wiring.** Dopamine
is modulatory, so it gets sign 0 for fast transmission — and then
`eliminate_zeros()` removes those edges entirely, making `DAN → MBON` look
absent. The build keeps two matrices: signed for dynamics, raw for anatomy.

## Pathway audit

Every link the design assumes was verified against the wiring before any
training. All present and thick:

| link | synapses | neurons used |
|---|---|---|
| photoreceptor R1-R6 → optic lobe | 242,275 | 3342/3354 → 2791 |
| optic lobe → visual projection | 6,299,357 | 66203 → 9182 |
| visual projection → central brain | 3,482,796 | 8578 → 13247 |
| central brain → Kenyon cells | 1,246,569 | 4949 → 4064 |
| **Kenyon cells → MBON** | **439,429** | 4062/4064 → 96/97 |
| MBON → descending | 4,073 | 34/97 → 106/1314 |
| central brain → descending | 2,528,400 | 15495 → 1302 |
| descending → motor | 327,332 | 1076 → 761 |
| **PAM (reward) → MBON** | **27,090** | 314/316 → 49/97 |
| **PPL1 (punishment) → MBON** | **10,819** | 16/16 → 81/97 |

**Dopaminergic compartments are real.** 95 of 97 MBONs receive dopaminergic
input, and the mean pairwise cosine similarity of their DAN input patterns is
**0.107** — nearly orthogonal. Different MBONs receive different dopamine, which
is what allows one learning rule to shape different behaviours.

## Findings from building it

**The brain decorrelates visual input the way a mushroom body should.** Showing
two maximally different scenes and measuring representational similarity down the
chain:

| stage | cos(scene A, scene B) |
|---|---|
| photoreceptors | 0.000 |
| optic lobe | 0.533 |
| central brain | 0.464 |
| **Kenyon cells** | **0.260** |
| MBON | 0.138 |

Progressive pattern separation, emerging from real wiring. Nothing in the model
was built to produce it.

**Fly photoreceptors are inhibitory and light-inverted.** They are histaminergic,
and light *hyperpolarises* them — brightness reduces transmitter release, so
downstream cells are driven by disinhibition. Driving them the intuitive way
round (bright pixel → more spiking) inhibits the whole brain and nothing
downstream fires at all. The retina has to be inverted on the way in, and
contrast-coded rather than absolute, or the photoreceptors saturate and the brain
goes blind while still looking healthy.

**The network has a razor-thin operating window.** With per-neuron input
normalisation and inhibition-dominated balance, tonic drive 0.03 leaves the
network silent and 0.08 saturates it — and saturation destroys discriminability
(motor-representation similarity rises from 0.04 to 0.30). 0.05 works.

**A random body mapping silently destroys learnability.** Partitioning descending
neurons into action pools at random gives every MBON near-identical influence on
every action — 0.906 mean cosine similarity between action columns. Plasticity
then happens and behaviour never changes, which is exactly what the first
training run showed. Clustering descending neurons by their MBON input signature
instead drops that to **0.409**, and is also closer to how real descending
neurons are organised.

## The main result: the representation, not the learning rule

The agent does not learn to play, and the reason is not the plasticity rule. It
is worth stating precisely, because it is the thing none of the other connectome
game demos test.

A probe (`src/train/probe.py`) asks a different question than "does the agent
improve?". An oracle with full game-state access supplies the correct action; a
linear decoder is then trained, *supervised*, to recover that action from the
brain's internal state. A supervised linear decoder is a far stronger learner
than any biological plasticity rule, so it upper-bounds what learning could
achieve on this representation.

| decoded from | mean per-action recall |
|---|---|
| the retina (the input itself) | **61.8%** |
| Kenyon cells (4,064) | 22.4% |
| MBON (97) | 20.7% |
| descending neurons (1,314) | 21.7% |
| chance | 20.0% |

The task-relevant information is fully present in what enters the brain and is
**gone within one or two synaptic stages.** Running that decoder in closed loop
scores a mean distance of **13.7 — identical to the random policy** — against an
oracle ceiling of **68.9**.

So the flat learning curves are a ceiling imposed by the representation. The
mushroom body never had a usable signal to learn from, and no learning rule
operating on this state could do better.

Quantified over 200 shared episode seeds (`out/run4_eval.json`), with bootstrap
intervals and permutation tests against the random policy:

| condition | mean distance | 95% CI | vs random | p |
|---|---|---|---|---|
| random policy | 13.8 | [12.9, 14.7] | 1.00x | — |
| untrained brain | 12.9 | [12.2, 13.7] | 0.94x | 0.14 |
| mushroom body (trained) | 13.4 | [12.7, 14.2] | 0.97x | 0.55 |

`mb vs untrained: p=0.36`. Nothing is distinguishable from chance, and the
oracle ceiling is 68.9.

Known gap in the harness: `FlyAgent.save` persists the plastic block, but the
linear-readout control's weights live on `ReadoutAgent` and are not saved, so
re-evaluating that checkpoint scores an untrained brain. Its real figure is the
final training evaluation, 14.2 — also indistinguishable from random.

This was not fixed by, though each of these measurably helped:

- **switching off membrane noise** — at 0.02 it was comparable to the retinal
  drive surviving input normalisation (KC decodability 16.5% → 28.2%)
- **heterogeneous spike thresholds** — with identical thresholds a uniform tonic
  drive of 0.05 integrates to 1.02 against a threshold of 1.00, so the network
  was a switch: 98.5% active or 2% active, nothing between, and a saturated
  network carries no information. Spreading thresholds put it at 42–58%.
- **retinotopic photoreceptor mapping** — the first version assigned
  photoreceptor *i* to pixel *i* % 360, a permutation of the visual field.
  Neighbouring ommatidia project into neighbouring columns and the columnar
  circuitry assumes that adjacency. Photoreceptors carry no ommatidial
  coordinates in the release, but the columnar cells they drive do (23,720 cells
  over 892 columns), so each photoreceptor's retinal position is recovered as the
  mean hex coordinate of the columns it projects into. Kenyon-cell decodability
  19.0% → 27.2%.

### Where the signal dies

Decodability measured at each stage of the pathway (`chance = 20%`):

| stage | n | mean per-action recall |
|---|---|---|
| retina (input) | 360 | 59.7% |
| photoreceptors | 3,354 | 46.0% |
| first synapse (direct optic lobe targets) | 2,791 | 32.2% |
| all optic lobe | 6,000 | 29.2% |
| visual projection | 6,000 | 24.3% |
| central brain | 6,000 | 17.9% |
| Kenyon cells | 4,064 | 15.7% |

There is no single broken link. The loss is **monotonic and roughly 25-30%
relative at every synapse**, and after five stages nothing is left. That is a
property of the channel, not a bug in one place.

The mechanism follows from the model's own construction. Each neuron's inputs
are normalised so that "all my inputs fired" equals 1, meaning a neuron's output
is driven by the *fraction* of its inputs that are active. A visual signal
arriving at a deep neuron is a handful of inputs among thousands, while the rest
carry tonic-driven background. Signal-to-noise at each stage is therefore
roughly (visual inputs)/(total inputs), and it compounds geometrically with
depth. No parameter setting escapes this, because the tonic drive is *required*
for the network to be active at all -- remove it and the network is silent.

### What was tried and did not rescue it

Each of these was implemented and measured, not assumed:

| change | effect |
|---|---|
| switching off membrane noise | KC 16.5% → 28.2%, helped |
| heterogeneous spike thresholds | fixed 98%/2% saturation → 42-58% active, helped |
| connectome-derived retinotopy | KC 19.0% → 27.2%, helped |
| transient (motion) photoreceptor drive | no improvement; best DN 27.6% |
| 8 vs 20 vs 40 brain steps per frame | no improvement; 16-23% throughout |
| membrane time constant 20ms vs 8ms | no improvement |

The first three improved the operating point substantially and none of them
changed the outcome: a supervised linear decoder still plays at exactly the
random baseline.

### Why it probably still fails

In rough order of suspicion:

1. **The optic lobe computes motion, not static frames.** The core operation of
   fly vision is elementary motion detection in T4/T5, built from temporal
   correlations between neighbouring columns. This model presents static
   per-frame intensities and runs 8 brain steps per frame with a spike trace that
   mixes frames together. The circuitry is wired for spatiotemporal correlations
   that the input never contains.
2. **A uniform tonic drive to all 166k neurons is unphysiological**, and the fact
   that it was *necessary* for anything to propagate is itself evidence the
   operating point is wrong. 3,354 photoreceptors cannot compete with a drive
   applied to every neuron in the brain.
3. **Current-based LIF with a single global gain may be too crude** to carry a
   signal through five synaptic stages without decorrelating it.
4. **The retina is a synthetic abstraction**, not fly-like imagery.

The honest summary: wiring a connectome up and reading motor neurons produces
*activity*, and activity is easy to mistake for function. Checking whether that
activity actually carries task information is a cheap test, and it is the test
that is missing from every "fly brain plays X" demo — including, until it was
run, this one.

## Super Mario Bros: one level yes, the game no

Subway Surfers was a poor target: closed, Android-only, full of adverts and menus.
The NES emulator (`gym-super-mario-bros`) is a library instead -- headless at ~574
fps, restartable with a function call, and it reports Mario's distance
continuously. The brain is the same one as everywhere else: 92,743 neurons and
3.69M synapses of photoreceptors plus optic lobe, with each neuron's resting drive
fitted against Mario's own frames. Only a linear map from pooled optic-lobe
activity to the seven buttons is learned, by evolution, because Mario has no
oracle to imitate.

**Level 1-1 works.** The evolved readout reaches x = 1944 (the level is ~3,200
wide). Random buttons average 539 and never beat 723 in 25 runs; every single
held button stalls at the first pipe (<= 315). Vision is doing real work:
forcing the retina to zero drops 1-1 from 1678 to 898, 2-1 from 640 to 308 and
3-1 from 1269 to 567. On 4-1 sighted and blind play are identical, so there it
uses no vision at all.

**Mario's imagery survives the optic lobe far better than the runner's did.**
Decoding where contrast sits in the visual field: 98% at the retina, 96% at the
first synapse, 87% across the optic lobe -- then 44% at the visual projection
neurons and chance in the central brain, Kenyon cells and descending neurons.
So the readout reads the optic lobe because that is where the signal is.

**It does not generalise.** Every attempt to train across levels failed on the
levels it had not seen:

| training | trained stages | never-seen stages |
|---|---|---|
| 4 stages | 2.1x random | 0.97x random |
| 22 stages, 831 generations overnight | 1.20x random | **0.88x random** |

Scores are distance divided by that stage's own random-play average (8 runs per
stage, all 32 measured), so 1.0 means as good as mashing buttons. The overnight
run split the game three ways -- 22 stages to train on, 5 to choose the
checkpoint (5-1/6-1/7-1/2-3/8-1 were the test set, 1-3/3-2/4-1/6-2/7-3 the
validation set) -- and its validation score of 1.61x was the best of about a
hundred checks on the same five stages, which is why it did not survive contact
with the test set.

Why it fails is visible in the policy. Its decisions win by a wide margin
(median logit gap 2.1-3.4), so the weight changes evolution makes almost never
flip one: per-stage scores on most levels were *identical* before and after 831
generations. And it dies early -- 20 decisions into 1-1 rather than the 1944 the
single-level policy reached. A linear readout over this optic lobe, trained by
evolution, can memorise a level but not learn the game. Getting further would
take a nonlinear readout trained with reinforcement learning on far more play
than was spent here.

Two bugs worth knowing about if you build on this:

- **Mario's position can read as 0xFFFF.** `x_pos` is two RAM bytes and during
  some transitions (seen on 4-4) they read as garbage. Taking the maximum
  position at face value scored one run at 81x random. `Mario.step` now ignores
  readings above 5,000 and any implausible jump.
- **nes-py and gym-super-mario-bros predate numpy 2.** `uint8 * 1024` and
  `byte * 0x100` overflow instead of widening, and old gym references
  `np.bool8`. Patched in the installed packages and shimmed in `src/nes/mario.py`.

Run it live, walking all 32 stages with the final policy, at
`http://localhost:8095/mario.html`:

```bash
python -u src/nes/mario_server.py --port 8095
```

## Finishing 1-1 by sight: PPO, a fixed eye, graded neurons

The evolved readout above could not finish a level. Three changes got the fly to
the flag, each found by measuring rather than guessing.

**1. Reinforcement learning instead of evolution** (`src/nes/ppo_fly.py`). PPO
with a small actor-critic network on top of the frozen connectome, 8 emulators in
parallel, a memory of the last 4 readouts and the last button pressed. The
connectome is not trained; only the network that turns its activity into buttons.

**2. The eye was 40% blind.** Photoreceptors carry no coordinates, so each is
placed where the optic-lobe columns it synapses onto sit. Those columns are
indexed on a *hexagonal* lattice, whose two axes are 60 degrees apart; treating
them as x and y put the eye on a slanted band, and 40% of the screen -- most of
the area ahead of and above Mario -- reached no photoreceptor at all. The eye was
also the upright 15-wide x 24-tall shape inherited from the phone game. Now the
axes are decorrelated, photoreceptors are tiled evenly over a wide 24 x 15 retina
(3,354 photoreceptors, 9-11 per pixel, every pixel seen), and the optic-lobe
maps use the same transform so they stay in register.

**3. Spiking neurons were destroying the picture.** `src/nes/probe_vision.py`
labels frames from the game's RAM (wall ahead, Goomba ahead) and asks how well a
linear readout recovers each label from each layer, on held-out episodes
(balanced accuracy, 50% = chance). For "wall ahead":

| layer | spiking LIF | graded |
|---|---|---|
| retina (input) | 83% | 83% |
| photoreceptors | 81% | 82% |
| first synapse | 70% | 61-77% |
| deeper optic lobe | 57-63% | 57-64% |
| what the policy reads | 63% | **76%** |

The signal is lost at every synapse after the photoreceptors, and giving the
spiking brain 10 or 25 ms per frame instead of 3 did not help. Photoreceptors and
most of the early optic lobe do not spike in a real fly; `GradedBrain` in
`src/brain/lif.py` runs the same wiring as rectified, saturating rate units with
a resting level, so that inhibitory input -- which is what photoreceptors send --
can move them. Deeper layers still mix the image away: the connectome gives
wiring but not synaptic strengths, which connectome-constrained models of the fly
visual system (Lappalainen et al., Nature 2024) fit by training. The policy
therefore reads the photoreceptors as a pixel map plus the first synapse and the
deeper optic lobe.

**Result, 10M frames (~4 h on a MacBook).** Training ended finishing 33 of its
last 40 attempts. The real test is top-choice play (no sampling) with each
episode started after a random 0-40 idle frames, so runs are not replays of one
deterministic game (`src/nes/eval_vision.py`, 10 episodes each):

| condition | flags | mean distance |
|---|---|---|
| sight | **4/10** | 2,445 |
| blind (blank retina) | 0/10 | 304 |
| scrambled (same pixels, random positions) | 0/10 | 85 |
| random buttons | 0/10 | 653 |

It plays by sight: blanking or scrambling the eye collapses it, scrambling to
below random. But 4 of 10 is the honest number. Started from the identical state
every time, it finishes every attempt -- that is one memorised trajectory, not
reliability. Its failures sit around x = 1,780-2,470, the pits and Goomba groups
in the second half. Next step: train with randomised starts so a memorised path
stops paying.

Watch training live, with the fly's eye, optic-lobe activity, neurons, where it
is attending and every reward and punishment, at `http://localhost:8097/training.html`:

```bash
python -u src/nes/ppo_fly.py --v2 --brain-steps 6 --levels 1-1 --frames 10e6 --rollout 512 --tag ppo_fly_v2
python -u src/nes/watch_training.py --tag ppo_fly_v2 --log out/ppo_fly_v2.log
```

The trained fly is `out/ppo_fly_v2.pt`.

## Where the model departs from biology

Stated plainly, because these are the things that would matter to anyone
checking the work:

1. **The plasticity sign.** The real rule is *depressing*: KC activity coincident
   with dopamine weakens the KC→MBON synapse, which produces correct behaviour
   because MBON output is avoidance-valenced. Our MBON→action mapping is
   anatomical and carries no valence, so a purely depressing rule inverts the
   reinforcement. The update is signed to make the mapping appetitive. Mechanism,
   site and compartment routing are the fly's; this sign convention is not.
2. **Glutamate treated as inhibitory.** Standard for *Drosophila* modelling
   (GluCl-α) but a minority of fly glutamatergic synapses are excitatory.
3. **Current-based LIF**, no conductance dynamics, no dendrites, no spike timing.
   The connectome constrains who talks to whom; it says nothing about biophysics,
   so adding detail there would be false precision.
4. **Global synaptic gain is fitted**, not measured. Synapse counts are not
   conductances, so some scale had to be chosen; it is calibrated to put the
   network in a sparse asynchronous regime.
5. **The body mapping is invented.** Which descending neuron drives which action
   is not known from anatomy, so it is clustered and then fixed.

## Transfer to real Subway Surfers

Training happens in the clone because the emulator cannot supply the experience:
adb screen capture runs at roughly 5–15 fps with 60–150 ms input latency, against
the clone's **83,000 steps/s**. The brain transfers via `src/android/bridge.py`,
which reduces real game frames to the same 15×24 contrast retina using the same
projective depth mapping.

The weakest link in the whole pipeline is `frame_to_retina`: everything upstream
is measured, that function is a hand-calibrated geometric approximation, and bad
lane geometry will make a working brain look broken. Calibrate it with
`python src/android/bridge.py`, which prints what the brain would see.

Getting the game onto the emulator is left to the user, via the Play Store on a
Google-APIs system image — pulling a commercial APK off a mirror is neither legal
nor safe.

## Data

MaleCNS v1.0, CC-BY. Google Research + HHMI Janelia FlyEM.
<https://male-cns.janelia.org/download/>

## Credits

Fly model: "Drosophila - Adult Fruit Fly - CT Scan"
(https://sketchfab.com/3d-models/drosophila-adult-fruit-fly-ct-scan-ad29b897bd2b4e27bb04ab9d31baa117)
by etainproject (https://sketchfab.com/etainproject), licensed under CC-BY-4.0
(http://creativecommons.org/licenses/by/4.0/).

Connectome: MaleCNS v1.0, Google Research and HHMI Janelia, CC-BY.
