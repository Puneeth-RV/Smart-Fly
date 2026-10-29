# How it works

A frame of Super Mario Bros travels through six stages before Mario moves. The
first four are fixed: they come from the connectome and are not trained. Only
the last two learn.

```
NES frame (240 x 256)
  │  game.retina          local contrast, 15 x 24 cells, playfield only
  ▼
retina
  │  eye                  each of 3,354 photoreceptors watches one cell
  ▼
photoreceptors ─┐
  │             │  brain  graded neurons on the real wiring, 6 steps per decision
  ▼             │
optic lobe ─────┘         89,389 neurons, 3.69M synapses between them all
  │  fly.features         activity averaged onto maps of the visual field: 1,344 numbers
  ▼
policy                    actor-critic network, memory of 4 decisions + last button   ← trained (PPO)
  ▼
one of 7 buttons, held for 4 frames
```

## 1. Retina

`smartfly/game.py`. The playfield (frame rows 48-224, below the score bar) is cut
into a 15 x 24 grid, and each cell takes the **standard deviation** of brightness
inside it, not the mean. Mario's world is flat colour with hard edges, so local
contrast marks pipes, blocks, gaps and enemies far better than brightness. The
grid is wide because the screen is: what matters most is what is coming from the
right.

## 2. Eye: where each photoreceptor looks

`smartfly/eye.py`. The connectome records which facet of the compound eye each
columnar optic-lobe cell serves, as hexagonal coordinates `(hex1, hex2)`.
Photoreceptors have no coordinate of their own, so each is placed at the
synapse-weighted average position of the cells it contacts.

Two corrections were needed before the fly could see the whole screen:

- **The hex axes are 60 degrees apart.** Used as x and y they put the eye on a
  slanted band, and 40% of the screen, including most of the area ahead of and
  above Mario, reached no photoreceptor. `HexToScreen` rotates onto the
  lattice's principal axes and spreads each axis along its own distribution.
- **Inferred positions cluster**, which still left holes. Photoreceptors are
  therefore tiled evenly: sorted across the eye into 24 strips, each strip
  sorted top to bottom into 15 cells. Every cell gets 9-11 photoreceptors and
  neighbours stay neighbours.

Each decision, a photoreceptor receives `0.2 + 0.8 × contrast + 1.5 × change
since the last decision`.

## 3. Brain: graded neurons on the wiring

`smartfly/brain.py`, `smartfly/connectome/`. The fly uses the R1-R6
photoreceptors and the optic lobe from the MaleCNS v1.0 connectome: 92,743
neurons, 3.69M synapses, with signs from each neuron's predicted transmitter
(acetylcholine excites; GABA, glutamate and histamine inhibit).

Each neuron is a leaky graded unit, `v ← v + (W·r + 0.5 + input − v) / 3`, with
output `r = clip(v, 0, 1)`:

- **Graded, not spiking.** Photoreceptors and most of the early fly optic lobe
  signal with continuous voltage. A spiking simulation lost the picture within a
  synapse or two (see [the journey](journey.md)).
- **Resting at 0.5.** Photoreceptor output is inhibitory. From a resting level of
  0 inhibition has nothing to push down and the optic lobe stayed silent.
- **Two chosen constants.** The connectome gives synapse counts and signs, not
  strengths. Inhibitory synapses count 4x and overall coupling is 0.5.

The visual system runs 6 steps per decision.

## 4. Features: what the policy reads

`smartfly/fly.py`. Optic-lobe neurons are assigned to a 20 x 16 map of the visual
field through the same `HexToScreen` transform as the eye, so the maps line up
with the screen. Per decision the fly produces:

| part | size | what |
|---|---|---|
| deep | 320 | optic-lobe activity, averaged per map cell |
| motion | 320 | change in `deep` since the last decision |
| photo | 24 | photoreceptor activity in 24 groups by index |
| first | 320 | the first layer behind the photoreceptors, per map cell |
| eye | 360 | photoreceptor activity laid out as the 15 x 24 retina |

`photo` and `first` turned out to carry little: the groups ignore position, and
photoreceptor inhibition holds the first layer close to zero. They are kept
because the trained policies expect exactly this layout.

What each layer encodes, from `python -m smartfly.probe` (4,000 decisions on 1-1,
balanced accuracy on held-out attempts, 50% = chance):

| layer | Goomba ahead | wall ahead |
|---|---|---|
| retina | 72% | 84% |
| photoreceptors | 59% | 87% |
| first layer | 57% | 65% |
| deeper optic lobe | 50% | 65% |
| what the policy reads | 61% | 85% |

Walls survive to the policy; Goombas, one or two retina cells across, barely do.

## 5. Policy

`smartfly/policy.py`. A two-layer MLP (256 units, tanh) with a policy head over
the 7 buttons and a value head. Its input is the last 4 feature vectors plus a
one-hot of the last button: short-term memory, so it can tell "mid-jump, keep
holding A" from "standing at a pipe, jump now".

## 6. Training

`smartfly/train.py`. PPO, 8 emulators in parallel, 512 decisions each per update,
4 epochs, clip 0.2, gamma 0.9, GAE lambda 0.95, entropy bonus 0.01 → 0.005. The
reward is the game's own (distance gained, minus time, minus death), clipped and
divided by 10, plus 5 for the flag.

- **Random starts.** Each attempt begins after 0-30 idle frames. The game is
  deterministic, so without this a policy can pass by memorising one path.
- **A short horizon** (gamma 0.9) credits the jump that cleared a pipe rather than
  everything in the next ten seconds.
- **20 s of patience.** An attempt is abandoned after 300 decisions with no new
  ground. 8 s was too short for 1-2's exit pipe.
- **Dead ends end the attempt.** In 1-2, running over the ceiling past the exit
  pipe leads to the Warp Zone, whose pipes need the Down button the fly does not
  have. It paid more distance than the real exit, so the policy kept choosing it.
  Passing that point without going through a pipe now ends the attempt and costs
  what dying does.

## The campaign

`smartfly/campaign.py` works through all 32 levels: train the current level in
1.5M-frame chunks (with 40% of practice reviewing levels already seen), test it
(10 top-choice attempts from random starts), and move on at 8/10. A level is
given 25M frames; if it is getting close (a recent test at 3/10 or better, or
distance rising), it gets 7.5M more, up to 50M; otherwise it is set aside and
revisited later.

## How it is measured

- **Top choice only.** Tests play the button the policy rates highest, never a
  random draw, so a pass means the fly knew what to do.
- **Random starts.** Every attempt starts after a random number of idle frames.
- **Ablations.** `smartfly.evaluate` repeats the test with the eye blanked
  (blind), with the same pixels shuffled into random positions (scrambled), and
  against random buttons.
- **Probes.** `smartfly.probe` labels frames from the game's memory ("Goomba
  ahead", "wall ahead") and asks how well a linear readout recovers each label
  from each layer, on attempts it was not fitted on.

## Reproducibility

On one machine every run is deterministic. Across machines it is not, and cannot
be made so cheaply: **the simulated optic lobe is chaotic.** Nudging its state by
1e-7 changes the fly's choices within about 15 decisions (one second of play),
and different CPUs round floating point differently by about that much. So the
same fly, from the same start, plays a different game on an Intel Linux server
than on an Apple-silicon Mac. Every result here is therefore a rate over varied
attempts, which is what carries over. The regression test compares only the
first decisions before the divergence sets in.
