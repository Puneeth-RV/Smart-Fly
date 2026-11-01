# The journey

The working fly came after several approaches that did not work. Each one was
measured rather than abandoned on a hunch, and each failure pointed at the next
fix. The code for all of them is in [`archive/`](../archive/).

## 1. The whole brain, learning with its own learning circuit

The first idea was the most biological: simulate all 166,197 neurons as spiking
cells, show them a simple three-lane runner game, and let the fly learn the way
flies do, through dopamine-gated plasticity in the mushroom body (Kenyon cell to
MBON synapses), with actions read from descending neurons.

It ran at laptop speed, and the brain did something real: two very different
scenes became progressively more distinct along the pathway, from cosine
similarity 0.53 in the optic lobe to 0.26 in Kenyon cells and 0.14 in MBONs, a
pattern separation that nothing in the model was built to produce.

**It did not learn.** Over 200 episodes, the trained fly scored 13.4 against 13.8
for random play, with an oracle ceiling of 68.9. A supervised probe showed why,
asking how well the correct action could be decoded from each stage:

| decoded from | recall (chance 20%) |
|---|---|
| the retina | 59.7% |
| photoreceptors | 46.0% |
| first synapse | 32.2% |
| optic lobe | 29.2% |
| visual projection neurons | 24.3% |
| central brain | 17.9% |
| Kenyon cells | 15.7% |

The information was lost by about a quarter at every synapse and gone before it
reached any learning circuit. No plasticity rule can learn from a signal that
is not there. Switching off noise, spreading spike thresholds and fixing the
eye's layout each helped, and none changed the outcome.

## 2. Subway Surfers on a phone

The game was then moved to the real Subway Surfers on an Android emulator,
driven over adb. Adverts, menus, crashes and slow screen capture made it
impossible to train on. Super Mario Bros on an NES emulator replaced it: a
library, headless, several hundred frames a second, restartable in a function
call, and it reports Mario's position every frame, so partial progress is
visible.

## 3. Mario by evolution

Since the signal survives in the optic lobe and dies after it, the next version
read the optic lobe directly: a linear map from its activity to the buttons,
evolved on distance.

It reached x = 1,944 on 1-1 (random play averages 539, and every single held
button stalls at the first pipe), and blanking the eye roughly halved its score,
so it was using vision. But it never finished a level, and trained across 22
levels it scored **0.88x random on levels it had not seen**. It memorised
levels; it did not learn to play.

## 4. Reinforcement learning, and three hidden problems

PPO with a small network on top of the frozen optic lobe reached the flag, but
mostly by luck: playing only its top choice, it walked into the first Goomba.
Finding out why took three measurements.

**The eye was 40% blind.** The connectome places optic-lobe columns on a
hexagonal lattice, and treating its two axes (60 degrees apart) as x and y put
the eye on a slanted band. Most of the screen ahead of and above Mario reached
no photoreceptor at all. Fixed by rotating onto the lattice's principal axes and
tiling photoreceptors evenly.

**Spiking neurons destroyed the picture.** Frames were labelled from the game's
memory ("wall ahead", "Goomba ahead") and a linear readout asked to recover the
label from each layer. For "wall ahead", 50% = chance:

| layer | spiking | graded |
|---|---|---|
| retina | 83% | 83% |
| photoreceptors | 81% | 82% |
| what the policy reads | 63% | **76%** |

Giving the spiking network more simulated time per frame did not help. A real
fly's early visual system mostly does not spike; it signals with graded voltage.
Switching to graded units on the same wiring kept the picture.

**"20 out of 20" was one memorised path.** With the eye fixed and graded
neurons, the fly finished 1-1 on 20 consecutive attempts. But the game is
deterministic and so was the start: it was replaying one trajectory. From starts
varied by a fraction of a second it finished 4 of 10. Training with random starts
took it to **10 of 10**.

The test that settled whether it was using its eyes at all: the same fly, blind
or with its pixels scrambled, fails at the first Goomba or barely moves.

## 5. The campaign, and the exit pipe

Level 1-2 ends not at a flagpole but by walking into a pipe in a wall. The fly
reached it and kept doing running jumps at it, 70-99% sure, because jumping was
what every earlier obstacle needed, until the 8-second no-progress limit
restarted the level. With 20 seconds it occasionally walked in, was rewarded, and
went from never finishing 1-2 to finishing half its tests.

Then it stalled, its tests swinging between 1 and 7 of 10 for 12M frames.
Saving the last screen of every failed attempt showed why: most ended in the
Warp Zone, reached by running along the ceiling past the exit pipe. Its pipes
need the Down button, which the fly does not have, so it ran into the far wall
until time ran out. Yet the ceiling route paid more distance than the exit,
whose pipe moves Mario ahead in a single step, so the policy kept taking it.
Now the attempt ends there, at the cost of a death. Within 7.5M frames it
passed, 9 of 10.

## 6. Bowser, and a jump the fly never tried

1-3 was set aside after 25M frames: every test ended at the same place. Saving
the failures again showed Mario jumping from a low treetop towards a tall one
and hitting its underside. A search over button sequences found that the jump
can only be made after a brief slow-down before take-off, which about 2% of
random sequences find. The fly had learned to run flat out everywhere, because
every earlier level paid for it, and never slowed down there.

1-4 came next. Its failures were nearly all at Bowser on the final bridge
(250 of the last 300), charging into him or his fire. Getting past him is a
matter of timing rather than a wrong incentive, so nothing was changed; it
passed after 28.6M frames, 8 of 10.

Back on 1-3, the fly now gets practice starts: half its attempts replay the
furthest route so far and hand control over at a random point along it. The
first version handed over only near the end of that route; once a lucky attempt
cleared the jump, practice moved past it and the jump was never learned.
Spread along the route, it cleared the jump after 16M frames, and from practice
starts beyond it reaches the flag. Making that jump reliably from the start of
the level is what remains.

## What carried through

- **Measure before fixing.** Every real fix here (the blind eye, the spiking
  loss, the memorised path, the exit pipe) was found by a probe, an ablation or
  watching the failure, not by tuning.
- **Activity is not function.** A wired-up connectome produces plenty of
  activity; whether that activity carries task information is a separate,
  cheap question, and here the answer changed the design twice.
- **Test the way it will be judged.** Top choice only, random starts, and the
  blind and scrambled controls.
