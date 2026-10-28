# Archive

Earlier experiments that led to the current fly, kept for reference. None of this
is needed to run Smart-Fly, and the code here imports the old `src/` layout, so it
does not run from this folder. To run any of it, check out commit `b1d320e` (the
last commit before the restructure), where everything is in place.

They are kept because the dead ends are part of how the working version was found.
The main README's *What was learned* section summarises what each taught.

## What is here

| Experiment | Code | Results | Outcome |
|---|---|---|---|
| **Full-brain spiking simulation.** All 166,197 neurons as leaky integrate-and-fire units, with homeostatic calibration. | `src/brain/lif.py` | | Ran at laptop speed, but spiking units lost the visual signal within one or two synapses. Replaced by graded units on the visual system only. |
| **The runner.** A three-lane endless runner played by the whole brain, learning through the fly's own mushroom body: dopamine-gated plasticity at the Kenyon cell to MBON synapses, plus imitation (DAgger) and readout controls. | `src/runner/`, `src/train/` | `out/run*`, `out/dagger*`, `out/fly_plays.gif` | Did not learn: 13.4 mean distance against 13.8 for random play over 200 seeds. A supervised probe showed why: the action was 61.8% decodable from the retina but at chance (~21%) from Kenyon cells, MBONs and descending neurons, so the signal was gone before reaching any learning circuit. |
| **Subway Surfers on an Android emulator.** The brain playing the real phone game over adb, plus a 3D scene of a fly holding a phone. | `src/android/`, `out/scene/index*` | | Abandoned: adverts, menus, crashes and slow screen capture. The NES emulator replaced it. |
| **Connectome in the browser.** The network exported and simulated live in a web page. | `src/android/export_live.py`, `out/scene/live*`, `out/scene/brain.js` | | Worked as a visual; not used further. |
| **Two-brain pursuit.** Two simulated flies chasing each other in an arena. | `src/connectome/export_pursuit.py`, `out/scene/pursuit*` | | Steering did not work: no usable left/right signal reached the descending neurons. |
| **Mario by evolution.** A linear readout of the spiking optic lobe evolved on distance. | `src/nes/mario.py`, `src/nes/mario_agent.py`, `src/nes/train_all.py`, `src/nes/mario_server.py`, `out/scene/mario.html` | `out/mario*`, `out/stage_baselines.json` | Reached x = 1,944 on 1-1 (random: 539), but never finished a level, and trained across 22 levels it scored 0.88x random on held-out ones. |
| **First PPO fly.** The working approach before the restructure: PPO, the full-field eye, graded neurons. | `src/nes/ppo_fly.py`, `eval_vision.py`, `probe_vision.py`, `campaign.py`, `fly_channel.py`, `site_server.py`, `watch_training.py`, `out/scene/training.html` | `out/ppo_fly_v2.json`, `out/probe_vision*.json` | Became the `smartfly` package, ported and checked to give identical decisions. |
| **Pathway audit.** Checks that the processed connectome's sensory-to-motor pathways exist. | `src/connectome/audit.py` | `out/pathway_audit.json` | Informational. |
| **Neuron skeletons and atlas.** Downloads neuron morphology for 3D rendering. | `src/connectome/fetch_skeletons.py`, `src/connectome/export_atlas.py` | | Used only by the 3D scenes. |

`run_local.sh` and `run_site.sh` started the old live pages; `requirements-old.txt`
was the old dependency list.
