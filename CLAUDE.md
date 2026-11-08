# CLAUDE.md

Notes for working on this repository with Claude Code.

## What this is

A fruit fly's visual system (R1-R6 photoreceptors + optic lobe, 92,743 neurons,
3.69M connections (29.1M synapses) from the MaleCNS v1.0 connectome) simulated as graded neurons and
shown Super Mario Bros. A small PPO policy reads optic-lobe activity and picks
buttons. The connectome part is fixed; only the policy learns. See
`docs/how-it-works.md` for the pipeline and `docs/journey.md` for what failed and why.

## Setup and commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m smartfly.compat            # NumPy 2 patches for nes-py / gym; required once per venv

ruff check smartfly tests scripts && ruff format --check smartfly tests scripts
pytest -m "not slow"                 # fast unit tests
pytest                               # + slow tests (emulator, data/visual_system.npz, models/fly-1-1.pt)

python -m smartfly.evaluate models/fly-campaign.pt --level 1-1 --attempts 10
python -m smartfly.web.server --checkpoint models/fly-campaign.pt   # http://localhost:8090
python -m smartfly.train --levels 1-1 --frames 10e6 --out runs/1-1
scripts/start.sh                     # campaign + site, detached; `scripts/start.sh site` for the site only
```

CI (`.github/workflows/ci.yml`) runs exactly the ruff and pytest commands above on
Ubuntu with CPU torch. Keep it green.

## Layout

- `smartfly/`: the package. `game.py` (emulator, frame to 15x24 retina), `eye.py`
  (hex coordinates to screen), `brain.py` (graded units), `fly.py` (eye + brain +
  1,344 features), `policy.py` (network, memory, `Player`), `train.py` (PPO),
  `campaign.py` (all 32 levels), `evaluate.py`, `probe.py`, `connectome/`, `web/`.
- `smartfly/paths.py`: every path. Use it instead of hard-coding paths.
- `site/`: the static frontend (plain HTML/CSS/JS, no build step).
- `models/`: committed trained flies; `results/`: committed measurements quoted in the README.
- `data/visual_system.npz`: committed (12.8 MB). `data/raw/`, `data/processed/` are ignored.
- `runs/`: training output, ignored. `runs/campaign/` holds the live campaign
  (`fly.pt`, `state.json`, `train.log`, `campaign.log`).
- `archive/`: earlier experiments, kept for the record. Excluded from ruff; do not modify.

## Things that are easy to get wrong

- **The optic lobe is chaotic.** A 1e-7 state difference changes choices within
  ~15 decisions, so runs are deterministic on one machine but differ across CPUs.
  Report results as rates over many attempts; never assert exact trajectories
  beyond the first few decisions in tests.
- **Feature layout is frozen.** Trained checkpoints expect the 1,344-feature
  layout (deep 320, motion 320, photo 24, first 320, eye 360) and its order.
  Changing `fly.py`, `eye.py`, `brain.py` constants or `game.retina` breaks every
  checkpoint in `models/`. If behaviour must change, check it against
  `tests/fly_1-1_first40.npz` first.
- **`--frames` is a total**, including the frames already in the starting
  checkpoint. Resuming a 41M-frame fly with `--frames 10e6` does nothing.
- **Testing means top choice only, random starts.** The game is deterministic;
  a fixed start lets a policy pass by memorising one path. Claims about vision
  need the blind and scrambled ablations (`smartfly.evaluate --conditions`).
- **Mario's x position glitches** (values > 5000 around jumps); `game.Mario` already
  handles it. Use `game.max_x`, not raw RAM.
- **Do not interfere with a running campaign.** Check `pgrep -f smartfly.campaign`
  before touching `runs/campaign/`, switching to code that changes training, or
  starting other heavy jobs. Each training chunk is a fresh `python -m smartfly.train`
  subprocess, so it picks up whatever code is checked out at that moment.

## Long runs on the Mac (MacBook Air, no fan)

Start long jobs detached (`start_new_session=True`) and hold `caffeinate -i -s -w <pid>`,
as `scripts/start.sh` does. The Mac must be plugged in with the lid open or the run
stalls. Throughput is about 2.7M frames/hour. It runs warm; if the user asks to stop
for heat, stop it.

## Git

- One feature branch per change (`feature/`, `fix/`, `docs/`, `chore/`,
  `refactor/`), several small commits, then `git merge --no-ff` into `main` and push.
- Delete a branch, locally and on the remote, once it is merged; the `--no-ff` merge
  commit keeps it visible in history. Commit named files only (no `git commit -a`):
  unfinished work left in the tree once got swept into main that way.
- **No Claude attribution anywhere**: no `Co-Authored-By`, `Claude-Session` or
  "Generated with Claude Code" lines in commits, PRs or files.
- The GitHub repo is public. Do not change its visibility without being asked.

## Writing and frontend style

- README and docs: plain, measured statements with numbers and how they were
  measured. No hype, no emoji.
- Frontend: professional and restrained, not "AI-looking". Readable at a glance,
  light and dark themes (Light / Dark / Auto toggle). No gradients-and-glow dashboards.
- Code: match the surrounding style; short docstrings, few comments, names that
  follow the maths (`W`, `Wt`, `v`, `rate`).
