"""Fast tests: no emulator, no connectome."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from smartfly import compat
from smartfly.brain import REST, GradedBrain
from smartfly.campaign import getting_close, level_mix, next_level
from smartfly.eye import HexToScreen, photoreceptor_pixels
from smartfly.game import ACTIONS, LEVELS, RETINA, retina
from smartfly.policy import ActorCritic, Memory, RunningNorm, silenced
from smartfly.probe import readout_accuracy
from smartfly.train import parse_levels
from smartfly.web.server import training_progress


# ------------------------------------------------------------------ game
def test_levels_and_actions():
    assert len(LEVELS) == 32 and LEVELS[0] == "1-1" and LEVELS[-1] == "8-4"
    assert len(ACTIONS) == 7


def test_retina_marks_edges_not_brightness():
    frame = np.full((240, 256, 3), 200, np.uint8)  # flat colour: no contrast anywhere
    assert retina(frame).shape == RETINA
    assert retina(frame).max() == 0
    frame[120:140, 128:136] = 0  # one dark block: contrast at its cells
    r = retina(frame)
    assert r.max() == pytest.approx(1.0)
    assert (r > 0).sum() <= 6  # a 20x8 px block straddles at most 3 x 2 cells


# ------------------------------------------------------------------- eye
def _hex_lattice(n=30):
    """Column coordinates on a slanted lattice, like the fly's (axes 60 degrees apart)."""
    a, b = np.meshgrid(np.arange(n), np.arange(n))
    return (a + 0.5 * b).ravel().astype(np.float32), b.ravel().astype(np.float32)


def test_hexmap_fills_the_screen():
    h1, h2 = _hex_lattice()
    u, v = HexToScreen(h1, h2)(h1, h2)
    assert u.min() == pytest.approx(0) and u.max() == pytest.approx(1)
    assert v.min() == pytest.approx(0) and v.max() == pytest.approx(1)
    # Evenly spread: every 10% band of each axis holds roughly 10% of columns.
    for w in (u, v):
        counts = np.histogram(w, bins=10, range=(0, 1))[0]
        assert counts.min() > 0.06 * w.size


def test_every_retina_pixel_gets_photoreceptors():
    rng = np.random.default_rng(0)
    h1, h2 = _hex_lattice()
    hexmap = HexToScreen(h1, h2)
    n_photo, rows, cols = 600, 5, 8
    targets = rng.integers(0, h1.size, (n_photo, 3))
    out = sp.csr_matrix(
        (np.ones(targets.size), (np.repeat(np.arange(n_photo), 3), targets.ravel())),
        shape=(n_photo, h1.size),
    )
    pixel = photoreceptor_pixels(out, h1, h2, hexmap, rows, cols)
    counts = np.bincount(pixel, minlength=rows * cols)
    assert counts.min() >= n_photo // (rows * cols) - 1
    assert counts.max() <= n_photo // (rows * cols) + 1


# ----------------------------------------------------------------- brain
def test_graded_brain_rests_and_responds():
    W = sp.csr_matrix(np.array([[0, 1.0], [0, 0]], np.float32))  # neuron 0 excites 1
    brain = GradedBrain(W)
    for _ in range(60):
        rest = brain.step(np.zeros(2, np.float32)).copy()
    assert rest[0] == pytest.approx(REST, abs=1e-3)
    for _ in range(60):
        driven = brain.step(np.array([0.4, 0], np.float32)).copy()
    assert driven[0] > rest[0] and driven[1] > rest[1]
    assert driven.max() <= 1.0 and driven.min() >= 0.0


def test_inhibitory_synapses_suppress():
    W = sp.csr_matrix(np.array([[0, -1.0], [0, 0]], np.float32))  # neuron 0 inhibits 1
    brain = GradedBrain(W)
    for _ in range(60):
        r = brain.step(np.zeros(2, np.float32))
    assert r[1] < r[0]


# ---------------------------------------------------------------- policy
def test_memory_keeps_newest_first_and_last_button():
    mem = Memory(n_features=3, k=2)
    x0 = mem.reset(np.array([1, 1, 1], np.float32))
    assert x0.tolist() == [1, 1, 1, 1, 1, 1] + [0] * 7
    x1 = mem.push(np.array([2, 2, 2], np.float32), action=5)
    assert x1[:6].tolist() == [2, 2, 2, 1, 1, 1]
    assert x1[6:].tolist() == [0, 0, 0, 0, 0, 1, 0]


def test_running_norm_standardises():
    rng = np.random.default_rng(1)
    norm = RunningNorm(4)
    for _ in range(50):
        norm.update(rng.normal(5, 2, (64, 4)))
    z = norm(rng.normal(5, 2, (1000, 4)))
    assert abs(z.mean()) < 0.1 and abs(z.std() - 1) < 0.1


def test_silenced_covers_each_group_in_every_memory_slot():
    layout = {"a": slice(0, 3), "b": slice(3, 5), "c": slice(5, 6)}
    idx = silenced(layout, n=6, k=2, groups=("a", "c"))
    assert sorted(idx.tolist()) == [0, 1, 2, 5, 6, 7, 8, 11]  # never b, never the last button
    assert silenced(layout, n=6, k=2, groups=()).size == 0


def test_actor_critic_shapes():
    net = ActorCritic(10)
    logits, value = net(__import__("torch").zeros(3, 10))
    assert logits.shape == (3, 7) and value.shape == (3,)


# ------------------------------------------------------------- training
def test_parse_levels():
    assert parse_levels("1-1") == [("1-1", 1.0)]
    assert parse_levels("1-2:0.6, 1-1:0.4") == [("1-2", 0.6), ("1-1", 0.4)]


def _campaign(**kw):
    st = {"order": LEVELS, "current": "1-2", "mastered": ["1-1"], "parked": [], "tests": []}
    return st | kw


def test_level_mix_reviews_old_levels():
    assert level_mix(_campaign(mastered=[])) == "1-2:1"
    assert level_mix(_campaign()) == "1-2:0.6,1-1:0.4000"


def test_next_level_skips_done_and_returns_to_parked():
    assert next_level(_campaign()) == "1-3"
    everything = [lv for lv in LEVELS if lv not in ("1-2", "4-4")]
    assert next_level(_campaign(mastered=everything, parked=["4-4"])) == "4-4"


def test_getting_close():
    class Args:
        extend_if = 3

    def tests(*flags_and_x):
        return [{"level": "1-2", "flags": f, "mean_x": x} for f, x in flags_and_x]

    assert getting_close(_campaign(tests=tests((2, 1900), (3, 2500), (5, 2600))), "1-2", Args)
    assert not getting_close(_campaign(tests=tests((0, 800), (0, 810), (1, 790))), "1-2", Args)
    assert getting_close(_campaign(tests=tests((0, 500), (0, 600), (0, 700))), "1-2", Args)


# ----------------------------------------------------------------- probe
def test_readout_accuracy_separable_vs_noise():
    rng = np.random.default_rng(2)
    y = rng.random(400) < 0.3
    signal = np.where(y[:, None], 1.0, -1.0) + rng.normal(0, 0.3, (400, 5))
    noise = rng.normal(0, 1, (400, 5))
    train, test = np.arange(300), np.arange(300, 400)
    assert readout_accuracy(signal.astype(np.float32), y, train, test) > 0.95
    assert abs(readout_accuracy(noise.astype(np.float32), y, train, test) - 0.5) < 0.15


# ------------------------------------------------------------------- web
def test_training_progress_parses_the_trainer_log(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        "=== 2026-10-06 20:00 1-2 ===\n"
        "update   58  frames 19.86M  episodes  1228  last-40 mean x   2318  max  3161  "
        "flags 18/40  total flags 433  ent 0.0070  (32 min)\n"
    )
    assert training_progress(log) == [{"frames": 19.86, "mean_x": 2318.0, "flags": 18, "of": 40}]


# ---------------------------------------------------------------- compat
def test_compat_fixes_are_well_formed():
    for package, name, old, new in compat.FIXES:
        assert package in ("nes_py", "gym_super_mario_bros") and name.endswith(".py")
        assert old != new and "int(" in new
