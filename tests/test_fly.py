"""Slow tests: the real visual system, the emulator and a trained fly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smartfly.paths import MODELS, VISUAL_SYSTEM

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not VISUAL_SYSTEM.exists(), reason="data/visual_system.npz missing"),
]
FIXTURE = Path(__file__).parent / "fly_1-1_first40.npz"


@pytest.fixture(scope="module")
def player():
    from smartfly.policy import Player

    return Player(MODELS / "fly-1-1.pt")


def test_visual_system_size(player):
    fly = player.fly
    assert fly.n == 92_743
    assert fly.photo.size == 3_354
    assert fly.n_features == 1_344
    assert np.bincount(fly.pixel, minlength=fly.n_pixels).min() >= 9  # whole screen seen


def test_trained_fly_decides_as_it_did(player):
    """Shown the recorded first screens of 1-1, the trained fly decides as it did.

    Guards the eye, brain, features and policy against changes that would
    silently alter what the fly learned.

    Only the first 10 decisions are compared. The simulated optic lobe is
    chaotic: a 1e-7 nudge to its state changes the fly's choices within ~15
    decisions, and different CPUs round floats differently by about that much.
    On one machine the whole run is deterministic.
    """
    ref = np.load(FIXTURE)
    player.reset()
    for t in range(10):
        d = player.decide(ref["retina"][t])
        np.testing.assert_allclose(d.features, ref["features"][t], atol=0.01)
        assert d.action == ref["actions"][t], f"decision {t}"


def test_brain_is_deterministic_on_one_machine(player):
    ref = np.load(FIXTURE)
    runs = []
    for _ in range(2):
        player.reset()
        runs.append([player.decide(r).action for r in ref["retina"][:25]])
    assert runs[0] == runs[1]


def test_blind_fly_does_not_get_far(player):
    from smartfly.evaluate import play

    blind = play(player, "1-1", attempts=1, condition="blind")
    assert blind[0]["x"] < 600 and not blind[0]["flag"]
