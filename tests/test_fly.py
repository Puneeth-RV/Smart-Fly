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
    """Shown the recorded first 40 screens of 1-1, the trained fly decides as it did.

    Guards the eye, brain, features and policy against any change that would
    silently alter what the fly learned. On the machine that recorded the
    fixture this is exact. Across CPUs, float rounding differs by ~1e-7 and the
    brain's clipping at 0 and 1 can amplify that in the odd map cell, so there
    the features must agree closely and the choices on 38 of 40 decisions.
    """
    ref = np.load(FIXTURE)
    player.reset()
    agree, worst, typical = 0, 0.0, []
    for t, retina in enumerate(ref["retina"]):
        d = player.decide(retina)
        diff = np.abs(d.features - ref["features"][t])
        worst, typical = max(worst, float(diff.max())), typical + [float(np.median(diff))]
        agree += d.action == ref["actions"][t]
        player._last = int(ref["actions"][t])  # remember the recorded button, as it did
    assert agree >= 38, f"{agree}/40 decisions match"
    assert max(typical) < 1e-4, "features drifted across the board"
    assert worst < 0.05, f"a feature differs by {worst}"


def test_blind_fly_does_not_get_far(player):
    from smartfly.evaluate import play

    blind = play(player, "1-1", attempts=1, condition="blind")
    assert blind[0]["x"] < 600 and not blind[0]["flag"]
