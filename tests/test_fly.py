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
    """The trained 1-1 fly makes the recorded decisions on its first 40 frames.

    Guards the whole chain -- emulator, retina, eye geometry, brain, features,
    policy -- against any change that would silently alter what it learned.
    """
    from smartfly.game import Mario

    ref = np.load(FIXTURE)
    game = Mario("1-1")
    retina = game.reset()
    player.reset()
    for t, action in enumerate(ref["actions"]):
        # Exact on one machine; across CPUs float rounding differs by ~1e-7, far
        # below the smallest gap between the fly's top two buttons here (0.017).
        np.testing.assert_allclose(retina, ref["retina"][t], atol=1e-5)
        d = player.decide(retina)
        np.testing.assert_allclose(d.features, ref["features"][t], atol=1e-4)
        assert d.action == action, f"decision {t}"
        for _ in range(player.frame_skip):
            retina, _, done, _ = game.step(d.action)
            if done:
                break
    game.close()


def test_blind_fly_does_not_get_far(player):
    from smartfly.evaluate import play

    blind = play(player, "1-1", attempts=1, condition="blind")
    assert blind[0]["x"] < 600 and not blind[0]["flag"]
