"""What does each layer of the fly's visual system still encode?

Plays 1-1 with right-leaning random buttons, and labels every decision from the
game's memory:

    goomba  an enemy 8-96 px ahead of Mario on screen
    wall    a solid tile (pipe, block, step) 1-3 tiles ahead at Mario's height

Then fits a linear readout for each label from each layer's activity and scores
it on attempts it was not fitted on (balanced accuracy: 50% is chance). A layer
far below the one before it is losing the picture.

    retina         the image fed to the photoreceptors
    photoreceptors the 3,354 photoreceptors themselves
    first layer    3,000 of the optic-lobe neurons they synapse onto
    optic lobe     3,000 optic-lobe neurons with known visual positions
    policy input   the feature vector the policy reads

    python -m smartfly.probe --decisions 3000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from smartfly.connectome.visual import load as load_visual_system
from smartfly.fly import Fly
from smartfly.game import Mario

BUTTON_ODDS = np.array([0.03, 0.25, 0.20, 0.25, 0.15, 0.10, 0.02])  # right-leaning random play
LABELS = ("goomba", "wall")


def _tile(ram: np.ndarray, col: int, row: int) -> int:
    if not 0 <= row < 13:
        return 0
    page = (col // 16) % 2
    return int(ram[0x500 + page * 208 + row * 16 + col % 16])


def label(ram: np.ndarray, x_level: int) -> list[bool]:
    """[goomba ahead, wall ahead] from the NES RAM."""
    mario_x, mario_y = int(ram[0x3AD]), int(ram[0xCE])
    goomba = any(ram[0x0F + i] and 8 <= int(ram[0x3AE + i]) - mario_x <= 96 for i in range(5))
    col = (x_level + 8) // 16
    rows = {(mario_y - 32) // 16, (mario_y - 32 + 15) // 16}
    wall = any(_tile(ram, col + k, r) for k in (1, 2, 3) for r in rows)
    return [goomba, wall]


def collect(decisions: int, seed: int = 0) -> tuple[dict, np.ndarray, np.ndarray]:
    vs = load_visual_system()
    fly = Fly(vs=vs)
    rng = np.random.default_rng(seed)
    has_pos = np.isfinite(vs.hex1[vs.whole_index[vs.optic_lobe]])
    optic = rng.choice(vs.optic_lobe[has_pos], 3000, replace=False)
    first = rng.choice(fly.first_neurons, min(3000, fly.first_neurons.size), replace=False)

    game = Mario("1-1")
    X: dict[str, list] = {
        k: [] for k in ("retina", "photoreceptors", "first layer", "optic lobe", "policy input")
    }
    Y, attempt_of, attempt = [], [], 0
    retina, x_level = game.reset(), 40
    fly.reset()
    while len(Y) < decisions:
        features = fly.see(retina)
        r = fly.brain.rate
        X["retina"].append(retina.ravel().copy())
        X["photoreceptors"].append(r[fly.photo].copy())
        X["first layer"].append(r[first].copy())
        X["optic lobe"].append(r[optic].copy())
        X["policy input"].append(features)
        Y.append(label(game.ram, x_level))
        attempt_of.append(attempt)
        action = int(rng.choice(7, p=BUTTON_ODDS))
        for _ in range(4):
            retina, _, done, info = game.step(action)
            x_level = int(info.get("x_pos", x_level))
            if done:
                break
        if game.done:
            attempt += 1
            retina = game.reset()
            fly.reset()
    return {k: np.array(v, np.float32) for k, v in X.items()}, np.array(Y), np.array(attempt_of)


def readout_accuracy(X: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray) -> float:
    """Class-balanced ridge classifier; regularisation chosen on held-out training attempts."""
    mu, sd = X[train].mean(0), X[train].std(0) + 1e-6
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1), np.float32)])
    target = np.where(y, 1.0, -1.0)
    weight = np.where(y, 0.5 / max(y[train].mean(), 1e-3), 0.5 / max(1 - y[train].mean(), 1e-3))

    def fit(idx, lam):
        A = Xs[idx] * weight[idx, None]
        return np.linalg.solve(Xs[idx].T @ A + lam * np.eye(Xs.shape[1]), A.T @ target[idx])

    def balanced_accuracy(w, idx):
        p = Xs[idx] @ w > 0
        return 0.5 * (
            (p & y[idx]).sum() / max(y[idx].sum(), 1)
            + (~p & ~y[idx]).sum() / max((~y[idx]).sum(), 1)
        )

    cut = int(len(train) * 0.75)
    lam = max(
        [1, 10, 100, 1000, 10000],
        key=lambda lam: balanced_accuracy(fit(train[:cut], lam), train[cut:]),
    )
    return float(balanced_accuracy(fit(train, lam), test))


def main() -> None:
    ap = argparse.ArgumentParser(description="What each layer of the visual system encodes.")
    ap.add_argument("--decisions", type=int, default=3000)
    ap.add_argument("--save", help="write the results as JSON here")
    args = ap.parse_args()
    X, Y, attempt_of = collect(args.decisions)
    split = attempt_of[int(len(attempt_of) * 0.7)]  # hold out the last ~30% of attempts
    train, test = np.flatnonzero(attempt_of < split), np.flatnonzero(attempt_of >= split)
    print(
        f"\n{len(Y)} decisions; "
        + ", ".join(f"{n} ahead {Y[:, i].mean():.0%} of the time" for i, n in enumerate(LABELS))
    )
    print(f"{'':16s}" + "".join(f"{n:>9s}" for n in LABELS))
    results = {}
    for layer, data in X.items():
        results[layer] = {
            n: readout_accuracy(data, Y[:, i], train, test) for i, n in enumerate(LABELS)
        }
        print(f"{layer:16s}" + "".join(f"{results[layer][n]:9.0%}" for n in LABELS), flush=True)
    if args.save:
        Path(args.save).write_text(json.dumps({"decisions": len(Y), "accuracy": results}, indent=1))


if __name__ == "__main__":
    main()
