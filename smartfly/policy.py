"""The trained part: a small network from the fly's features to the seven buttons.

The connectome is not trained. The policy is an actor-critic MLP (two hidden
layers of 256) reading the fly's features from its last four decisions plus the
button it pressed last -- short-term memory, without which it could not tell
"mid-jump, keep holding" from "standing at a pipe, jump now".

`Player` loads a trained fly from a checkpoint and plays one decision at a time;
evaluation, the site and the campaign's tests all use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from smartfly.fly import Fly
from smartfly.game import ACTIONS

N_ACTIONS = len(ACTIONS)


class ActorCritic(nn.Module):
    def __init__(self, n_inputs: int, n_actions: int = N_ACTIONS, hidden: int = 256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_inputs, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


class Memory:
    """The last `k` feature vectors (newest first) and the last button, as one input."""

    def __init__(self, n_features: int, k: int = 4):
        self.buf = np.zeros((k, n_features), np.float32)
        self.last = np.zeros(N_ACTIONS, np.float32)

    def reset(self, features: np.ndarray) -> np.ndarray:
        self.buf[:] = features
        self.last[:] = 0
        return self.input()

    def push(self, features: np.ndarray, action: int) -> np.ndarray:
        self.buf = np.roll(self.buf, 1, axis=0)
        self.buf[0] = features
        self.last[:] = 0
        self.last[int(action)] = 1
        return self.input()

    def input(self) -> np.ndarray:
        return np.concatenate([self.buf.ravel(), self.last])


class RunningNorm:
    """Running mean and variance, so the policy sees standardised features."""

    def __init__(self, n: int):
        self.mean, self.var, self.count = np.zeros(n), np.ones(n), 1e-4

    def update(self, x: np.ndarray) -> None:
        bm, bv, bn = x.mean(0), x.var(0), len(x)
        delta, total = bm - self.mean, self.count + bn
        self.mean = self.mean + delta * bn / total
        self.var = (self.var * self.count + bv * bn + delta**2 * self.count * bn / total) / total
        self.count = total

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -6, 6)

    def state(self) -> tuple:
        return self.mean, self.var, self.count

    def load(self, state: tuple) -> None:
        self.mean, self.var, self.count = state


@dataclass
class Decision:
    action: int
    probs: np.ndarray  # probability of each button
    value: float  # reward the fly expects from here
    features: np.ndarray  # what the fly's visual system produced
    attention: np.ndarray | None = None  # per map cell, which parts of view drove the choice


class Player:
    """A trained fly, playing its top choice (or sampling) one decision at a time."""

    def __init__(self, checkpoint: str | Path, fly: Fly | None = None):
        ck = torch.load(checkpoint, weights_only=False)
        args = ck["args"]
        self.checkpoint = Path(checkpoint)
        self.frames_trained = int(ck.get("frames", 0))
        self.frame_skip = int(args.get("frame_skip", 4))
        self.fly = fly or Fly(steps=int(args.get("brain_steps", 6)))
        self.k = int(args.get("stack", 4))
        self.memory = Memory(self.fly.n_features, self.k)
        self.net = ActorCritic(self.fly.n_features * self.k + N_ACTIONS)
        self.net.load_state_dict(ck["net"])
        self.net.eval()
        self.norm = RunningNorm(self.fly.n_features * self.k + N_ACTIONS)
        self.norm.load(ck["norm"])
        self._last: int | None = None

    def reset(self) -> None:
        self.fly.reset()
        self._last = None

    def decide(
        self,
        retina: np.ndarray,
        greedy: bool = True,
        attention: bool = False,
        rng: np.random.Generator | None = None,
    ) -> Decision:
        """Show the fly a retina image and get its choice."""
        f = self.fly.see(retina)
        obs = self.memory.reset(f) if self._last is None else self.memory.push(f, self._last)
        x = torch.from_numpy(self.norm(obs).astype(np.float32))[None]
        if attention:
            x.requires_grad_(True)
            logits, value = self.net(x)
        else:
            with torch.no_grad():
                logits, value = self.net(x)
        probs = torch.softmax(logits, -1)[0].detach().numpy().astype(np.float64)
        probs /= probs.sum()
        a = (
            int(probs.argmax())
            if greedy
            else int((rng or np.random.default_rng()).choice(len(probs), p=probs))
        )
        att = None
        if attention:
            logits[0, a].backward()
            att = self._attention((x.grad[0] * x[0]).abs().detach().numpy())
        self._last = a
        return Decision(a, probs, float(value[0].detach()), f, att)

    def _attention(self, saliency: np.ndarray) -> np.ndarray:
        """Gradient x input of the chosen button, summed per map cell over memory."""
        n, lay = self.fly.n_features, self.fly.layout
        total = np.zeros(self.fly.n_cells)
        for j in range(self.k):
            for part in ("deep", "motion", "first"):
                s = lay[part]
                total += saliency[j * n + s.start : j * n + s.stop]
        return total / (total.max() + 1e-9)
