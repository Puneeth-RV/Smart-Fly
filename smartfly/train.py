"""Train the fly to play with PPO. The connectome stays fixed; only the policy learns.

Eight emulators run in parallel worker processes, each with its own copy of the
fly. Every update collects `rollout` decisions from each, then takes four PPO
epochs over them. The reward is the game's own (progress right, minus time and
death), clipped, plus 50 for reaching the flag, all scaled by 1/10.

Two choices that mattered:

* Random idle starts (`--start-jitter`). The game is deterministic, so from a
  fixed start a policy can pass by replaying one memorised path: the first 1-1
  fly finished 20/20 from a fixed start and 4/10 from varied ones. Trained with
  varied starts it passed 10/10.
* A short horizon (gamma 0.9), so credit lands on the jump that cleared the
  pipe rather than being smeared over the next ten seconds.

    python -m smartfly.train --levels 1-1 --frames 10e6 --out runs/1-1
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from smartfly.paths import RUNS

FLAG_BONUS = 50.0
# Decisions without new ground before an attempt is abandoned (15 per second).
# 8 s was too short for 1-2, which ends by walking into a sideways pipe: the fly
# kept jumping at it, as every earlier obstacle needed, until time ran out.
STUCK_LIMIT = 300
# Routes with no way out for a fly without a Down button, ended and penalised
# like a death as soon as Mario passes this distance without a pipe warp. In 1-2,
# running over the ceiling past the exit pipe (entered at 2,643) leads to the
# Warp Zone; it paid more distance than the exit, and 40% of failed attempts
# ended there, stuck against its wall.
DEAD_ENDS = {"1-2": 2720}
DEAD_END_PENALTY = 25.0  # what dying costs in the game


def parse_levels(spec: str) -> list[tuple[str, float]]:
    """'1-1:0.6,1-2:0.4' -> [('1-1', 0.6), ('1-2', 0.4)]; weights default to 1."""
    out = []
    for tok in spec.split(","):
        name, _, w = tok.strip().partition(":")
        out.append((name, float(w) if w else 1.0))
    return out


# ------------------------------------------------------------------ workers
def _worker(conn, levels, steps, frame_skip, stack, vision, start_jitter, seed):
    from smartfly.fly import Fly
    from smartfly.game import Mario
    from smartfly.policy import Memory

    fly = Fly(steps=steps)
    memory = Memory(fly.n_features, stack)
    rng = np.random.default_rng(seed)
    names = [lv for lv, _ in levels]
    weights = np.array([w for _, w in levels], np.float64)
    weights /= weights.sum()
    games: dict[str, Mario] = {}
    ep: dict = {}

    def look(retina: np.ndarray) -> np.ndarray:
        if vision == "blind":
            retina = np.zeros_like(retina)
        elif vision == "scrambled":
            retina = rng.permutation(retina.ravel()).reshape(retina.shape)
        return fly.see(retina)

    def new_attempt() -> np.ndarray:
        level = names[int(rng.choice(len(names), p=weights))]
        if level not in games:
            games[level] = Mario(level)
        game = games[level]
        retina = game.reset()
        for _ in range(int(rng.integers(0, start_jitter + 1))):
            retina, _, done, _ = game.step(0)
            if done:
                retina = game.reset()
        fly.reset()
        ep.update(game=game, level=level, best=0, since=0, ret=0.0, steps=0)
        return memory.reset(look(retina))

    conn.send(new_attempt())
    while True:
        cmd, action = conn.recv()
        if cmd == "close":
            break
        game = ep["game"]
        reward, done, flag = 0.0, False, False
        for _ in range(frame_skip):
            retina, r, d, info = game.step(int(action))
            reward += float(np.clip(r, -25, 15))
            flag |= bool(info.get("flag_get"))
            if d:
                done = True
                break
        if flag:
            reward += FLAG_BONUS
        if not game.warped and game.max_x > DEAD_ENDS.get(ep["level"], np.inf):
            reward -= DEAD_END_PENALTY
            done = True
        ep["steps"] += 1
        ep["ret"] += reward
        if game.max_x > ep["best"]:
            ep["best"], ep["since"] = game.max_x, 0
        else:
            ep["since"] += 1
        done |= ep["since"] >= STUCK_LIMIT
        summary = None
        if done:
            summary = {
                "level": ep["level"],
                "x": int(game.max_x),
                "flag": flag,
                "ret": ep["ret"],
                "steps": ep["steps"],
            }
            obs = new_attempt()
        else:
            obs = memory.push(look(retina), action)
        conn.send((obs, reward, done, summary))
    for game in games.values():
        game.close()


class ParallelGames:
    """N worker processes, stepped together."""

    def __init__(self, n, levels, steps, frame_skip, stack, vision, start_jitter):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        base = int(time.time()) % 10000
        for i in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(
                target=_worker,
                daemon=True,
                args=(b, levels, steps, frame_skip, stack, vision, start_jitter, 1000 + i + base),
            )
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        self.obs = np.stack([c.recv() for c in self.conns])

    def step(self, actions):
        for c, a in zip(self.conns, actions):
            c.send(("step", int(a)))
        res = [c.recv() for c in self.conns]
        self.obs = np.stack([r[0] for r in res])
        return (
            self.obs,
            np.array([r[1] for r in res], np.float32),
            np.array([r[2] for r in res], np.float32),
            [r[3] for r in res],
        )

    def close(self):
        for c in self.conns:
            with contextlib.suppress(BrokenPipeError, OSError):
                c.send(("close", 0))
        for p in self.procs:
            p.join(timeout=5)


# ---------------------------------------------------------------------- PPO
def train(args: argparse.Namespace) -> None:
    import torch

    from smartfly.policy import ActorCritic, RunningNorm

    torch.set_num_threads(2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "fly.pt"
    levels = parse_levels(args.levels)
    print(f"levels {levels}  vision {args.vision}  envs {args.envs}", flush=True)
    games = ParallelGames(
        args.envs,
        levels,
        args.brain_steps,
        args.frame_skip,
        args.stack,
        args.vision,
        args.start_jitter,
    )
    d = games.obs.shape[1]
    net = ActorCritic(d)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    norm = RunningNorm(d)
    total_steps = 0
    source = ckpt if ckpt.exists() and args.resume else (Path(args.init) if args.init else None)
    if source:
        s = torch.load(source, weights_only=False)
        net.load_state_dict(s["net"])
        opt.load_state_dict(s["opt"])
        norm.load(s["norm"])
        total_steps = int(s.get("frames", 0)) // args.frame_skip
        print(
            f"continuing from {source} at {total_steps * args.frame_skip / 1e6:.2f}M frames",
            flush=True,
        )

    if total_steps * args.frame_skip >= args.frames:
        print(
            f"already at {total_steps * args.frame_skip / 1e6:.2f}M frames, the --frames total; "
            "nothing to do",
            flush=True,
        )
    gamma, lam, clip, vf_coef = args.gamma, 0.95, 0.2, 0.5
    T, N = args.rollout, args.envs
    budget = args.frames
    episodes: list[dict] = []
    update, t0 = 0, time.time()
    while total_steps * args.frame_skip < budget:
        update += 1
        frac = min(1.0, total_steps * args.frame_skip / budget)
        ent_coef = args.ent_start + (args.ent_end - args.ent_start) * frac
        obs_b = np.zeros((T, N, d), np.float32)
        act_b = np.zeros((T, N), np.int64)
        logp_b, rew_b, done_b, val_b = (np.zeros((T, N), np.float32) for _ in range(4))
        for t in range(T):
            norm.update(games.obs)
            o = norm(games.obs).astype(np.float32)
            with torch.no_grad():
                logits, v = net(torch.from_numpy(o))
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
            obs_b[t], act_b[t] = o, a.numpy()
            logp_b[t], val_b[t] = dist.log_prob(a).numpy(), v.numpy()
            _, r, dn, eps = games.step(a.numpy())
            rew_b[t], done_b[t] = r / 10.0, dn
            episodes += [e for e in eps if e]
        total_steps += T * N

        with torch.no_grad():
            _, last_v = net(torch.from_numpy(norm(games.obs).astype(np.float32)))
        last_v = last_v.numpy()
        adv = np.zeros((T, N), np.float32)
        gae = np.zeros(N, np.float32)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else val_b[t + 1]
            delta = rew_b[t] + gamma * nv * (1 - done_b[t]) - val_b[t]
            gae = delta + gamma * lam * (1 - done_b[t]) * gae
            adv[t] = gae
        ret = adv + val_b

        B = T * N
        o_t = torch.from_numpy(obs_b.reshape(B, d))
        a_t = torch.from_numpy(act_b.reshape(B))
        lp_t = torch.from_numpy(logp_b.reshape(B))
        ad = adv.reshape(B)
        ad_t = torch.from_numpy((ad - ad.mean()) / (ad.std() + 1e-8))
        rt_t = torch.from_numpy(ret.reshape(B))
        for _ in range(4):
            perm = torch.randperm(B)
            for i in range(0, B, 512):
                idx = perm[i : i + 512]
                logits, v = net(o_t[idx])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = (dist.log_prob(a_t[idx]) - lp_t[idx]).exp()
                pg = -torch.min(
                    ratio * ad_t[idx], ratio.clamp(1 - clip, 1 + clip) * ad_t[idx]
                ).mean()
                loss = (
                    pg + vf_coef * ((v - rt_t[idx]) ** 2).mean() - ent_coef * dist.entropy().mean()
                )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        recent = episodes[-40:]
        if recent:
            xs = [e["x"] for e in recent]
            print(
                f"update {update:4d}  frames {total_steps * args.frame_skip / 1e6:5.2f}M  "
                f"episodes {len(episodes):5d}  last-40 mean x {np.mean(xs):6.0f}  "
                f"max {max(xs):5d}  flags {sum(e['flag'] for e in recent)}/{len(recent)}  "
                f"total flags {sum(e['flag'] for e in episodes)}  ent {ent_coef:.4f}  "
                f"({(time.time() - t0) / 60:.0f} min)",
                flush=True,
            )
        saved_args = vars(args) | {"v2": True}
        torch.save(
            {
                "net": net.state_dict(),
                "opt": opt.state_dict(),
                "norm": norm.state(),
                "args": saved_args,
                "frames": total_steps * args.frame_skip,
            },
            ckpt,
        )
        (out / "episodes.json").write_text(
            json.dumps({"episodes": episodes[-2000:], "frames": total_steps * args.frame_skip})
        )
    games.close()
    print("done", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--levels", default="1-1", help="comma list, optional weights: 1-1:0.6,1-2:0.4")
    ap.add_argument(
        "--frames",
        type=float,
        required=True,
        help="stop at this many frames in total, counting any the starting fly already had",
    )
    ap.add_argument("--out", default=str(RUNS / "train"), help="directory for fly.pt and logs")
    ap.add_argument("--resume", action="store_true", help="continue from OUT/fly.pt")
    ap.add_argument("--init", help="start from another checkpoint's weights")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--rollout", type=int, default=512, help="decisions per game per update")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--ent-start", type=float, default=0.01)
    ap.add_argument("--ent-end", type=float, default=0.005)
    ap.add_argument("--stack", type=int, default=4, help="decisions of memory")
    ap.add_argument("--brain-steps", type=int, default=6, help="simulation steps per decision")
    ap.add_argument("--frame-skip", type=int, default=4, help="frames each button is held")
    ap.add_argument("--start-jitter", type=int, default=30, help="idle frames before control, 0..N")
    ap.add_argument("--vision", default="normal", choices=["normal", "blind", "scrambled"])
    return ap


if __name__ == "__main__":
    train(build_parser().parse_args())
