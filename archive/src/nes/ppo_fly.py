"""Train the fly to finish Mario levels: frozen connectome eyes, PPO motor policy.

What changed from the evolution runs, and why
---------------------------------------------
Evolution scored each candidate by one number -- how far Mario got -- so it could
not tell which jump or which button press mattered, and it was tuning a linear
map whose decisions won by such wide margins that most weight changes never
flipped one. It learned single levels and nothing that transferred.

PPO learns from every step: actions that led to progress are made more likely,
ones that led to death less likely. It is the standard method that finishes
Mario levels, and it trains a small neural network rather than a linear map.

What is the fly and what is not
-------------------------------
The visual system is the real connectome and stays frozen: photoreceptors through
optic lobe, 92,743 neurons, 3.69M synapses, simulated as spiking neurons with the
calibration fitted on Mario's own frames. The policy that turns optic-lobe
activity into buttons is learned. With a network this size doing the deciding,
much of the skill lives in the learned part, so `--features` can swap the
fly's activity for a blank or scrambled input to test how much the fly
contributes.

Environments run in worker processes, one brain each, all loading the same
calibration so every worker runs the identical brain.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "out"

FLAG_BONUS = 50.0          # the library gives nothing for finishing; we do
# Decisions without new ground before an attempt is given up (15 a second).
# Was 120 (8 s). 1-2 ends by walking into a sideways pipe in a wall, and the
# fly reached it and kept jumping at it -- jumping is what every earlier
# obstacle needed -- until the 8 s ran out. 300 (20 s) leaves room for the
# occasional plain walk right that enters the pipe and gets rewarded.
STUCK_LIMIT = 300
N_ACTIONS = 7


class Memory:
    """Short-term memory: the last few optic-lobe readouts plus the last button.

    One frame plus one frame of change could not tell the policy where it was in
    a jump. Mario's jump height depends on how long A is held, and greedy play
    walked into the first pipe and stayed there. With the last k readouts and its
    own previous button, the policy can tell "mid-jump, keep holding" from
    "standing at a pipe, press now".
    """
    def __init__(self, d, k):
        self.d, self.k = d, k
        self.buf = np.zeros((k, d), np.float32)
        self.last = np.zeros(N_ACTIONS, np.float32)

    def reset(self, f):
        self.buf[:] = f; self.last[:] = 0
        return self.obs()

    def push(self, f, a):
        self.buf = np.roll(self.buf, 1, axis=0); self.buf[0] = f
        self.last[:] = 0; self.last[int(a)] = 1
        return self.obs()

    def obs(self):
        return np.concatenate([self.buf.ravel(), self.last])


# ---------------------------------------------------------------- workers
def make_fly(brain_steps, frame_skip, calib, v2):
    """The fly as trained. v2: wide 15x24 eye covering the whole screen and
    smooth-signal (graded) neurons; v1: the spiking brain with its calibration."""
    from nes.mario_agent import MarioFly
    if v2:
        return MarioFly(brain_steps=brain_steps, frame_skip=frame_skip, graded=True,
                        retina=(15, 24)), (15, 24)
    agent = MarioFly(brain_steps=brain_steps, frame_skip=frame_skip)
    z = np.load(calib)
    agent.brain.bias = z["bias"].astype(np.float32)
    agent.brain.v_th = z["v_th"].astype(np.float32)
    return agent, (24, 15)


def _worker(conn, levels, calib, brain_steps, frame_skip, features_mode, seed, stack, v2,
            start_jitter=0):
    import warnings
    warnings.filterwarnings("ignore")
    from nes.mario import Mario
    from nes.mario_agent import MarioFly

    agent, retina_shape = make_fly(brain_steps, frame_skip, calib, v2)
    rng = np.random.default_rng(seed)
    mem = Memory(agent.n_feat, stack)
    envs = {}

    def make(level):
        if level not in envs:
            envs[level] = Mario(f"SuperMarioBros-{level}-v0", retina=retina_shape)
        return envs[level]

    state = {}

    def observe(retina):
        if features_mode == "blind":
            retina = np.zeros_like(retina)
        elif features_mode == "shuffled":
            r = retina.ravel().copy(); rng.shuffle(r); retina = r.reshape(retina.shape)
        agent.ext = agent._drive(retina)
        for _ in range(agent.brain_steps):
            agent.brain.step(agent.ext)
        return agent.features()

    names = [lv for lv, _ in levels]
    probs = np.array([w for _, w in levels], np.float64); probs /= probs.sum()

    def reset():
        level = names[int(rng.choice(len(names), p=probs))]
        m = make(level)
        retina = m.reset()
        # Random idle start. The game is deterministic, so without this every
        # attempt begins identically and the policy can pass by memorising one
        # trajectory: the first 1-1 fly finished 20/20 from a fixed start and
        # 4/10 when starts varied by a fraction of a second.
        for _ in range(int(rng.integers(0, start_jitter + 1))):
            retina, _, d, _ = m.step(0)
            if d:
                retina = m.reset()
        agent.brain.reset(); agent.prev_px = None
        agent.prev_grid = np.zeros(agent.n_grid, dtype=np.float32)
        state.update(m=m, level=level, best=0, since=0, ret=0.0, steps=0)
        return mem.reset(observe(retina))

    obs = reset()
    conn.send(obs)
    while True:
        cmd, a = conn.recv()
        if cmd == "close":
            break
        m = state["m"]
        rew, done, flag = 0.0, False, False
        for _ in range(frame_skip):
            retina, r, d, info = m.step(int(a))
            rew += float(np.clip(r, -25, 15))
            if info.get("flag_get"):
                flag = True
            if d:
                done = True
                break
        if flag:
            rew += FLAG_BONUS
        state["steps"] += 1
        state["ret"] += rew
        if m.max_x > state["best"]:
            state["best"], state["since"] = m.max_x, 0
        else:
            state["since"] += 1
        if state["since"] >= STUCK_LIMIT:
            done = True
        ep = None
        if done:
            ep = {"level": state["level"], "x": int(m.max_x), "flag": bool(flag),
                  "ret": state["ret"], "steps": state["steps"]}
            obs = reset()
        else:
            obs = mem.push(observe(retina), a)
        conn.send((obs, rew, done, ep))
    for m in envs.values():
        m.env.close()


class VecEnv:
    def __init__(self, n, levels, calib, brain_steps, frame_skip, features_mode, stack, v2,
                 start_jitter=0):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for i in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, daemon=True,
                            args=(b, levels, calib, brain_steps, frame_skip,
                                  features_mode, 1000 + i + int(time.time()) % 10000,
                                  stack, v2, start_jitter))
            p.start(); self.conns.append(a); self.procs.append(p)
        self.obs = np.stack([c.recv() for c in self.conns])

    def step(self, actions):
        for c, a in zip(self.conns, actions):
            c.send(("step", int(a)))
        res = [c.recv() for c in self.conns]
        self.obs = np.stack([r[0] for r in res])
        return (self.obs, np.array([r[1] for r in res], np.float32),
                np.array([r[2] for r in res], np.float32), [r[3] for r in res])

    def close(self):
        for c in self.conns:
            try: c.send(("close", 0))
            except Exception: pass
        for p in self.procs:
            p.join(timeout=5)


# -------------------------------------------------------------------- PPO
class RunningNorm:
    """Running mean/variance, so the policy sees standardised optic-lobe input."""
    def __init__(self, d):
        self.mean, self.var, self.n = np.zeros(d), np.ones(d), 1e-4

    def update(self, x):
        bm, bv, bn = x.mean(0), x.var(0), len(x)
        delta, tot = bm - self.mean, self.n + bn
        self.mean = self.mean + delta * bn / tot
        self.var = (self.var * self.n + bv * bn + delta ** 2 * self.n * bn / tot) / tot
        self.n = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -6, 6)


def build_net(d, n_actions):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(d, 256), nn.Tanh(),
                                      nn.Linear(256, 256), nn.Tanh())
            self.pi = nn.Linear(256, n_actions)
            self.v = nn.Linear(256, 1)
            nn.init.orthogonal_(self.pi.weight, 0.01); nn.init.zeros_(self.pi.bias)

        def forward(self, x):
            h = self.body(x)
            return self.pi(h), self.v(h).squeeze(-1)
    return Net()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1-1",
                    help="comma list; optional weights as 1-1:0.6,1-2:0.4")
    ap.add_argument("--start-jitter", type=int, default=0,
                    help="idle frames before control, drawn from 0..N per attempt")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--rollout", type=int, default=256)
    ap.add_argument("--hours", type=float, default=0.0,
                    help="wall-clock limit; 0 = none. Prefer --frames: a sleeping "
                         "laptop burns wall-clock time without training")
    ap.add_argument("--frames", type=float, default=0.0,
                    help="stop after this many game frames (e.g. 8e6)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.9,
                    help="short horizon: credit lands on the jump that cleared the "
                         "pipe, not smeared over the next ten seconds")
    ap.add_argument("--stack", type=int, default=4,
                    help="optic-lobe readouts remembered, newest first")
    ap.add_argument("--ent-start", type=float, default=0.01)
    ap.add_argument("--ent-end", type=float, default=0.005,
                    help="entropy bonus at the end of the frame budget. The bonus "
                         "keeps the policy exploring early, but held constant it also "
                         "keeps it hedging: measured at 1.30 nats against 1.95 for "
                         "uniform, its top choice alone did worse than its dice.")
    ap.add_argument("--brain-steps", type=int, default=3)
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--features", default="fly", choices=["fly", "blind", "shuffled"])
    ap.add_argument("--calibration", default=str(OUT / "mario_final_calibration.npz"))
    ap.add_argument("--tag", default="ppo_fly")
    ap.add_argument("--v2", action="store_true",
                    help="wide eye over the whole screen + graded neurons")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(2)
    levels = []
    for tok in args.levels.split(","):
        name, _, w = tok.strip().partition(":")
        levels.append((name, float(w) if w else 1.0))
    print(f"levels: {levels}   features: {args.features}   envs: {args.envs}")
    env = VecEnv(args.envs, levels, args.calibration, args.brain_steps,
                 args.frame_skip, args.features, args.stack, args.v2, args.start_jitter)
    d = env.obs.shape[1]; A = 7
    net = build_net(d, A)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    norm = RunningNorm(d)
    ck = OUT / f"{args.tag}.pt"
    if args.resume and ck.exists():
        s = torch.load(ck, weights_only=False)
        net.load_state_dict(s["net"]); opt.load_state_dict(s["opt"])
        norm.mean, norm.var, norm.n = s["norm"]
        print(f"resumed {ck.name}")

    gamma, lam, clip, vf_c = args.gamma, 0.95, 0.2, 0.5
    T, N = args.rollout, args.envs
    episodes, total_steps, upd = [], 0, 0
    t_end = time.time() + args.hours * 3600 if args.hours > 0 else float("inf")
    t0 = time.time()
    # Budget by frames played, not wall-clock: an overnight run on a laptop that
    # went to sleep with the lid closed spent its whole 3-hour allowance asleep
    # and trained for about 15 minutes of it.
    frame_budget = args.frames if args.frames > 0 else float("inf")
    if args.resume and ck.exists():
        total_steps = int(s.get("frames", 0) // args.frame_skip)
    while time.time() < t_end and total_steps * args.frame_skip < frame_budget:
        upd += 1
        # Anneal the entropy bonus from exploring toward committing, over the budget.
        frac = min(1.0, total_steps * args.frame_skip / frame_budget) \
            if frame_budget < float("inf") else min(1.0, upd / 400)
        ent_c = args.ent_start + (args.ent_end - args.ent_start) * frac
        obs_b = np.zeros((T, N, d), np.float32); act_b = np.zeros((T, N), np.int64)
        logp_b = np.zeros((T, N), np.float32); rew_b = np.zeros((T, N), np.float32)
        done_b = np.zeros((T, N), np.float32); val_b = np.zeros((T, N), np.float32)
        for t in range(T):
            norm.update(env.obs)
            o = norm(env.obs).astype(np.float32)
            with torch.no_grad():
                logits, v = net(torch.from_numpy(o))
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
            obs_b[t], act_b[t] = o, a.numpy()
            logp_b[t], val_b[t] = dist.log_prob(a).numpy(), v.numpy()
            _, r, dn, eps = env.step(a.numpy())
            rew_b[t], done_b[t] = r / 10.0, dn
            episodes += [e for e in eps if e]
        total_steps += T * N
        with torch.no_grad():
            _, last_v = net(torch.from_numpy(norm(env.obs).astype(np.float32)))
        last_v = last_v.numpy()
        adv = np.zeros((T, N), np.float32); gae = np.zeros(N, np.float32)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else val_b[t + 1]
            delta = rew_b[t] + gamma * nv * (1 - done_b[t]) - val_b[t]
            gae = delta + gamma * lam * (1 - done_b[t]) * gae
            adv[t] = gae
        ret = adv + val_b
        B = T * N
        o_t = torch.from_numpy(obs_b.reshape(B, d)); a_t = torch.from_numpy(act_b.reshape(B))
        lp_t = torch.from_numpy(logp_b.reshape(B)); ad = adv.reshape(B)
        ad_t = torch.from_numpy((ad - ad.mean()) / (ad.std() + 1e-8))
        rt_t = torch.from_numpy(ret.reshape(B))
        for _ in range(4):
            perm = torch.randperm(B)
            for i in range(0, B, 512):
                idx = perm[i:i + 512]
                logits, v = net(o_t[idx])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = (dist.log_prob(a_t[idx]) - lp_t[idx]).exp()
                pg = -torch.min(ratio * ad_t[idx],
                                ratio.clamp(1 - clip, 1 + clip) * ad_t[idx]).mean()
                loss = pg + vf_c * ((v - rt_t[idx]) ** 2).mean() - ent_c * dist.entropy().mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()

        recent = episodes[-40:]
        if recent:
            xs = [e["x"] for e in recent]; fl = sum(e["flag"] for e in recent)
            print(f"update {upd:4d}  frames {total_steps * args.frame_skip / 1e6:5.2f}M  "
                  f"episodes {len(episodes):5d}  last-40 mean x {np.mean(xs):6.0f}  "
                  f"max {max(xs):5d}  flags {fl}/{len(recent)}  "
                  f"total flags {sum(e['flag'] for e in episodes)}  ent {ent_c:.4f}  "
                  f"({(time.time()-t0)/60:.0f} min)", flush=True)
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                    "norm": (norm.mean, norm.var, norm.n), "args": vars(args),
                    "frames": total_steps * args.frame_skip}, ck)
        (OUT / f"{args.tag}.json").write_text(json.dumps(
            {"episodes": episodes[-2000:], "updates": upd,
             "frames": total_steps * args.frame_skip}, indent=0))
    env.close()
    print("done")


if __name__ == "__main__":
    main()
