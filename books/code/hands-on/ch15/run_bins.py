# -*- coding: utf-8 -*-
"""Chapter 15 experiment: how many torque bins are enough on Pendulum?

Same DQN as run_pendulum_discrete.py, only the bin count changes.
3, 7 and 15 bins, 2 seeds, 200 episodes each.

    python books/code/hands-on/ch15/run_bins.py

Writes results/pendulum_bins<b>_s<seed>.json
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from rlkit import Result, evaluate, pick_device, set_seed

RESULTS = HERE.parent / "results"


class QNet(nn.Module):
    def __init__(self, obs_dim=3, n_actions=7, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def train(bins: int, seed: int, episodes=200, gamma=0.99, lr=1e-3,
          batch=64, target_sync=200) -> Result:
    torques = np.linspace(-2.0, 2.0, bins)
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = pick_device()
    env = gym.make("Pendulum-v1")
    eval_env = gym.make("Pendulum-v1")

    q = QNet(3, bins).to(device)
    target = QNet(3, bins).to(device)
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buf = []
    eps_start, eps_end, decay = 1.0, 0.05, 6000
    step = 0

    res = Result(name="pendulum_bins%d_s%d" % (bins, seed),
                 config={"bins": bins, "seed": seed, "episodes": episodes,
                         "gamma": gamma, "lr": lr, "batch": batch,
                         "target_sync": target_sync})

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        done, total = False, 0.0
        while not done:
            eps = max(eps_end, eps_start - (eps_start - eps_end) * step / decay)
            if rng.random() < eps:
                action = int(rng.integers(bins))
            else:
                with torch.no_grad():
                    z = torch.as_tensor(obs, dtype=torch.float32, device=device)
                action = int(q(z).argmax().item())
            nxt, reward, term, trunc, _ = env.step([torques[action]])
            done = term or trunc
            buf.append((obs, action, reward, nxt, float(term)))
            if len(buf) > 50000:
                buf.pop(0)
            obs = nxt
            total += reward

            if len(buf) >= 1000:
                items = [buf[i] for i in rng.integers(0, len(buf), batch)]
                s, a, r, s2, d = zip(*items)
                s = torch.as_tensor(np.asarray(s, dtype=np.float32), device=device)
                a = torch.as_tensor(np.asarray(a, dtype=np.int64), device=device)
                r = torch.as_tensor(np.asarray(r, dtype=np.float32), device=device)
                s2 = torch.as_tensor(np.asarray(s2, dtype=np.float32), device=device)
                d = torch.as_tensor(np.asarray(d, dtype=np.float32), device=device)
                with torch.no_grad():
                    best = q(s2).argmax(dim=1, keepdim=True)
                    q_next = target(s2).gather(1, best).squeeze(1)
                    y = r + gamma * (1.0 - d) * q_next
                pred = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()
                if step % target_sync == 0:
                    target.load_state_dict(q.state_dict())
            step += 1

        res.returns.append(total)

    used = set()
    counts = {i: 0 for i in range(bins)}
    switches = 0
    last = None

    def greedy(o):
        nonlocal switches, last
        with torch.no_grad():
            z = torch.as_tensor(o, dtype=torch.float32, device=device)
            idx = int(q(z).argmax().item())
        used.add(idx)
        counts[idx] += 1
        if last is not None and idx != last:
            switches += 1
        last = idx
        return np.array([torques[idx]], dtype=np.float32)

    res.extra["eval_return"] = evaluate(eval_env, episodes=20, policy_fn=greedy)
    res.extra["actions_used"] = sorted(used)
    res.extra["action_counts"] = counts
    res.extra["switches"] = switches
    env.close()
    eval_env.close()
    return res


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    for bins in (3, 7, 15):
        for seed in (0, 1):
            t0 = time.time()
            res = train(bins, seed)
            res.save(RESULTS)
            print("bins%-3d s%d eval=%7.1f used=%s (%.0fs)"
                  % (bins, seed, res.extra["eval_return"],
                     res.extra["actions_used"], time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
