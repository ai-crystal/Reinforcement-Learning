# -*- coding: utf-8 -*-
"""第 7 章：在网络结构相同的前提下，逐个关掉 DQN 的小部件。

variants:
    full      double DQN + target net + grad clip
    no_double max over the target net (classic DQN)
    no_clip   double DQN + target net, no gradient clipping

运行： python books/code/hands-on/ch07_ablation.py
输出： results/dqn7_<variant>_s<seed>.json
"""
from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from rlkit import Result, evaluate, pick_device, set_seed

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def train(variant: str, seed: int, episodes: int = 250,
          gamma: float = 0.99, lr: float = 5e-4, batch: int = 64,
          target_sync: int = 200) -> Result:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = pick_device()
    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")

    q = QNet(4, 2).to(device)
    target = QNet(4, 2).to(device)
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buf = deque(maxlen=50000)
    eps_start, eps_end, decay = 1.0, 0.02, 5000
    step = 0

    res = Result(name="dqn7_%s_s%d" % (variant, seed),
                 config={"variant": variant, "seed": seed, "episodes": episodes,
                         "gamma": gamma, "lr": lr, "batch": batch,
                         "target_sync": target_sync})
    clip_fracs = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        done, total, length = False, 0.0, 0
        while not done:
            eps = max(eps_end, eps_start - (eps_start - eps_end) * step / decay)
            if rng.random() < eps:
                action = int(rng.integers(2))
            else:
                with torch.no_grad():
                    x = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    action = int(q(x).argmax().item())
            nxt, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            buf.append((obs, action, reward, nxt, float(term)))
            obs = nxt
            total += reward
            length += 1

            if len(buf) >= 1000:
                idx = rng.integers(0, len(buf), size=batch)
                items = [buf[i] for i in idx]
                s, a, r, s2, d = zip(*items)
                s = torch.as_tensor(np.asarray(s), dtype=torch.float32, device=device)
                a = torch.as_tensor(np.asarray(a), dtype=torch.int64, device=device)
                r = torch.as_tensor(np.asarray(r), dtype=torch.float32, device=device)
                s2 = torch.as_tensor(np.asarray(s2), dtype=torch.float32, device=device)
                d = torch.as_tensor(np.asarray(d), dtype=torch.float32, device=device)
                with torch.no_grad():
                    if variant == "no_double":
                        q_next = target(s2).max(dim=1).values
                    else:
                        best = q(s2).argmax(dim=1, keepdim=True)
                        q_next = target(s2).gather(1, best).squeeze(1)
                    y = r + gamma * (1.0 - d) * q_next
                pred = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, y)
                opt.zero_grad()
                loss.backward()
                if variant != "no_clip":
                    total_norm = nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                    clip_fracs.append(float(total_norm > 5.0))
                else:
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        q.parameters(), float("inf"))
                    clip_fracs.append(float(total_norm > 5.0))
                opt.step()
                if step % target_sync == 0:
                    target.load_state_dict(q.state_dict())
            step += 1

        res.returns.append(total)
        res.lengths.append(length)

    def greedy(o):
        with torch.no_grad():
            x = torch.as_tensor(o, dtype=torch.float32, device=device)
            return int(q(x).argmax().item())

    res.extra["eval_return"] = evaluate(eval_env, greedy, episodes=20)
    res.extra["clip_frac"] = round(float(np.mean(clip_fracs)), 4) if clip_fracs else 0.0
    res.extra["device"] = str(device)
    env.close()
    eval_env.close()
    return res


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    variants = sys.argv[1:] or ["full", "no_double", "no_clip"]
    for variant in variants:
        for seed in (0, 1):
            t0 = time.time()
            res = train(variant, seed)
            path = res.save(RESULTS)
            print("%-10s seed=%d  eval=%.1f  clip=%.3f  (%.0fs) -> %s"
                  % (variant, seed, res.extra["eval_return"],
                     res.extra["clip_frac"], time.time() - t0, path.name), flush=True)


if __name__ == "__main__":
    main()
