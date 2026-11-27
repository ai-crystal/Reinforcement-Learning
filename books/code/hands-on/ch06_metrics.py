# -*- coding: utf-8 -*-
"""第 6 章：把训练过程中的几类指标一次采齐。

* 每局回报与回合长度
* 每 25 局做一次贪心评估，记录评估回报
* 每局两个动作各选了多少次
* 每局结束时的探索率

运行： python books/code/hands-on/ch06_metrics.py
输出： results/ch06_metrics_s<seed>.json
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

from rlkit import Result, pick_device, set_seed

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


def greedy_eval(q, env, device, episodes=10, seed=9000):
    """固定的评估种子，便于不同检查点之间直接比较。"""
    total = 0.0
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32, device=device)
                action = int(q(x).argmax().item())
            obs, reward, term, trunc, _ = env.step(action)
            total += reward
            done = term or trunc
    return total / episodes


def train(seed: int, episodes: int = 200, eval_every: int = 25) -> Result:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = pick_device()

    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")
    q = QNet(4, 2).to(device)
    target = QNet(4, 2).to(device)
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=5e-4)
    buf = deque(maxlen=50000)
    batch = 64
    gamma = 0.99
    sync = 200
    eps_start, eps_end, decay = 1.0, 0.02, 5000
    step = 0

    res = Result(name="ch06_metrics_s%d" % seed,
                 config={"seed": seed, "episodes": episodes,
                         "eval_every": eval_every, "lr": 5e-4,
                         "gamma": gamma, "target_sync": sync})
    eval_points = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        done, total, length = False, 0.0, 0
        counts = [0, 0]
        while not done:
            eps = max(eps_end, eps_start - (eps_start - eps_end) * step / decay)
            if rng.random() < eps:
                action = int(rng.integers(2))
            else:
                with torch.no_grad():
                    x = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    action = int(q(x).argmax().item())
            counts[action] += 1
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
                    best = q(s2).argmax(dim=1, keepdim=True)
                    q_next = target(s2).gather(1, best).squeeze(1)
                    y = r + gamma * (1.0 - d) * q_next
                pred = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()
                if step % sync == 0:
                    target.load_state_dict(q.state_dict())
            step += 1

        res.returns.append(total)
        res.lengths.append(length)
        res.extra.setdefault("actions", []).append(counts)
        res.extra.setdefault("eps", []).append(round(eps, 4))

        if (ep + 1) % eval_every == 0:
            score = greedy_eval(q, eval_env, device)
            eval_points.append([ep + 1, round(score, 2)])
            print("seed %d  ep %3d  train %.1f  eval %.1f"
                  % (seed, ep + 1, total, score), flush=True)

    res.extra["eval_points"] = eval_points
    res.extra["device"] = str(device)
    env.close()
    eval_env.close()
    return res


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2]
    for seed in seeds:
        t0 = time.time()
        res = train(seed)
        path = res.save(RESULTS)
        print("ch06 seed=%d  done (%.0fs) -> %s" % (seed, time.time() - t0, path.name),
              flush=True)


if __name__ == "__main__":
    main()
