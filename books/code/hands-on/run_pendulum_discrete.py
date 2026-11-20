# -*- coding: utf-8 -*-
"""第 16 章实验：把连续动作离散化，再用 DQN 跑 Pendulum。

连续扭矩被切成 7 档，其他部分和 CartPole 上的 DQN 一样。
这不是最优做法，只是用来看清“离散化”能做到什么程度。

运行： python books/code/hands-on/run_pendulum_discrete.py
输出： results/pendulum_dqn_s<seed>.json
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
TORQUES = np.linspace(-2.0, 2.0, 7)


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


def train(seed: int, episodes: int = 300, gamma=0.99, lr=1e-3,
          batch=64, target_sync=200) -> Result:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = pick_device()
    env = gym.make("Pendulum-v1")
    eval_env = gym.make("Pendulum-v1")

    q = QNet().to(device)
    target = QNet().to(device)
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buf = deque(maxlen=50000)
    eps_start, eps_end, decay = 1.0, 0.05, 6000
    step = 0

    res = Result(name="pendulum_dqn_s%d" % seed,
                 config={"seed": seed, "episodes": episodes, "gamma": gamma,
                         "lr": lr, "torques": TORQUES.tolist()})

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        done, total = False, 0.0
        while not done:
            eps = max(eps_end, eps_start - (eps_start - eps_end) * step / decay)
            if rng.random() < eps:
                action = int(rng.integers(len(TORQUES)))
            else:
                with torch.no_grad():
                    z = torch.as_tensor(obs, dtype=torch.float32, device=device)
                action = int(q(z).argmax().item())
            nxt, reward, term, trunc, _ = env.step([TORQUES[action]])
            done = term or trunc
            buf.append((obs, action, reward, nxt, float(term)))
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

    def greedy(o):
        with torch.no_grad():
            z = torch.as_tensor(o, dtype=torch.float32, device=device)
            idx = int(q(z).argmax().item())
        return np.array([TORQUES[idx]], dtype=np.float32)

    res.extra["eval_return"] = evaluate(eval_env, greedy, episodes=20)
    res.extra["device"] = str(device)
    env.close()
    eval_env.close()
    return res


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    for seed in (0, 1, 2):
        t0 = time.time()
        res = train(seed)
        res.save(RESULTS)
        print("s%d eval=%.1f (%.0fs)" % (seed, res.extra["eval_return"],
                                         time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
