# -*- coding: utf-8 -*-
"""第 6 章：动作分布要看条件分布，不能只看总体比例。

CartPole 的最优策略左右各占一半，总体动作比例看不出任何东西。
这里按杆子倾角把状态分桶，统计贪心策略在每个桶里选右推的比例。

运行： python books/code/hands-on/ch06_action_state.py
输出： results/ch06_action_state.json
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from rlkit import pick_device, set_seed

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


class QNet(nn.Module):
    def __init__(self, obs_dim: int = 4, n_actions: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    set_seed(0)
    rng = np.random.default_rng(0)
    device = pick_device()
    env = gym.make("CartPole-v1")
    q = QNet().to(device)
    target = QNet().to(device)
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=5e-4)
    buf = deque(maxlen=50000)
    gamma, batch, sync = 0.99, 64, 200
    eps_start, eps_end, decay = 1.0, 0.02, 5000
    step = 0

    for ep in range(150):
        obs, _ = env.reset(seed=ep)
        done = False
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
    env.close()

    # 用贪心策略跑固定局面，按杆子倾角分桶
    eval_env = gym.make("CartPole-v1")
    edges = np.linspace(-0.22, 0.22, 12)
    total = np.zeros(len(edges) - 1)
    right = np.zeros(len(edges) - 1)
    with torch.no_grad():
        for i in range(20):
            obs, _ = eval_env.reset(seed=500 + i)
            done = False
            while not done:
                x = torch.as_tensor(obs, dtype=torch.float32, device=device)
                action = int(q(x).argmax().item())
                k = int(np.clip(np.digitize(obs[2], edges) - 1, 0, len(total) - 1))
                total[k] += 1
                right[k] += action
                obs, _, term, trunc, _ = eval_env.step(action)
                done = term or trunc
    eval_env.close()

    bins = []
    for k in range(len(total)):
        if total[k] < 5:
            continue
        bins.append({
            "angle_mid": round(float((edges[k] + edges[k + 1]) / 2), 4),
            "count": int(total[k]),
            "right_share": round(float(right[k] / total[k]), 3),
        })
    payload = {"episodes": 150, "bins": bins,
               "overall_right_share": round(float(right.sum() / total.sum()), 3)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ch06_action_state.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
