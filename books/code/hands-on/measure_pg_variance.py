# -*- coding: utf-8 -*-
"""第 8 章：策略梯度的方差，和基线到底压掉了什么。

固定一个初始策略，采一批轨迹，各自算一次梯度，看这些梯度的方向和大小有多分散。
运行： python books/code/hands-on/measure_pg_variance.py
输出： results/pg_variance.json
"""
from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from rlkit import set_seed

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
GAMMA = 0.99


class Policy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.logits = nn.Linear(hidden, n_actions)

    def forward(self, x):
        return self.logits(self.body(x))


def discounted_returns(rewards):
    out = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for i in reversed(range(len(rewards))):
        running = rewards[i] + GAMMA * running
        out[i] = running
    return out


def gradient(policy, obs, acts, weights):
    """对 sum(logp * weight) 求一次梯度。"""
    policy.zero_grad()
    x = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
    a = torch.as_tensor(np.asarray(acts), dtype=torch.int64)
    logp = torch.distributions.Categorical(logits=policy(x)).log_prob(a)
    loss = -(logp * torch.as_tensor(weights, dtype=torch.float32)).sum()
    loss.backward()
    grads = [p.grad.detach().clone().flatten() for p in policy.parameters()
             if p.grad is not None]
    return torch.cat(grads), float(loss.item())


def main() -> None:
    set_seed(0)
    env = gym.make("CartPole-v1")
    policy = Policy(4, 2)
    rng = np.random.default_rng(0)

    trajs = []
    for ep in range(100):
        obs, _ = env.reset(seed=ep)
        ob, ac, rew, val = [], [], [], []
        done = False
        while not done:
            x = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                dist = torch.distributions.Categorical(logits=policy(x))
                v = float(policy(x).mean().item())   # 未训练的价值估计
                a = int(dist.sample().item())
            ob.append(obs)
            ac.append(a)
            val.append(v)
            obs, r, term, trunc, _ = env.step(a)
            rew.append(r)
            done = term or trunc
        trajs.append((ob, ac, rew, np.asarray(val, dtype=np.float32)))
    env.close()

    returns = [float(discounted_returns(rew).mean()) for _, _, rew, _ in trajs]
    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))

    grads = {k: [] for k in ["no_baseline", "batch_mean", "value_net"]}
    for ob, ac, rew, V in trajs:
        R = discounted_returns(rew)
        g, _ = gradient(policy, ob, ac, R)
        grads["no_baseline"].append(g)
        g, _ = gradient(policy, ob, ac, R - mean_ret)
        grads["batch_mean"].append(g)
        g, _ = gradient(policy, ob, ac, R - V)
        grads["value_net"].append(g)

    out = {
        "episodes": len(trajs),
        "explained_variance_mean_return": mean_ret,
        "std_return": std_ret,
        "mean_length": float(np.mean([len(r) for _, _, r, _ in trajs])),
        "variants": {},
    }
    for key, gs in grads.items():
        G = torch.stack(gs)
        mean_g = G.mean(dim=0)
        norm = float(mean_g.norm().item())
        spread = float((G - mean_g).norm(dim=1).mean().item())
        cos = torch.nn.functional.cosine_similarity(G, mean_g.expand_as(G), dim=1)
        out["variants"][key] = {
            "mean_grad_norm": norm,
            "mean_deviation": spread,
            "noise_to_signal": spread / (norm + 1e-12),
            "mean_cos_to_mean": float(cos.mean().item()),
        }

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "pg_variance.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
