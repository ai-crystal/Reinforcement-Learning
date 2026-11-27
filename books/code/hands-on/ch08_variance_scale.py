# -*- coding: utf-8 -*-
"""第 8 章：批次大小怎么改变策略梯度的信噪比。

固定一个未训练的策略，采 160 条轨迹放进池子。对每个批次大小 k，
随机抽 60 次，每次取 k 条轨迹求平均梯度，看这 60 个梯度围绕全池均值散得多开。

噪声按 1/sqrt(k) 缩小，信号不动，所以信噪比应该随着 k 增大而下降。

运行： python books/code/hands-on/ch08_variance_scale.py
输出： results/pg_variance_scale.json
"""
from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rlkit import set_seed

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
GAMMA = 0.99


class Policy(torch.nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(obs_dim, hidden),
                                        torch.nn.ReLU())
        self.logits = torch.nn.Linear(hidden, n_actions)

    def forward(self, x):
        return self.logits(self.body(x))


def discounted_returns(rewards, gamma: float = GAMMA):
    out = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for i in reversed(range(len(rewards))):
        running = rewards[i] + gamma * running
        out[i] = running
    return out


def flat_grad(policy, obs, acts, weights):
    policy.zero_grad()
    x = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
    a = torch.as_tensor(np.asarray(acts), dtype=torch.int64)
    logp = torch.distributions.Categorical(logits=policy(x)).log_prob(a)
    loss = -(logp * torch.as_tensor(np.asarray(weights), dtype=torch.float32)).sum()
    loss.backward()
    return torch.cat([p.grad.detach().flatten() for p in policy.parameters()
                      if p.grad is not None])


def main() -> None:
    set_seed(0)
    env = gym.make("CartPole-v1")
    policy = Policy(4, 2)
    rng = np.random.default_rng(0)

    trajs = []
    for ep in range(160):
        obs, _ = env.reset(seed=ep)
        ob, ac, rew = [], [], []
        done = False
        while not done:
            x = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                dist = torch.distributions.Categorical(logits=policy(x))
                a = int(dist.sample().item())
            ob.append(obs)
            ac.append(a)
            obs, r, term, trunc, _ = env.step(a)
            rew.append(r)
            done = term or trunc
        trajs.append((ob, ac, rew))
    env.close()

    # 每条轨迹在两种权重下的梯度：原始回报，以及减去全池均值
    mean_ret = float(np.mean([discounted_returns(r).mean() for _, _, r in trajs]))
    grads = {"no_baseline": [], "batch_mean": []}
    for ob, ac, rew in trajs:
        R = discounted_returns(rew)
        grads["no_baseline"].append(flat_grad(policy, ob, ac, R))
        grads["batch_mean"].append(flat_grad(policy, ob, ac, R - mean_ret))

    out = {"pool_episodes": len(trajs), "mean_return": mean_ret,
           "std_return": float(np.std(
               [discounted_returns(r).mean() for _, _, r in trajs])),
           "repeats": 60, "cells": []}

    for key, G in grads.items():
        G = torch.stack(G)
        mean_g = G.mean(dim=0)
        signal = float(mean_g.norm().item())
        for k in (1, 8, 32, 128):
            devs, norms = [], []
            for _ in range(out["repeats"]):
                idx = rng.choice(len(G), size=k, replace=False)
                g = G[idx].mean(dim=0)
                devs.append(float((g - mean_g).norm().item()))
                norms.append(float(g.norm().item()))
            out["cells"].append({
                "variant": key,
                "batch": k,
                "signal_norm": signal,
                "mean_deviation": float(np.mean(devs)),
                "noise_to_signal": float(np.mean(devs) / (signal + 1e-12)),
                "mean_batch_norm": float(np.mean(norms)),
            })
            print("%-12s k=%3d  dev=%.2f  snr=%.2f"
                  % (key, k, np.mean(devs), np.mean(devs) / signal), flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "pg_variance_scale.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", RESULTS / "pg_variance_scale.json")


if __name__ == "__main__":
    main()
