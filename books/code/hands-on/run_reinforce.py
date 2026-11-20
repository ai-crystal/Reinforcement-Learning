# -*- coding: utf-8 -*-
"""REINFORCE：最朴素的策略梯度，带与不带基线的对比。

运行： python books/code/hands-on/run_reinforce.py
输出： results/reinforce_<variant>_s<seed>.json
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from rlkit import Result, evaluate, pick_device, set_seed

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


class Policy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.logits = nn.Linear(hidden, n_actions)

    def forward(self, x):
        return self.logits(self.body(x))


class Value(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def discounted_returns(rewards, gamma: float) -> np.ndarray:
    """从后往前算每一步之后能拿到的折扣回报。"""
    out = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for i in reversed(range(len(rewards))):
        running = rewards[i] + gamma * running
        out[i] = running
    return out


def train(variant: str, seed: int, episodes: int = 800,
          gamma: float = 0.99, lr: float = 5e-3) -> Result:
    set_seed(seed)
    device = pick_device()
    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)

    policy = Policy(obs_dim, n_actions).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    baseline = Value(obs_dim).to(device) if variant == "baseline" else None
    opt_v = torch.optim.Adam(baseline.parameters(), lr=1e-2) if baseline else None

    res = Result(name="reinforce_%s_s%d" % (variant, seed),
                 config={"variant": variant, "seed": seed, "gamma": gamma,
                         "lr": lr, "episodes": episodes})

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        logps, rewards, values = [], [], []
        done, total = False, 0.0
        while not done:
            x = torch.as_tensor(obs, dtype=torch.float32, device=device)
            dist = torch.distributions.Categorical(logits=policy(x))
            action = dist.sample()
            logps.append(dist.log_prob(action))
            if baseline is not None:
                values.append(baseline(x))
            obs, reward, term, trunc, _ = env.step(int(action.item()))
            rewards.append(reward)
            total += reward
            done = term or trunc

        returns = torch.as_tensor(discounted_returns(rewards, gamma), device=device)
        logp = torch.stack(logps)
        advantage = returns
        if baseline is not None:
            values = torch.stack(values)
            advantage = returns - values.detach()
            value_loss = nn.functional.mse_loss(values, returns)
            opt_v.zero_grad()
            value_loss.backward()
            opt_v.step()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        loss = -(logp * advantage).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        res.returns.append(total)
        res.lengths.append(len(rewards))

    def greedy(o):
        with torch.no_grad():
            x = torch.as_tensor(o, dtype=torch.float32, device=device)
            return int(policy(x).argmax().item())

    res.extra["eval_return"] = evaluate(eval_env, greedy, episodes=20)
    res.extra["device"] = str(device)
    env.close()
    eval_env.close()
    return res


def main() -> None:
    variants = sys.argv[1:] or ["plain", "baseline"]
    RESULTS.mkdir(exist_ok=True)
    for variant in variants:
        for seed in [0, 1, 2]:
            t0 = time.time()
            res = train(variant, seed)
            path = res.save(RESULTS)
            print("%-9s seed=%d  eval=%.1f  (%.0fs)  -> %s"
                  % (variant, seed, res.extra["eval_return"], time.time() - t0,
                     path.name), flush=True)


if __name__ == "__main__":
    main()
