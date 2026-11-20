# -*- coding: utf-8 -*-
"""PPO：裁剪的目标函数 + 优势估计，在 CartPole 上跑通。

运行： python books/code/hands-on/run_ppo.py
输出： results/ppo_s<seed>.json
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


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)


def gae(rewards, values, dones, gamma: float = 0.99, lam: float = 0.95):
    """广义优势估计，从后往前算。"""
    adv = np.zeros(len(rewards), dtype=np.float32)
    last = 0.0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    returns = adv + np.asarray(values[:len(rewards)], dtype=np.float32)
    return adv, returns


def train(seed: int, updates: int = 120, steps_per_update: int = 1024,
          gamma: float = 0.99, lam: float = 0.95, clip: float = 0.2,
          lr: float = 3e-4, epochs: int = 10, minibatch: int = 128) -> Result:
    set_seed(seed)
    device = pick_device()
    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)

    model = ActorCritic(obs_dim, n_actions).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    res = Result(name="ppo_s%d" % seed,
                 config={"seed": seed, "updates": updates,
                         "steps_per_update": steps_per_update, "gamma": gamma,
                         "lam": lam, "clip": clip, "lr": lr})
    obs, _ = env.reset(seed=seed)
    ep_return, ep_len = 0.0, 0

    for update in range(updates):
        obs_buf, act_buf, logp_buf = [], [], []
        rew_buf, done_buf, val_buf = [], [], []
        for _ in range(steps_per_update):
            x = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, value = model(x)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
            nxt, reward, term, trunc, _ = env.step(int(action.item()))
            obs_buf.append(obs)
            act_buf.append(int(action.item()))
            logp_buf.append(float(dist.log_prob(action).item()))
            rew_buf.append(reward)
            done_buf.append(float(term))
            val_buf.append(float(value.item()))
            ep_return += reward
            ep_len += 1
            obs = nxt
            if term or trunc:
                res.returns.append(ep_return)
                res.lengths.append(ep_len)
                ep_return, ep_len = 0.0, 0
                obs, _ = env.reset()

        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=device)
            _, last_value = model(x)
        values = val_buf + [float(last_value.item())]
        adv, returns = gae(rew_buf, values, done_buf, gamma, lam)
        adv_t = torch.as_tensor(adv, device=device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.as_tensor(returns, device=device)
        obs_t = torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32, device=device)
        act_t = torch.as_tensor(act_buf, device=device)
        logp_t = torch.as_tensor(logp_buf, device=device)

        n = len(obs_buf)
        for _ in range(epochs):
            order = torch.randperm(n, device=device)
            for start in range(0, n, minibatch):
                idx = order[start:start + minibatch]
                logits, value = model(obs_t[idx])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(act_t[idx])
                ratio = (new_logp - logp_t[idx]).exp()
                surr1 = ratio * adv_t[idx]
                surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.functional.mse_loss(value, ret_t[idx])
                entropy = dist.entropy().mean()
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()

    def greedy(o):
        with torch.no_grad():
            x = torch.as_tensor(o, dtype=torch.float32, device=device)
            logits, _ = model(x)
            return int(logits.argmax().item())

    res.extra["eval_return"] = evaluate(eval_env, greedy, episodes=20)
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
        print("ppo seed=%d  eval=%.1f  (%.0fs)  -> %s"
              % (seed, res.extra["eval_return"], time.time() - t0, path.name),
              flush=True)


if __name__ == "__main__":
    main()
