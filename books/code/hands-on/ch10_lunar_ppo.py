# -*- coding: utf-8 -*-
"""第 10 章：把在 CartPole 上调好的 PPO 原样搬到 LunarLander，看它怎么挣扎。

刻意只给一个很小的预算（一百次更新，约二十万步），记录评估回报、
每局动作分布和坠毁比例，用来讨论换环境后的训练预算问题。

运行： python books/code/hands-on/ch10_lunar_ppo.py
输出： results/lunar_ppo_s<seed>.json
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
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)


def gae(rewards, values, dones, gamma=0.999, lam=0.95):
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


def train(seed: int, updates: int = 100, steps_per_update: int = 2048,
          gamma=0.999, lam=0.95, clip=0.2, lr=3e-4, epochs=10,
          minibatch=256, reward_scale=0.1) -> Result:
    set_seed(seed)
    device = pick_device()
    env = gym.make("LunarLander-v3")
    eval_env = gym.make("LunarLander-v3")
    model = ActorCritic(8, 4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    res = Result(name="lunar_ppo_s%d" % seed,
                 config={"seed": seed, "updates": updates,
                         "steps_per_update": steps_per_update, "gamma": gamma,
                         "lam": lam, "clip": clip, "lr": lr,
                         "reward_scale": reward_scale})
    obs, _ = env.reset(seed=seed)
    ep_return, ep_len, ep_actions = 0.0, 0, [0, 0, 0, 0]
    eval_hist = []
    steps_total = 0

    for update in range(updates):
        obs_buf, act_buf, logp_buf = [], [], []
        rew_buf, done_buf, val_buf = [], [], []
        for _ in range(steps_per_update):
            x = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, value = model(x)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
            a = int(action.item())
            nxt, reward, term, trunc, _ = env.step(a)
            obs_buf.append(obs)
            act_buf.append(a)
            logp_buf.append(float(dist.log_prob(action).item()))
            rew_buf.append(reward * reward_scale)
            done_buf.append(float(term))
            val_buf.append(float(value.item()))
            ep_return += reward
            ep_len += 1
            ep_actions[a] += 1
            obs = nxt
            steps_total += 1
            if term or trunc:
                res.returns.append(ep_return)
                res.lengths.append(ep_len)
                res.extra.setdefault("actions", []).append(list(ep_actions))
                ep_return, ep_len, ep_actions = 0.0, 0, [0, 0, 0, 0]
                obs, _ = env.reset()

        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=device)
            _, last_value = model(x)
        values = val_buf + [float(last_value.item())]
        adv, returns = gae(rew_buf, values, done_buf, gamma, lam)
        adv_t = torch.as_tensor(adv, device=device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.as_tensor(returns, device=device)
        obs_t = torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32,
                                device=device)
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

        if (update + 1) % 10 == 0:
            def greedy(o):
                with torch.no_grad():
                    z = torch.as_tensor(o, dtype=torch.float32, device=device)
                    logits, _ = model(z)
                    return int(logits.argmax().item())
            score = evaluate(eval_env, greedy, episodes=5, seed=7000)
            eval_hist.append([steps_total, round(score, 2)])
            recent = res.returns[-10:]
            print("seed %d  upd %3d  steps %6d  train10 %7.1f  eval %7.1f"
                  % (seed, update + 1, steps_total, np.mean(recent), score), flush=True)

    res.extra["eval_points"] = eval_hist
    res.extra["steps_total"] = steps_total
    res.extra["device"] = str(device)
    env.close()
    eval_env.close()
    return res


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1]
    for seed in seeds:
        t0 = time.time()
        res = train(seed)
        path = res.save(RESULTS)
        print("lunar ppo seed=%d  (%.0fs) -> %s"
              % (seed, time.time() - t0, path.name), flush=True)


if __name__ == "__main__":
    main()
