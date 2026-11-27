# -*- coding: utf-8 -*-
"""第 9 章：PPO 的健康指标。学习率正常与过大各跑一遍，记录 KL、裁剪比例、熵。

运行： python books/code/hands-on/ch09_health.py
输出： results/ppo9_lr<lr>_s<seed>.json
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


def gae(rewards, values, dones, gamma=0.99, lam=0.95):
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


def train(seed: int, lr: float, updates: int = 120, steps_per_update: int = 1024,
          gamma=0.99, lam=0.95, clip=0.2, epochs=10, minibatch=128) -> Result:
    set_seed(seed)
    device = pick_device()
    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")
    model = ActorCritic(4, 2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    res = Result(name="ppo9_lr%s_s%d" % (lr, seed),
                 config={"seed": seed, "lr": lr, "updates": updates,
                         "steps_per_update": steps_per_update, "clip": clip})
    obs, _ = env.reset(seed=seed)
    ep_return, ep_len = 0.0, 0
    kl_hist, clip_hist, ent_hist, eval_hist = [], [], [], []

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
        kls, clips, ents = [], [], []
        for _ in range(epochs):
            order = torch.randperm(n, device=device)
            for start in range(0, n, minibatch):
                idx = order[start:start + minibatch]
                logits, value = model(obs_t[idx])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(act_t[idx])
                log_ratio = new_logp - logp_t[idx]
                ratio = log_ratio.exp()
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
                kls.append(float(((ratio - 1) - log_ratio).mean().item()))
                clips.append(float((torch.abs(ratio - 1) > clip).float().mean().item()))
                ents.append(float(entropy.item()))

        kl_hist.append(round(float(np.mean(kls)), 5))
        clip_hist.append(round(float(np.mean(clips)), 4))
        ent_hist.append(round(float(np.mean(ents)), 4))

        if (update + 1) % 20 == 0:
            def greedy(o):
                with torch.no_grad():
                    z = torch.as_tensor(o, dtype=torch.float32, device=device)
                    logits, _ = model(z)
                    return int(logits.argmax().item())
            score = evaluate(eval_env, greedy, episodes=10)
            eval_hist.append([update + 1, round(score, 2)])
            print("lr=%g seed=%d  upd %3d  kl %.4f  clip %.3f  ent %.3f  eval %.1f"
                  % (lr, seed, update + 1, kl_hist[-1], clip_hist[-1],
                     ent_hist[-1], score), flush=True)

    res.extra["kl"] = kl_hist
    res.extra["clip_frac"] = clip_hist
    res.extra["entropy"] = ent_hist
    res.extra["eval_points"] = eval_hist
    res.extra["device"] = str(device)
    env.close()
    eval_env.close()
    return res


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    specs = [(3e-4, 0), (3e-4, 1), (1e-2, 0), (1e-2, 1)]
    if len(sys.argv) == 3:
        specs = [(float(sys.argv[1]), int(sys.argv[2]))]
    for lr, seed in specs:
        t0 = time.time()
        res = train(seed, lr)
        path = res.save(RESULTS)
        print("lr=%g seed=%d  (%.0fs) -> %s" % (lr, seed, time.time() - t0, path.name),
              flush=True)


if __name__ == "__main__":
    main()
