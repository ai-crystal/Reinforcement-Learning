# -*- coding: utf-8 -*-
"""第 7 章的补充实验：目标网络同步间隔、奖励尺度。

运行： python books/code/hands-on/sweep_dqn_sync.py
输出： results/dqn_sync<k>_s<seed>.json, results/dqn_scale<s>_s<seed>.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rlkit import Result, evaluate, set_seed
import run_dqn_cartpole as base

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def train_scaled(scale: float, seed: int, episodes: int = 250) -> Result:
    """奖励整体乘一个系数，其余完全不变，看会不会学崩。"""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cpu")
    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")

    q = base.QNet(4, 2).to(device)
    target = base.QNet(4, 2).to(device)
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=5e-4)
    buffer = base.Replay()
    eps_start, eps_end, decay = 1.0, 0.02, 5000
    step = 0

    res = Result(name="dqn_scale%s_s%d" % (scale, seed),
                 config={"seed": seed, "reward_scale": scale,
                         "lr": 5e-4, "target_sync": 200})

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        done, total, length = False, 0.0, 0
        while not done:
            eps = max(eps_end, eps_start - (eps_start - eps_end) * step / decay)
            if rng.random() < eps:
                action = int(rng.integers(2))
            else:
                with torch.no_grad():
                    t = torch.as_tensor(obs, dtype=torch.float32)
                    action = int(q(t).argmax().item())
            nxt, reward, term, trunc, _ = env.step(action)
            reward = reward * scale
            done = term or trunc
            buffer.add(obs, action, reward, nxt, float(term))
            obs = nxt
            total += reward
            length += 1

            if len(buffer) >= 1000:
                s, a, r, s2, d = buffer.sample(64, rng)
                s = torch.as_tensor(s)
                a = torch.as_tensor(a)
                r = torch.as_tensor(r)
                s2 = torch.as_tensor(s2)
                d = torch.as_tensor(d)
                with torch.no_grad():
                    best = q(s2).argmax(dim=1, keepdim=True)
                    q_next = target(s2).gather(1, best).squeeze(1)
                    y = r + 0.99 * (1.0 - d) * q_next
                pred = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = torch.nn.functional.smooth_l1_loss(pred, y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()
                if step % 200 == 0:
                    target.load_state_dict(q.state_dict())
            step += 1

        res.returns.append(total / scale)   # 换算回步数，方便比较
        res.lengths.append(length)

    def greedy(o):
        with torch.no_grad():
            t = torch.as_tensor(o, dtype=torch.float32)
            return int(q(t).argmax().item())

    res.extra["eval_return"] = evaluate(eval_env, greedy, episodes=20) * scale
    res.extra["eval_in_steps"] = res.extra["eval_return"] / scale
    env.close()
    eval_env.close()
    return res


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "sync"

    if mode == "sync":
        for sync in [50, 200, 1000]:
            for seed in [0, 1]:
                res = base.train("full", seed, episodes=250, target_sync=sync)
                res.name = "dqn_sync%d_s%d" % (sync, seed)
                res.config["target_sync"] = sync
                print(res.name, "eval=%.1f" % res.extra["eval_return"],
                      flush=True)
                res.save(RESULTS)
    else:
        for scale in [0.01, 1.0, 100.0]:
            for seed in [0, 1]:
                res = train_scaled(scale, seed)
                print(res.name, "eval_steps=%.1f" % res.extra["eval_in_steps"],
                      flush=True)
                res.save(RESULTS)


if __name__ == "__main__":
    main()
