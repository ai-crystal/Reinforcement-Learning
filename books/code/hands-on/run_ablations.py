# -*- coding: utf-8 -*-
"""第 15 章实验：DQN 的三个改进，逐个加上去看效果。

variants:
    vanilla  target network, max over the target net (no double)
    double   online net picks the action, target net scores it
    dueling  double + dueling head (value + advantage)

运行： python books/code/hands-on/run_ablations.py
输出： results/dqn_<variant>_s<seed>.json
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


class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DuelingNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.value = nn.Linear(hidden, 1)
        self.adv = nn.Linear(hidden, n_actions)

    def forward(self, x):
        h = self.body(x)
        v = self.value(h)
        a = self.adv(h)
        return v + a - a.mean(dim=-1, keepdim=True)


class Replay:
    def __init__(self, capacity=50000):
        self.buf = deque(maxlen=capacity)

    def add(self, *item):
        self.buf.append(item)

    def sample(self, batch, rng):
        idx = rng.integers(0, len(self.buf), size=batch)
        return zip(*[self.buf[i] for i in idx])

    def __len__(self):
        return len(self.buf)


def stack(items):
    s, a, r, s2, d = items
    return (np.asarray(s, dtype=np.float32), np.asarray(a, dtype=np.int64),
            np.asarray(r, dtype=np.float32), np.asarray(s2, dtype=np.float32),
            np.asarray(d, dtype=np.float32))


def train(variant: str, seed: int, episodes: int = 250,
          gamma=0.99, lr=1e-3, batch=64, target_sync=200) -> Result:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = pick_device()
    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")

    maker = DuelingNet if variant == "dueling" else QNet
    q = maker(4, 2).to(device)
    target = maker(4, 2).to(device)
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buffer = Replay()
    eps_start, eps_end, decay = 1.0, 0.02, 5000
    step = 0

    res = Result(name="dqn_%s_s%d" % (variant, seed),
                 config={"variant": variant, "seed": seed, "episodes": episodes,
                         "gamma": gamma, "lr": lr, "batch": batch,
                         "target_sync": target_sync})

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        done, total, length = False, 0.0, 0
        while not done:
            eps = max(eps_end, eps_start - (eps_start - eps_end) * step / decay)
            if rng.random() < eps:
                action = int(rng.integers(2))
            else:
                with torch.no_grad():
                    z = torch.as_tensor(obs, dtype=torch.float32, device=device)
                action = int(q(z).argmax().item())
            nxt, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            buffer.add(obs, action, reward, nxt, float(term))
            obs = nxt
            total += reward
            length += 1

            if len(buffer) >= 1000:
                s, a, r, s2, d = stack(tuple(buffer.sample(batch, rng)))
                s = torch.as_tensor(s, device=device)
                a = torch.as_tensor(a, device=device)
                r = torch.as_tensor(r, device=device)
                s2 = torch.as_tensor(s2, device=device)
                d = torch.as_tensor(d, device=device)
                with torch.no_grad():
                    if variant == "vanilla":
                        q_next = target(s2).max(dim=1).values
                    else:
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
        res.lengths.append(length)

    def greedy(o):
        with torch.no_grad():
            z = torch.as_tensor(o, dtype=torch.float32, device=device)
            return int(q(z).argmax().item())

    res.extra["eval_return"] = evaluate(eval_env, greedy, episodes=20)
    res.extra["device"] = str(device)
    env.close()
    eval_env.close()
    return res


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    variants = sys.argv[1:] or ["vanilla", "dueling"]
    for variant in variants:
        for seed in (0, 1, 2):
            t0 = time.time()
            res = train(variant, seed)
            res.save(RESULTS)
            print("%-8s s%d eval=%.1f (%.0fs)" % (variant, seed,
                                                   res.extra["eval_return"],
                                                   time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
