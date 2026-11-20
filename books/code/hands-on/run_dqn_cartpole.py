# -*- coding: utf-8 -*-
"""DQN 在 CartPole 上，附带两组消融：去掉目标网络、去掉经验回放。

运行： python books/code/hands-on/run_dqn_cartpole.py
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
    """把四个传感器读数映射到两个动作的 Q 值。"""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class Replay:
    """定长经验池，满了就挤掉最老的。"""

    def __init__(self, capacity: int = 50000):
        self.buf = deque(maxlen=capacity)

    def add(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch: int, rng: np.random.Generator):
        idx = rng.integers(0, len(self.buf), size=batch)
        items = [self.buf[i] for i in idx]
        return _stack(items)

    def __len__(self):
        return len(self.buf)


def _stack(items):
    s, a, r, s2, d = zip(*items)
    return (np.asarray(s, dtype=np.float32), np.asarray(a, dtype=np.int64),
            np.asarray(r, dtype=np.float32), np.asarray(s2, dtype=np.float32),
            np.asarray(d, dtype=np.float32))


def train(variant: str, seed: int, episodes: int = 400,
          gamma: float = 0.99, lr: float = 5e-4, batch: int = 64,
          target_sync: int = 200) -> Result:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = pick_device()

    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)

    q = QNet(obs_dim, n_actions).to(device)
    target = QNet(obs_dim, n_actions).to(device)
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)

    buffer = Replay()
    recent = deque(maxlen=batch)      # 只用连续样本的对照组
    eps_start, eps_end, decay = 1.0, 0.02, 5000
    step = 0

    result = Result(name="dqn_%s_s%d" % (variant, seed),
                    config={"variant": variant, "seed": seed, "gamma": gamma,
                            "lr": lr, "batch": batch, "target_sync": target_sync,
                            "episodes": episodes})

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        done, total, length = False, 0.0, 0
        while not done:
            eps = max(eps_end, eps_start - (eps_start - eps_end) * step / decay)
            if rng.random() < eps:
                action = int(rng.integers(n_actions))
            else:
                with torch.no_grad():
                    qs = q(torch.as_tensor(obs, dtype=torch.float32, device=device))
                action = int(qs.argmax().item())

            nxt, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            transition = (obs, action, reward, nxt, float(term))
            buffer.add(*transition)
            recent.append(transition)
            obs = nxt
            total += reward
            length += 1

            if len(buffer) >= 1000:
                batch_items = list(recent) if variant == "no_replay" else None
                if batch_items is None:
                    s, a, r, s2, d = buffer.sample(batch, rng)
                else:
                    s, a, r, s2, d = _stack(batch_items)

                s = torch.as_tensor(s, device=device)
                a = torch.as_tensor(a, device=device)
                r = torch.as_tensor(r, device=device)
                s2 = torch.as_tensor(s2, device=device)
                d = torch.as_tensor(d, device=device)

                with torch.no_grad():
                    if variant == "no_target":
                        q_next = q(s2).max(dim=1).values
                    else:
                        # double DQN: the online net picks the action,
                        # the target net scores it
                        best = q(s2).argmax(dim=1, keepdim=True)
                        q_next = target(s2).gather(1, best).squeeze(1)
                    y = r + gamma * (1.0 - d) * q_next
                pred = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

                if variant != "no_target" and step % target_sync == 0:
                    target.load_state_dict(q.state_dict())
            step += 1

        result.returns.append(total)
        result.lengths.append(length)

    def greedy(o):
        with torch.no_grad():
            t = torch.as_tensor(o, dtype=torch.float32, device=device)
            return int(q(t).argmax().item())

    result.extra["eval_return"] = evaluate(eval_env, greedy, episodes=20)
    result.extra["device"] = str(device)
    env.close()
    eval_env.close()
    return result


def main() -> None:
    variants = sys.argv[1:] or ["full", "no_target", "no_replay"]
    seeds = [0, 1, 2]
    RESULTS.mkdir(exist_ok=True)
    for variant in variants:
        for seed in seeds:
            t0 = time.time()
            res = train(variant, seed)
            path = res.save(RESULTS)
            print("%-10s seed=%d  eval=%.1f  (%.0fs)  -> %s"
                  % (variant, seed, res.extra["eval_return"], time.time() - t0,
                     path.name), flush=True)


if __name__ == "__main__":
    main()
