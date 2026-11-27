# -*- coding: utf-8 -*-
"""Chapter 14 extra experiment: uniform replay vs prioritized replay.

Both variants use double estimation, lr 5e-4, batch 64, target sync 200,
250 episodes. Only the sampling distribution differs.

    python books/code/hands-on/ch14/run_per.py

Writes results/dqn_per_<variant>_s<seed>.json
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from rlkit import Result, evaluate, pick_device, set_seed

RESULTS = HERE.parent / "results"


class QNet(nn.Module):
    def __init__(self, obs_dim=4, n_actions=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class UniformBuffer:
    """Plain ring buffer, every item equally likely."""

    def __init__(self, capacity, alpha=0.6):
        self.cap = capacity
        self.data = []
        self.pos = 0

    def add(self, item, td_error=None):
        if len(self.data) < self.cap:
            self.data.append(item)
        else:
            self.data[self.pos] = item
        self.pos = (self.pos + 1) % self.cap

    def sample(self, batch, rng, beta=1.0):
        idx = rng.integers(0, len(self.data), size=batch)
        items = [self.data[i] for i in idx]
        weights = np.ones(batch, dtype=np.float32)
        return items, idx, weights

    def update(self, idx, td_error):
        return None

    def __len__(self):
        return len(self.data)


class PerBuffer(UniformBuffer):
    """Prioritized buffer: priority p = (|td| + eps) ** alpha.

    Sampling is proportional to p, importance weight (1 / N p) ** beta
    corrects the bias, then the weights are normalized by their max so the
    gradient scale stays close to the uniform case.
    """

    def __init__(self, capacity, alpha=0.6, eps=1e-6):
        super().__init__(capacity)
        self.alpha = alpha
        self.eps = eps
        self.prio = np.zeros(capacity, dtype=np.float64)
        self.max_prio = 1.0

    def add(self, item, td_error=None):
        p = self.max_prio if td_error is None else (abs(td_error) + self.eps) ** self.alpha
        if len(self.data) < self.cap:
            self.prio[len(self.data)] = p
        else:
            self.prio[self.pos] = p
        self.max_prio = max(self.max_prio, p)
        super().add(item)

    def sample(self, batch, rng, beta=1.0):
        p = self.prio[:len(self.data)]
        prob = p / p.sum()
        idx = rng.choice(len(self.data), size=batch, p=prob)
        items = [self.data[i] for i in idx]
        w = (len(self.data) * prob[idx]) ** (-beta)
        w = w / w.max()
        return items, idx, w.astype(np.float32)

    def update(self, idx, td_error):
        p = (np.abs(td_error) + self.eps) ** self.alpha
        for i, v in zip(idx, p):
            self.prio[i] = v
        self.max_prio = max(self.max_prio, float(p.max()))


def train(variant: str, seed: int, episodes: int = 250, gamma=0.99,
          lr=5e-4, batch=64, target_sync=200, alpha=0.6,
          beta_start=0.4) -> Result:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = pick_device()
    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")

    q = QNet().to(device)
    target = QNet().to(device)
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buf = PerBuffer(50000, alpha=alpha) if variant == "per" else UniformBuffer(50000)

    eps_start, eps_end, decay = 1.0, 0.02, 5000
    step = 0
    res = Result(name="dqn_per_%s_s%d" % (variant, seed),
                 config={"variant": variant, "seed": seed, "episodes": episodes,
                         "gamma": gamma, "lr": lr, "batch": batch,
                         "target_sync": target_sync, "alpha": alpha,
                         "beta_start": beta_start})

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
            buf.add((obs, action, reward, nxt, float(term)))
            obs = nxt
            total += reward
            length += 1

            if len(buf) >= 1000:
                beta = min(1.0, beta_start + (1.0 - beta_start) * step / 20000)
                items, idx, w = buf.sample(batch, rng, beta=beta)
                s, a, r, s2, d = zip(*items)
                s = torch.as_tensor(np.asarray(s, dtype=np.float32), device=device)
                a = torch.as_tensor(np.asarray(a, dtype=np.int64), device=device)
                r = torch.as_tensor(np.asarray(r, dtype=np.float32), device=device)
                s2 = torch.as_tensor(np.asarray(s2, dtype=np.float32), device=device)
                d = torch.as_tensor(np.asarray(d, dtype=np.float32), device=device)
                wt = torch.as_tensor(w, dtype=torch.float32, device=device)

                with torch.no_grad():
                    best = q(s2).argmax(dim=1, keepdim=True)
                    q_next = target(s2).gather(1, best).squeeze(1)
                    y = r + gamma * (1.0 - d) * q_next
                pred = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
                td = (pred - y).detach().cpu().numpy()
                loss = (wt * nn.functional.smooth_l1_loss(pred, y, reduction="none")).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()
                buf.update(idx, td)
                if step % target_sync == 0:
                    target.load_state_dict(q.state_dict())
            step += 1

        res.returns.append(total)
        res.lengths.append(length)

    def greedy(o):
        with torch.no_grad():
            z = torch.as_tensor(o, dtype=torch.float32, device=device)
            return int(q(z).argmax().item())

    res.extra["eval_return"] = evaluate(eval_env, episodes=20, policy_fn=greedy)
    res.extra["device"] = str(device)
    res.extra["steps"] = step
    env.close()
    eval_env.close()
    return res


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    variants = sys.argv[1:] or ["base", "per"]
    for variant in variants:
        for seed in (0, 1, 2):
            t0 = time.time()
            res = train(variant, seed)
            res.save(RESULTS)
            print("%-5s s%d eval=%6.1f (%.0fs)" % (variant, seed,
                                                   res.extra["eval_return"],
                                                   time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
