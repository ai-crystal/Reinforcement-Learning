# -*- coding: utf-8 -*-
"""Chapter 17 experiment: batch size and target-sync frequency.

Fixed lr 5e-4, 250 episodes, 2 seeds per cell. Only one knob moves at a
time, so the columns can be read as small one-factor sweeps.

    python books/code/hands-on/ch17/run_sweeps.py

Writes results/sweep_<name>_s<seed>.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import gymnasium as gym
import numpy as np
import torch
from torch import nn

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


def train(name: str, seed: int, episodes=250, gamma=0.99, lr=5e-4,
          batch=64, target_sync=200) -> Result:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = pick_device()
    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")

    q = QNet().to(device)
    target = QNet().to(device)
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buf = []
    eps_start, eps_end, decay = 1.0, 0.02, 5000
    step = 0
    updates = 0

    res = Result(name="sweep_%s_s%d" % (name, seed),
                 config={"name": name, "seed": seed, "episodes": episodes,
                         "gamma": gamma, "lr": lr, "batch": batch,
                         "target_sync": target_sync})

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        done, total = False, 0.0
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
            buf.append((obs, action, reward, nxt, float(term)))
            if len(buf) > 50000:
                buf.pop(0)
            obs = nxt
            total += reward

            if len(buf) >= 1000:
                items = [buf[i] for i in rng.integers(0, len(buf), batch)]
                s, a, r, s2, d = zip(*items)
                s = torch.as_tensor(np.asarray(s, dtype=np.float32), device=device)
                a = torch.as_tensor(np.asarray(a, dtype=np.int64), device=device)
                r = torch.as_tensor(np.asarray(r, dtype=np.float32), device=device)
                s2 = torch.as_tensor(np.asarray(s2, dtype=np.float32), device=device)
                d = torch.as_tensor(np.asarray(d, dtype=np.float32), device=device)
                with torch.no_grad():
                    best = q(s2).argmax(dim=1, keepdim=True)
                    q_next = target(s2).gather(1, best).squeeze(1)
                    y = r + gamma * (1.0 - d) * q_next
                pred = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()
                updates += 1
                if step % target_sync == 0:
                    target.load_state_dict(q.state_dict())
            step += 1

        res.returns.append(total)

    def greedy(o):
        with torch.no_grad():
            z = torch.as_tensor(o, dtype=torch.float32, device=device)
            return int(q(z).argmax().item())

    res.extra["eval_return"] = evaluate(eval_env, episodes=20, policy_fn=greedy)
    res.extra["updates"] = updates
    res.extra["steps"] = step
    env.close()
    eval_env.close()
    return res


CELLS = [
    ("batch32", dict(batch=32)),
    ("batch64", dict(batch=64)),
    ("batch128", dict(batch=128)),
    ("sync50", dict(target_sync=50)),
    ("sync200", dict(target_sync=200)),
    ("sync1000", dict(target_sync=1000)),
]


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    only = sys.argv[1:]
    rows = []
    for name, kw in CELLS:
        if only and name not in only:
            continue
        evals = []
        for seed in (0, 1):
            t0 = time.time()
            res = train(name, seed, **kw)
            res.save(RESULTS)
            evals.append(res.extra["eval_return"])
            print("%-9s s%d eval=%6.1f (%.0fs)" % (name, seed,
                                                   res.extra["eval_return"],
                                                   time.time() - t0), flush=True)
        rows.append({"cell": name, "evals": evals,
                     "mean": float(np.mean(evals))})
        print("--", name, "mean %.1f" % rows[-1]["mean"], flush=True)
    out = RESULTS / "sweeps.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
