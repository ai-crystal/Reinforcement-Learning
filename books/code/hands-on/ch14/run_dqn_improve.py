# -*- coding: utf-8 -*-
"""第 15 章实验：DQN 的三个改进在 CartPole 上的对照。

四个配置，各 3 个种子，每个 250 局：
    base    DQN，目标网络 + max 取下一状态的最大值
    double  在 base 上换成 Double DQN
    dueling 再接上 Dueling 结构
    per     再接上优先经验回放

用法：
    python books/code/hands-on/ch15/run_dqn_improve.py base 0
    python books/code/hands-on/ch15/run_dqn_improve.py all
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import torch
from torch import nn

import gymnasium as gym

from rlkit import Result, evaluate, pick_device, set_seed

RESULTS = HERE.parent / "results"
VARIANTS = ("base", "double", "dueling", "per")


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DuelingNet(nn.Module):
    """把 Q 拆成「这个状态多好」加「动作比平均好多少」。"""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value = nn.Linear(hidden, 1)
        self.adv = nn.Linear(hidden, n_actions)

    def forward(self, x):
        h = self.body(x)
        v = self.value(h)
        a = self.adv(h)
        return v + a - a.mean(dim=1, keepdim=True)


class Replay:
    def __init__(self, capacity: int = 50000):
        self.cap = capacity
        self.s = np.zeros((capacity, 4), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, 4), dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add(self, s, a, r, s2, d):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch: int, rng):
        idx = rng.integers(0, self.size, size=batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]

    def __len__(self):
        return self.size


class SumTree:
    """按优先级采样的数组实现，内部是一棵求和二叉树。"""

    def __init__(self, capacity: int):
        self.cap = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.size = 0
        self.ptr = 0

    def _up(self, idx: int, delta: float) -> None:
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def add(self, priority: float, item) -> None:
        leaf = self.ptr + self.cap - 1
        self.data[self.ptr] = item
        self._up(leaf, priority - self.tree[leaf])
        self.tree[leaf] = priority
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def update(self, leaves, priorities) -> None:
        for leaf, p in zip(leaves, priorities):
            self._up(int(leaf), float(p) - self.tree[int(leaf)])
            self.tree[int(leaf)] = float(p)

    def _leaf_of(self, value: float) -> int:
        idx = 0
        while idx < self.cap - 1:
            left = 2 * idx + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        return idx

    def sample(self, batch: int, rng):
        total = self.tree[0]
        seg = total / batch
        idxs, leaves = [], []
        for i in range(batch):
            v = rng.uniform(seg * i, seg * (i + 1))
            leaf = self._leaf_of(min(v, total - 1e-9))
            idxs.append(leaf - self.cap + 1)
            leaves.append(leaf)
        return np.asarray(idxs), np.asarray(leaves)


def flat(items):
    s, a, r, s2, d = zip(*items)
    return (np.asarray(s, dtype=np.float32), np.asarray(a, dtype=np.int64),
            np.asarray(r, dtype=np.float32), np.asarray(s2, dtype=np.float32),
            np.asarray(d, dtype=np.float32))


def train(variant: str, seed: int, episodes: int = 250,
          gamma: float = 0.99, lr: float = 5e-4, batch: int = 64,
          target_sync: int = 200, alpha: float = 0.6, beta: float = 0.4) -> Result:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = pick_device()

    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")
    obs_dim = int(env.observation_space.shape[0])
    n_actions = int(env.action_space.n)

    use_double = variant in ("double", "dueling", "per")
    use_dueling = variant in ("dueling", "per")
    use_per = variant == "per"

    net_cls = DuelingNet if use_dueling else QNet
    q = net_cls(obs_dim, n_actions).to(device)
    target = net_cls(obs_dim, n_actions).to(device)
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)

    if use_per:
        buf = SumTree(50000)
    else:
        buf = Replay()

    eps_start, eps_end, decay = 1.0, 0.02, 5000
    step = 0
    result = Result(name="dqn_improve_%s_s%d" % (variant, seed),
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
            item = (obs, action, reward, nxt, float(term))
            if use_per:
                # 新样本先给当前最大优先级，保证至少被抽到一次
                if buf.size == 0:
                    buf.add(1.0, item)
                else:
                    buf.add(float(buf.tree.max()), item)
            else:
                buf.add(*item)
            obs = nxt
            total += reward
            length += 1

            if len(buf) >= 1000:
                if use_per:
                    b = min(1.0, beta + (1.0 - beta) * step / 60000)
                    idx, leaves = buf.sample(batch, rng)
                    items = [buf.data[i] for i in idx]
                    s, a, r, s2, d = flat(items)
                    weights = (len(buf) * (buf.tree[leaves] / buf.tree[0])) ** (-b)
                    weights = np.asarray(weights / weights.max(), dtype=np.float32)
                    w = torch.as_tensor(weights, device=device)
                else:
                    s, a, r, s2, d = buf.sample(batch, rng)
                    w = None

                s = torch.as_tensor(s, device=device)
                a = torch.as_tensor(a, device=device)
                r = torch.as_tensor(r, device=device)
                s2 = torch.as_tensor(s2, device=device)
                d = torch.as_tensor(d, device=device)

                with torch.no_grad():
                    if use_double:
                        best = q(s2).argmax(dim=1, keepdim=True)
                        q_next = target(s2).gather(1, best).squeeze(1)
                    else:
                        q_next = target(s2).max(dim=1).values
                    y = r + gamma * (1.0 - d) * q_next

                pred = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
                td = pred - y
                if w is None:
                    loss = nn.functional.smooth_l1_loss(pred, y)
                else:
                    loss = (w * nn.functional.smooth_l1_loss(pred, y, reduction="none")).mean()

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

                if use_per:
                    buf.update(leaves, td.detach().abs().cpu().numpy() + 1e-6)

                if step % target_sync == 0:
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
    torch.set_num_threads(1)
    args = sys.argv[1:]
    RESULTS.mkdir(exist_ok=True)
    if not args or args[0] == "all":
        jobs = [(v, s) for v in VARIANTS for s in (0, 1, 2)]
    elif len(args) == 2:
        jobs = [(args[0], int(args[1]))]
    else:
        raise SystemExit("usage: run_dqn_improve.py base 0 | all")

    for variant, seed in jobs:
        t0 = time.time()
        res = train(variant, seed)
        path = res.save(RESULTS)
        print("%-8s seed=%d  eval=%5.1f  (%4.0fs)  -> %s"
              % (variant, seed, res.extra["eval_return"], time.time() - t0, path.name),
              flush=True)


if __name__ == "__main__":
    main()
