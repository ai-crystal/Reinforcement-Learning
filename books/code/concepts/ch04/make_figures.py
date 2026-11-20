# -*- coding: utf-8 -*-
"""第 4 章插图：十臂老虎机上的 ε-贪心对比（真实实验，可复现）。

运行： python books/code/concepts/ch04/make_figures.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figstyle import DEEP, GOLD, TEAL, canvas, save


def run_bandit(eps: float, steps: int, runs: int, k: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    means = rng.normal(0.0, 1.0, size=(runs, k))
    Q = np.zeros((runs, k))
    N = np.zeros((runs, k))
    avg = np.zeros(steps)
    for t in range(steps):
        greedy = Q.argmax(axis=1)
        explore = rng.random(runs) < eps
        choice = np.where(explore, rng.integers(0, k, size=runs), greedy)
        reward = rng.normal(means[np.arange(runs), choice], 1.0)
        N[np.arange(runs), choice] += 1
        n = N[np.arange(runs), choice]
        Q[np.arange(runs), choice] += (reward - Q[np.arange(runs), choice]) / n
        avg[t] = reward.mean()
    return avg


fig, ax = canvas(6.4, 3.2)
steps = 1000
for eps, color, name in ((0.0, GOLD, "ε = 0（完全贪心）"),
                         (0.01, TEAL, "ε = 0.01"),
                         (0.1, DEEP, "ε = 0.1")):
    curve = run_bandit(eps, steps, runs=300, seed=7)
    kern = np.ones(11) / 11
    smooth = np.convolve(curve, kern, mode="valid")
    x = np.arange(len(smooth)) + 5
    ax.plot(x, smooth, color=color, lw=1.4, label=name)

ax.set_xlabel("投币次数")
ax.set_ylabel("这 11 次的平均奖励")
ax.set_xlim(0, steps)
ax.legend(frameon=False, loc="lower right")


save(fig, "concepts", "04", "fig04-1.png")
