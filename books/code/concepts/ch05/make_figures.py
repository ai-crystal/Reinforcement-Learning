# -*- coding: utf-8 -*-
"""第 5 章插图：网格世界的状态价值（值迭代真实计算）。

环境设定：
  5×5 网格，右上角出口 +1，右下角陷阱 −1，两者都是终止状态
  每走一步有 −0.04 的移动代价
  动作有 10% 的概率滑向另外三个方向中的一个（各 1/30）
  折扣因子 0.9

运行： python books/code/concepts/ch05/make_figures.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt

from figstyle import INK, WHITE, save, setup

N = 5
GAMMA = 0.9
STEP_COST = -0.04
SLIP = 0.1
GOAL = (0, N - 1)
TRAP = (N - 1, N - 1)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

V = np.zeros((N, N))


def move(r, c, dr, dc):
    nr, nc = r + dr, c + dc
    if nr < 0 or nr >= N or nc < 0 or nc >= N:
        return r, c
    return nr, nc


def reward_of(cell):
    if cell == GOAL:
        return 1.0
    if cell == TRAP:
        return -1.0
    return STEP_COST


for _ in range(500):
    new = V.copy()
    for r in range(N):
        for c in range(N):
            if (r, c) in (GOAL, TRAP):
                continue
            best = -1e9
            for i, act in enumerate(ACTIONS):
                value = 0.0
                others = [a for j, a in enumerate(ACTIONS) if j != i]
                outcomes = [(act, 1.0 - SLIP)] + [(a, SLIP / 3.0) for a in others]
                for a, p in outcomes:
                    nr, nc = move(r, c, *a)
                    value += p * (reward_of((nr, nc)) + GAMMA * V[nr, nc])
                best = max(best, value)
            new[r, c] = best
    if np.max(np.abs(new - V)) < 1e-10:
        V = new
        break
    V = new

V[GOAL] = 1.0
V[TRAP] = -1.0

setup()
fig, ax = plt.subplots(figsize=(5.0, 5.0))
fig.subplots_adjust(left=0.04, right=0.96, top=0.94, bottom=0.06)
im = ax.imshow(V, cmap="RdYlGn", vmin=-1, vmax=1)
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)
for r in range(N):
    for c in range(N):
        if (r, c) == GOAL:
            txt, col = "出口\n+1", WHITE
        elif (r, c) == TRAP:
            txt, col = "陷阱\n−1", WHITE
        else:
            txt, col = "%.2f" % V[r, c], INK
        ax.text(c, r, txt, ha="center", va="center", fontsize=11.5, color=col)

for spine in ax.spines.values():
    spine.set_visible(False)

save(fig, "concepts", "05", "fig05-1.png")
print(np.round(V, 3))
