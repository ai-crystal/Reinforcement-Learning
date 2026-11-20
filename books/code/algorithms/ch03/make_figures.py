# -*- coding: utf-8 -*-
"""第 3 章插图：值迭代的收敛过程（真实计算）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from figstyle import DEEP, GOLD, GRAY, canvas, save

N, GAMMA = 5, 0.9
GOAL, TRAP = (0, N - 1), (N - 1, N - 1)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SLIP, STEP_COST = 0.1, -0.04


def move(r, c, dr, dc):
    nr, nc = r + dr, c + dc
    if 0 <= nr < N and 0 <= nc < N:
        return nr, nc
    return r, c


def reward_of(cell):
    return 1.0 if cell == GOAL else (-1.0 if cell == TRAP else STEP_COST)


V = np.zeros((N, N))
errors, deltas = [], []
for sweep in range(120):
    new = V.copy()
    for r in range(N):
        for c in range(N):
            if (r, c) in (GOAL, TRAP):
                continue
            best = -1e9
            for i, act in enumerate(ACTIONS):
                others = [a for j, a in enumerate(ACTIONS) if j != i]
                outcomes = [(act, 1 - SLIP)] + [(a, SLIP / 3) for a in others]
                q = 0.0
                for a, p in outcomes:
                    nr, nc = move(r, c, *a)
                    q += p * (reward_of((nr, nc)) + GAMMA * V[nr, nc])
                best = max(best, q)
            new[r, c] = best
    delta = float(np.max(np.abs(new - V)))
    V = new
    deltas.append(delta)
    if delta < 1e-10:
        break

fig, ax = canvas(6.4, 3.0)
ax.semilogy(np.arange(1, len(deltas) + 1), deltas, color=DEEP, lw=1.6)
ax.set_xlabel("扫描轮数")
ax.set_ylabel("这一轮的最大变化量（对数刻度）")
ax.set_xlim(1, len(deltas))
ax.axhline(1e-2, color=GRAY, lw=0.8, ls="--")

save(fig, "algorithms", "03", "fig03-1.png")
print("sweeps", len(deltas), "final", deltas[-1])
