# -*- coding: utf-8 -*-
"""第 5 章插图：Q 学习的训练曲线与学到的策略。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from figstyle import (DEEP, GOLD, GRAY, INK, LGRAY, canvas, save, setup)
from gridworld import ACTIONS, MAP
from plots import load, moving

data = load("gridworld_q.json")[0]
history = data["history"]
Q = np.asarray(data["Q"])
h, w = len(MAP), len(MAP[0])

# ---------------------------------------------------------------- 图 5-1
setup()
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))
fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.20, wspace=0.30)

success = moving(history["success"], 25)
axes[0].plot(np.arange(len(success)) + 12, success * 100, color=DEEP, lw=1.4)
axes[0].set_xlabel("训练回合")
axes[0].set_ylabel("成功率（%）")
axes[0].set_xlim(0, 500)
axes[0].set_ylim(0, 105)

steps = moving(history["steps"], 25)
axes[1].plot(np.arange(len(steps)) + 12, steps, color=GOLD, lw=1.4)
axes[1].set_xlabel("训练回合")
axes[1].set_ylabel("每局走的步数")
axes[1].set_xlim(0, 500)
axes[1].set_ylim(0, 110)

save(fig, "hands-on", "05", "fig05-1.png")

# ---------------------------------------------------------------- 图 5-2
setup()
fig, ax = plt.subplots(figsize=(6.0, 3.5))
fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03)

values = Q.max(axis=1).reshape(h, w)
vmax = max(abs(values).max(), 1e-6)
for r in range(h):
    for c in range(w):
        y = h - 1 - r
        ch = MAP[r][c]
        if ch == "#":
            ax.add_patch(Rectangle((c, y), 1, 1, facecolor="#8D9AA5",
                                   edgecolor="white", lw=1.4))
            continue
        v = values[r, c] / vmax
        color = plt.cm.RdYlGn((v + 1) / 2)
        ax.add_patch(Rectangle((c, y), 1, 1, facecolor=color,
                               edgecolor="white", lw=1.4))
        if ch == "G":
            ax.text(c + 0.5, y + 0.5, "终点", ha="center", va="center",
                    fontsize=10, color=INK)
            continue
        state = r * w + c
        a = int(np.argmax(Q[state]))
        dr, dc = ACTIONS[a]
        cx, cy = c + 0.5, y + 0.5
        ax.annotate("", xy=(cx + dc * 0.32, cy - dr * 0.32),
                    xytext=(cx - dc * 0.20, cy + dr * 0.20),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))

ax.set_xlim(0, w)
ax.set_ylim(0, h)
ax.set_aspect("equal")
ax.axis("off")

save(fig, "hands-on", "05", "fig05-2.png")
