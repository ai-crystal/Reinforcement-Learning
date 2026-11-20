# -*- coding: utf-8 -*-
"""第 4 章插图：网格世界的地图。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from figstyle import DEEP, DEEP_BG, GOLD, GOLD_L, GRAY, INK, LGRAY, RED, save, setup
from gridworld import MAP

h, w = len(MAP), len(MAP[0])

setup()
fig, ax = plt.subplots(figsize=(6.0, 3.4))
fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.10)

for r in range(h):
    for c in range(w):
        ch = MAP[r][c]
        if ch == "#":
            ax.add_patch(Rectangle((c, h - 1 - r), 1, 1, facecolor="#8D9AA5",
                                   edgecolor="white", lw=1.4))
        else:
            ax.add_patch(Rectangle((c, h - 1 - r), 1, 1, facecolor="white",
                                   edgecolor=LGRAY, lw=0.8))
        if ch == "S":
            ax.text(c + 0.5, h - 1 - r + 0.5, "起点", ha="center", va="center",
                    fontsize=11, color=DEEP, weight="bold")
        elif ch == "G":
            ax.add_patch(Rectangle((c, h - 1 - r), 1, 1, facecolor=GOLD_L,
                                   edgecolor="white", lw=1.4))
            ax.text(c + 0.5, h - 1 - r + 0.5, "终点\n+1", ha="center", va="center",
                    fontsize=10, color=INK)
        elif ch == "X":
            ax.add_patch(Rectangle((c, h - 1 - r), 1, 1, facecolor="#EBD9D6",
                                   edgecolor="white", lw=1.4))
            ax.text(c + 0.5, h - 1 - r + 0.5, "陷阱\n−1", ha="center", va="center",
                    fontsize=9.5, color=RED)

ax.set_xlim(0, w)
ax.set_ylim(0, h)
ax.set_aspect("equal")
ax.axis("off")

save(fig, "hands-on", "04", "fig04-1.png")
