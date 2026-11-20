# -*- coding: utf-8 -*-
"""第 3 章插图。运行： python books/code/concepts/ch03/make_figures.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figstyle import (DEEP, DEEP_L, GOLD, GRAY, INK, LGRAY, PALETTE, arrow,
                      box, canvas, diagram, label, save)
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------- 图 3-1
fig, ax = diagram(6.8, 2.9, xlim=(0, 100), ylim=(0, 50))

xs = [4, 20, 36, 52, 68]
for i, x in enumerate(xs):
    box(ax, x, 32, 11, 8, "s%d" % (i + 1), fc="white", ec=GRAY, fs=8.8)
    if i:
        arrow(ax, (x - 4.4, 36.0), (x - 0.5, 36.0), color=GRAY, mutation=8)

box(ax, 82, 32, 15, 8, "奖励 +10", fc="#FBF3E3", ec=GOLD, fs=8.4)
arrow(ax, (79.4, 36.0), (81.4, 36.0), color=GOLD, mutation=9)

# 每一步分到的信用，越靠后越多
weights = [1.0, 1.7, 2.7, 4.2, 6.4]
for i, (x, h) in enumerate(zip(xs, weights)):
    alpha = 0.35 + 0.13 * i
    ax.add_patch(Rectangle((x + 4.2, 16), 3.4, h, facecolor=DEEP, alpha=alpha,
                           edgecolor="none", zorder=3))
label(ax, 4, 27.0, "各步分到的信用", fs=7.4, color=GRAY, ha="left")
label(ax, 50, 44.0, "奖励沿路径回传，越靠近终点的决定分到越多", fs=8.0, color=INK)
label(ax, 50, 10.0, "越早的决定，影响越间接", fs=7.4, color=GRAY)

save(fig, "concepts", "03", "fig03-1.png")

# ---------------------------------------------------------------- 图 3-2
fig, ax = canvas(6.4, 3.0)
t = np.arange(0, 121)
for g, c, name in ((0.90, PALETTE[0], "γ = 0.90"),
                   (0.95, PALETTE[1], "γ = 0.95"),
                   (0.99, PALETTE[2], "γ = 0.99"),
                   (0.999, PALETTE[5], "γ = 0.999")):
    ax.plot(t, g ** t, color=c, lw=1.5, label=name)
ax.set_xlabel("距离当前这一步的步数")
ax.set_ylabel("该步奖励剩下的权重")
ax.set_ylim(0, 1.02)
ax.set_xlim(0, 120)
ax.legend(frameon=False, loc="upper right")
ax.axhline(0.5, color=LGRAY, lw=0.7, ls="--", zorder=0)

save(fig, "concepts", "03", "fig03-2.png")
