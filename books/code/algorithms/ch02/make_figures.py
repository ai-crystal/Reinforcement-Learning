# -*- coding: utf-8 -*-
"""第 2 章插图：三种备份图。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt

from figstyle import DEEP, GOLD, GRAY, INK, save, setup

R_S, R_A = 0.125, 0.07


def state(ax, x, y, text=""):
    ax.add_patch(plt.Circle((x, y), R_S, facecolor="white", edgecolor=DEEP, lw=1.4))
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=8, color=INK)


def action(ax, x, y, color=DEEP, filled=True):
    ax.add_patch(plt.Circle((x, y), R_A, facecolor=color if filled else "white",
                            edgecolor=color, lw=1.2))


def edge(ax, p0, p1, color=GRAY, lw=0.9, arrow=False):
    if arrow:
        ax.annotate("", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))
    else:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, zorder=0)


setup()
fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.4))
titles = ["$V^\\pi$：按策略分叉", "$Q^\\pi$：按环境分叉", "$V^*$：取最好的动作"]
for ax, title in zip(axes, titles):
    ax.set_xlim(-0.15, 2.45)
    ax.set_ylim(0.05, 2.75)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9.2, color=INK, pad=4)

# 左：状态 -> 三个动作 -> 每个动作两个下一状态
ax = axes[0]
state(ax, 0.15, 1.4, "s")
for dy in (0.62, 0.0, -0.62):
    action(ax, 0.9, 1.4 + dy)
    edge(ax, (0.30, 1.4), (0.82, 1.4 + dy))
    for ddy in (0.16, -0.16):
        state(ax, 1.75, 1.4 + dy + ddy)
        edge(ax, (0.98, 1.4 + dy), (1.60, 1.4 + dy + ddy))

# 中：动作 -> 三个下一状态 -> 每个状态两个动作
ax = axes[1]
action(ax, 0.15, 1.4)
for dy in (0.62, 0.0, -0.62):
    state(ax, 0.9, 1.4 + dy, "s'")
    edge(ax, (0.23, 1.4), (0.75, 1.4 + dy))
    for ddy in (0.16, -0.16):
        action(ax, 1.72, 1.4 + dy + ddy)
        edge(ax, (1.05, 1.4 + dy), (1.64, 1.4 + dy + ddy))

# 右：只保留最好的动作
ax = axes[2]
state(ax, 0.15, 1.4, "s")
for dy in (0.62, -0.62):
    action(ax, 0.9, 1.4 + dy, color="#B9C2C9", filled=False)
    edge(ax, (0.30, 1.4), (0.82, 1.4 + dy), color="#C9D0D5")
action(ax, 0.9, 1.4, color=GOLD)
state(ax, 1.8, 1.4, "s'")
edge(ax, (0.98, 1.4), (1.64, 1.4), color=GOLD, lw=1.6, arrow=True)
ax.text(1.38, 0.95, "只留最好的一条", fontsize=7.4, color=GOLD, ha="center")

fig.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.02, wspace=0.10)
save(fig, "algorithms", "02", "fig02-1.png")
