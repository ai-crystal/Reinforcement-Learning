# -*- coding: utf-8 -*-
"""第 16 章插图：分布式强化学习的三段结构（示意图，不是实验数据）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figstyle import (BG, DEEP, DEEP_L, DEEP_XL, GOLD, GRAY, LGRAY, WHITE,
                      arrow, box, diagram, label, line, rect, save)

fig, ax = diagram(6.8, 3.2)

# ---- 左侧：actor 池 ----
actor_y = (31.6, 22.6, 13.6, 4.6)
for i, y in enumerate(actor_y):
    box(ax, 2, y, 18, 6.6, "actor %d" % (i + 1), fc=DEEP_XL, ec=DEEP_L, fs=7.6)

# ---- 中间：共享回放池 ----
box(ax, 38, 6, 20, 32.2, "", fc=WHITE, ec=DEEP, lw=1.0)
label(ax, 48, 35.6, "共享回放池", fs=7.8, weight="bold")
for i in range(4):
    rect(ax, 40.2, 8.4 + i * 6.4, 15.6, 3.8, fc=DEEP_XL, ec="none", zorder=4)
    label(ax, 48, 10.3 + i * 6.4, "经验 + 优先级", fs=6.4, color=DEEP, zorder=5)

# ---- 右侧：learner ----
box(ax, 74, 17.2, 22, 9.6, "learner\n采样并更新网络", fc=DEEP, ec=DEEP, tc=WHITE, fs=7.6)

# ---- 箭头：actor -> 回放池 ----
for y in actor_y:
    arrow(ax, (20, y + 3.3), (37.4, y + 3.3), color=DEEP_L, lw=0.9)

# ---- 箭头：回放池 -> learner ----
arrow(ax, (58, 22), (73.4, 22), color=GOLD, lw=1.1)
label(ax, 65.6, 24.2, "采样一批", fs=6.8, color=GOLD)

# ---- 箭头：参数广播回 actor ----
line(ax, (85, 26.8), (85, 45.6), color=GRAY, lw=0.9, ls="--")
line(ax, (85, 45.6), (11, 45.6), color=GRAY, lw=0.9, ls="--")
arrow(ax, (11, 45.6), (11, 38.6), color=GRAY, lw=0.9, ls="--")
label(ax, 48, 47.6, "参数同步（每 N 步）", fs=6.8, color=GRAY)

label(ax, 22, 1.4, "每个 actor 各持一份环境副本", fs=6.6, color=GRAY)

save(fig, "algorithms", "16", "fig16-1.png")
