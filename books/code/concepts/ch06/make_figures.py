# -*- coding: utf-8 -*-
"""第 6 章插图：策略迭代的循环。

运行： python books/code/concepts/ch06/make_figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figstyle import (DEEP, DEEP_BG, DEEP_L, GOLD, GRAY, INK, WHITE, arrow, box,
                      diagram, label, line, save)

fig, ax = diagram(6.4, 2.5, xlim=(0, 100), ylim=(0, 44))

box(ax, 6, 18, 26, 12, fc=DEEP_BG, ec=DEEP_L, rounded=1.4)
label(ax, 19, 26.0, "当前的策略", fs=8.6, weight="bold")
label(ax, 19, 21.6, "每个状态选哪个动作", fs=7.4, color=GRAY)

box(ax, 38, 18, 24, 12, fc=WHITE, ec=GRAY, rounded=1.4)
label(ax, 50, 26.0, "策略评估", fs=8.6, weight="bold")
label(ax, 50, 21.6, "算出每个状态的价值", fs=7.4, color=GRAY)

box(ax, 68, 18, 24, 12, fc=WHITE, ec=GRAY, rounded=1.4)
label(ax, 80, 26.0, "策略改进", fs=8.6, weight="bold")
label(ax, 80, 21.6, "换成价值更高的动作", fs=7.4, color=GRAY)

arrow(ax, (32.4, 24.0), (37.6, 24.0), color=DEEP, mutation=11)
arrow(ax, (62.4, 24.0), (67.6, 24.0), color=DEEP, mutation=11)

# 回到评估：从改进下方绕回
line(ax, (80.0, 17.4), (80.0, 7.0), color=GOLD, lw=1.1)
line(ax, (80.0, 7.0), (50.0, 7.0), color=GOLD, lw=1.1)
arrow(ax, (50.0, 7.0), (50.0, 17.2), color=GOLD, mutation=11)
label(ax, 65.0, 3.8, "策略变了，价值要重算", fs=7.4, color=GOLD)

label(ax, 50, 37.0, "评估和改进交替进行，价值越来越准，策略越来越好", fs=8.2, color=INK)

save(fig, "concepts", "06", "fig06-1.png")
