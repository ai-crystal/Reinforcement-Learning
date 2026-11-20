# -*- coding: utf-8 -*-
"""第 1 章插图。运行： python books/code/concepts/ch01/make_figures.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figstyle import (DEEP, DEEP_BG, DEEP_L, DEEP_XL, GOLD, GOLD_L, GRAY, INK,
                      WHITE, arrow, box, diagram, label, save)

# ---------------------------------------------------------------- 图 1-1
fig, ax = diagram(6.8, 3.0, xlim=(0, 100), ylim=(0, 50))

# 左：有标准答案
box(ax, 3, 41, 43, 7, "监督学习：有标准答案", fc=DEEP_XL, ec=DEEP_L, fs=8.6, weight="bold")
box(ax, 4, 28, 17, 8, "样本（带标签）", fc=WHITE, ec=GRAY, fs=7.8)
box(ax, 28, 28, 15, 8, "模型", fc=DEEP_BG, ec=DEEP_L, fs=8.6)
box(ax, 28, 13, 15, 7, "预测", fc=WHITE, ec=GRAY, fs=8.2)
box(ax, 4, 13, 17, 7, "与标签比较", fc=WHITE, ec=GRAY, fs=7.8)
arrow(ax, (21.4, 32.0), (27.6, 32.0), color=GRAY)
label(ax, 24.5, 34.0, "训练", fs=7.2, color=GRAY)
arrow(ax, (35.5, 27.6), (35.5, 20.4), color=GRAY)
arrow(ax, (27.6, 16.5), (21.4, 16.5), color=GRAY)
arrow(ax, (13.0, 20.4), (29.0, 27.7), color=DEEP, rad=-0.22)
label(ax, 8.0, 24.4, "更新参数", fs=7.2, color=DEEP, ha="left")

# 右：只有后果
box(ax, 53, 41, 44, 7, "强化学习：只有后果", fc=GOLD_L, ec=GOLD, fs=8.6, weight="bold")
box(ax, 60, 29, 30, 8, "智能体（当前策略）", fc=DEEP_BG, ec=DEEP_L, fs=8.2)
box(ax, 60, 12, 30, 8, "环境", fc=WHITE, ec=GRAY, fs=8.6)
arrow(ax, (65.0, 28.6), (65.0, 20.4), color=DEEP, mutation=10)
label(ax, 63.4, 24.5, "动作", fs=7.6, color=DEEP, ha="right")
arrow(ax, (85.0, 20.4), (85.0, 28.6), color=GOLD, mutation=10)
label(ax, 86.6, 24.5, "状态、奖励", fs=7.4, color=GOLD, ha="left")

save(fig, "concepts", "01", "fig01-1.png")
