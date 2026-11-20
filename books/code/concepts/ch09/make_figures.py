# -*- coding: utf-8 -*-
"""第 9 章插图：价值路线与策略路线。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figstyle import (DEEP, DEEP_BG, DEEP_L, GOLD, GOLD_L, GRAY, INK, WHITE,
                      arrow, box, diagram, label, save)

fig, ax = diagram(6.8, 3.0, xlim=(0, 100), ylim=(0, 48))

# 左：价值路线
box(ax, 2, 36, 44, 7, "价值路线：先打分", fc=DEEP_BG, ec=DEEP_L, fs=8.4, weight="bold")
box(ax, 4, 22, 15, 9, "状态", fc=WHITE, ec=GRAY, fs=7.8)
box(ax, 24, 22, 18, 9, "给每个动作\n打分", fc=DEEP_BG, ec=DEEP_L, fs=7.4)
box(ax, 4, 7, 15, 9, "选分数最高\n的动作", fc=WHITE, ec=GRAY, fs=7.4)
arrow(ax, (19.4, 26.5), (23.6, 26.5), color=GRAY)
arrow(ax, (33.0, 21.6), (15.0, 16.4), color=GRAY)
label(ax, 30.0, 14.0, "只能比较有限个候选", fs=7.0, color=GRAY, ha="left")

# 右：策略路线
box(ax, 54, 36, 44, 7, "策略路线：直接调概率", fc=GOLD_L, ec=GOLD, fs=8.4, weight="bold")
box(ax, 56, 22, 15, 9, "状态", fc=WHITE, ec=GRAY, fs=7.8)
box(ax, 76, 22, 20, 9, "输出一组概率\n左 3 成，右 7 成", fc=GOLD_L, ec=GOLD, fs=7.2)
arrow(ax, (71.4, 26.5), (75.6, 26.5), color=GRAY)
box(ax, 56, 7, 40, 9, "按概率抽一个动作，结果好坏再回头调概率",
    fc=WHITE, ec=GRAY, fs=7.2)
arrow(ax, (86.0, 21.6), (86.0, 16.4), color=GOLD, mutation=9)

label(ax, 76, 2.6, "连续动作和随机策略都装得下", fs=7.0, color=GRAY)

save(fig, "concepts", "09", "fig09-1.png")
