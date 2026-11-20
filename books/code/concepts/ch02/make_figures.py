# -*- coding: utf-8 -*-
"""第 2 章插图。运行： python books/code/concepts/ch02/make_figures.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figstyle import (BG, DEEP, DEEP_BG, DEEP_L, DEEP_XL, GOLD, GOLD_L, GRAY,
                      INK, LGRAY, WHITE, arrow, box, diagram, label, line, save)

# ---------------------------------------------------------------- 图 2-1
fig, ax = diagram(6.6, 2.7, xlim=(0, 100), ylim=(0, 50))

box(ax, 5, 15, 26, 22, fc=DEEP_BG, ec=DEEP_L, rounded=1.6)
label(ax, 18, 31.4, "智能体", fs=10.0, weight="bold")
label(ax, 18, 26.6, "策略：看到状态", fs=7.6, color=INK)
label(ax, 18, 22.8, "决定动作", fs=7.6, color=INK)
label(ax, 18, 18.6, "（可能带一点随机）", fs=7.0, color=GRAY)

box(ax, 66, 15, 27, 22, fc=WHITE, ec=GRAY, rounded=1.6)
label(ax, 79.5, 31.4, "环境", fs=10.0, weight="bold")
label(ax, 79.5, 26.6, "按自己的规则变化", fs=7.6, color=INK)
label(ax, 79.5, 22.8, "并给出奖励", fs=7.6, color=INK)
label(ax, 79.5, 18.6, "（规则不受智能体控制）", fs=7.0, color=GRAY)

# 上面：动作
arrow(ax, (32.6, 30.0), (64.4, 30.0), color=DEEP, lw=1.3, mutation=11)
label(ax, 48.5, 33.0, "动作 a", fs=8.2, color=DEEP)

# 下面：状态与奖励
arrow(ax, (64.4, 21.0), (32.6, 21.0), color=GOLD, lw=1.3, mutation=11)
label(ax, 48.5, 17.6, "状态 s' 与奖励 r", fs=8.2, color=GOLD)

label(ax, 48.5, 44.0, "第 t 步", fs=7.6, color=GRAY)
label(ax, 48.5, 7.0, "第 t+1 步：重复这个过程", fs=7.6, color=GRAY)

save(fig, "concepts", "02", "fig02-1.png")
