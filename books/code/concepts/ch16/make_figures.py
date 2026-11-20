# -*- coding: utf-8 -*-
"""第 16 章插图：自对弈的循环。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figstyle import (DEEP, DEEP_BG, DEEP_L, GOLD, GOLD_L, GRAY, INK, WHITE,
                      arrow, box, diagram, label, line, save)

fig, ax = diagram(6.6, 2.9, xlim=(0, 100), ylim=(0, 46))

box(ax, 4, 25, 22, 12, "", fc=DEEP_BG, ec=DEEP_L)
label(ax, 15, 33.0, "当前版本", fs=8.6, weight="bold")
label(ax, 15, 28.4, "正在训练的策略", fs=6.8, color=GRAY)

box(ax, 4, 7, 22, 12, "", fc=WHITE, ec=GRAY)
label(ax, 15, 15.0, "对手池", fs=8.6, weight="bold")
label(ax, 15, 10.4, "历史版本 + 噪声", fs=6.8, color=GRAY)

box(ax, 36, 15, 18, 12, "", fc=GOLD_L, ec=GOLD)
label(ax, 45, 23.0, "对局", fs=8.8, weight="bold")
label(ax, 45, 18.4, "成千上万盘", fs=6.8, color=GRAY)

box(ax, 62, 15, 16, 12, "", fc=WHITE, ec=GRAY)
label(ax, 70, 23.0, "胜负", fs=8.6, weight="bold")
label(ax, 70, 18.4, "只信结果", fs=6.8, color=GRAY)

box(ax, 84, 25, 14, 12, "", fc=DEEP_BG, ec=DEEP_L)
label(ax, 91, 33.0, "新版本", fs=8.2, weight="bold")
label(ax, 91, 28.4, "更强一点", fs=6.8, color=GRAY)

arrow(ax, (26.4, 31.0), (35.6, 24.6), color=GRAY)
arrow(ax, (26.4, 13.0), (35.6, 17.4), color=GRAY)
arrow(ax, (54.4, 21.0), (61.6, 21.0), color=GRAY)
arrow(ax, (78.4, 21.0), (84.4, 24.6), color=DEEP, mutation=9)

# 新版本进入对手池
line(ax, (91.0, 24.6), (91.0, 3.2), color=DEEP, lw=1.0)
line(ax, (91.0, 3.2), (15.0, 3.2), color=DEEP, lw=1.0)
arrow(ax, (15.0, 3.2), (15.0, 6.6), color=DEEP, mutation=9)
label(ax, 52.0, 1.2, "新版本也进入对手池", fs=6.8, color=DEEP)

label(ax, 50, 43.2, "自对弈：让对手和自己一起变强", fs=7.8, color=INK)

save(fig, "concepts", "16", "fig16-1.png")
