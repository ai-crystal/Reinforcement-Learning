# -*- coding: utf-8 -*-
"""第 17 章插图：模仿与试错两条路径。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figstyle import (DEEP, DEEP_BG, DEEP_L, GOLD, GOLD_L, GRAY, INK, WHITE,
                      arrow, box, diagram, label, line, save)


def card(ax, x, y, w, h, title, note, fc=WHITE, ec=GRAY):
    box(ax, x, y, w, h, "", fc=fc, ec=ec, rounded=1.0)
    label(ax, x + w / 2, y + h - 3.2, title, fs=8.2, weight="bold")
    label(ax, x + w / 2, y + 3.0, note, fs=6.8, color=GRAY)


fig, ax = diagram(6.8, 2.9, xlim=(0, 100), ylim=(0, 46))

# 上：模仿
card(ax, 2, 31, 20, 11, "专家示范", "人做的记录")
card(ax, 30, 31, 22, 11, "模型照着学", "监督学习", DEEP_BG, DEEP_L)
card(ax, 62, 31, 22, 11, "会做类似的事", "但没见过出错怎么办")
arrow(ax, (22.4, 36.5), (29.6, 36.5), color=GRAY)
arrow(ax, (52.4, 36.5), (61.6, 36.5), color=GRAY)
label(ax, 45.0, 44.4, "模仿：看别人怎么做", fs=8.2, color=INK)

# 下：试错
card(ax, 2, 9, 20, 11, "自己去做", "先试一步")
card(ax, 30, 9, 22, 11, "拿到结果", "好或坏", GOLD_L, GOLD)
card(ax, 62, 9, 22, 11, "调动作的概率", "好的多来，坏的少来")
arrow(ax, (22.4, 14.5), (29.6, 14.5), color=GRAY)
arrow(ax, (52.4, 14.5), (61.6, 14.5), color=GRAY)
label(ax, 45.0, 23.6, "试错：从后果里学", fs=8.2, color=INK)

line(ax, (73.0, 8.6), (73.0, 4.2), color=GOLD, lw=1.0, ls=(0, (4, 2)))
line(ax, (73.0, 4.2), (12.0, 4.2), color=GOLD, lw=1.0, ls=(0, (4, 2)))
arrow(ax, (12.0, 4.2), (12.0, 8.6), color=GOLD, ls=(0, (4, 2)), mutation=8)
label(ax, 42.0, 1.4, "重复很多次", fs=6.8, color=GOLD)

save(fig, "concepts", "17", "fig17-1.png")
