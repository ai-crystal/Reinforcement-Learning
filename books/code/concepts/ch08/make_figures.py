# -*- coding: utf-8 -*-
"""第 8 章插图：蒙特卡洛树搜索与自我对弈循环。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figstyle import (DEEP, DEEP_BG, DEEP_L, GOLD, GOLD_L, GRAY, INK, WHITE,
                      arrow, box, diagram, label, save)

# ---------------------------------------------------------------- 图 8-1
fig, ax = diagram(6.4, 2.9, xlim=(0, 100), ylim=(0, 48))

items = [
    (10, 27, "选择", "沿统计量往下走", DEEP_BG, DEEP_L),
    (57, 27, "扩展", "加一个新分支", DEEP_BG, DEEP_L),
    (57, 5, "模拟", "快速下完一盘", WHITE, GRAY),
    (10, 5, "回传", "把胜负记回去", WHITE, GRAY),
]
for x, y, n, note, fc, ec in items:
    box(ax, x, y, 33, 12, "", fc=fc, ec=ec, rounded=1.2)
    label(ax, x + 16.5, y + 8.2, n, fs=9.4, weight="bold")
    label(ax, x + 16.5, y + 3.8, note, fs=7.2, color=GRAY)

arrow(ax, (43.6, 33.0), (56.4, 33.0), color=DEEP, mutation=10)
arrow(ax, (73.5, 26.6), (73.5, 17.4), color=DEEP, mutation=10)
arrow(ax, (56.4, 11.0), (43.6, 11.0), color=DEEP, mutation=10)
arrow(ax, (26.5, 17.4), (26.5, 26.6), color=DEEP, mutation=10)

label(ax, 50, 43.0, "重复成千上万次，统计量向有希望的分支倾斜", fs=8.0, color=INK)

save(fig, "concepts", "08", "fig08-1.png")

# ---------------------------------------------------------------- 图 8-2
fig, ax = diagram(6.6, 3.0, xlim=(0, 100), ylim=(0, 48))

box(ax, 2, 30, 22, 10, "策略网络：往哪里想", fc=DEEP_BG, ec=DEEP_L, fs=7.6)
box(ax, 2, 8, 22, 10, "价值网络：局面好不好", fc=GOLD_L, ec=GOLD, fs=7.6)
box(ax, 33, 19, 22, 10, "搜索：把两者结合", fc=WHITE, ec=GRAY, fs=7.8)
arrow(ax, (24.4, 35.0), (32.6, 27.4), color=GRAY)
arrow(ax, (24.4, 13.0), (32.6, 20.6), color=GRAY)

box(ax, 62, 19, 16, 10, "对局与胜负", fc=WHITE, ec=GRAY, fs=7.6)
arrow(ax, (55.4, 24.0), (61.6, 24.0), color=GRAY)

box(ax, 62, 3, 16, 10, "训练网络", fc=DEEP_BG, ec=DEEP_L, fs=7.8)
arrow(ax, (70.0, 18.6), (70.0, 13.4), color=GRAY)
arrow(ax, (61.6, 8.0), (24.6, 8.0), color=DEEP, mutation=10)
arrow(ax, (13.0, 8.0), (13.0, 29.6), color=DEEP, mutation=10)
label(ax, 43.0, 5.4, "自我对弈产生更好的数据，两个网络一起更新", fs=7.0, color=DEEP)

box(ax, 83, 19, 15, 10, "更强的搜索\n与判断", fc="#F4F6F7", ec=GRAY, fs=7.4,
    ls=(0, (3, 2)))
arrow(ax, (80.4, 24.0), (82.6, 24.0), color=GRAY, ls=(0, (3, 2)))

save(fig, "concepts", "08", "fig08-2.png")
