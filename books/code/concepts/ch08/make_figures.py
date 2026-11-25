# -*- coding: utf-8 -*-
"""第 8 章插图：蒙特卡洛树搜索与自我对弈循环。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figstyle import (DEEP, DEEP_BG, DEEP_L, GOLD, GOLD_L, GRAY, INK, WHITE,
                      arrow, box, diagram, label, line, save)

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
label(ax, 43.0, 5.4, "自我对弈产生更好的数据，再拿去训练网络", fs=7.0, color=DEEP)

box(ax, 83, 19, 15, 10, "更强的搜索\n与判断", fc="#F4F6F7", ec=GRAY, fs=7.4,
    ls=(0, (3, 2)))
arrow(ax, (80.4, 24.0), (82.6, 24.0), color=GRAY, ls=(0, (3, 2)))

save(fig, "concepts", "08", "fig08-2.png")

# ---------------------------------------------------------------- 图 8-3
fig, ax = diagram(6.8, 3.2, xlim=(0, 100), ylim=(0, 50))


def card(x, y, w, h, title, note, fc=WHITE, ec=GRAY):
    box(ax, x, y, w, h, "", fc=fc, ec=ec, rounded=1.0)
    label(ax, x + w / 2, y + h - 3.0, title, fs=7.6, weight="bold")
    label(ax, x + w / 2, y + 2.6, note, fs=6.8, color=GRAY)


label(ax, 24, 46.0, "AlphaGo：三步串行", fs=8.4, color=INK, weight="bold")
card(3, 33, 18, 8, "人类棋谱", "三千万个局面")
card(25, 33, 18, 8, "策略网络", "模仿人类选点", DEEP_BG, DEEP_L)
card(25, 21, 18, 8, "自我对弈强化", "赢的走法加概率", DEEP_BG, DEEP_L)
card(3, 21, 18, 8, "策略网络（定稿）", "之后不再变")
card(3, 9, 18, 8, "用它生成对局", "输赢有标签")
card(25, 9, 18, 8, "价值网络", "预测胜率", GOLD_L, GOLD)
arrow(ax, (21.4, 37.0), (24.6, 37.0), color=GRAY)
arrow(ax, (34.0, 32.6), (34.0, 29.4), color=GRAY)
arrow(ax, (24.6, 25.0), (21.4, 25.0), color=GRAY)
arrow(ax, (12.0, 20.6), (12.0, 17.4), color=GRAY)
arrow(ax, (21.4, 13.0), (24.6, 13.0), color=GRAY)
label(ax, 24, 3.2, "后一步开始时，前一步已经冻结", fs=7.0, color=GRAY)

label(ax, 76, 46.0, "AlphaGo Zero：两个头一起训练", fs=8.4, color=INK, weight="bold")
card(52, 30, 18, 10, "共享网络", "卷积主干", DEEP_BG, DEEP_L)
card(78, 32, 19, 8, "策略头", "输出落子概率", GOLD_L, GOLD)
card(78, 22, 19, 8, "价值头", "输出胜率", GOLD_L, GOLD)
card(54, 10, 40, 8, "自我对弈 + 搜索", "每步 1600 次模拟")
arrow(ax, (70.4, 36.0), (77.6, 36.0), color=GRAY)
arrow(ax, (70.4, 33.0), (77.6, 26.0), color=GRAY)
arrow(ax, (87.5, 21.6), (87.5, 17.4), color=GRAY)
arrow(ax, (54.0, 14.0), (46.0, 14.0), color=GRAY)
arrow(ax, (46.0, 14.0), (46.0, 35.0), color=GRAY)
arrow(ax, (46.0, 35.0), (51.6, 35.0), color=GRAY)
label(ax, 47.6, 23.0, "访问分布和胜负同时当目标", fs=6.8, color=DEEP, ha="left")

save(fig, "concepts", "08", "fig08-3.png")

# ---------------------------------------------------------------- 图 8-4
fig, ax = diagram(6.6, 3.0, xlim=(0, 100), ylim=(0, 48))

box(ax, 2, 30, 16, 9, "", fc=WHITE, ec=GRAY)
label(ax, 10, 36.4, "棋盘局面", fs=7.4)
label(ax, 10, 32.4, "输入", fs=6.8, color=GRAY)

box(ax, 24, 25, 22, 19, "", fc=DEEP_BG, ec=DEEP_L)
label(ax, 35, 39.0, "共享主干", fs=7.8, weight="bold")
label(ax, 35, 35.0, "卷积层", fs=6.8, color=GRAY)
label(ax, 35, 31.4, "只有一份参数", fs=6.8, color=GRAY)

box(ax, 54, 35, 22, 9, "", fc=GOLD_L, ec=GOLD)
label(ax, 65, 41.4, "策略头", fs=7.4, weight="bold")
label(ax, 65, 37.4, "落子概率", fs=6.8, color=GRAY)

box(ax, 54, 24, 22, 9, "", fc=GOLD_L, ec=GOLD)
label(ax, 65, 30.4, "价值头", fs=7.4, weight="bold")
label(ax, 65, 26.4, "胜率", fs=6.8, color=GRAY)

box(ax, 82, 35, 16, 9, "", fc=WHITE, ec=GRAY)
label(ax, 90, 41.4, "策略损失", fs=7.0)
label(ax, 90, 37.4, "对比搜索分布", fs=6.4, color=GRAY)

box(ax, 82, 24, 16, 9, "", fc=WHITE, ec=GRAY)
label(ax, 90, 30.4, "价值损失", fs=7.0)
label(ax, 90, 26.4, "对比真实胜负", fs=6.4, color=GRAY)

arrow(ax, (18.4, 34.5), (23.6, 34.5), color=GRAY)
arrow(ax, (46.4, 39.5), (53.6, 39.5), color=GRAY)
arrow(ax, (46.4, 28.5), (53.6, 28.5), color=GRAY)
arrow(ax, (76.4, 39.5), (81.6, 39.5), color=GRAY)
arrow(ax, (76.4, 28.5), (81.6, 28.5), color=GRAY)

line(ax, (90.0, 23.6), (90.0, 8.0), color=DEEP, lw=0.9, ls=(0, (3, 2)))
line(ax, (90.0, 8.0), (35.0, 8.0), color=DEEP, lw=0.9, ls=(0, (3, 2)))
arrow(ax, (35.0, 8.0), (35.0, 24.6), color=DEEP, ls=(0, (3, 2)), mutation=8)
label(ax, 63.0, 5.6, "两个损失各自反传，梯度在主干里相加", fs=6.8, color=DEEP)

save(fig, "concepts", "08", "fig08-4.png")
