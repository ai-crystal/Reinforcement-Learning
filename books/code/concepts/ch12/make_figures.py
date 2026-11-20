# -*- coding: utf-8 -*-
"""第 12 章插图：RLHF 的三个阶段。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figstyle import (DEEP, DEEP_BG, DEEP_L, GOLD, GOLD_L, GRAY, INK, WHITE,
                      arrow, box, diagram, label, save)

fig, ax = diagram(6.8, 2.9, xlim=(0, 100), ylim=(0, 46))

stages = [
    (3, "第一步　监督微调", "拿一批好回答\n让模型照着学", DEEP_BG, DEEP_L, "学会怎样好好回答"),
    (35, "第二步　奖励模型", "人来比较两个回答\n训练一个打分模型", GOLD_L, GOLD, "学会判断好不好"),
    (67, "第三步　强化学习", "模型自己生成\n按分数调整行为", DEEP_BG, DEEP_L, "学会偏向高分回答"),
]
for x, title, body, fc, ec, note in stages:
    box(ax, x, 20, 28, 18, "", fc=fc, ec=ec, rounded=1.4)
    label(ax, x + 14, 33.4, title, fs=8.2, weight="bold")
    label(ax, x + 14, 26.6, body, fs=7.4, color=INK)
    label(ax, x + 14, 13.0, note, fs=7.2, color=GRAY)

arrow(ax, (31.4, 29.0), (34.6, 29.0), color=DEEP, mutation=10)
arrow(ax, (63.4, 29.0), (66.6, 29.0), color=DEEP, mutation=10)

label(ax, 50, 43.0, "预训练模型在第一步学会格式，第二步得到评价标准，第三步改变行为", fs=7.8, color=INK)

save(fig, "concepts", "12", "fig12-1.png")
