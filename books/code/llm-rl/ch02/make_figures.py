# -*- coding: utf-8 -*-
"""第 2 章插图：从偏好比较到奖励模型。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figstyle import (DEEP, DEEP_BG, DEEP_L, DEEP_XL, GOLD, GOLD_L, GRAY, INK,
                      WHITE, arrow, box, diagram, label, save)

fig, ax = diagram(6.8, 2.9, xlim=(0, 100), ylim=(0, 46))

box(ax, 2, 30, 26, 12, "", fc=WHITE, ec=GRAY, rounded=1.2)
label(ax, 15, 38.4, "同一个问题", fs=7.4, color=GRAY)
label(ax, 15, 34.4, "回答 A 与回答 B", fs=8.0, color=INK)

box(ax, 36, 30, 22, 12, "", fc=GOLD_L, ec=GOLD, rounded=1.2)
label(ax, 47, 38.4, "人来判断", fs=7.4, color=GRAY)
label(ax, 47, 34.2, "哪个更好", fs=8.4, color=INK)

box(ax, 66, 30, 30, 12, "", fc=DEEP_BG, ec=DEEP_L, rounded=1.2)
label(ax, 81, 38.4, "奖励模型", fs=7.4, color=GRAY)
label(ax, 81, 34.2, "学会给回答打分", fs=8.2, color=INK)

arrow(ax, (28.4, 36.0), (35.6, 36.0), color=GRAY)
arrow(ax, (58.4, 36.0), (65.6, 36.0), color=GRAY)

box(ax, 36, 8, 22, 12, "", fc=WHITE, ec=GRAY, rounded=1.2)
label(ax, 47, 16.4, "比较结果", fs=7.4, color=GRAY)
label(ax, 47, 12.2, "A 更好 / B 更好", fs=8.0, color=INK)
arrow(ax, (47.0, 29.6), (47.0, 20.4), color=GOLD, mutation=9)
label(ax, 48.6, 25.0, "作为训练数据", fs=7.0, color=GOLD, ha="left")

label(ax, 82, 20.0, "打分时只看单个回答", fs=7.6, color=GRAY)
label(ax, 82, 16.0, "不需要人来现场比较", fs=7.6, color=GRAY)

save(fig, "llm-rl", "02", "fig02-1.png")
