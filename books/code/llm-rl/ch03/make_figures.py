# -*- coding: utf-8 -*-
"""第 3 章插图：PPO 训练大模型时的四个模型。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figstyle import (DEEP, DEEP_BG, DEEP_L, GOLD, GOLD_L, GRAY, INK, WHITE,
                      arrow, box, diagram, label, save)

fig, ax = diagram(6.8, 3.0, xlim=(0, 100), ylim=(0, 48))

box(ax, 3, 17, 24, 14, "", fc=DEEP_BG, ec=DEEP_L, rounded=1.2)
label(ax, 15, 26.4, "策略模型", fs=9.0, weight="bold")
label(ax, 15, 21.6, "唯一要训练的", fs=7.2, color=GRAY)

items = [
    (34, "奖励模型", "给整段回答打分", GOLD_L, GOLD),
    (20, "参考模型", "冻结的旧版本，用来算 KL", "#F4F6F7", GRAY),
    (6, "价值模型", "估计每一步的价值", WHITE, GRAY),
]
for y, name, note, fc, ec in items:
    box(ax, 47, y, 34, 11, "", fc=fc, ec=ec, rounded=1.2)
    label(ax, 64, y + 7.6, name, fs=8.2, weight="bold")
    label(ax, 64, y + 3.4, note, fs=7.0, color=GRAY)

for y in (39, 25, 11):
    arrow(ax, (27.4, 24.0), (46.6, y), color=GRAY)

label(ax, 36.5, 40.6, "生成回答", fs=7.0, color=GRAY)
label(ax, 36.4, 23.4, "比 KL", fs=7.0, color=GRAY)
label(ax, 36.4, 8.4, "估值", fs=7.0, color=GRAY)

label(ax, 50, 1.6, "四个模型同时在显存里，这是 RLHF 最贵的地方", fs=7.8, color=INK)

save(fig, "llm-rl", "03", "fig03-1.png")
