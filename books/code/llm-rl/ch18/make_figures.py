# -*- coding: utf-8 -*-
"""第 18 章插图：拒绝采样蒸馏的流水线。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figstyle import (DEEP, DEEP_BG, DEEP_L, GOLD, GOLD_L, GRAY, LGRAY, INK,
                      WHITE, arrow, box, diagram, label, save)

fig, ax = diagram(6.8, 3.15, xlim=(0, 100), ylim=(0, 52))

label(ax, 50, 48.6, "数据全部来自强模型的采样，验证器决定留下什么", fs=7.9, color=INK)

top = [
    (2.0, "强模型", "对每道题采样", DEEP_BG, DEEP_L),
    (36.0, "每题 k 条", "k 取 8 到 16", WHITE, GRAY),
    (70.0, "验证器筛选", "只留判对的，产率一成到五成", GOLD_L, GOLD),
]
bot = [
    (2.0, "评测", "四项指标一起报", WHITE, GRAY),
    (36.0, "监督微调", "学推理过程", DEEP_BG, DEEP_L),
    (70.0, "去重与分层", "按差距挑题", GOLD_L, GOLD),
]

for x, name, sub, fc, ec in top:
    box(ax, x, 28, 28, 13, "", fc=fc, ec=ec, rounded=1.2)
    label(ax, x + 14, 36.2, name, fs=8.8, weight="bold")
    label(ax, x + 14, 31.4, sub, fs=7.1, color=GRAY)

for x, name, sub, fc, ec in bot:
    box(ax, x, 6, 28, 13, "", fc=fc, ec=ec, rounded=1.2)
    label(ax, x + 14, 14.2, name, fs=8.8, weight="bold")
    label(ax, x + 14, 9.4, sub, fs=7.1, color=GRAY)

arrow(ax, (30.4, 34.5), (35.6, 34.5), color=DEEP)
arrow(ax, (64.4, 34.5), (69.6, 34.5), color=DEEP)
arrow(ax, (84.0, 27.6), (84.0, 19.4), color=GOLD)
arrow(ax, (69.6, 12.5), (64.4, 12.5), color=GOLD)
arrow(ax, (35.6, 12.5), (30.4, 12.5), color=DEEP)

label(ax, 50, 1.6, "学生做不出的题才有信息量：先估学生的通过率，再挑差距大的",
      fs=6.9, color=INK)

save(fig, "llm-rl", "18", "fig18-1.png")
