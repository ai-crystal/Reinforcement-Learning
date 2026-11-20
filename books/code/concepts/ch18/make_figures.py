# -*- coding: utf-8 -*-
"""第 18 章插图：世界模型的循环。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figstyle import (DEEP, DEEP_BG, DEEP_L, GOLD, GOLD_L, GRAY, INK, WHITE,
                      arrow, box, diagram, label, line, save)


def card(ax, x, y, w, h, title, note, fc=WHITE, ec=GRAY):
    box(ax, x, y, w, h, "", fc=fc, ec=ec, rounded=1.0)
    label(ax, x + w / 2, y + h - 3.2, title, fs=8.2, weight="bold")
    label(ax, x + w / 2, y + 3.0, note, fs=6.8, color=GRAY)


fig, ax = diagram(6.8, 3.0, xlim=(0, 100), ylim=(0, 48))

card(ax, 3, 27, 22, 11, "真实环境", "机器人、驾驶、游戏")
card(ax, 36, 27, 22, 11, "世界模型", "预测下一步", DEEP_BG, DEEP_L)
card(ax, 69, 27, 26, 11, "想象里练习", "大量虚拟轨迹", GOLD_L, GOLD)
card(ax, 36, 6, 22, 10, "策略", "学会怎么动")
card(ax, 69, 6, 26, 10, "回到真实环境", "部署，再收集数据", DEEP_BG, DEEP_L)

arrow(ax, (25.4, 32.5), (35.6, 32.5), color=GRAY)
label(ax, 30.2, 35.0, "数据", fs=6.8, color=GRAY)
arrow(ax, (58.4, 32.5), (68.6, 32.5), color=GRAY)
arrow(ax, (82.0, 26.6), (82.0, 16.4), color=GOLD, mutation=9)
arrow(ax, (68.6, 11.0), (58.4, 11.0), color=GRAY)
arrow(ax, (47.0, 16.4), (47.0, 26.6), color=DEEP, mutation=9)
label(ax, 48.4, 21.2, "改进策略", fs=6.8, color=DEEP, ha="left")

line(ax, (82.0, 6.0), (82.0, 3.0), color=GRAY, lw=0.9, ls=(0, (4, 2)))
line(ax, (82.0, 3.0), (25.0, 3.0), color=GRAY, lw=0.9, ls=(0, (4, 2)))
arrow(ax, (25.0, 3.0), (25.0, 26.6), color=GRAY, ls=(0, (4, 2)), mutation=8)
label(ax, 53.0, 1.2, "部署后再收集数据，回到起点", fs=6.8, color=GRAY)

save(fig, "concepts", "18", "fig18-1.png")
