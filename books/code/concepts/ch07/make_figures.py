# -*- coding: utf-8 -*-
"""第 7 章插图：DQN 的三件套。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figstyle import (DEEP, DEEP_BG, DEEP_L, GOLD, GOLD_L, GRAY, INK, WHITE,
                      arrow, box, diagram, label, line, save)

fig, ax = diagram(6.8, 3.1, xlim=(0, 100), ylim=(0, 46))

box(ax, 4, 33, 14, 9, "当前帧", fc=WHITE, ec=GRAY, fs=8.0)
box(ax, 23, 33, 16, 9, "当前网络", fc=DEEP_BG, ec=DEEP_L, fs=8.4)
box(ax, 44, 33, 14, 9, "各动作 Q 值", fc=WHITE, ec=GRAY, fs=7.6)

box(ax, 4, 20, 14, 9, "下一帧", fc=WHITE, ec=GRAY, fs=8.0)
box(ax, 23, 20, 16, 9, "目标网络", fc=GOLD_L, ec=GOLD, fs=8.4)
box(ax, 44, 20, 14, 9, "目标值", fc=WHITE, ec=GRAY, fs=7.8)

arrow(ax, (18.4, 37.5), (22.6, 37.5), color=GRAY)
arrow(ax, (39.4, 37.5), (43.6, 37.5), color=GRAY)
arrow(ax, (18.4, 24.5), (22.6, 24.5), color=GRAY)
arrow(ax, (39.4, 24.5), (43.6, 24.5), color=GRAY)

box(ax, 62, 24, 15, 12, "相减\n得到误差", fc=WHITE, ec=GRAY, fs=8.0)
arrow(ax, (58.4, 37.0), (62.6, 34.0), color=GRAY)
arrow(ax, (58.4, 24.5), (61.6, 26.5), color=GRAY)
label(ax, 59.8, 39.4, "估计", fs=7.0, color=GRAY)
label(ax, 60.0, 20.4, "学习目标", fs=7.0, color=GRAY)

box(ax, 82, 24, 15, 12, "反向传播\n更新参数", fc=DEEP_BG, ec=DEEP_L, fs=7.6)
arrow(ax, (77.4, 30.0), (81.6, 30.0), color=DEEP)

# 回放池：绕开下方方框，从左侧通道接到当前网络
box(ax, 25, 3, 38, 8, "经验回放池（把样本打散）", fc="#F4F6F7", ec=GRAY, fs=7.6,
    ls=(0, (3, 2)))
line(ax, (25.0, 7.0), (20.8, 7.0), color=DEEP, lw=1.0, ls=(0, (3, 2)))
line(ax, (20.8, 7.0), (20.8, 31.2), color=DEEP, lw=1.0, ls=(0, (3, 2)))
line(ax, (20.8, 31.2), (31.0, 31.2), color=DEEP, lw=1.0, ls=(0, (3, 2)))
arrow(ax, (31.0, 31.2), (31.0, 32.6), color=DEEP, ls=(0, (3, 2)), mutation=8)
label(ax, 27.0, 8.8, "训练时随机抽取", fs=6.8, color=DEEP, ha="left")

save(fig, "concepts", "07", "fig07-1.png")
