# -*- coding: utf-8 -*-
"""第 14 章插图：多轮智能体的循环。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figstyle import (DEEP, DEEP_BG, DEEP_L, GOLD, GOLD_L, GRAY, INK, WHITE,
                      arrow, box, diagram, label, save)


def card(ax, x, y, w, h, title, note, fc=WHITE, ec=GRAY):
    box(ax, x, y, w, h, "", fc=fc, ec=ec, rounded=1.0)
    label(ax, x + w / 2, y + h - 3.4, title, fs=8.2, weight="bold")
    label(ax, x + w / 2, y + 3.2, note, fs=6.8, color=GRAY)


fig, ax = diagram(6.8, 3.0, xlim=(0, 100), ylim=(0, 48))

card(ax, 2, 26, 20, 12, "任务", "改好这个 bug")
card(ax, 30, 26, 24, 12, "模型决定下一步", "选工具、填参数", DEEP_BG, DEEP_L)
card(ax, 62, 26, 26, 12, "工具与环境", "搜索、读文件、跑测试")
card(ax, 62, 6, 26, 10, "结果与反馈", "报错、超时、通过", GOLD_L, GOLD)
card(ax, 30, 6, 24, 10, "评分", "任务是否完成")

arrow(ax, (22.4, 32.0), (29.6, 32.0), color=GRAY)
arrow(ax, (54.4, 32.0), (61.6, 32.0), color=GRAY)
label(ax, 58.0, 34.4, "调用", fs=7.0, color=GRAY)
arrow(ax, (75.0, 25.6), (75.0, 16.4), color=GRAY)
arrow(ax, (61.6, 11.0), (54.4, 11.0), color=GRAY)
arrow(ax, (42.0, 16.4), (42.0, 25.6), color=DEEP, mutation=9)
label(ax, 43.6, 21.0, "看到结果\n再决定", fs=6.8, color=DEEP, ha="left")

label(ax, 50, 43.0, "一轮任务里有很多步决策，奖励往往只在最后给出", fs=7.8, color=INK)

save(fig, "concepts", "14", "fig14-1.png")
