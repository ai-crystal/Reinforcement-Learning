# -*- coding: utf-8 -*-
"""第 11 章插图：多轮智能体的循环。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figstyle import (DEEP, DEEP_BG, DEEP_L, GOLD, GOLD_L, GRAY, INK, WHITE,
                      arrow, box, diagram, label, save)

fig, ax = diagram(6.8, 3.0, xlim=(0, 100), ylim=(0, 48))

box(ax, 3, 26, 26, 12, "", fc=DEEP_BG, ec=DEEP_L, rounded=1.2)
label(ax, 16, 34.4, "模型", fs=8.6, weight="bold")
label(ax, 16, 30.0, "决定下一步做什么", fs=7.2, color=GRAY)

box(ax, 40, 26, 26, 12, "", fc=WHITE, ec=GRAY, rounded=1.2)
label(ax, 53, 34.4, "工具与环境", fs=8.6, weight="bold")
label(ax, 53, 30.0, "搜索、代码、文件", fs=7.2, color=GRAY)

box(ax, 40, 6, 26, 12, "", fc=GOLD_L, ec=GOLD, rounded=1.2)
label(ax, 53, 14.4, "结果与反馈", fs=8.6, weight="bold")
label(ax, 53, 10.0, "成功、报错、超时", fs=7.2, color=GRAY)

box(ax, 76, 6, 22, 12, "", fc="#F4F6F7", ec=GRAY, rounded=1.2)
label(ax, 87, 14.4, "整轮评价", fs=8.4, weight="bold")
label(ax, 87, 10.0, "任务是否完成", fs=7.2, color=GRAY)

arrow(ax, (29.4, 32.0), (39.6, 32.0), color=GRAY)
label(ax, 34.5, 34.4, "调用", fs=7.0, color=GRAY)
arrow(ax, (53.0, 25.6), (53.0, 18.4), color=GRAY)
arrow(ax, (39.6, 12.0), (29.4, 12.0), color=DEEP, mutation=9)
arrow(ax, (29.0, 12.0), (29.0, 25.6), color=DEEP, mutation=9)
label(ax, 30.6, 18.6, "看到反馈再决定", fs=7.0, color=DEEP, ha="left")
arrow(ax, (66.4, 12.0), (75.6, 12.0), color=GOLD, mutation=9)

label(ax, 50, 43.0, "一轮任务里有很多步决策，奖励只在最后给出", fs=7.8, color=INK)

save(fig, "llm-rl", "11", "fig11-1.png")
