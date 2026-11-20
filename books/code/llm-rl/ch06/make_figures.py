# -*- coding: utf-8 -*-
"""第 6 章插图：可验证奖励的流程。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figstyle import (DEEP, DEEP_BG, DEEP_L, GOLD, GOLD_L, GRAY, INK, RED,
                      WHITE, arrow, box, diagram, label, save)

fig, ax = diagram(6.8, 2.6, xlim=(0, 100), ylim=(0, 44))

box(ax, 2, 24, 20, 12, "", fc=WHITE, ec=GRAY, rounded=1.2)
label(ax, 12, 32.4, "题目", fs=7.4, color=GRAY)
label(ax, 12, 28.0, "3+4 等于几", fs=8.0, color=INK)

box(ax, 27, 24, 22, 12, "", fc=DEEP_BG, ec=DEEP_L, rounded=1.2)
label(ax, 38, 32.4, "模型生成", fs=7.4, color=GRAY)
label(ax, 38, 28.0, "推理 + 答案", fs=8.0, color=INK)

box(ax, 54, 24, 20, 12, "", fc="#F4F6F7", ec=GRAY, rounded=1.2)
label(ax, 64, 32.4, "程序校验", fs=7.4, color=GRAY)
label(ax, 64, 28.0, "比对或跑测试", fs=7.6, color=INK)

box(ax, 79, 24, 18, 12, "", fc=GOLD_L, ec=GOLD, rounded=1.2)
label(ax, 88, 32.4, "奖励", fs=7.4, color=GRAY)
label(ax, 88, 28.0, "对 1 / 错 0", fs=8.0, color=INK)

arrow(ax, (22.4, 30.0), (26.6, 30.0), color=GRAY)
arrow(ax, (49.4, 30.0), (53.6, 30.0), color=GRAY)
arrow(ax, (74.4, 30.0), (78.6, 30.0), color=GRAY)

arrow(ax, (88.0, 23.6), (88.0, 12.0), color=GOLD, mutation=9)
arrow(ax, (88.0, 12.0), (38.6, 12.0), color=GOLD, mutation=9)
arrow(ax, (38.0, 12.0), (38.0, 23.6), color=GOLD, mutation=9)
label(ax, 63.0, 9.0, "用这个奖励更新模型", fs=7.4, color=GOLD)

label(ax, 50, 40.0, "答案能被程序判对错时，奖励不需要人来给", fs=7.8, color=INK)

save(fig, "llm-rl", "06", "fig06-1.png")
