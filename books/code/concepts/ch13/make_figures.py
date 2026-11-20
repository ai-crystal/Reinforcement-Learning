# -*- coding: utf-8 -*-
"""第 13 章插图：PPO 与 GRPO 的对比。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figstyle import (DEEP, DEEP_BG, DEEP_L, GOLD, GOLD_L, GRAY, INK, WHITE,
                      arrow, box, diagram, label, save)

fig, ax = diagram(6.8, 3.0, xlim=(0, 100), ylim=(0, 48))

# 左：PPO
box(ax, 2, 36, 44, 6, "PPO：需要两个网络", fc="#F4F6F7", ec=GRAY, fs=8.0)
box(ax, 3, 24, 16, 8, "策略网络", fc=DEEP_BG, ec=DEEP_L, fs=7.8)
box(ax, 3, 12, 16, 8, "价值网络", fc=GOLD_L, ec=GOLD, fs=7.8)
box(ax, 27, 18, 18, 8, "优势估计\n当前值减基准", fc=WHITE, ec=GRAY, fs=7.2)
arrow(ax, (19.4, 28.0), (26.6, 24.0), color=GRAY)
arrow(ax, (19.4, 16.0), (26.6, 20.0), color=GRAY)
label(ax, 15.0, 6.0, "两个模型都要放进显存", fs=7.0, color=GRAY)

# 右：GRPO
box(ax, 52, 36, 46, 6, "GRPO：一组回答互相比较", fc="#F4F6F7", ec=GRAY, fs=8.0)
box(ax, 53, 24, 16, 8, "策略网络", fc=DEEP_BG, ec=DEEP_L, fs=7.8)
box(ax, 74, 17, 22, 15, "", fc=WHITE, ec=GRAY, rounded=1.0)
label(ax, 85, 29.0, "同一道题的三个回答", fs=7.0, color=INK)
label(ax, 85, 25.6, "答 1　　答 2　　答 3", fs=6.8, color=GRAY)
box(ax, 74, 4, 22, 9, "组内平均分当基准", fc=GOLD_L, ec=GOLD, fs=7.2)
arrow(ax, (69.4, 28.0), (73.6, 26.0), color=GRAY)
arrow(ax, (85.0, 16.6), (85.0, 13.4), color=GRAY)
label(ax, 62.0, 6.0, "只用一个网络，省显存", fs=7.0, color=GRAY)

save(fig, "concepts", "13", "fig13-1.png")
