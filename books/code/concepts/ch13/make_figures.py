# -*- coding: utf-8 -*-
"""第 13 章插图：可验证奖励的流程，以及 PPO 与 GRPO 的对比。"""
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

save(fig, "concepts", "13", "fig13-2.png")

# ---------------------------------------------------------------- 图 13-1
fig, ax = diagram(6.8, 2.8, xlim=(0, 100), ylim=(0, 46))


def card(x, y, w, h, title, note, fc=WHITE, ec=GRAY):
    box(ax, x, y, w, h, "", fc=fc, ec=ec, rounded=1.0)
    label(ax, x + w / 2, y + h - 3.2, title, fs=8.0, weight="bold")
    label(ax, x + w / 2, y + 3.0, note, fs=6.8, color=GRAY)


card(2, 24, 18, 12, "一道题", "有标准答案")
card(24, 24, 22, 12, "模型生成", "推理过程和答案")
card(50, 24, 18, 12, "程序校验", "比对或跑测试")
card(72, 24, 18, 12, "奖励", "对 1，错 0", GOLD_L, GOLD)
arrow(ax, (20.4, 30.0), (23.6, 30.0), color=GRAY)
arrow(ax, (46.4, 30.0), (49.6, 30.0), color=GRAY)
arrow(ax, (68.4, 30.0), (71.6, 30.0), color=GRAY)
card(36, 6, 26, 10, "更新模型", "让答对的概率变大", DEEP_BG, DEEP_L)
arrow(ax, (81.0, 23.6), (81.0, 11.0), color=GOLD, mutation=9)
arrow(ax, (81.0, 11.0), (62.4, 11.0), color=GOLD, mutation=9)
arrow(ax, (36.0, 11.0), (36.0, 23.6), color=GOLD, mutation=9)
label(ax, 50, 43.0, "答案能被程序判对错时，奖励不需要人来给", fs=7.6, color=INK)

save(fig, "concepts", "13", "fig13-1.png")
