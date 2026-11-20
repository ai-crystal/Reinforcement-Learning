# -*- coding: utf-8 -*-
"""第 7 章插图：GRPO 的组内比较。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figstyle import (DEEP, DEEP_BG, DEEP_L, GOLD, GOLD_L, GRAY, INK, RED,
                      WHITE, arrow, box, diagram, label, save)

fig, ax = diagram(6.8, 2.8, xlim=(0, 100), ylim=(0, 46))

box(ax, 2, 17, 18, 12, "", fc=WHITE, ec=GRAY, rounded=1.2)
label(ax, 11, 25.4, "同一道题", fs=7.2, color=GRAY)
label(ax, 11, 21.0, "采样 8 次", fs=8.0, color=INK)

rows = [("答 1", 28, 1), ("答 2", 21, 0), ("答 3", 14, 1), ("答 4", 7, 0)]
for name, y, ok in rows:
    box(ax, 26, y, 18, 6, "", fc=DEEP_BG if ok else "#F4F6F7",
        ec=DEEP_L if ok else GRAY, rounded=0.8)
    label(ax, 35, y + 3.0, name, fs=7.6, color=INK)
    label(ax, 48, y + 3.0, "对 +1" if ok else "错 0", fs=7.2,
          color=DEEP if ok else GRAY, ha="left")

arrow(ax, (20.4, 23.0), (25.6, 22.0), color=GRAY)
label(ax, 23.0, 26.5, "", fs=7)

box(ax, 58, 16, 18, 12, "", fc=GOLD_L, ec=GOLD, rounded=1.2)
label(ax, 67, 24.4, "组内平均分", fs=7.4, color=GRAY)
label(ax, 67, 20.0, "0.5", fs=9.0, color=INK)

arrow(ax, (54.4, 22.0), (57.6, 22.0), color=GOLD)
label(ax, 60.0, 25.0, "", fs=7)

label(ax, 80.0, 26.0, "高于均值 → 概率上调", fs=7.0, color=DEEP, ha="left")
label(ax, 80.0, 21.0, "低于均值 → 概率下调", fs=7.0, color=RED, ha="left")
label(ax, 80.0, 16.0, "不需要价值网络", fs=7.0, color=GRAY, ha="left")

label(ax, 50, 41.0, "同一道题的一组回答互相比较，平均分就是基准", fs=7.8, color=INK)

save(fig, "llm-rl", "07", "fig07-1.png")
