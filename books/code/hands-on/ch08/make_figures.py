# -*- coding: utf-8 -*-
"""第 8 章插图：REINFORCE 有基线与无基线的对比。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from figstyle import DEEP, GOLD, GRAY, canvas, save
from plots import band, load

fig, ax = canvas(6.4, 3.1)
ax.axhline(500, color=GRAY, lw=0.8, ls="--")

band(ax, load("reinforce_plain_s*.json"), DEEP, "只减均值（无基线）", window=20)
band(ax, load("reinforce_baseline_s*.json"), GOLD, "减去学出来的价值", window=20)

ax.set_xlabel("训练回合")
ax.set_ylabel("每局回报（20 回合滑动平均）")
ax.set_xlim(0, 500)
ax.set_ylim(0, 520)
ax.legend(frameon=False, loc="lower right", fontsize=9)

save(fig, "hands-on", "08", "fig08-1.png")
