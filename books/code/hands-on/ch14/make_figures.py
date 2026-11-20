# -*- coding: utf-8 -*-
"""第 15 章插图：原版 DQN、双估计、Dueling 的对比（真实实验）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from figstyle import DEEP, GOLD, GRAY, PLUM, canvas, save
from plots import band, load

fig, ax = canvas(6.4, 3.1)
ax.axhline(500, color=GRAY, lw=0.8, ls="--")

band(ax, load("dqn_vanilla_s*.json"), GOLD, "原版（目标网络取最大）", window=10)
band(ax, load("dqn_full_s*.json"), DEEP, "加双估计（double DQN）", window=10)
band(ax, load("dqn_dueling_s*.json"), PLUM, "再加 Dueling 结构", window=10)

ax.set_xlabel("训练回合")
ax.set_ylabel("每局回报（10 回合滑动平均）")
ax.set_xlim(0, 250)
ax.set_ylim(0, 520)
ax.legend(frameon=False, loc="upper left", fontsize=9)

save(fig, "hands-on", "14", "fig14-1.png")
