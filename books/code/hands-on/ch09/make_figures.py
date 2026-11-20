# -*- coding: utf-8 -*-
"""第 9 章插图：PPO 在 CartPole 上的学习曲线。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from figstyle import DEEP, GRAY, canvas, save
from plots import band, load

fig, ax = canvas(6.4, 3.1)
ax.axhline(500, color=GRAY, lw=0.8, ls="--")

band(ax, load("ppo_s*.json"), DEEP, "PPO（3 个种子）", window=30)

ax.set_xlabel("训练回合")
ax.set_ylabel("每局回报（30 回合滑动平均）")
ax.set_xlim(0, 1000)
ax.set_ylim(0, 520)
ax.legend(frameon=False, loc="lower right", fontsize=9)

save(fig, "hands-on", "09", "fig09-1.png")
