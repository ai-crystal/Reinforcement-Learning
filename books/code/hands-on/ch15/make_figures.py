# -*- coding: utf-8 -*-
"""第 16 章插图：把连续扭矩切成 7 档后用 DQN 训练 Pendulum（真实实验）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from figstyle import DEEP, GOLD, GRAY, canvas, save
from plots import band, load

fig, ax = canvas(6.4, 3.0)
band(ax, load("pendulum_dqn_s*.json"), DEEP, "离散化后的 DQN（3 个种子）", window=15)
ax.axhline(-1200, color=GRAY, lw=0.8, ls=":")
ax.text(10, -1150, "随机乱推大约在 −1200", fontsize=8.6, color=GRAY)

ax.set_xlabel("训练回合")
ax.set_ylabel("每局回报（15 回合滑动平均）")
ax.set_xlim(0, 300)
ax.set_ylim(-1400, 0)
ax.legend(frameon=False, loc="lower right", fontsize=9)

save(fig, "hands-on", "15", "fig15-1.png")
