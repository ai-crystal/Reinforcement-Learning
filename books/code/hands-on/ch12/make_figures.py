# -*- coding: utf-8 -*-
"""第 12 章插图：三种典型的“学不动”曲线，全部来自真实实验。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from figstyle import DEEP, GOLD, GRAY, PALETTE, TEAL, canvas, save
from plots import load, moving

fig, ax = canvas(6.4, 3.1)

# 完全不动：去掉目标网络
curve = moving(load("dqn_no_target_s0.json")[0]["returns"], 15)
ax.plot(np.arange(len(curve)) + 7, curve, color=GOLD, lw=1.4,
        label="一直停在随机水平")

# 学会了又掉下去：去掉经验回放的一个种子
curve = moving(load("dqn_no_replay_s0.json")[0]["returns"], 15)
ax.plot(np.arange(len(curve)) + 7, curve, color=TEAL, lw=1.4,
        label="短暂学会，随后崩掉")

# 稳定上升：PPO
curve = moving(load("ppo_s0.json")[0]["returns"], 30)
ax.plot(np.arange(len(curve)) + 15, curve, color=DEEP, lw=1.6,
        label="逐步上升并稳住")

ax.set_xlabel("训练回合")
ax.set_ylabel("每局回报（滑动平均）")
ax.set_xlim(0, 700)
ax.set_ylim(0, 520)
ax.legend(frameon=False, loc="upper left", fontsize=9)

save(fig, "hands-on", "12", "fig12-1.png")
