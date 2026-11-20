# -*- coding: utf-8 -*-
"""第 6 章插图：原始曲线与滑动平均的差别。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from figstyle import DEEP, GOLD, GRAY, canvas, save
from plots import load, moving

runs = load("dqn_full_s0.json")
returns = np.asarray(runs[0]["returns"])

fig, ax = canvas(6.4, 3.1)
ax.plot(np.arange(len(returns)), returns, color=GRAY, lw=0.7, alpha=0.65,
        label="每局回报（原始）")
ax.plot(np.arange(len(moving(returns, 20))) + 10, moving(returns, 20),
        color=DEEP, lw=1.7, label="20 局滑动平均")
ax.set_xlabel("训练回合")
ax.set_ylabel("回报")
ax.set_xlim(0, len(returns))
ax.legend(frameon=False, loc="upper left")

save(fig, "hands-on", "06", "fig06-1.png")
