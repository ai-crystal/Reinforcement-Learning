# -*- coding: utf-8 -*-
"""第 11 章插图：同一个算法，三个种子。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from figstyle import DEEP, GOLD, GRAY, PLUM, canvas, save
from plots import load, moving

fig, ax = canvas(6.4, 3.1)
ax.axhline(500, color=GRAY, lw=0.8, ls="--")
for run, color in zip(load("dqn_full_s*.json"), (DEEP, GOLD, PLUM)):
    curve = moving(run["returns"], 15)
    ax.plot(np.arange(len(curve)) + 7, curve, color=color, lw=1.4,
            label="seed %d（最终评估 %.0f）" % (run["config"]["seed"],
                                            run["extra"]["eval_return"]))
ax.set_xlabel("训练回合")
ax.set_ylabel("每局回报（15 回合滑动平均）")
ax.set_xlim(0, 250)
ax.set_ylim(0, 520)
ax.legend(frameon=False, loc="upper left", fontsize=8.6)

save(fig, "hands-on", "11", "fig11-1.png")
