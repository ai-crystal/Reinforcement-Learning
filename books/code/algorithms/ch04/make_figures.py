# -*- coding: utf-8 -*-
"""第 4 章插图：蒙特卡洛与时序差分的误差曲线。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np

from figstyle import DEEP, GOLD, GRAY, PALETTE, TEAL, canvas, save

data = json.loads((Path(__file__).resolve().parents[1] / "results" / "mc_td.json")
                  .read_text(encoding="utf-8"))

fig, ax = canvas(6.4, 3.0)
ax.plot(data["mc_0.1"], color=GOLD, lw=1.5, label="蒙特卡洛，步长 0.1")
ax.plot(data["td_0.1"], color=DEEP, lw=1.5, label="时序差分，步长 0.1")
ax.plot(data["mc_0.05"], color=GOLD, lw=1.1, ls="--", label="蒙特卡洛，步长 0.05")
ax.plot(data["td_0.05"], color=DEEP, lw=1.1, ls="--", label="时序差分，步长 0.05")
ax.set_xlabel("回合数")
ax.set_ylabel("价值估计的均方根误差")
ax.set_xlim(0, 200)
ax.legend(frameon=False, loc="upper right", fontsize=8.6)

save(fig, "algorithms", "04", "fig04-1.png")
