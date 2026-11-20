# -*- coding: utf-8 -*-
"""第 5 章插图：悬崖行走上的 Q 学习与 SARSA。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np

from figstyle import DEEP, GOLD, canvas, save

data = json.loads((Path(__file__).resolve().parents[1] / "results" / "cliff.json")
                  .read_text(encoding="utf-8"))

fig, ax = canvas(6.4, 3.0)
for key, color, name in (("q", GOLD, "Q 学习"), ("sarsa", DEEP, "SARSA")):
    curve = np.asarray(data[key], dtype=float)
    smooth = np.convolve(curve, np.ones(10) / 10, mode="valid")
    ax.plot(np.arange(len(smooth)) + 5, smooth, color=color, lw=1.5, label=name)

ax.set_xlabel("回合")
ax.set_ylabel("每局累积奖励（10 回合滑动平均）")
ax.set_xlim(0, 500)
ax.set_ylim(-140, 0)
ax.legend(frameon=False, loc="lower right", fontsize=9)

save(fig, "algorithms", "05", "fig05-1.png")
