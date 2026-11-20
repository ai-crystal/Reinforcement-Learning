# -*- coding: utf-8 -*-
"""第 2 章插图：随机策略在 CartPole 上的表现。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from figstyle import DEEP, GOLD, GRAY, canvas, label, save
from plots import load

runs = load("random_cartpole.json")
returns = np.asarray(runs[0]["returns"])

fig, ax = canvas(6.4, 3.0)
bins = np.arange(5, 80, 5)
ax.hist(returns, bins=bins, color=DEEP, alpha=0.75, edgecolor="white")
ax.axvline(returns.mean(), color=GOLD, lw=1.4)
ax.text(returns.mean() + 2, ax.get_ylim()[1] * 0.9, "平均 %.1f 步" % returns.mean(),
        fontsize=9, color=GOLD)

ax.set_xlabel("每局坚持的步数")
ax.set_ylabel("局数")
ax.set_xlim(0, 80)

save(fig, "hands-on", "02", "fig02-1.png")
