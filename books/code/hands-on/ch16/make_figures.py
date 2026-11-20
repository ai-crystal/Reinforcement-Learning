# -*- coding: utf-8 -*-
"""第 17 章插图：稀疏奖励与势函数塑形（真实实验，5 个种子）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from figstyle import DEEP, GOLD, GRAY, canvas, save
from plots import load, moving

fig, ax = canvas(6.4, 3.0)
for pattern, color, name in (("sparse_plain_s*.json", GRAY, "只有到终点才给奖励"),
                             ("sparse_shaped_s*.json", DEEP, "加势函数塑形")):
    runs = load(pattern)
    curves = np.array([moving(r["returns"], 20) for r in runs])
    mean = curves.mean(axis=0)
    ax.plot(np.arange(len(mean)) + 10, mean * 100, color=color, lw=1.7,
            label="%s（%d 个种子）" % (name, len(runs)))

ax.set_xlabel("训练回合")
ax.set_ylabel("最近 20 局的成功率（%）")
ax.set_xlim(0, 400)
ax.set_ylim(-3, 105)
ax.text(60, 3.5, "只有到终点才给奖励的那组，5 个种子始终是 0", fontsize=8.6, color=GRAY)
ax.legend(frameon=False, loc="center right", fontsize=9)

save(fig, "hands-on", "16", "fig16-1.png")
