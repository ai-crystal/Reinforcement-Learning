# -*- coding: utf-8 -*-
"""第 10 章插图：PPO 的裁剪目标（按公式画的曲线，不是实验数据）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from figstyle import DEEP, GOLD, GRAY, canvas, save

r = np.linspace(0, 2, 401)
eps = 0.2

fig, ax = canvas(6.2, 2.9)
for a, color, name in ((1.0, DEEP, "优势为正：$\\hat{A}=1$"),
                       (-1.0, GOLD, "优势为负：$\\hat{A}=-1$")):
    clipped = np.minimum(r * a, np.clip(r, 1 - eps, 1 + eps) * a)
    ax.plot(r, clipped, color=color, lw=1.8, label=name)
    ax.plot(r, r * a, color=color, lw=0.9, ls="--", alpha=0.55)

ax.axvline(1 - eps, color=GRAY, lw=0.8, ls=":")
ax.axvline(1 + eps, color=GRAY, lw=0.8, ls=":")
ax.text(1 - eps, -0.14, "1−ε", ha="center", va="top", fontsize=8.6, color=GRAY)
ax.text(1 + eps, -0.14, "1+ε", ha="center", va="top", fontsize=8.6, color=GRAY)
ax.axhline(0, color=GRAY, lw=0.6)

ax.set_xlabel("新旧策略的概率比 $r_t(\\theta)$")
ax.set_ylabel("裁剪后的目标")
ax.set_xlim(0, 2)
ax.set_ylim(-1.75, 1.35)
ax.legend(frameon=False, loc="upper left", fontsize=9)

save(fig, "algorithms", "10", "fig10-1.png")
