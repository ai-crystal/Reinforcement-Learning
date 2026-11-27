# -*- coding: utf-8 -*-
"""第 6 章第二张图：同一个种子、三种窗口，看平滑的代价。"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from figstyle import DEEP, DEEP_L, GOLD, GRAY, canvas, save

HERE = Path(__file__).resolve().parents[1]
runs = json.loads((HERE / "results" / "ch06_metrics_s0.json").read_text(encoding="utf-8"))
returns = np.asarray(runs["returns"], dtype=float)


def moving(values, window):
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


fig, ax = canvas(6.6, 3.2)
ax.plot(np.arange(len(returns)), returns, color=GRAY, lw=0.6, alpha=0.55,
        label="每局回报")

colors = {5: GOLD, 20: DEEP, 60: DEEP_L}
for w in (5, 20, 60):
    curve = moving(returns, w)
    x = np.arange(len(curve)) + w // 2
    ax.plot(x, curve, color=colors[w], lw=1.6, label="窗口 %d" % w)
    hit = np.where(curve >= 200)[0]
    if len(hit):
        ex = hit[0] + w // 2
        ax.plot([ex], [200], marker="o", ms=4, color=colors[w], zorder=5)
        if w == 5:
            ax.annotate("5/20 窗口：%d 局" % ex, xy=(ex, 200), xytext=(ex - 46, 244),
                        fontsize=9, color=colors[w])
        if w == 60:
            ax.annotate("60 窗口：%d 局" % ex, xy=(ex, 200), xytext=(ex + 8, 172),
                        fontsize=9, color=colors[w])

ax.axhline(500, color=GRAY, lw=0.8, ls="--")
ax.text(118, 512, "CartPole 回报上限", fontsize=9, color=GRAY)
ax.set_xlabel("训练回合")
ax.set_ylabel("回报")
ax.set_xlim(0, len(returns))
ax.set_ylim(0, 560)
ax.legend(frameon=False, loc="lower right", fontsize=9, ncol=2)

save(fig, "hands-on", "06", "fig06-2.png")
