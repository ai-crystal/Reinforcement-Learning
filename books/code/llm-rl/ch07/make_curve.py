# -*- coding: utf-8 -*-
"""第 7 章第二张插图：玩具 GRPO 的学习曲线（真实运行结果）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json

import numpy as np

from figstyle import DEEP, GOLD, GRAY, PALETTE, TEAL, canvas, save

data = json.loads((Path(__file__).resolve().parents[1] / "results" / "toy_grpo.json")
                  .read_text(encoding="utf-8"))

fig, ax = canvas(6.4, 2.9)
for (key, run), color in zip(data.items(), (DEEP, GOLD, TEAL)):
    curve = np.convolve(run["acc"], np.ones(20) / 20, mode="valid")
    ax.plot(np.arange(len(curve)) + 10, curve * 100, color=color, lw=1.4,
            label="seed %s" % key[-1])

ax.set_xlabel("更新步数")
ax.set_ylabel("组内正确率（%）")
ax.set_xlim(0, 600)
ax.set_ylim(0, 105)
ax.legend(frameon=False, loc="lower right", fontsize=9)

save(fig, "llm-rl", "07", "fig07-2.png")
