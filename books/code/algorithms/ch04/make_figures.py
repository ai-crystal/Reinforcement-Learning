# -*- coding: utf-8 -*-
"""第 4 章插图之二：n 步回报的误差随 n 的变化（真实计算）。

数据来自五状态随机游走，200 局、30 个种子，
三档步长下的均方根误差，最右一列（n=100）等价于蒙特卡洛。
重新生成数据可参考本文件的 docstring，脚本只负责画图。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json

import numpy as np

from figstyle import DEEP, GOLD, TEAL, canvas, save

data = json.loads((Path(__file__).resolve().parents[1] / "results" / "nstep.json")
                  .read_text(encoding="utf-8"))

fig, ax = canvas(6.4, 3.0)
for key, color in (("0.02", DEEP), ("0.05", TEAL), ("0.1", GOLD)):
    ax.plot(data["n"], data[key], marker="o", ms=3.6, color=color, lw=1.4,
            label="步长 %s" % key)

ax.set_xscale("log")
ax.set_xticks([1, 2, 3, 5, 8, 100])
ax.set_xticklabels(["1", "2", "3", "5", "8", "MC"])
ax.set_xlabel("自举之前的步数 $n$")
ax.set_ylabel("两百局之后的均方根误差")
ax.set_ylim(0, 0.18)
ax.legend(frameon=False, loc="upper left", fontsize=9)

save(fig, "algorithms", "04", "fig04-2.png")
