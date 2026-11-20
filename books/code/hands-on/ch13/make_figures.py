# -*- coding: utf-8 -*-
"""第 14 章插图：串行与并行环境的采样速度（真实计时）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

from figstyle import DEEP, GOLD, GRAY, canvas, save

data = json.loads((Path(__file__).resolve().parents[1] / "results" / "vector_speed.json")
                  .read_text(encoding="utf-8"))

labels = ["1 个环境", "4 个环境", "8 个环境"]
keys = ["envs_1", "envs_4", "envs_8"]
values = [data[k]["steps_per_sec"] / 1000 for k in keys]

fig, ax = canvas(6.0, 2.9)
bars = ax.bar(labels, values, color=[GRAY, GOLD, DEEP], width=0.5)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 3, "%.0f" % v, ha="center",
            fontsize=9.5, color="#333333")
ax.set_ylabel("每秒采样步数（千步）")
ax.set_ylim(0, max(values) * 1.2)

save(fig, "hands-on", "13", "fig13-1.png")
