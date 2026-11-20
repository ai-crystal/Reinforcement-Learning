# -*- coding: utf-8 -*-
"""第 7 章插图：DQN 与两组消融的真实曲线。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from figstyle import DEEP, GOLD, GRAY, PALETTE, TEAL, canvas, label, save
from plots import band, load

fig, ax = canvas(6.4, 3.2)
ax.axhline(500, color=GRAY, lw=0.8, ls="--")

for pattern, color, name in (
        ("dqn_full_s*.json", DEEP, "完整 DQN"),
        ("dqn_no_target_s*.json", GOLD, "去掉目标网络"),
        ("dqn_no_replay_s*.json", TEAL, "去掉经验回放")):
    runs = load(pattern)
    band(ax, runs, color, "%s（%d 个种子）" % (name, len(runs)), window=10)

ax.set_xlabel("训练回合")
ax.set_ylabel("每局回报（10 回合滑动平均）")
ax.set_xlim(0, 250)
ax.set_ylim(0, 520)
ax.legend(frameon=False, loc="upper left", fontsize=9)

save(fig, "hands-on", "07", "fig07-1.png")
