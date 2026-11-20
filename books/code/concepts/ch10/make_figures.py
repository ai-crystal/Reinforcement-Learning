# -*- coding: utf-8 -*-
"""第 10 章插图：仿真与现实的差距及常用对策。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figstyle import (DEEP, DEEP_BG, DEEP_L, GOLD, GOLD_L, GRAY, INK, WHITE,
                      arrow, box, diagram, label, line, save)

fig, ax = diagram(6.8, 3.2, xlim=(0, 100), ylim=(0, 46))

box(ax, 4, 24, 26, 16, "", fc=DEEP_BG, ec=DEEP_L, rounded=1.4)
label(ax, 17, 36.0, "仿真", fs=9.4, weight="bold")
label(ax, 17, 31.6, "摩擦是常数", fs=7.2, color=GRAY)
label(ax, 17, 28.4, "电机理想响应", fs=7.2, color=GRAY)
label(ax, 17, 25.2, "渲染近似真实", fs=7.2, color=GRAY)

box(ax, 70, 24, 26, 16, "", fc="#FBF3E3", ec=GOLD, rounded=1.4)
label(ax, 83, 36.0, "现实", fs=9.4, weight="bold")
label(ax, 83, 31.6, "摩擦随时变化", fs=7.2, color=GRAY)
label(ax, 83, 28.4, "有延迟和死区", fs=7.2, color=GRAY)
label(ax, 83, 25.2, "光照、磨损、噪声", fs=7.2, color=GRAY)

# 中间的差距
arrow(ax, (30.6, 32.0), (69.4, 32.0), color=RED if False else GRAY, ls=(0, (5, 3)),
      style="<|-|>", mutation=9)
label(ax, 50, 34.6, "现实鸿沟", fs=8.0, color=GRAY)
label(ax, 50, 29.4, "策略在左边学会，到右边失效", fs=7.0, color=GRAY)

# 对策
box(ax, 4, 6, 28, 11, "域随机化\n训练时把参数抖开", fc=WHITE, ec=DEEP_L, fs=7.8)
box(ax, 36, 6, 28, 11, "系统辨识\n用真机数据把仿真调准", fc=WHITE, ec=DEEP_L, fs=7.8)
box(ax, 68, 6, 28, 11, "在线适应\n运行中根据观测自我修正", fc=WHITE, ec=DEEP_L, fs=7.8)

for x in (18, 50, 82):
    arrow(ax, (x, 17.6), (x, 23.4), color=DEEP, ls=(0, (3, 2)), mutation=8)

save(fig, "concepts", "10", "fig10-1.png")
