# -*- coding: utf-8 -*-
"""第 18 章插图：基于模型的循环（示意图，不是实验数据）。"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figstyle import (BG, DEEP, DEEP_L, DEEP_XL, GOLD, GOLD_L, GRAY, WHITE,
                      arrow, box, diagram, label, save)

fig, ax = diagram(6.6, 3.6)

CX, CY, RX, RY = 50.0, 24.0, 31.0, 14.0
W, H = 18.0, 8.4

# 五边形上顺时针排列：真实环境 -> 真实数据 -> 世界模型 -> 想象 -> 策略
nodes = [
    ("真实环境", 162, DEEP_XL, DEEP_L),
    ("真实数据\n缓冲", 234, BG, GRAY),
    ("世界模型\n集成", 306, DEEP_XL, DEEP_L),
    ("想象滚推演\nk 步", 18, GOLD_L, GOLD),
    ("策略与价值", 90, BG, GRAY),
]
pos = []
for _, deg, _, _ in nodes:
    a = math.radians(deg)
    pos.append((CX + RX * math.cos(a), CY + RY * math.sin(a)))

for (text, _, fc, ec), (x, y) in zip(nodes, pos):
    box(ax, x - W / 2, y - H / 2, W, H, text, fc=fc, ec=ec, fs=7.4)


def edge_point(p, q):
    """从 p 出发沿 pq 方向，与以 p 为中心、W x H 的矩形边界的交点。"""
    dx, dy = q[0] - p[0], q[1] - p[1]
    sx = (W / 2) / abs(dx) if dx else 1e9
    sy = (H / 2) / abs(dy) if dy else 1e9
    t = min(sx, sy)
    return (p[0] + dx * t, p[1] + dy * t)


captions = ["交互", "拟合", "生成分支", "λ 回报", "部署新策略"]
for i in range(len(nodes)):
    p, q = pos[i], pos[(i + 1) % len(nodes)]
    a = edge_point(p, q)
    b = edge_point(q, p)
    arrow(ax, a, b, color=DEEP, lw=1.0, rad=0.0, shrinkA=0.0, shrinkB=0.0)
    mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
    vx, vy = mx - CX, my - CY
    norm = math.hypot(vx, vy) or 1.0
    label(ax, mx + vx / norm * 5.2, my + vy / norm * 4.4, captions[i],
          fs=6.8, color=GRAY)

label(ax, CX, CY + 2.6, "模型误差在这里累积", fs=7.2, color=GOLD)
label(ax, CX, CY - 2.6, "推演越长，误差越大", fs=6.4, color=GRAY)

save(fig, "algorithms", "18", "fig18-1.png")
