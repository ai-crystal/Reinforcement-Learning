# -*- coding: utf-8 -*-
"""第 13 章插图：预训练、SFT 与强化学习的三段分工。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figstyle import (DEEP, DEEP_BG, DEEP_L, GOLD, GOLD_L, GRAY, INK, WHITE,
                      arrow, box, diagram, label, save)

fig, ax = diagram(6.8, 3.05, xlim=(0, 100), ylim=(0, 50))

label(ax, 50, 46.6, "同一套自回归策略，三段用不同的数据和目标", fs=7.9, color=INK)

stages = [
    (2.0, "预训练", "万亿 token，预测下一个 token",
     "得到：语言与常识", "风险：底子不够，后面补不上",
     DEEP_BG, DEEP_L),
    (35.5, "SFT", "示范数据，只算回答部分",
     "得到：听得懂指令", "风险：只会那几种句式",
     WHITE, GRAY),
    (69.0, "RL", "自己采样，奖励打分",
     "得到：做对的概率", "风险：全对或全错，没有信号",
     GOLD_L, GOLD),
]

for x, name, sub, got, risk, fc, ec in stages:
    box(ax, x, 27, 29, 13.5, "", fc=fc, ec=ec, rounded=1.2)
    label(ax, x + 14.5, 35.6, name, fs=9.0, weight="bold")
    label(ax, x + 14.5, 30.4, sub, fs=7.1, color=GRAY)
    label(ax, x + 14.5, 23.2, got, fs=7.4, color=DEEP, weight="bold")
    label(ax, x + 14.5, 19.0, risk, fs=7.0, color=GRAY)

arrow(ax, (31.4, 33.7), (35.1, 33.7), color=DEEP)
arrow(ax, (64.9, 33.7), (68.6, 33.7), color=DEEP)

label(ax, 50, 10.4, "三段不是流水线：跑完一轮 RL，格式会漂，通常要回头补一轮 SFT",
      fs=7.2, color=INK)
label(ax, 50, 5.6, "判断当前该做哪一段，只看模型现在的短板在知识、在形式，还是在做对的概率上",
      fs=6.9, color=GRAY)

save(fig, "llm-rl", "13", "fig13-1.png")
