# -*- coding: utf-8 -*-
"""第 9 章插图：多数投票的收益，取决于错误之间是否独立（真实运行结果）。

三种错误结构，单次正确率都设在 0.4 附近：
  * independent  每次采样独立犯错，错误答案各不相同
  * heterogeneous 题目难度不同，同一题的多次采样共享难度
  * shared       同一题的错误答案是同一个，采样只是重复同一个错
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from figstyle import DEEP, GOLD, GRAY, RED, TEAL, canvas, save

NQ = 3000
TRIALS = 12
NS = [1, 3, 5, 7, 9, 11, 15, 21, 31, 41, 51, 63]


def majority(votes):
    # votes: (NQ, n) int array; return the most frequent answer per row
    out = np.zeros(votes.shape[0], dtype=int)
    for i, row in enumerate(votes):
        vals, cnts = np.unique(row, return_counts=True)
        out[i] = vals[np.argmax(cnts)]
    return out


def run(kind, rng):
    accs = []
    for n in NS:
        acc = 0.0
        for _ in range(TRIALS):
            if kind == "independent":
                p = 0.4
                wrong = rng.integers(1, 1000, size=(NQ, n))
                votes = np.where(rng.random((NQ, n)) < p, 0, wrong)
            elif kind == "heterogeneous":
                q = rng.beta(1.2, 1.8, size=NQ)
                wrong = rng.integers(1, 1000, size=(NQ, n))
                votes = np.where(rng.random((NQ, n)) < q[:, None], 0, wrong)
            else:
                good = rng.random(NQ) < 0.4
                same_wrong = rng.integers(1, 1000, size=NQ)
                votes = np.where(good[:, None], 0, same_wrong[:, None])
            acc += (majority(votes) == 0).mean()
        accs.append(acc / TRIALS)
    return np.array(accs)


rng = np.random.default_rng(7)
curves = {}
for kind in ("independent", "heterogeneous", "shared"):
    curves[kind] = run(kind, rng)
    print(kind, [round(float(a), 4) for a in curves[kind]])

fig, ax = canvas(6.4, 2.9)
style = {
    "independent": ("错误互相独立", DEEP),
    "heterogeneous": ("题目难度不同", GOLD),
    "shared": ("每次都错在同一个答案", RED),
}
for kind, (label, color) in style.items():
    ax.plot(NS, curves[kind] * 100, color=color, lw=1.5, marker="o", ms=3.2,
            label=label)

ax.set_xscale("log")
ax.set_xticks([1, 3, 9, 31, 63])
ax.set_xticklabels(["1", "3", "9", "31", "63"])
ax.set_xlabel("采样次数 n（对数刻度）")
ax.set_ylabel("多数投票正确率（%）")
ax.set_ylim(35, 102)
ax.legend(frameon=False, loc="center right", fontsize=9)

save(fig, "llm-rl", "09", "fig09-1.png")
