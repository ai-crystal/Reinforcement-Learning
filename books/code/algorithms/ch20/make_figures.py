# -*- coding: utf-8 -*-
"""第 20 章插图：二十个种子上的算法表现分布，以及置信区间随种子数收窄。

实验是十臂赌博机，epsilon-greedy，epsilon=0.1，步长用样本平均。
全部数字来自本脚本的真实运行，结果写入 results/bandit_seeds.json。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
from figstyle import DEEP, DEEP_L, GOLD, GRAY, setup, save

K = 10          # arms
STEPS = 1000    # steps per run
EPS = 0.1
SEEDS = 20

RESULT = Path(__file__).resolve().parents[1] / "results" / "bandit_seeds.json"


def run(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    means = rng.normal(0.0, 1.0, size=K)
    q = np.zeros(K)
    n = np.zeros(K)
    reward = np.zeros(STEPS)
    for t in range(STEPS):
        if rng.random() < EPS:
            a = rng.integers(K)
        else:
            a = int(np.argmax(q))
        r = rng.normal(means[a], 1.0)
        n[a] += 1
        q[a] += (r - q[a]) / n[a]
        reward[t] = r
    return reward


def boot_ci(x: np.ndarray, n_boot: int = 4000, alpha: float = 0.05):
    """bootstrap 95% CI of the mean of x (last axis are seeds)."""
    rng = np.random.default_rng(0)
    idx = rng.integers(0, x.shape[-1], size=(n_boot, x.shape[-1]))
    means = x[..., idx].mean(axis=-1)
    return np.percentile(means, 100 * alpha / 2), np.percentile(means, 100 * (1 - alpha / 2))


def smooth(x: np.ndarray, w: int = 20) -> np.ndarray:
    """moving average along the last axis, edge padded so length is kept"""
    w = int(w)
    if w % 2 == 0:
        w += 1
    pad = w // 2
    k = np.ones(w) / w
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)], mode="edge")
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), -1, xp)


def carlo_t(n: int) -> float:
    # 97.5% two-sided t quantile, small table is enough for n <= 20
    table = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
             8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.093}
    return table[n]


def main() -> None:
    R = np.stack([run(s) for s in range(SEEDS)])          # (seeds, steps)
    final = R[:, -100:].mean(axis=1)                      # per-seed level at the end
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps({
        "config": {"k": K, "steps": STEPS, "eps": EPS, "seeds": SEEDS},
        "final_per_seed": [round(float(v), 4) for v in final],
        "final_mean": round(float(final.mean()), 4),
        "final_std": round(float(final.std(ddof=1)), 4),
        "final_min": round(float(final.min()), 4),
        "final_max": round(float(final.max()), 4),
    }, indent=2), encoding="utf-8")

    setup()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.17, wspace=0.45)

    # ---- left: where the 20 seeds end up ----
    ax = axes[0]
    rng2 = np.random.default_rng(7)
    ys = rng2.uniform(-0.16, 0.16, size=SEEDS)
    ax.scatter(final, ys, s=26, color=DEEP_L, edgecolor=DEEP, linewidth=0.6, zorder=3)
    ax.axvline(final.mean(), color=GOLD, linewidth=1.8, zorder=4)
    lo, hi = np.percentile(final, [2.5, 97.5])
    ax.plot([lo, hi], [0.34, 0.34], color=GOLD, linewidth=1.4, zorder=4)
    ax.plot([lo, lo], [0.29, 0.39], color=GOLD, linewidth=1.0)
    ax.plot([hi, hi], [0.29, 0.39], color=GOLD, linewidth=1.0)
    ax.set_xlabel("最后 100 步的平均奖励")
    ax.set_yticks([])
    ax.set_ylim(-0.5, 0.62)
    ax.annotate("均值", (final.mean(), -0.36), color=GOLD, fontsize=10,
                ha="center", va="top")
    ax.annotate("2.5 到 97.5 分位", ((lo + hi) / 2, 0.42), color=GOLD, fontsize=9.2,
                ha="center", va="bottom")
    ax.text(0.02, 0.965, "20 个种子", transform=ax.transAxes,
            fontsize=10, color=GRAY, va="top")

    # ---- right: t-interval half width vs number of seeds ----
    ax = axes[1]
    ns = [2, 3, 5, 8, 10, 15, 20]
    rng = np.random.default_rng(1)
    measured = []
    for n in ns:
        widths = []
        for _ in range(300):
            pick = rng.choice(SEEDS, size=n, replace=False)
            v = final[pick]
            widths.append(carlo_t(n) * v.std(ddof=1) / np.sqrt(n))
        measured.append(float(np.mean(widths)))
    ax.plot(ns, measured, color=DEEP, marker="o", markersize=3.4, label="实测半宽")
    ref = [carlo_t(20) * final.std(ddof=1) / np.sqrt(n) for n in ns]
    ax.plot(ns, ref, color=GRAY, linestyle="--", label=r"同一个标准差按 $1/\sqrt{n}$ 外推")
    ax.set_xlabel("种子数")
    ax.set_ylabel("95% 区间半宽", labelpad=2)
    ax.set_xticks(ns)
    ax.legend(loc="upper right", frameon=False)

    save(fig, "algorithms", "20", "fig20-1.png")
    print("final mean %.3f std %.3f min %.3f max %.3f"
          % (final.mean(), final.std(ddof=1), final.min(), final.max()))
    for n, w0, w1 in zip(ns, measured, ref):
        print("n=%2d measured %.3f ref %.3f" % (n, w0, w1))


if __name__ == "__main__":
    main()
