# -*- coding: utf-8 -*-
"""第 5 章的补充扫描：带打滑时学习率的影响，以及探索率不衰减的代价。

运行： python books/code/hands-on/sweep_gridworld_noisy.py
输出： results/grid_sweep_noisy.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sweep_gridworld import train

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def summarize(rec: dict) -> dict:
    steps = np.asarray(rec["steps_curve"], dtype=float)
    return {
        "success_rate": rec["success_rate"],
        "avg_steps_when_success": rec["avg_steps_when_success"],
        "episode_to_90pct": rec["episode_to_90pct"],
        "train_steps_first50": float(steps[:50].mean()),
        "train_steps_last50": float(steps[-50:].mean()),
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    out = {}
    for slip in [0.0, 0.3]:
        for alpha in [0.02, 0.1, 0.5, 0.9]:
            key = "slip%s_alpha%s" % (slip, alpha)
            out[key] = summarize(train(slip=slip, alpha=alpha, seed=0))
            print(key, out[key], flush=True)

    # 探索率固定不衰减，看训练期步数的差别
    for start, end in [(0.3, 0.3), (0.3, 0.02), (1.0, 0.02)]:
        key = "eps%s_%s" % (start, end)
        out[key] = summarize(train(eps_start=start, eps_end=end, seed=0))
        print(key, out[key], flush=True)

    (RESULTS / "grid_sweep_noisy.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
