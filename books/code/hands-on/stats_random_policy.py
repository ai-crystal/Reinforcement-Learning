# -*- coding: utf-8 -*-
"""随机策略在 CartPole 上的基线统计，第 2 章用。

运行： python books/code/hands-on/stats_random_policy.py
输出： results/random_cartpole.json
"""
from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def rollout(episodes: int = 500, seed: int = 0) -> dict:
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)
    lengths = []
    for ep in range(episodes):
        env.reset(seed=ep)
        done, n = False, 0
        while not done:
            _, _, term, trunc, _ = env.step(int(rng.integers(2)))
            n += 1
            done = term or trunc
        lengths.append(n)
    env.close()

    arr = np.asarray(lengths, dtype=float)
    return {
        "episodes": episodes,
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "max": int(arr.max()),
        "min": int(arr.min()),
        "share_over_50": float((arr > 50).mean()),
        "share_over_100": float((arr > 100).mean()),
        "first_50_mean": float(arr[:50].mean()),
        "last_50_mean": float(arr[-50:].mean()),
        "lengths": [int(v) for v in lengths],
    }


def main() -> None:
    payload = rollout()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "random_cartpole.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    for key in ["mean", "median", "std", "p10", "p90", "max",
                "share_over_50", "share_over_100", "first_50_mean",
                "last_50_mean"]:
        print("%-16s %s" % (key, payload[key]))


if __name__ == "__main__":
    main()
