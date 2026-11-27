# -*- coding: utf-8 -*-
"""第 10 章：先摸清 LunarLander 的底细，再谈调参。

跑随机策略拿基线，顺便统计观测范围和每一步奖励的构成。
运行： python books/code/hands-on/check_lunar.py
输出： results/lunar_random.json
"""
from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    env = gym.make("LunarLander-v3")
    rng = np.random.default_rng(0)
    returns, lengths, steps_r = [], [], []
    obs_acc = []

    for ep in range(100):
        obs, _ = env.reset(seed=ep)
        done, total, n = False, 0.0, 0
        while not done:
            obs_acc.append(obs)
            a = int(rng.integers(env.action_space.n))
            obs, r, term, trunc, _ = env.step(a)
            total += r
            steps_r.append(r)
            n += 1
            done = term or trunc
        returns.append(total)
        lengths.append(n)
    env.close()

    O = np.asarray(obs_acc, dtype=np.float64)
    R = np.asarray(returns)
    S = np.asarray(steps_r)
    payload = {
        "episodes": 100,
        "return_mean": float(R.mean()),
        "return_std": float(R.std()),
        "return_min": float(R.min()),
        "return_max": float(R.max()),
        "length_mean": float(np.mean(lengths)),
        "step_reward_mean": float(S.mean()),
        "step_reward_min": float(S.min()),
        "step_reward_max": float(S.max()),
        "step_reward_abs_mean": float(np.abs(S).mean()),
        "obs_min": O.min(axis=0).round(3).tolist(),
        "obs_max": O.max(axis=0).round(3).tolist(),
        "obs_std": O.std(axis=0).round(3).tolist(),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "lunar_random.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
