# -*- coding: utf-8 -*-
"""第 14 章实验：串行环境与并行环境的采样速度对比。

用随机策略各跑固定步数，记录墙钟时间，算每秒采样步数。

运行： python books/code/hands-on/run_vector_speed.py
输出： results/vector_speed.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
STEPS = 20000


def run_serial(steps: int) -> float:
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    t0 = time.time()
    for _ in range(steps):
        obs, _, term, trunc, _ = env.step(int(rng.integers(2)))
        if term or trunc:
            obs, _ = env.reset()
    elapsed = time.time() - t0
    env.close()
    return elapsed


def run_vector(n_envs: int, steps: int) -> float:
    env = gym.vector.SyncVectorEnv(
        [lambda: gym.make("CartPole-v1") for _ in range(n_envs)])
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    t0 = time.time()
    # 每个环境走 steps/n_envs 步，总步数一致
    per_env = max(1, steps // n_envs)
    for _ in range(per_env):
        actions = rng.integers(0, 2, size=n_envs)
        obs, _, term, trunc, _ = env.step(actions)
        done = np.logical_or(term, trunc)
        if done.any():
            obs, _ = env.reset()
    elapsed = time.time() - t0
    env.close()
    return elapsed


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    data = {}
    for n in (1, 4, 8):
        # 预热一次，避免把首次创建环境的时间算进去
        run_vector(n, 200)
        t = run_vector(n, STEPS)
        data["envs_%d" % n] = {"seconds": round(t, 3),
                               "steps_per_sec": round(STEPS / t, 1)}
        print(n, data["envs_%d" % n], flush=True)
    (RESULTS / "vector_speed.json").write_text(json.dumps(data), encoding="utf-8")


if __name__ == "__main__":
    main()
