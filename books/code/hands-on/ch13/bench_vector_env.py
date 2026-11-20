# -*- coding: utf-8 -*-
"""第 14 章实验：串行、同步并行、异步并行三种采样的墙钟时间。

每个采样器都攒够 STEPS 步，计时后折算成每秒步数和相对串行的加速比。
结果写 results/vecenv_bench.json。

用法：
    python books/code/hands-on/ch14/bench_vector_env.py
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

RESULTS = HERE.parent / "results"
N_ENVS = 8
STEPS = 20000


def serial(n_steps: int, seed: int = 0) -> float:
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=seed)
    t0 = time.perf_counter()
    for i in range(n_steps):
        action = i % 2
        obs, reward, term, trunc, _ = env.step(action)
        if term or trunc:
            env.reset(seed=seed + i)
    dt = time.perf_counter() - t0
    env.close()
    return dt


def make_env(seed: int):
    def thunk():
        env = gym.make("CartPole-v1")
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env
    return thunk


def vec_run(kind: str, n_steps: int, base_seed: int = 0) -> float:
    makers = [make_env(base_seed + i) for i in range(N_ENVS)]
    envs = SyncVectorEnv(makers) if kind == "sync" else AsyncVectorEnv(makers)
    obs, _ = envs.reset(seed=[base_seed + i for i in range(N_ENVS)])
    rng = np.random.default_rng(base_seed)
    t0 = time.perf_counter()
    done_steps = 0
    while done_steps < n_steps:
        actions = rng.integers(0, 2, size=N_ENVS)
        obs, reward, term, trunc, _ = envs.step(actions)
        done_steps += N_ENVS
    dt = time.perf_counter() - t0
    envs.close()
    return dt


def main() -> None:
    rows = []
    for name in ("serial", "sync", "async"):
        times = []
        for repeat in range(3):
            if name == "serial":
                dt = serial(STEPS, seed=repeat)
            else:
                dt = vec_run(name, STEPS, base_seed=repeat * 100)
            times.append(dt)
        best = min(times)
        rows.append({
            "name": name,
            "steps": STEPS,
            "envs": 1 if name == "serial" else N_ENVS,
            "seconds": round(best, 3),
            "steps_per_sec": round(STEPS / best, 1),
            "runs": [round(t, 3) for t in times],
        })
        print("%-7s %8.3fs  %8.1f steps/s" % (name, best, STEPS / best), flush=True)

    base = rows[0]["seconds"]
    for row in rows:
        row["speedup"] = round(base / row["seconds"], 2)

    payload = {
        "env": "CartPole-v1",
        "n_envs": N_ENVS,
        "steps": STEPS,
        "python": platform.python_version(),
        "gymnasium": gym.__version__,
        "rows": rows,
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "vecenv_bench.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
