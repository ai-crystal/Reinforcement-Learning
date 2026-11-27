# -*- coding: utf-8 -*-
"""Chapter 13 extra measurement: same loop, sync vs async vector envs.

The loop style follows run_vector_speed.py so the numbers are comparable
inside this file only: same step count, same random action sampling, same
reset handling.

    python books/code/hands-on/ch13/bench_async_overhead.py

Writes results/vector_async.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
STEPS = 20000


def make_env(seed: int):
    def thunk():
        env = gym.make("CartPole-v1")
        env.reset(seed=seed)
        return env
    return thunk


def run(kind: str, n_envs: int, steps: int) -> float:
    if kind == "serial":
        env = gym.make("CartPole-v1")
        rng = np.random.default_rng(0)
        obs, _ = env.reset(seed=0)
        t0 = time.perf_counter()
        for _ in range(steps):
            obs, _, term, trunc, _ = env.step(int(rng.integers(2)))
            if term or trunc:
                obs, _ = env.reset()
        dt = time.perf_counter() - t0
        env.close()
        return dt

    makers = [make_env(i) for i in range(n_envs)]
    cls = gym.vector.SyncVectorEnv if kind == "sync" else gym.vector.AsyncVectorEnv
    env = cls(makers)
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    per_env = max(1, steps // n_envs)
    t0 = time.perf_counter()
    for _ in range(per_env):
        actions = rng.integers(0, 2, size=n_envs)
        obs, _, term, trunc, _ = env.step(actions)
        if np.logical_or(term, trunc).any():
            obs, _ = env.reset()
    dt = time.perf_counter() - t0
    env.close()
    return dt


def main() -> None:
    data = {}
    for kind, n_envs in (("serial", 1), ("sync", 8), ("async", 8)):
        run(kind, n_envs, 2000)                      # warm up
        times = [run(kind, n_envs, STEPS) for _ in range(3)]
        best = min(times)
        data["%s_%d" % (kind, n_envs)] = {
            "seconds": round(best, 3),
            "steps_per_sec": round(STEPS / best, 1),
            "runs": [round(t, 3) for t in times],
        }
        print("%-6s envs=%d %6.3fs %8.1f steps/s"
              % (kind, n_envs, best, STEPS / best), flush=True)

    base = data["serial_1"]["seconds"]
    for key, row in data.items():
        row["speedup_vs_serial"] = round(base / row["seconds"], 2)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "vector_async.json"
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
