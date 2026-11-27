# -*- coding: utf-8 -*-
"""网格世界上的参数扫描，第 5 章用。

分别扫学习率、折扣因子、探索衰减和打滑概率，看成功率与步数怎么变。
运行： python books/code/hands-on/sweep_gridworld.py
输出： results/grid_sweep_<tag>.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gridworld import GridWorld

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def epsilon_at(step: int, total: int, start: float, end: float) -> float:
    return max(end, start - (start - end) * step / total)


def train(episodes: int = 500, seed: int = 0, alpha: float = 0.1,
          gamma: float = 0.95, eps_start: float = 0.3,
          eps_end: float = 0.02, slip: float = 0.0) -> dict:
    rng = np.random.default_rng(seed)
    env = GridWorld(slip=slip)
    Q = np.zeros((env.obs_dim, env.n_actions))
    success, steps = [], []

    for ep in range(episodes):
        state, _ = env.reset()
        done, total, n = False, 0.0, 0
        while not done:
            eps = epsilon_at(ep * 200 + n, episodes * 200, eps_start, eps_end)
            if rng.random() < eps:
                action = int(rng.integers(env.n_actions))
            else:
                action = int(np.argmax(Q[state]))
            out = env.step(action)
            target = out.reward
            if not out.terminated:
                target += gamma * np.max(Q[out.state])
            Q[state, action] += alpha * (target - Q[state, action])
            state = out.state
            total += out.reward
            n += 1
            done = out.terminated or out.truncated
        success.append(1 if total > 0.5 else 0)
        steps.append(n)

    # 训练结束后用贪心策略单独评估，不探索
    eval_env = GridWorld(slip=slip)
    wins, ok_steps = 0, []
    for _ in range(100):
        state, _ = eval_env.reset()
        done, n = False, 0
        while not done:
            state, reward, term, trunc, _ = eval_env.step(int(np.argmax(Q[state])))
            n += 1
            done = term or trunc
        if reward > 0.5:
            wins += 1
            ok_steps.append(n)

    reach = next((i for i in range(len(success) - 20)
                  if np.mean(success[i:i + 20]) >= 0.9), None)
    return {
        "config": {"episodes": episodes, "seed": seed, "alpha": alpha,
                   "gamma": gamma, "eps_start": eps_start,
                   "eps_end": eps_end, "slip": slip},
        "success_rate": wins / 100.0,
        "avg_steps_when_success": float(np.mean(ok_steps)) if ok_steps else None,
        "episode_to_90pct": reach,
        "final_success_100": float(np.mean(success[-100:])),
        "success_curve": success,
        "steps_curve": steps,
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    grid = {
        "alpha": [0.01, 0.1, 0.5, 0.9],
        "gamma": [0.5, 0.9, 0.95, 0.99],
        "eps": [(0.3, 0.3), (0.3, 0.02), (0.05, 0.02)],
        "slip": [0.0, 0.1, 0.3],
    }
    out = {}
    for alpha in grid["alpha"]:
        out["alpha_%s" % alpha] = train(alpha=alpha, seed=0)
    for gamma in grid["gamma"]:
        out["gamma_%s" % gamma] = train(gamma=gamma, seed=0)
    for start, end in grid["eps"]:
        out["eps_%s_%s" % (start, end)] = train(eps_start=start, eps_end=end, seed=0)
    for slip in grid["slip"]:
        out["slip_%s" % slip] = train(slip=slip, seed=0)

    (RESULTS / "grid_sweep.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8")
    for key, rec in out.items():
        print("%-14s success=%.2f steps=%s reach90=%s"
              % (key, rec["success_rate"], rec["avg_steps_when_success"],
                 rec["episode_to_90pct"]))


if __name__ == "__main__":
    main()
