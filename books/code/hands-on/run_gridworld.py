# -*- coding: utf-8 -*-
"""表格 Q 学习：在网格世界里学会走到终点。

运行： python books/code/hands-on/run_gridworld.py
输出： results/gridworld_q.json，训练曲线与学到的 Q 表
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gridworld import ACTIONS, GridWorld

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def epsilon_at(step: int, total: int) -> float:
    """从 0.3 线性降到 0.02。"""
    start, end = 0.3, 0.02
    return max(end, start - (start - end) * step / total)


def train(episodes: int = 500, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    env = GridWorld()
    n_states, n_actions = env.obs_dim, env.n_actions
    Q = np.zeros((n_states, n_actions))

    history = {"return": [], "steps": [], "success": []}
    for ep in range(episodes):
        state, _ = env.reset()
        done, total, steps = False, 0.0, 0
        while not done:
            eps = epsilon_at(ep * 200 + steps, episodes * 200)
            if rng.random() < eps:
                action = int(rng.integers(n_actions))
            else:
                action = int(np.argmax(Q[state]))
            out = env.step(action)
            target = out.reward
            if not out.terminated:
                target += 0.95 * np.max(Q[out.state])
            Q[state, action] += 0.1 * (target - Q[state, action])
            state = out.state
            total += out.reward
            steps += 1
            done = out.terminated or out.truncated
        history["return"].append(total)
        history["steps"].append(steps)
        history["success"].append(1 if total > 0.5 else 0)
    return history, Q


def evaluate(Q: np.ndarray, episodes: int = 100) -> dict:
    env = GridWorld()
    wins, steps = 0, []
    for _ in range(episodes):
        state, _ = env.reset()
        done, n = False, 0
        while not done:
            state, reward, term, trunc, _ = env.step(int(np.argmax(Q[state])))
            done = term or trunc
            n += 1
        if reward > 0.5:
            wins += 1
            steps.append(n)
    return {"success_rate": wins / episodes,
            "avg_steps_when_success": float(np.mean(steps)) if steps else None}


def random_baseline(episodes: int = 200, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    env = GridWorld()
    wins = 0
    for _ in range(episodes):
        _, _ = env.reset()
        done, total = False, 0.0
        while not done:
            _, reward, term, trunc, _ = env.step(int(rng.integers(env.n_actions)))
            total += reward
            done = term or trunc
        wins += 1 if total > 0.5 else 0
    return {"success_rate": wins / episodes}


def main() -> None:
    history, Q = train()
    summary = evaluate(Q)
    RESULTS.mkdir(exist_ok=True)
    payload = {
        "history": history,
        "evaluation": summary,
        "random": random_baseline(),
        "Q": Q.round(4).tolist(),
    }
    (RESULTS / "gridworld_q.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print("evaluation:", summary)
    print("random:", payload["random"])


if __name__ == "__main__":
    main()
