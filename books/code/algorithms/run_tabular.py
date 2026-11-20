# -*- coding: utf-8 -*-
"""两个小实验：随机游走上的蒙特卡洛与时序差分，悬崖行走上的 Q 学习与 SARSA。

运行： python books/code/algorithms/run_tabular.py
输出： results/mc_td.json, results/cliff.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


# ---------------------------------------------------------------- 随机游走

def true_values(n_states: int = 5) -> np.ndarray:
    """0 和 n+1 是终止状态，中间状态的真实价值是从 i 出发最终到右端的概率。"""
    return np.arange(1, n_states + 1) / (n_states + 1.0)


def random_walk_episode(rng, n_states: int = 5):
    state = (n_states + 1) // 2
    states, rewards = [state], []
    while True:
        state += 1 if rng.random() < 0.5 else -1
        states.append(state)
        if state == 0:
            rewards.append(0.0)
            return states, rewards
        if state == n_states + 1:
            rewards.append(1.0)
            return states, rewards
        rewards.append(0.0)


def mc_td(alpha: float, episodes: int, seed: int, method: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    V = np.full(7, 0.5)
    V[0] = V[6] = 0.0
    errors = []
    truth = true_values()
    for _ in range(episodes):
        states, rewards = random_walk_episode(rng)
        if method == "mc":
            G = 0.0
            for t in reversed(range(len(rewards))):
                G = rewards[t] + G
                s = states[t]
                if 0 < s < 6:
                    V[s] += alpha * (G - V[s])
        else:
            for t in range(len(rewards)):
                s, s2 = states[t], states[t + 1]
                if 0 < s < 6:
                    target = rewards[t]
                    if 0 < s2 < 6:
                        target += V[s2]
                    V[s] += alpha * (target - V[s])
        errors.append(float(np.sqrt(np.mean((V[1:6] - truth) ** 2))))
    return np.array(errors)


# ---------------------------------------------------------------- 悬崖行走

CLIFF = (4, 12)
CLIFF_START = (3, 0)
CLIFF_GOAL = (3, 11)
CLIFF_ZONE = [(3, c) for c in range(1, 11)]


def cliff_step(state, action, rng):
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    r, c = state
    dr, dc = moves[action]
    nr, nc = min(max(r + dr, 0), CLIFF[0] - 1), min(max(c + dc, 0), CLIFF[1] - 1)
    if (nr, nc) in CLIFF_ZONE:
        return CLIFF_START, -100.0, False
    if (nr, nc) == CLIFF_GOAL:
        return (nr, nc), -1.0, True
    return (nr, nc), -1.0, False


def eps_greedy(Q, state, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(4))
    return int(np.argmax(Q[state[0], state[1]]))


def cliff_run(method: str, episodes: int = 500, alpha: float = 0.5,
              eps: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((CLIFF[0], CLIFF[1], 4))
    returns = []
    for _ in range(episodes):
        state = CLIFF_START
        action = eps_greedy(Q, state, eps, rng)
        total = 0.0
        while True:
            nxt, reward, done = cliff_step(state, action, rng)
            nxt_action = eps_greedy(Q, nxt, eps, rng)
            if method == "q":
                target = reward
                if not done:
                    target += 0.95 * np.max(Q[nxt[0], nxt[1]])
            else:
                target = reward
                if not done:
                    target += 0.95 * Q[nxt[0], nxt[1], nxt_action]
            Q[state[0], state[1], action] += alpha * (
                target - Q[state[0], state[1], action])
            total += reward
            state = nxt
            action = nxt_action
            if done:
                break
        returns.append(total)
    return returns, Q


def main() -> None:
    RESULTS.mkdir(exist_ok=True)

    # 随机游走：MC 与 TD 在不同步长下的误差
    curve = {}
    for method in ("mc", "td"):
        for alpha in (0.05, 0.1, 0.15):
            runs = np.array([mc_td(alpha, 200, seed, method) for seed in range(30)])
            curve["%s_%s" % (method, alpha)] = runs.mean(axis=0).round(4).tolist()
    (RESULTS / "mc_td.json").write_text(json.dumps(curve), encoding="utf-8")

    # 悬崖行走：两种算法的每局奖励
    cliff = {}
    for method in ("q", "sarsa"):
        runs = [cliff_run(method, seed=s)[0] for s in range(10)]
        cliff[method] = np.array(runs).mean(axis=0).round(2).tolist()
    (RESULTS / "cliff.json").write_text(json.dumps(cliff), encoding="utf-8")
    print("saved", RESULTS)


if __name__ == "__main__":
    main()
