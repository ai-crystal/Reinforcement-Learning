# -*- coding: utf-8 -*-
"""第 17 章实验：稀疏奖励与势函数塑形（稀疏网格）。

地图 10x10，四周是墙，中间几道隔墙，起点 (1,1)，终点 (8,8)，最多 200 步。
plain   只有到终点才给 +1
shaped  额外的势函数奖励：离终点越近，回报越高

运行： python books/code/hands-on/run_sparse_grid.py
输出： results/sparse_<variant>_s<seed>.json
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from rlkit import Result

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

H = W = 10
START, GOAL = (1, 1), (8, 8)
WALLS = {(r, 5) for r in range(1, 8)} | {(5, c) for c in range(3, 9)}
WALLS.discard((5, 5))
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_STEPS = 200


def step(state, action):
    dr, dc = ACTIONS[action]
    nr, nc = state[0] + dr, state[1] + dc
    if not (0 < nr < H - 1 and 0 < nc < W - 1) or (nr, nc) in WALLS:
        nr, nc = state
    done = (nr, nc) == GOAL
    return (nr, nc), done


def potential(state) -> float:
    return -(abs(state[0] - GOAL[0]) + abs(state[1] - GOAL[1]))


def train(variant: str, seed: int, episodes: int = 400,
          alpha=0.1, gamma=0.99) -> Result:
    rng = np.random.default_rng(seed)
    Q = np.zeros((H, W, 4))
    res = Result(name="sparse_%s_s%d" % (variant, seed),
                 config={"variant": variant, "seed": seed, "episodes": episodes})

    for ep in range(episodes):
        state = START
        eps = max(0.02, 0.3 * (1 - ep / episodes))
        steps, done = 0, False
        while not done and steps < MAX_STEPS:
            if rng.random() < eps:
                action = int(rng.integers(4))
            else:
                action = int(np.argmax(Q[state[0], state[1]]))
            nxt, done = step(state, action)
            if done:
                reward = 1.0
            elif variant == "shaped":
                reward = gamma * potential(nxt) - potential(state)
            else:
                reward = 0.0
            target = reward
            if not done:
                target += gamma * np.max(Q[nxt[0], nxt[1]])
            Q[state[0], state[1], action] += alpha * (
                target - Q[state[0], state[1], action])
            state = nxt
            steps += 1
        res.returns.append(1.0 if done else 0.0)
        res.lengths.append(steps)

    # 贪心评估
    wins, total_steps = 0, 0
    for _ in range(20):
        state, done, steps = START, False, 0
        while not done and steps < MAX_STEPS:
            action = int(np.argmax(Q[state[0], state[1]]))
            state, done = step(state, action)
            steps += 1
        wins += 1 if done else 0
        total_steps += steps
    res.extra["success_rate"] = wins / 20.0
    res.extra["avg_steps"] = total_steps / 20.0
    return res


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    variants = sys.argv[1:] or ["plain", "shaped"]
    for variant in variants:
        for seed in (0, 1, 2, 3, 4):
            t0 = time.time()
            res = train(variant, seed)
            res.save(RESULTS)
            print("%-6s s%d success=%.2f steps=%.1f (%.0fs)"
                  % (variant, seed, res.extra["success_rate"],
                     res.extra["avg_steps"], time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
