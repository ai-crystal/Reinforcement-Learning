# -*- coding: utf-8 -*-
"""Chapter 16 baseline: how often does a random policy reach the goal?

5000 episodes of uniform random actions on the same 10x10 maze used by
run_sparse_grid.py, 200 steps per episode. Also reports the shortest path
length by breadth-first search.

    python books/code/hands-on/ch16/random_baseline.py

Writes results/sparse_random_baseline.json
"""
from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import run_sparse_grid as G

RESULTS = HERE.parent / "results"


def main() -> None:
    rng = np.random.default_rng(0)
    n = 5000
    succ, steps = 0, []
    for _ in range(n):
        s = G.START
        for t in range(G.MAX_STEPS):
            s, done = G.step(s, int(rng.integers(4)))
            if done:
                succ += 1
                steps.append(t + 1)
                break

    dist = {G.START: 0}
    queue = deque([G.START])
    while queue:
        cur = queue.popleft()
        for a in range(4):
            nxt, _ = G.step(cur, a)
            if nxt not in dist and nxt != cur:
                dist[nxt] = dist[cur] + 1
                queue.append(nxt)

    payload = {
        "episodes": n,
        "success": succ,
        "success_rate": round(succ / n, 4),
        "mean_steps_on_success": round(float(np.mean(steps)), 2) if steps else None,
        "shortest_path": dist.get(G.GOAL),
        "walls": sorted(G.WALLS),
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "sparse_random_baseline.json"
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
