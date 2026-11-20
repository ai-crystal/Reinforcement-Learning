# -*- coding: utf-8 -*-
"""一个最小的网格世界，接口照 gymnasium 的样子写。

地图用字符画描述：
    S 起点，G 终点（+1），X 陷阱（-1），# 墙，. 空地
每一步奖励 -0.01，走到终点或陷阱后回合结束。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAP = [
    "S..#....",
    ".#.#.##.",
    ".#...#..",
    ".#.#.#.#",
    "...#...G",
]

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # 上 下 左 右
ACTION_NAMES = ["上", "下", "左", "右"]


@dataclass
class StepResult:
    state: int
    reward: float
    terminated: bool
    truncated: bool
    info: dict

    def __iter__(self):
        """允许像 gymnasium 那样直接解包五个返回值。"""
        return iter((self.state, self.reward, self.terminated,
                     self.truncated, self.info))


class GridWorld:
    def __init__(self, slip: float = 0.0, max_steps: int = 200):
        self.grid = [list(row) for row in MAP]
        self.h, self.w = len(self.grid), len(self.grid[0])
        self.slip = slip
        self.max_steps = max_steps
        self.start = self._find("S")
        self.obs_dim = self.h * self.w
        self.n_actions = len(ACTIONS)
        self.reset()

    def _find(self, ch: str) -> tuple[int, int]:
        for r in range(self.h):
            for c in range(self.w):
                if self.grid[r][c] == ch:
                    return r, c
        raise ValueError("map has no " + ch)

    def reset(self, seed: int | None = None):
        if seed is not None:
            np.random.seed(seed)
        self.pos = self.start
        self.steps = 0
        return self._state(), {}

    def _state(self) -> int:
        r, c = self.pos
        return r * self.w + c

    def _cell(self, r: int, c: int) -> str:
        if 0 <= r < self.h and 0 <= c < self.w:
            return self.grid[r][c]
        return "#"

    def step(self, action: int) -> StepResult:
        dr, dc = ACTIONS[action]
        if self.slip > 0 and np.random.rand() < self.slip:
            dr, dc = ACTIONS[np.random.randint(len(ACTIONS))]
        r, c = self.pos
        nr, nc = r + dr, c + dc
        if self._cell(nr, nc) == "#":
            nr, nc = r, c
        self.pos = (nr, nc)
        self.steps += 1

        cell = self._cell(nr, nc)
        reward, terminated = -0.01, False
        if cell == "G":
            reward, terminated = 1.0, True
        elif cell == "X":
            reward, terminated = -1.0, True
        truncated = self.steps >= self.max_steps
        return StepResult(self._state(), reward, terminated, truncated, {})

    def render_text(self) -> str:
        rows = []
        for r in range(self.h):
            line = []
            for c in range(self.w):
                if (r, c) == self.pos:
                    line.append("A")
                else:
                    line.append(self.grid[r][c])
            rows.append("".join(line))
        return "\n".join(rows)


if __name__ == "__main__":
    env = GridWorld()
    obs, _ = env.reset()
    print(env.render_text())
    for _ in range(6):
        out = env.step(np.random.randint(4))
        print("action ->", out.state, out.reward, out.terminated)
