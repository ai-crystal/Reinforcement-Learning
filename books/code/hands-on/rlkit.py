# -*- coding: utf-8 -*-
"""第二册用到的小工具箱。

只依赖 gymnasium、numpy、torch，能在 CPU 上跑，也能用显卡。
本文件同时是书里的代码样本，改动时请同步第 2 册正文。
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------- 基础工具

def set_seed(seed: int) -> None:
    """把三个随机源都固定住，保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    """有显卡就用显卡，没有就退回 CPU。

    设环境变量 RLKIT_DEVICE=cpu 可以强制用 CPU，方便对比或复现。
    """
    forced = os.environ.get("RLKIT_DEVICE", "").lower()
    if forced == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def smooth(values, window: int = 10) -> np.ndarray:
    """滑动平均，只为了让曲线好看，不改变数据本身。"""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


@dataclass
class Result:
    """一次实验的记录：曲线 + 关键指标 + 配置。"""
    name: str
    returns: list = field(default_factory=list)
    lengths: list = field(default_factory=list)
    config: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def save(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        path = root / (self.name + ".json")
        payload = {
            "name": self.name,
            "returns": [round(float(v), 4) for v in self.returns],
            "lengths": [int(v) for v in self.lengths],
            "config": self.config,
            "extra": self.extra,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path


def load_results(root: Path, pattern: str) -> list[dict]:
    out = []
    for path in sorted(root.glob(pattern)):
        out.append(json.loads(path.read_text(encoding="utf-8")))
    return out


# ---------------------------------------------------------------- 评估

@torch.no_grad()
def evaluate(env, policy_fn, episodes: int = 20, seed: int = 1234) -> float:
    """用固定种子跑若干局，返回平均回报。

    policy_fn(obs) 返回一个动作，内部要不要探索由调用方决定；
    评估时应该关掉随机性。
    """
    total = 0.0
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
    return total / episodes
