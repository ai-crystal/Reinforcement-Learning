# -*- coding: utf-8 -*-
"""把 results/ 里的曲线画成书里的插图时共用的小工具。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def load(pattern: str) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(RESULTS.glob(pattern))]


def moving(values, window: int = 10) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def band(ax, runs: list[dict], color, label: str, window: int = 10,
         key: str = "returns", fill: bool = True):
    """多组实验的中位数曲线，外面包一层四分位区间。"""
    curves = [moving(r[key], window) for r in runs]
    n = min(len(c) for c in curves)
    curves = np.array([c[:n] for c in curves])
    x = np.arange(n) + window // 2
    mid = np.median(curves, axis=0)
    ax.plot(x, mid, color=color, lw=1.5, label=label)
    if fill and len(runs) > 1:
        lo = np.percentile(curves, 25, axis=0)
        hi = np.percentile(curves, 75, axis=0)
        ax.fill_between(x, lo, hi, color=color, alpha=0.16, linewidth=0)
    return mid
