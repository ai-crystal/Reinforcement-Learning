# -*- coding: utf-8 -*-
"""第 9 章的补充实验：裁剪范围与 GAE 的 lambda。

运行： python books/code/hands-on/sweep_ppo_clip.py
输出： results/ppo_clip<k>_s<seed>.json, results/ppo_lam<k>_s<seed>.json
"""
from __future__ import annotations

import sys
from pathlib import Path

from rlkit import Result

import run_ppo

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "clip"

    if mode == "clip":
        for clip in [0.02, 0.2, 0.5]:
            for seed in [0, 1]:
                res = run_ppo.train(seed, clip=clip)
                res.name = "ppo_clip%s_s%d" % (clip, seed)
                res.config["clip"] = clip
                print(res.name, "eval=%.1f" % res.extra["eval_return"],
                      flush=True)
                res.save(RESULTS)
    else:
        for lam in [0.0, 0.95, 1.0]:
            for seed in [0, 1]:
                res = run_ppo.train(seed, lam=lam)
                res.name = "ppo_lam%s_s%d" % (lam, seed)
                res.config["lam"] = lam
                print(res.name, "eval=%.1f" % res.extra["eval_return"],
                      flush=True)
                res.save(RESULTS)


if __name__ == "__main__":
    main()
