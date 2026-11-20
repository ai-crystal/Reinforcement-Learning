# -*- coding: utf-8 -*-
"""一个玩具级的 GRPO 演示。

任务：五道一位数加法。策略是一张查表（每个题目一组 logits），
不涉及语言模型，但更新方式和 GRPO 完全一致：
同一道题采样一组答案，用组内平均分做基准，按相对好坏更新概率。
奖励是答案是否正确，属于可验证奖励。

运行： python books/code/llm-rl/toy_grpo.py
输出： results/toy_grpo.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

TASKS = [(3, 4), (5, 2), (6, 7), (8, 1), (2, 9)]
ANSWERS = list(range(20))          # 候选答案 0..19
GROUP = 8                          # 每题采样几个答案
STEPS = 600
LR = 0.35


def sample(rng, logits, k):
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return rng.choice(len(ANSWERS), size=k, p=p)


def run(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    theta = np.zeros((len(TASKS), len(ANSWERS)))
    acc_curve, reward_curve = [], []

    for step in range(STEPS):
        task = step % len(TASKS)
        a, b = TASKS[task]
        correct = a + b
        picks = sample(rng, theta[task], GROUP)
        rewards = (picks == correct).astype(float)

        # GRPO 的核心：用组内均值做基准
        adv = rewards - rewards.mean()
        if adv.std() > 0:
            adv = adv / (adv.std() + 1e-8)

        for idx, sign in zip(picks, adv):
            one_hot = np.zeros(len(ANSWERS))
            one_hot[idx] = 1.0
            p = np.exp(theta[task] - theta[task].max())
            p /= p.sum()
            theta[task] += LR * sign * (one_hot - p)    # 策略梯度

        acc_curve.append(float(rewards.mean()))
        reward_curve.append(float(rewards.mean()))

    final = final_accuracy(theta, rng)
    return {"acc": acc_curve, "final": final}


def final_accuracy(theta, rng, samples: int = 400) -> float:
    hit = 0
    for task, (a, b) in enumerate(TASKS):
        picks = sample(rng, theta[task], samples)
        hit += float((picks == a + b).mean())
    return hit / len(TASKS)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    out = {}
    for seed in range(3):
        res = run(seed)
        out["seed%d" % seed] = res
        print("seed %d  final %.3f" % (seed, res["final"]))
    (RESULTS / "toy_grpo.json").write_text(json.dumps(out), encoding="utf-8")


if __name__ == "__main__":
    main()
