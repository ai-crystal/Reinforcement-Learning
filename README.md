# 强化学习系列

> 概念 · 动手 · 算法 · 大模型，四册

从一次试错，到一个会推理的模型。

智能体做的事情，说起来只有一句：不知道答案，先做，看结果，再调整。

这套书按四册展开同一件事。第一册把概念和全景讲清楚，第二册在游戏里真的训练起来，第三册拆开算法的推导与实现，第四册回到今天最热的题目：大模型怎样用强化学习学会推理。

## 四册

| 册 | 书名 | 页数 | 一句话 | 适合谁 |
| --- | --- | --- | --- | --- |
| 01 | [概念篇 · 从老虎机到推理模型的全景图](books/concepts/README.md) | 149 | 21 章全景：概念、真实案例、大模型、多智能体、具身智能，不写公式 | 想看懂技术新闻、想纵览全貌的人 |
| 02 | [动手篇 · 在游戏里训练一个会决策的程序](books/hands-on/README.md) | 105 | 从 CartPole 到 PPO，再加并行采样、连续动作、稀疏奖励与部署 | 会一点 Python、想让程序真的学会一件事的人 |
| 03 | [算法篇 · 从贝尔曼方程到 PPO 与离线强化学习](books/algorithms/README.md) | 77 | 贝尔曼方程、时序差分、策略梯度、GAE、SAC 的推导与代码 | 要读懂论文、要自己改算法的人 |
| 04 | [大模型篇 · 从人类反馈到推理模型](books/llm-rl/README.md) | 112 | RLHF、DPO、GRPO、RLVR，再到奖励模型实战、TRL 实现与蒸馏 | 关心大模型训练、想做对齐与 Agent 的人 |

四册可以按顺序读，也可以从最关心的一册开始。第二册之后各册会反复用到第一册的术语，遇到不熟的概念回去翻一页就够。

## 在线阅读

- [概念篇](https://ai-crystal.github.io/Reinforcement-Learning/#/books/concepts/README)
- [动手篇](https://ai-crystal.github.io/Reinforcement-Learning/#/books/hands-on/README)
- [算法篇](https://ai-crystal.github.io/Reinforcement-Learning/#/books/algorithms/README)
- [大模型篇](https://ai-crystal.github.io/Reinforcement-Learning/#/books/llm-rl/README)

每一章都有独立地址，可以直接分享给朋友。

## 下载 PDF

- [概念篇 PDF](output/rl-v1-concepts.pdf)
- [动手篇 PDF](output/rl-v2-hands-on.pdf)
- [算法篇 PDF](output/rl-v3-algorithms.pdf)
- [大模型篇 PDF](output/rl-v4-llm.pdf)

## 书里的代码

实验代码在 `books/code/`，每一章一个目录，图和曲线都由脚本生成，跑一遍就能复现书里的结果。绘图脚本负责插图，训练脚本负责曲线，两者都不依赖书写工具。

作者：Dr. Dezeming，2026
