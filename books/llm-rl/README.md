# 大模型篇

> 从人类反馈到推理模型

这一册把强化学习放到语言模型上。状态是已经生成的文本，动作是下一个词，奖励来自人类偏好或者一段检查答案的程序。

同名的方法在这里换了样子。PPO 要同时放四个模型，DPO 把强化学习化简成一个分类问题，GRPO 干脆去掉价值网络，可验证奖励让推理能力第一次变成可以直接训练的东西。

共 19 章 · 113 页

[从第 1 章开始](chapters/ch01.md)　·　<a href="../../output/v4-llm.pdf" target="_blank" rel="noopener" data-router-ignore>下载本册 PDF</a>

## 目录

- [第 1 章　一个 token 就是一步动作](chapters/ch01.md)
- [第 2 章　奖励模型：把偏好变成一个分数](chapters/ch02.md)
- [第 3 章　PPO 训练大模型的工程现场](chapters/ch03.md)
- [第 4 章　DPO：把强化学习化简成分类](chapters/ch04.md)
- [第 5 章　偏好优化的变体：IPO、KTO、ORPO、SimPO](chapters/ch05.md)
- [第 6 章　RLVR：答案能检查时发生了什么](chapters/ch06.md)
- [第 7 章　GRPO：去掉价值网络](chapters/ch07.md)
- [第 8 章　R1 式训练：冷启动、拒绝采样与蒸馏](chapters/ch08.md)
- [第 9 章　测试时计算：采样、投票与搜索](chapters/ch09.md)
- [第 10 章　奖励黑客与对齐失败](chapters/ch10.md)
- [第 11 章　Agentic RL：多轮、工具与环境](chapters/ch11.md)
- [第 12 章　对齐的代价：谄媚、退化与多样性](chapters/ch12.md)
- [第 13 章　预训练、SFT 与 RL 的分工](chapters/ch13.md)
- [第 14 章　奖励模型训练实战](chapters/ch14.md)
- [第 15 章　PPO 实现：用 TRL 跑一遍](chapters/ch15.md)
- [第 16 章　DPO 的数学细节](chapters/ch16.md)
- [第 17 章　可验证奖励的数据工程](chapters/ch17.md)
- [第 18 章　蒸馏与小模型复现](chapters/ch18.md)
- [第 19 章　评测、复现与收尾](chapters/ch19.md)

---

[返回系列首页](../../README.md)
