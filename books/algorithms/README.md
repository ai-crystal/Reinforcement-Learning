# 算法篇

> 从贝尔曼方程到 PPO 与离线强化学习

这一册回到公式，把前两册"这样做能行"的地方补上"为什么必须这样"。

从马尔可夫决策过程开始，推贝尔曼方程，讲清动态规划、时序差分、Q 学习之间的关系。然后进入函数逼近，解释换掉表格之后训练为什么变得不稳定。策略梯度部分给出定理的推导，说明优势函数和广义优势估计在降低什么。再往后是连续动作、离线强化学习、基于模型的方法，最后一章讲怎么核对结果。

共 20 章 · 108 页

[从第 1 章开始](chapters/ch01.md)　·　<a href="../../output/rl-v3-algorithms.pdf" target="_blank" rel="noopener" data-router-ignore>下载本册 PDF</a>

## 公式怎么读

每个公式都配一句人话解释。卡住时可以回到第一册找直觉，也可以跳到后面的代码看它在实现什么。书里的两个实验（随机游走、悬崖行走）都用真实运行结果画图。

## 目录

- [第 1 章　马尔可夫决策过程](chapters/ch01.md)
- [第 2 章　回报、价值与贝尔曼方程](chapters/ch02.md)
- [第 3 章　动态规划：策略迭代与值迭代](chapters/ch03.md)
- [第 4 章　蒙特卡洛与时序差分](chapters/ch04.md)
- [第 5 章　Q 学习与 SARSA：离轨与在轨](chapters/ch05.md)
- [第 6 章　函数逼近与致命三要素](chapters/ch06.md)
- [第 7 章　DQN 家族：每一个改动在修什么](chapters/ch07.md)
- [第 8 章　策略梯度定理与方差缩减](chapters/ch08.md)
- [第 9 章　Actor-Critic 与广义优势估计](chapters/ch09.md)
- [第 10 章　TRPO 与 PPO：为什么要裁剪](chapters/ch10.md)
- [第 11 章　连续动作：DDPG 与 SAC](chapters/ch11.md)
- [第 12 章　离线强化学习](chapters/ch12.md)
- [第 13 章　基于模型与探索](chapters/ch13.md)
- [第 14 章　奖励设计与奖励黑客](chapters/ch14.md)
- [第 15 章　约束强化学习与安全](chapters/ch15.md)
- [第 16 章　分布式强化学习](chapters/ch16.md)
- [第 17 章　多智能体算法](chapters/ch17.md)
- [第 18 章　基于模型的进阶](chapters/ch18.md)
- [第 19 章　逆强化学习与模仿](chapters/ch19.md)
- [第 20 章　评估、复现与样本效率](chapters/ch20.md)

---

[返回系列首页](../../README.md)
