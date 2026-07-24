# Experiment Notes

> 关键问题、根因、修复、结论。一句话能说完的不写两句。

---

## 时间线

| 日期 | 里程碑 | 关键指标变化 |
|------|--------|------------|
| 07-23 | 修复配置问题，训练能跑 | AUC 0.50 → 0.70 |
| 07-23 | 发现 AUC 天花板（假阴性 ~60%） | 引入 BPR + Recall@K |
| 07-23 | BPR+Embedding 得 Recall@20=0.007 | 确认 Embedding 架构行不通 |
| 07-23 | 切到 Linear-multi-hot 架构 | Recall@20: 0.007 → 0.18 |
| 07-24 | VAE 修复为 SPINRec 原版 autoencoder | AUC 0.71 → 0.97 |
| 07-24 | 三种模型评估加速 | VAE eval: 112s → 6s |

---

## 核心发现

### 1. Embedding vs Linear-multi-hot（架构即天花板）

SPINRec 用 `Linear(num_items, hidden)` 输入 multi-hot 用户向量，每个物品是独立特征。
我们用 `Embedding(user_id)` 把用户压成 32 维向量 — 梯度信号太密，SIF/DVF 无区分度。

| 指标 | Embedding | Linear |
|------|-----------|--------|
| Recall@20 | 0.007 | 0.18 |
| 可解释性 | embedding 维度无意义 | 每个维度 = 一个物品 |

### 2. 三种模型需不同训练策略（SPINRec 的设计）

| 模型 | 训练范式 | 不能混用 |
|------|---------|---------|
| LinearRec | pairwise (BPR/BCE) | — |
| LinearNCF | pairwise (BPR) | — |
| LinearVAE | autoencoder (CE + KL) | 不能塞进 BPR 循环 |

强行统一训练策略会压死 VAE（我们 VAE 初版 AUC 0.71，修复后 0.97）。

### 3. 当前三轮训练结果（Pinterest 全量）

| 模型 | AUC | Recall@20 | NDCG@20 | 收敛速度 |
|------|-----|-----------|---------|---------|
| MLP | 0.93 | 0.18 | 0.27 | 10 epoch 内 |
| NCF | 0.94 | 0.10 | 0.13 | 需更强正则 |
| VAE | 0.97 | 待测 | 待测 | 30+ epoch |

### 4. BPR 下 val_loss 与 AACC 不反映过拟合

BPR 优化 `pos > neg` 不控绝对值，val_BCE 必然上涨。scheduler/early_stop 应盯 AUC 而非 BCE loss。
VAE 的 softmax 输出套在 BPR evaluate 里 acc=0.5 是 cosmetic issue，AUC 不受影响。

---

## 评估加速方案

| 模型 | 旧 | 新 | 空间安全 |
|------|-----|-----|---------|
| LinearRec | 19 forward | 1 matmul (128×n_items) | ≤ 100M items |
| LinearVAE | 19 forward | 1 forward | ≤ 50M items |
| LinearNCF | 19 forward | ~10 forward @ 1024 batch | 任意大小 |

---

## 旧版 Bug（已修复）

| Bug | 位置 | 影响 |
|-----|------|------|
| `make_loaders` 体损坏 | functions | return 后跟不可达代码 |
| IF `zip(flat, params)` | benchmark | 标量×张量矩阵，分数无意义 |
| VAE 丢 KL loss | train_model | `(logit, mu, logvar)` 丢弃后两个 |
| DVF/TracIn 共享计时 | benchmark | 共用 `t0`，报告相同耗时 |
| `evaluate_model`×3 | functions | 重复定义，仅末次生效 |
| `neg_ratio` 硬编码 | data_prep | 用户改不了负采样比例 |
| VAE 错塞进 BPR 训练循环 | functions | AUC 在 0.71 停滞 |

---

*最后更新: 2026-07-24*
