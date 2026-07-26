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
| 07-25 | 余弦退火 + 周期早停 | ReplaceLR → CosineAnnealing |
| 07-25 | RecDataset 去 clone + GPU matrix | 20s/epoch → 14s/epoch |
| 07-26 | 发现 evaluate_ranking 未屏蔽训练集物品 | Recall 被训练物品占坑压低 |

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

## 余弦退火 + 周期早停

### 问题

`ReduceLROnPlateau` 盯 BPR val_loss，但 CSWR 下 loss 每 cycle 重启会跳涨 → 误触发降 lr 或早停。

### 修复

- `ReduceLROnPlateau` → `CosineAnnealingWarmRestarts(T_0, T_mult)`
- early_stop 改为周期级：连续 2 个 cosine cycle 无改善则停
- `cycle_improved` 跟踪整个周期内是否有任何 epoch 破纪录
- 每模型独立 `cosine_T0`：MLP=5, NCF=5, VAE=10

### 效果

- MLP 15 epoch 内自动停，不浪费
- NCF 可跑满 15-25 epoch
- VAE 50 epoch 自然收敛

---

## 三模型差异化训练路径

| 模型 | 训练路径 | 早停依据 | cosine_T0 |
|------|---------|---------|-----------|
| MLP | BPR pairwise | val_loss（周期级） | 5 |
| NCF | BPR pairwise | val_loss（周期级） | 5 |
| VAE | `train_one_epoch` (CE+KL autoencoder) | train_loss（周期级） | 10 |

VAE 走独立的 `train_one_epoch`——重建完整用户向量，不再塞进 pairwise batch 循环。

---

## 训练速度优化

| 改动 | 位置 | 效果 |
|------|------|------|
| `__getitem__` 返回 `int` 而非 `clone()` tensor | RecDataset | 消除 153,600 次 clone/epoch |
| `user_item_matrix` 预加载到 GPU | pipeline | 消除 CPU→GPU 拷贝 |
| VAE 走单次 GPU matmul | evaluate_ranking | 19 forward → 1 forward |
| MLP 走 `u @ weight` | evaluate_ranking | 19 forward → 1 matmul |

总：20s/epoch → 14s/epoch（MLP/NCF），VAE eval 112s → 6s。

---

## evaluate_ranking 未屏蔽训练集物品（待修）

SPINRec 原文 `masked_fill(train_matrix.bool(), -inf)`。当前代码对全部物品排序取 top-K，训练物品天然高分占坑，Recall 被系统性压低。

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
