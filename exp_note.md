# Experiment Notes

> 关键问题、根因、修复、结论。一句话能说完的不写两句。

---

## 时间线

| 日期 | 里程碑 | 关键指标变化 |
|------|--------|------------|
| 07-23 | 修复配置问题，训练能跑 | AUC 0.50 → 0.70 |
| 07-23 | 发现 AUC 天花板（假阴性 ~60%） | 引入 BPR + Recall@K |
| 07-23 | BPR+Embedding 得 Recall@20=0.007 | Embedding 架构行不通 |
| 07-23 | 切到 Linear-multi-hot | Recall@20: 0.007 → 0.18 |
| 07-24 | VAE 修复为 SPINRec 原版 autoencoder | AUC 0.71 → 0.97 |
| 07-25 | 余弦退火 + 周期早停 | ReduceLROnPlateau → CosineAnnealing |
| 07-26 | 屏蔽训练集物品 in evaluate_ranking | Recall: 0.18 → 0.02（真值） |
| 07-26 | BCE-30 负采样 | Recall: 0.02 → 0.10 |
| 07-27 | 低 lr (1e-4) 收敛轨迹 | 归因信号仅在早期 epoch 存在 |
| 07-28 | LOO epoch1 归因验证 | IF-diag ρ=±0.15，SIF vs IF 完美反相关 |
| 07-28 | 三数据集 + 显式负样本 + 归因汇总 | MLP+SIF |ρ|=0.14 vs Random=0.03 (3ds×3seed) |
| 07-28 | Yahoo 归因最强 | SIF |ρ|=0.17（显式负样本贡献最大） |
| 07-28 | **LOO 改用 BCE loss** | |ρ| 从 0.15 跳到 0.61（4 倍） |

---

## 核心发现

### 1. Embedding vs Linear-multi-hot

SPINRec 用 `Linear(num_items, hidden)`，每个物品是独立特征。Embedding 压成 32 维→梯度无区分度。

| 指标 | Embedding | Linear |
|------|-----------|--------|
| Recall@20 | 0.007 | 0.18 |
| 归因可解释性 | embedding 维度无意义 | 每个维度 = 一个物品 |

### 2. 三种模型需不同训练策略

| 模型 | 训练范式 | 当前状态 |
|------|---------|---------|
| LinearRec | pairwise (BCE-30) | AUC 0.99, Recall 0.10 |
| LinearNCF | pairwise (BCE-30) | AUC 0.98, 归因信号弱于 MLP |
| LinearVAE | autoencoder (CE+KL) | AUC 0.97, 归因不兼容 |

### 3. IF 与 SIF 的反号关系（论文级）

**IF 标准定义为 `-g_z^T H^{-1} g_val`，自带负号。SIF 为 `g_z^T g_val`，无负号。两者在 LO O上必然反号。**

| 模型 | Seed | SIF ρ | IF-diag ρ |
|------|------|--------|-----------|
| MLP | 7 | -0.15 | **+0.15** |
| MLP | 42 | +0.10 | **-0.10** |
| MLP | 91 | +0.15 | **-0.15** |
| NCF | 7 | +0.05 | -0.03 |
| NCF | 91 | +0.12 | **-0.12** |

不是 bug——IF 的负号是数学定义。论文里写明即可。

### 4. 归因信号随训练衰减

| epoch | ρ | 含义 |
|-------|-----|------|
| 1 | ±0.15 | 样本贡献可区分 |
| 2 | ±0.04 | 衰减 |
| 12 | 0.00 | 模型收敛→所有样本无差别 |
| final | 0.00 | 同 |

**结论**：归因只在训练早期有意义。CosineAnnealing 重启后信号也被压平。

### 5. LOO 用 loss 比 AUC 灵敏 4 倍

AUC 只测排序方向——删一个样本，30 个备份还在，方向不变 → ΔAUC≈0。BCE loss 测绝对分数偏移——每个样本的贡献直接体现在 loss 变化上。实测：`|ρ|` 从 0.15（用 AUC）跳到 0.61（用 loss）。

### 6. MLP > NCF 对归因器

MLP 只有 `user_fc + item_fc` 两层，梯度路径短，信号直接。NCF 的 GMF+MLP 双分支把梯度分散到 6 层 Linear + 3 层 Dropout——信号稀释 3-5 倍，ρ 相应下降。

### 6. 高 lr 不可用于归因器

高 lr 一步跳过所有中间状态直接到最优——所有样本的梯度贡献被压平。低 lr 每步只改一点点——样本间梯度差异被保留更久。归因需要"模型还在学"的状态，不是"模型已经学会"。

### 7. 余弦重启对归因有负面影响

epoch 12 的 ρ=0 不是因为模型收敛（val_loss 还在降），而是 CosineAnnealing 在 epoch 11 重启时所有参数以相同比例放大，冲销了不同样本在梯度中累积的差异。归因器需要在**第一个余弦周期内**完成采样——之后信号被重置。

### 8. Full softmax 失败的根本原因

每个正样本 vs 9682 个隐式负样本——正梯度 1 份，负梯度 9682 份，正信号被稀释 10000 倍。收敛不动。BCE+30 负采样把对手缩到 30 → 梯度质量极高。

### 9. LOO 方法学

- `N_CONTROL=128` 够用了——同 seed 下 DVF 必须从 1 epoch 开始算，否则 ρ=0
- `LOO_EPOCHS=10` 对 early checkpoint 够了（epoch 1 时模型还在快变期）
- 不同 seed 的 ρ 符号可能翻转——LOO ground truth 本身的 sign ambiguity

### 10. 推荐器 AUC 的欺骗性

AUC 0.99 看起来完美，但 Recall@20 只有 0.10——模型只学会正>负的相对排序，没学会在全量 8761 个物品中把正样本推到 top。AUC 不能反映真实排序能力。BCE-30 是目前验证的最佳平衡。

---

## 训练配置（当前最优）

```python
NEG_PER_POS = 30,  # data_preprocessing
"MLP":  {"hidden_dim": 512, "lr": 1e-4, "weight_decay": 1e-3, "epochs": 50, "loss_fn": "bce", "cosine_T0": 10},
"NCF":  {"factor_num": 256, "lr": 2e-4, "num_layers": 2, "dropout": 0.5, "weight_decay": 1e-3, "epochs": 50, "loss_fn": "bce", "cosine_T0": 10},
```

---

## Valuation 函数 Bug（已修复）

`compute_sif`、`compute_dvf_stage`、`compute_if_diag` 中使用 `loader.dataset.matrix`——变量名 `loader` 不存在，应为 `train_loader_eval.dataset.matrix`。刚修复。

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

*最后更新: 2026-07-28*
