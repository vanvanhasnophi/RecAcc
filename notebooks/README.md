# Notebooks (Refactored)

SPINRec 风格推荐系统训练与归因评估管线。本目录于 2026-07-23 对原 `notebooks/` 进行重构，
取其精华去其糟粕。

## 目录结构

```
notebooks/
├── README.md
│
├── 📦 共享函数库（核心依赖）
│   ├── functions.ipynb               # 模型定义、训练、评估、归因（SIF/DVF）
│   └── baselines_functions.ipynb     # 传统基线：Influence Function、SHAP
│
├── 🔄 数据预处理
│   └── data_preprocessing.ipynb      # 解析 → 缓存 PreparedData
│
├── 🏋️ 推荐器训练
│   └── rec_training.ipynb            # MLP / VAE / NCF 训练管线
│
├── 📊 归因评测
│   └── evaluate.ipynb               # SIF/DVF 估值 + 归因器对比（含 LOO Ground Truth）
│
├── 🎯 统一 Benchmark
│   └── benchmark.ipynb              # 反事实评估（Deletion/Insertion 曲线、NDCG@K、统计检验）
│
└── 🧪 快速验证
    ├── MLP.ipynb                     # MLP 3-epoch 快速验证
    ├── NCF.ipynb                     # NCF 3-epoch 快速验证
    ├── VAE.ipynb                     # VAE 3-epoch 快速验证
    ├── benchmark MLP.ipynb           # MLP 专用 benchmark 变体
    └── benchmark_VAE.ipynb           # VAE 专用 benchmark 变体
```

## 执行顺序

```
1. data_preprocessing.ipynb   →  生成 log/notebook_cache/prepared_data.pt
2. rec_training.ipynb         →  训练模型，输出到 log/notebook_runs/step2_training/
3. evaluate.ipynb             →  归因器评测 + LOO Ground Truth
4. benchmark.ipynb            →  统一反事实评估协议
```

## 相比原 notebooks/ 的主要修复

### Bug 修复
- **`make_loaders` 函数体损坏** — 已完全重写，移除不可达代码
- **VAE 缺失 KL 散度** — 训练时自动计算 β-VAE loss，支持 warmup
- **Influence Function `zip(flat, params)` 致命错误** — 改用 element-wise 对角 Fisher 近似
- **DVF/TracIn 共用计时器** — 每个方法独立计时
- **函数被定义 3 次** — 每个函数唯一定义，清除冗余

### 设计改进
- **Ground Truth 方案** — 新增 LOO Control Set（64 样本 × 3 seed 平均）作为可靠基准
- **类别平衡阈值** — `class_balance_threshold` 从 3.0 降至 1.5（可配置）
- **负采样比例** — `train_neg_ratio` 从硬编码 1 恢复为默认 4
- **归因方法** — 聚焦 SIF / SIF+ / IF-diag / DVF，移除无效的 Shapley 和 IF-cg

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NUM_BUCKETS` | 50000 | 哈希桶数量 |
| `vae_beta` | 0.01 | VAE KL 散度权重 |
| `vae_beta_warmup_epochs` | 5 | KL loss 线性 warmup 轮数 |
| `class_balance_threshold` | 1.5 | pos_weight 启用的最小正负比例 |
| `N_CONTROL` | 64 | LOO Ground Truth 控制集大小 |

## 从 git 恢复原 notebooks/

```bash
git checkout HEAD -- notebooks/
```

## 模型架构

| 模型 | 输入 | 结构 | 输出 |
|------|------|------|------|
| MLPRec4L | dense + sparse embeddings | 4-layer MLP (512→256→128→64) | binary logit |
| VAERec4L | dense + sparse embeddings | 4-layer encoder → (μ,σ) → 4-layer decoder | (logit, μ, logvar) |
| NCFRec4L | user_id, item_id embeddings | GMF branch + 4-layer MLP branch | binary logit |

## 归因方法

| 方法 | 复杂度 | 原理 |
|------|--------|------|
| SIF | O(N·P) | `dot(grad_z, grad_val)` |
| SIF+ | O(N·P) | SIF 经 Fisher 对角归一化 |
| IF-diag | O(N·P) | `-grad_val · diag(H)^{-1} · grad_z` |
| DVF | O(T·N·P) | 沿训练轨迹的 SIF 梯形积分 |

## Ground Truth 评估

不依赖循环验证。对 64 样本控制集做 Leave-One-Out 重训（3 seed 平均），
计算各方法排序与真实边际贡献的 Spearman 相关系数。
