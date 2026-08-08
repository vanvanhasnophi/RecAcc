# Notebooks

推荐系统训练与归因评估管线。基于 SPINRec 风格，采用 Linear-multi-hot 架构。

## 目录结构

```
notebooks/
├── README.md
├── functions.ipynb               # 共享库：模型 / 训练 / 评估 / 归因（SIF/DVF/IF-diag）
├── data_preprocessing.ipynb      # step 1：数据预处理 → prepared_data 缓存
├── rec_training.ipynb            # step 2：MLP / NCF 训练管线
└── attribution_benchmark.ipynb   # step 3：LOO Ground Truth + 归因 benchmark
```

## 执行顺序

```
1. data_preprocessing.ipynb   →  log/notebook_cache/prepared_data_{ds}.pt
2. rec_training.ipynb         →  log/notebook_runs/step2_training/{timestamp}_{ds}/
3. attribution_benchmark.ipynb →  log/notebook_runs/attribution_benchmark/
```

## 模型架构

| 模型 | 输入 | 结构 | 参数量级 |
|------|------|------|---------|
| LinearRec (MLP) | user multi-hot [n_items] | 2-layer Linear (n_items→512) | ~1-4M |
| LinearNCF | user multi-hot [n_items] | GMF + 2-layer MLP | ~2-6M |

## 归因方法

| 方法 | 复杂度 | 原理 |
|------|--------|------|
| SIF | O(Nd) | `g_val · g_z` |
| IF-diag | O(Nd) | `-g_val · g_z / diag(F)` |
| DVF | O(TNd) | 沿训练轨迹 SIF 梯形积分 |
| Random | O(1) | 随机基线 |

## Ground Truth 评估

64 样本 LOO：每个控制样本单独重训（5 seed 平均），以 BCE loss 差异作为真实边际贡献。
各方法分数与 Ground Truth 的 Spearman 秩相关系数作为归因质量指标。
