# Experiment Notes

> 关键问题、根因、修复、结论。一句话能说完的不写两句。

---

## 2026-07-23 · 无法训练 (AUC ≈ 0.50)

**主因**：样本量不足。4000 样本 × 1500 用户，每用户仅 ~2.6 次交互/epoch，纯 MF 也学不动。

**放大器**：
- `user_disjoint` split → val 用户从未被训练 → CF 不可能泛化
- `neg_ratio=4` → 20/80 不平衡 → `pos_weight=4` → weighted loss 卡死随机基线

**修复**：`random` split + `neg_ratio=1` → 60K 全量 AUC ≈ 0.70。

---

## 2026-07-23 · AUC 天花板 ≈ 0.70

**根因**：隐式反馈 + 随机负采样 → "负样本"中 ~60% 是假阴性（用户未见过 ≠ 不喜欢）。

**定量**：`AUC = (1-f)×1.0 + f×0.5, AUC≈0.70 → f≈0.60`

**突破方向**：
1. 评估指标：AUC → Recall@K / NDCG@K（leave-one-out，不需负样本）
2. 损失函数：BCE → BPR（只学排序，不给负样本打绝对分）
3. 负采样：random → popularity-based → hard negative mining

**启示**：隐式反馈下 AUC 不能作为唯一指标；数据越稀疏天花板越低；`user_disjoint` 对纯 CF 致命。

---

## 旧版 Bug（已修复）

| Bug | 位置 | 一句话 |
|-----|------|--------|
| `make_loaders` 体损坏 | functions | return 后跟不可达代码 |
| IF `zip(flat, params)` | benchmark | 标量×张量矩阵，分数无意义 |
| VAE 丢 KL loss | train_model | `(logit, mu, logvar)` 丢弃后两个 |
| DVF/TracIn 共享计时 | benchmark | 共用 `t0`，报告相同耗时 |
| `evaluate_model`×3 | functions | 重复定义，仅末次生效 |
| `neg_ratio` 硬编码 | data_prep | 用户改不了负采样比例 |

---

*最后更新: 2026-07-23*
