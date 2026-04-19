#!/usr/bin/env python3
"""
统计无表头 CSV 中稀疏特征的唯一值数量（流式处理）。

用法示例:
python scripts/count_unique_sparse.py \
  --input DATASET/Eleme/20220307_01.csv \
  --feature-names label user_id gender visit_city avg_price is_supervip ctr_30 ord_30 total_amt_30 shop_id item_id ... \
  --sparse-cols user_id shop_id item_id gender \
  --chunksize 200000 \
  --topk 20 \
  --output unique_counts.json
"""
import argparse
import json
from collections import Counter, defaultdict
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser(description="Count unique values for sparse columns in a no-header CSV")
    p.add_argument("--input", required=True, help="输入 CSV 文件（无表头）")
    p.add_argument("--feature-names", nargs='+', required=True,
                   help="按列顺序列出所有特征名（必须覆盖 CSV 的每一列）")
    p.add_argument("--sparse-cols", nargs='+', required=True,
                   help="要统计的稀疏特征，可用列名或0-based列索引（混合也可）")
    p.add_argument("--chunksize", type=int, default=65536, help="pandas chunksize")
    p.add_argument("--topk", type=int, default=0, help="若>0，同时输出每列 top-K 频次")
    p.add_argument("--output", default="", help="若提供则以 JSON 保存统计结果")
    return p.parse_args()

def normalize_sparse_cols(specs, feature_names):
    names = []
    for s in specs:
        # 尝试解析为整数索引
        try:
            idx = int(s)
            if idx < 0 or idx >= len(feature_names):
                raise ValueError(f"index {idx} out of range")
            names.append(feature_names[idx])
        except ValueError:
            if s not in feature_names:
                raise ValueError(f"Column name '{s}' not found in feature_names")
            names.append(s)
    return names

def main():
    args = parse_args()
    feature_names = args.feature_names
    sparse_names = normalize_sparse_cols(args.sparse_cols, feature_names)

    unique_sets = {name: set() for name in sparse_names}
    counters = {name: Counter() for name in sparse_names} if args.topk > 0 else None
    total_rows = 0
    for chunk in pd.read_csv(args.input, header=None, names=feature_names, chunksize=args.chunksize, dtype=str):
        total_rows += len(chunk)
        for name in sparse_names:
            col = chunk[name].fillna("").astype(str)
            unique_sets[name].update(col.unique().tolist())
            if counters is not None:
                counters[name].update(col.values)
    results = {}
    for name in sparse_names:
        uniq_count = len(unique_sets[name])
        res = {"unique_count": uniq_count}
        if counters is not None:
            res["topk"] = counters[name].most_common(args.topk)
        results[name] = res

    summary = {
        "input": args.input,
        "total_rows_processed": total_rows,
        "results": results
    }

    out_json = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out_json)
        print(f"Saved results to {args.output}")
    else:
        print(out_json)

if __name__ == "__main__":
    main()