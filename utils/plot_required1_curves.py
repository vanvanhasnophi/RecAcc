#!/usr/bin/env python3
"""Plot required Figure #1: deletion/insertion curves from benchmark_curves.csv.

Usage example:
    python scripts/plot_required1_curves.py
    python scripts/plot_required1_curves.py \
        --input log/notebook_runs/step5_unified_benchmark/benchmark_curves.csv \
        --output paper/figures/fig_main_deletion_insertion.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


COLOR_MAP = {
    "sif": "#1f77b4",
    "sif_plus": "#2ca02c",
    "influence": "#d62728",
    "dvf": "#ff7f0e",
    "tracin": "#9467bd",
}

LABEL_MAP = {
    "sif": "SIF",
    "sif_plus": "SIF+",
    "influence": "Influence Function",
    "dvf": "DVF",
    "tracin": "TracIn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot deletion/insertion curves")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "log/notebook_runs/step5_unified_benchmark/benchmark_curves.csv",
        help="Path to benchmark_curves.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "paper/figures/fig_main_deletion_insertion.png",
        help="Output image path",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI",
    )
    return parser.parse_args()


def _aggregate(curve_df: pd.DataFrame, curve_name: str) -> pd.DataFrame:
    subset = curve_df[curve_df["curve"] == curve_name].copy()
    if subset.empty:
        raise ValueError(f"No rows found for curve={curve_name!r}")

    agg = (
        subset.groupby(["method", "x"], as_index=False)["y"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "y_mean", "std": "y_std"})
    )
    agg["y_std"] = agg["y_std"].fillna(0.0)
    return agg


def _plot_one(ax: plt.Axes, agg: pd.DataFrame, title: str) -> None:
    methods = sorted(agg["method"].unique().tolist())
    for method in methods:
        m = agg[agg["method"] == method].sort_values("x")
        color = COLOR_MAP.get(method, None)
        label = LABEL_MAP.get(method, method)
        x = m["x"].to_numpy()
        y = m["y_mean"].to_numpy()
        s = m["y_std"].to_numpy()

        ax.plot(x, y, marker="o", linewidth=1.8, label=label, color=color)
        ax.fill_between(x, y - s, y + s, alpha=0.15, color=color)

    ax.set_title(title)
    ax.set_xlabel("Fraction")
    ax.set_ylabel("Utility Gain")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)


def main() -> None:
    args = parse_args()
    args.input = args.input.resolve()
    args.output = args.output.resolve()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    expected_cols = {"seed", "model", "method", "curve", "x", "y"}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    deletion = _aggregate(df, "deletion")
    insertion = _aggregate(df, "insertion")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    plt.subplots_adjust(top=0.82, wspace=0.22)
    _plot_one(axes[0], deletion, "Deletion Curve (mean ± std)")
    _plot_one(axes[1], insertion, "Insertion Curve (mean ± std)")

    # unified legend on top, outside axes to avoid overlap with titles
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), frameon=False)
    # reduce title fontsize and move it down to avoid collision
    for ax in axes:
        ax.title.set_fontsize(11)
        ax.title.set_y(1.02)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")

    # Optional vector figure for paper quality if only extension differs.
    pdf_out = args.output.with_suffix(".pdf")
    fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {args.output}")
    print(f"Saved: {pdf_out}")


if __name__ == "__main__":
    main()
