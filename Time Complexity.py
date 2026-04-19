import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)
plt.rcParams['font.family'] = 'DejaVu Sans'  # 或 'SimHei' 以支持中文

methods = [
    ("Shapley",           "exponential", None,    "#d62728", True),
    ("TMC-Shapley",       "poly_train",  None,    "#ff7f0e", True),
    ("Influence Function","cubic",       None,    "#9467bd", False),
    ("DVF",               "linear_T",    None,    "#1f77b4", False),
    ("TracIn",            "linear_T",    None,    "#17becf", False),
    ("SIF",               "linear",      None,    "#2ca02c", False),
    ("SIF+",              "linear",      None,    "#bcbd22", False),
]

N_vals = np.logspace(2, 5, 300)
d = 1000
T = 9
K = 200          
tmc_factor = 50  # 额外重训练成本因子，可调大（如 100）以体现实际开销

def complexity(kind, N, d, T, K):
    if kind == "exponential":
        return 2 ** np.minimum(N / 1000, 40)
    elif kind == "poly_train":
        return tmc_factor * K * N * np.log(N)
    elif kind == "cubic":
        return N * d ** 2
    elif kind == "linear_T":
        return T * N * d
    elif kind == "linear":
        return N * d

fig, ax = plt.subplots(figsize=(12, 7))

# 先计算所有曲线，用于确定 y 轴范围
all_y = []
for name, kind, _, _, _ in methods:
    y = complexity(kind, N_vals, d, T, K)
    all_y.extend(y)
ymin = np.min(all_y) * 0.5  # 留出下方空间，让 SIF 曲线可见
ymax = np.max(all_y) * 1.2

for name, kind, _, color, retrain in methods:
    y = complexity(kind, N_vals, d, T, K)
    ls = "--" if retrain else "-"
    lw = 2.5 if retrain else 2.2
    alpha = 0.85
    # 为 SIF 增加标记，提升辨识度
    if name == "SIF":
        ax.plot(N_vals, y, label=name, color=color, linewidth=lw,
                linestyle=ls, alpha=alpha, marker='D', markersize=3, markevery=50)
    else:
        ax.plot(N_vals, y, label=name, color=color, linewidth=lw,
                linestyle=ls, alpha=alpha)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of Training Samples $N$", fontsize=14)
ax.set_ylabel("Computational Cost (relative units)", fontsize=14)
ax.set_title("Time Complexity Comparison of Data Valuation Methods\n"
             r"($d=10^3$, $T=9$ checkpoints, $K=200$ TMC rounds)",
             fontsize=15, pad=15)
ax.set_ylim(ymin, ymax)  # 关键：防止 SIF 被压底

# 网格细化
ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)

# 标注复杂度公式，使用箭头指向曲线末端
annotations = {
    "Shapley":            r"$O(2^N)$",
    "TMC-Shapley":        r"$O(KN\log N)$ + retraining",
    "Influence Function": r"$O(Nd^2)$",
    "DVF":                r"$O(TNd)$",
    "TracIn":             r"$O(TNd)$",
    "SIF":                r"$O(Nd)$",
    "SIF+":               r"$O(Nd)$",
}

for name, kind, _, color, _ in methods:
    y = complexity(kind, N_vals[-1], d, T, K)
    ax.annotate(annotations[name],
                xy=(N_vals[-1], y),
                xytext=(10, 0),
                textcoords="offset points",
                color=color,
                fontsize=9,
                va="center",
                arrowprops=dict(arrowstyle="-", color=color, alpha=0.5, lw=0.8))

# 图例：将线型说明单独列出
solid_patch = mpatches.Patch(color="gray", label="No retraining required", alpha=0.6)
dashed_patch = mpatches.Patch(color="gray", linestyle="--", label="Requires retraining",
                              fill=False, linewidth=1.5)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles + [solid_patch, dashed_patch],
          labels + ["No retraining", "Requires retraining"],
          loc="upper left", fontsize=10, frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig("complexity_comparison.png", dpi=300, bbox_inches="tight", facecolor='white')
plt.show()