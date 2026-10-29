"""Plot consolidated Master Comparison across all 4 Sets
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

csv_path = r"d:\code\Mangoleaf\all_models_evaluations\master_evaluation_metrics.csv"
df = pd.read_csv(csv_path)

plt.figure(figsize=(14, 7))
plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})

palette = {
    "Teacher (Master)": "#1B365D",
    "Child 1 (Plain Student)": "#7F8C8D",
    "Child 2 (Distilled Student)": "#2A78D6",
    "Child 3 (XKD Student)": "#1BAF7A"
}

fig, ax = plt.subplots(figsize=(15, 7.5))
bars = sns.barplot(
    data=df,
    x="Set",
    y="Test Accuracy (%)",
    hue="Role",
    palette=palette,
    edgecolor="black",
    linewidth=1.2,
    alpha=0.95,
    ax=ax
)

ax.set_ylim(94, 100.5)
ax.set_ylabel("Test Accuracy (%)", fontsize=12, fontweight="bold")
ax.set_xlabel("Architectural Evaluation Set", fontsize=12, fontweight="bold")
ax.set_title(
    "Cross-Architecture Knowledge Distillation & XKD Performance Across 4 Sets\n(Leak-Free Group-Aware Evaluation)",
    fontsize=14, fontweight="bold", pad=15
)
ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.legend(title="Model Role", title_fontsize='11', fontsize='10', framealpha=0.95, loc="lower right")

for p in ax.patches:
    h = p.get_height()
    if h > 0:
        ax.annotate(f"{h:.2f}%",
                    (p.get_x() + p.get_width() / 2., h),
                    ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold',
                    xytext=(0, 3), textcoords='offset points')

plt.tight_layout()
out_path = r"d:\code\Mangoleaf\all_models_evaluations\figures\all_sets_master_comparison.png"
plt.savefig(out_path, dpi=300)
plt.close()
print("Saved master comparison bar chart ->", out_path)
