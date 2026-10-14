import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image

from src.config import (
    FIGURES_DIR,
    RESULTS_DIR,
    TABLES_DIR,
    CLASS_NAMES,
    MASKS_DIR,
    SPLITS_DIR
)

# Set high-quality styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "font.family": "sans-serif"
})

def plot_training_curves():
    """
    Fig 2: Training loss and validation accuracy curves for Teacher and Students.
    """
    teacher_path = os.path.join(RESULTS_DIR, "teacher_swin_metrics.json")
    stu_a_path = os.path.join(RESULTS_DIR, "student_plain_metrics.json")
    stu_b_path = os.path.join(RESULTS_DIR, "student_distilled_metrics.json")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Palette
    c_teacher = "#1f77b4" # Blue
    c_stu_a = "#d62728"   # Red
    c_stu_b = "#2ca02c"   # Green

    plotted_any = False
    for path, label, color in [
        (teacher_path, "Teacher (Swin-T)", c_teacher),
        (stu_a_path, "Student A (Plain)", c_stu_a),
        (stu_b_path, "Student B (Distilled)", c_stu_b)
    ]:
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            hist = data.get("history", {})
            epochs = range(1, len(hist.get("train_loss", [])) + 1)

            if len(epochs) > 0:
                plotted_any = True
                axes[0].plot(epochs, hist["train_loss"], label=f"{label} (Train)", color=color, linestyle="--", alpha=0.7)
                axes[0].plot(epochs, hist["val_loss"], label=f"{label} (Val)", color=color, linewidth=2)

                axes[1].plot(epochs, [a * 100 for a in hist["val_acc"]], label=label, color=color, linewidth=2)

    if plotted_any:
        axes[0].set_title("(a) Cross-Entropy / Distillation Loss Trajectory")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        axes[1].set_title("(b) Validation Accuracy Trajectory")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation Accuracy (%)")
        axes[1].legend()

        plt.tight_layout()
        out_fig = os.path.join(FIGURES_DIR, "fig2_training_curves.png")
        plt.savefig(out_fig, dpi=300)
        plt.close()
        print(f"Generated Figure 2: {out_fig}")

def plot_robustness_curves():
    """
    Fig 5: Robustness decay curves across the 5 corruption types.
    """
    res_path = os.path.join(RESULTS_DIR, "robustness_benchmark_results.json")
    if not os.path.exists(res_path):
        return

    with open(res_path, "r") as f:
        data = json.load(f)

    corruptions = ["gaussian_blur", "brightness_contrast", "gaussian_noise", "jpeg_compression", "occlusion"]
    titles = ["Gaussian Blur", "Brightness/Contrast", "Gaussian Noise", "JPEG Compression", "Occlusion"]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), dpi=300, sharey=True)
    severities = [0, 1, 2, 3] # 0 = clean

    models = [
        ("Teacher (Swin-T)", "#1f77b4", "o-"),
        ("Student A (Plain)", "#d62728", "s--"),
        ("Student B (Distilled)", "#2ca02c", "^-")
    ]

    for i, (corr, title) in enumerate(zip(corruptions, titles)):
        ax = axes[i]
        for m_name, color, style in models:
            if m_name in data and corr in data[m_name]:
                clean_acc = data[m_name]["clean"]["accuracy"] * 100
                sev1 = data[m_name][corr]["severity_1"] * 100
                sev2 = data[m_name][corr]["severity_2"] * 100
                sev3 = data[m_name][corr]["severity_3"] * 100
                accs = [clean_acc, sev1, sev2, sev3]
                ax.plot(severities, accs, style, label=m_name, color=color, linewidth=2, markersize=6)

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Severity Level")
        ax.set_xticks(severities)
        ax.set_xticklabels(["Clean", "1", "2", "3"])
        if i == 0:
            ax.set_ylabel("Test Accuracy (%)")
            ax.legend()

    plt.suptitle("Model Accuracy Under Distribution Shifts & Corruptions", fontsize=15, y=1.03)
    plt.tight_layout()
    out_fig = os.path.join(FIGURES_DIR, "fig5_robustness_curves.png")
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 5: {out_fig}")

def plot_agreement_vs_robustness_correlation():
    """
    Fig 6: Scatter plot of Teacher-Student Explanation Agreement vs Corruption Robustness Retention.
    """
    agree_path = os.path.join(RESULTS_DIR, "teacher_agreement_records.csv")
    if not os.path.exists(agree_path):
        return

    df_agree = pd.read_csv(agree_path)
    # Per-class mean SSIM and Spearman
    class_stats = df_agree.groupby("class_name").agg({
        "ssim_student_A": "mean",
        "ssim_student_B": "mean",
        "spearman_student_A": "mean",
        "spearman_student_B": "mean"
    }).reset_index()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Plot Student A and Student B points
    x_vals = class_stats["ssim_student_B"]
    y_vals = class_stats["spearman_student_B"]

    sns.regplot(
        x=class_stats["ssim_student_B"],
        y=class_stats["spearman_student_B"],
        scatter_kws={"s": 90, "color": "#2ca02c", "alpha": 0.8},
        line_kws={"color": "#1b5e20", "linewidth": 2},
        ax=ax,
        label="Student B (Distilled) Classes"
    )

    sns.regplot(
        x=class_stats["ssim_student_A"],
        y=class_stats["spearman_student_A"],
        scatter_kws={"s": 90, "color": "#d62728", "alpha": 0.8},
        line_kws={"color": "#b71c1c", "linewidth": 2, "linestyle": "--"},
        ax=ax,
        label="Student A (Plain) Classes"
    )

    for i, row in class_stats.iterrows():
        ax.annotate(row["class_name"], (row["ssim_student_B"] + 0.005, row["spearman_student_B"]), fontsize=9)

    ax.set_title("Preservation of Diagnostic Reasoning: Teacher-Student Agreement", fontweight="bold")
    ax.set_xlabel("Structural Similarity Index (SSIM) vs Swin-T Teacher")
    ax.set_ylabel("Spearman Rank Correlation (ρ) vs Swin-T Teacher")
    ax.legend(loc="lower right")

    plt.tight_layout()
    out_fig = os.path.join(FIGURES_DIR, "fig6_agreement_correlation.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Generated Figure 6: {out_fig}")

def generate_all_figures():
    plot_training_curves()
    plot_robustness_curves()
    plot_agreement_vs_robustness_correlation()

if __name__ == "__main__":
    generate_all_figures()
