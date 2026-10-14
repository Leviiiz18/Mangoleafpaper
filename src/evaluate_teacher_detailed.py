import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize

from src.config import (
    CLASS_NAMES,
    NUM_CLASSES,
    CACHE_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    RESULTS_DIR
)

def evaluate_and_plot_teacher():
    cache_path = os.path.join(CACHE_DIR, "teacher_logits.pt")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Teacher logits cache not found at {cache_path}")

    cache = torch.load(cache_path, map_location="cpu")
    test_data = cache["test"]
    logits = test_data["logits"].numpy()
    labels = test_data["labels"].numpy()

    # Softmax probabilities
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)

    # 1. Overall Metrics
    overall_acc = accuracy_score(labels, preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")

    print("\n" + "="*50)
    print("--- TEACHER (SWIN-T) DETAILED PERFORMANCE REPORT ---")
    print("="*50)
    print(f"Overall Test Accuracy: {overall_acc * 100:.2f}%")
    print(f"Macro Precision:       {macro_p * 100:.2f}%")
    print(f"Macro Recall:          {macro_r * 100:.2f}%")
    print(f"Macro F1-Score:        {macro_f1 * 100:.2f}%")
    print(f"Weighted F1-Score:     {weighted_f1 * 100:.2f}%")
    print("="*50)

    # 2. Per-Class Metrics Table
    p_per_class, r_per_class, f1_per_class, support = precision_recall_fscore_support(labels, preds, average=None)
    per_class_records = []
    for i, c_name in enumerate(CLASS_NAMES):
        per_class_records.append({
            "Class Name": c_name,
            "Support (Images)": int(support[i]),
            "Precision (%)": round(p_per_class[i] * 100, 2),
            "Recall (%)": round(r_per_class[i] * 100, 2),
            "F1-Score (%)": round(f1_per_class[i] * 100, 2)
        })

    df_per_class = pd.DataFrame(per_class_records)
    print("\n--- Per-Class Performance Breakdown ---")
    print(df_per_class.to_string(index=False))

    # Save to CSV
    csv_path = os.path.join(TABLES_DIR, "teacher_per_class_metrics.csv")
    df_per_class.to_csv(csv_path, index=False)

    # 3. Confusion Matrix Plot
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=300)

    # Raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[0],
        cbar=True
    )
    axes[0].set_title("(a) Confusion Matrix (Raw Sample Counts)", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel("Predicted Class", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("True Class", fontsize=11, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)

    # Normalized percentages
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2%',
        cmap='Greens',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[1],
        cbar=True
    )
    axes[1].set_title("(b) Normalized Confusion Matrix (% Accuracy per Class)", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlabel("Predicted Class", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("True Class", fontsize=11, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    cm_fig_path = os.path.join(FIGURES_DIR, "teacher_confusion_matrix.png")
    plt.savefig(cm_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved Confusion Matrix figure to: {cm_fig_path}")

    # 4. Multi-class ROC Curves
    labels_bin = label_binarize(labels, classes=list(range(NUM_CLASSES)))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(labels_bin.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))

    for i, color in zip(range(NUM_CLASSES), colors):
        ax.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"{CLASS_NAMES[i]} (AUC = {roc_auc[i]:.4f})"
        )

    ax.plot(
        fpr["micro"],
        tpr["micro"],
        color="deeppink",
        linestyle=":",
        linewidth=3,
        label=f"Micro-average ROC (AUC = {roc_auc['micro']:.4f})"
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7)
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight='bold')
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, fontweight='bold')
    ax.set_title("Multi-Class Receiver Operating Characteristic (ROC) Curves\nTeacher Model (Swin Transformer)", fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    roc_fig_path = os.path.join(FIGURES_DIR, "teacher_roc_curves.png")
    plt.savefig(roc_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curves figure to: {roc_fig_path}")

    # 5. Multi-class Precision-Recall (PR) Curves
    precision_dict = dict()
    recall_dict = dict()
    ap_dict = dict()

    for i in range(NUM_CLASSES):
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(labels_bin[:, i], probs[:, i])
        ap_dict[i] = average_precision_score(labels_bin[:, i], probs[:, i])

    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    for i, color in zip(range(NUM_CLASSES), colors):
        ax.plot(
            recall_dict[i],
            precision_dict[i],
            color=color,
            lw=2,
            label=f"{CLASS_NAMES[i]} (AP = {ap_dict[i]:.4f})"
        )

    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("Recall", fontsize=12, fontweight='bold')
    ax.set_ylabel("Precision", fontsize=12, fontweight='bold')
    ax.set_title("Multi-Class Precision-Recall (PR) Curves\nTeacher Model (Swin Transformer)", fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    pr_fig_path = os.path.join(FIGURES_DIR, "teacher_pr_curves.png")
    plt.savefig(pr_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PR curves figure to: {pr_fig_path}")

    # Save summary JSON
    summary_data = {
        "overall_test_accuracy": float(overall_acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "micro_roc_auc": float(roc_auc["micro"]),
        "per_class_metrics": per_class_records
    }
    with open(os.path.join(RESULTS_DIR, "teacher_detailed_report.json"), "w") as f:
        json.dump(summary_data, f, indent=2)

    return summary_data, df_per_class

if __name__ == "__main__":
    evaluate_and_plot_teacher()
