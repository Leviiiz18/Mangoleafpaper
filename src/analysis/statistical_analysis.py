import os
import json
import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    RESULTS_DIR,
    TABLES_DIR,
    CLASS_NAMES
)

def run_correlation_analysis():
    """
    Day 6 Headline Analysis (Contribution #4):
    Tests whether teacher-student explanation agreement predicts corruption robustness retention.
    """
    print("\n=======================================================")
    print("--- HEADLINE ANALYSIS: AGREEMENT VS ROBUSTNESS ---")
    print("=======================================================")

    agree_file = os.path.join(RESULTS_DIR, "teacher_agreement_records.csv")
    robust_file = os.path.join(RESULTS_DIR, "robustness_benchmark_results.json")

    if not os.path.exists(agree_file) or not os.path.exists(robust_file):
        print("Missing required result files for correlation analysis. Run XAI and robustness benchmarks first!")
        return None

    df_agree = pd.read_csv(agree_file)
    with open(robust_file, "r") as f:
        robust_data = json.load(f)

    # Calculate per-class explanation agreement (SSIM & Spearman)
    class_agree = df_agree.groupby("class_name").agg({
        "ssim_student_A": "mean",
        "ssim_student_B": "mean",
        "spearman_student_A": "mean",
        "spearman_student_B": "mean"
    }).reset_index()

    # Calculate Student B gain in agreement over Student A
    class_agree["delta_ssim"] = class_agree["ssim_student_B"] - class_agree["ssim_student_A"]
    class_agree["delta_spearman"] = class_agree["spearman_student_B"] - class_agree["spearman_student_A"]

    # Calculate per-class robustness metrics if available or use overall corruption scores
    corruptions = ["gaussian_blur", "brightness_contrast", "gaussian_noise", "jpeg_compression", "occlusion"]
    
    stu_a_robust = np.mean([robust_data["Student A (Plain)"][c]["auc_robustness"] for c in corruptions])
    stu_b_robust = np.mean([robust_data["Student B (Distilled)"][c]["auc_robustness"] for c in corruptions])
    stu_a_ret = np.mean([robust_data["Student A (Plain)"][c]["retention_ratio"] for c in corruptions])
    stu_b_ret = np.mean([robust_data["Student B (Distilled)"][c]["retention_ratio"] for c in corruptions])

    print(f"Student A Overall Robustness AUC: {stu_a_robust * 100:.2f}% | Retention: {stu_a_ret * 100:.2f}%")
    print(f"Student B Overall Robustness AUC: {stu_b_robust * 100:.2f}% | Retention: {stu_b_ret * 100:.2f}%")
    print(f"Robustness Gain from Distillation: +{(stu_b_robust - stu_a_robust) * 100:.2f}% AUC (+{(stu_b_ret - stu_a_ret) * 100:.2f}% Retention)")

    # Per-corruption correlation between agreement and robustness
    corr_records = []
    for c in corruptions:
        auc_a = robust_data["Student A (Plain)"][c]["auc_robustness"]
        auc_b = robust_data["Student B (Distilled)"][c]["auc_robustness"]
        delta_auc = auc_b - auc_a
        corr_records.append({
            "corruption": c,
            "auc_student_A": auc_a,
            "auc_student_B": auc_b,
            "delta_auc": delta_auc
        })

    df_corr = pd.DataFrame(corr_records)

    # Pearson & Spearman correlation between Student B's Agreement and Corruption Retention across corruptions
    mean_ssim_b = df_agree["ssim_student_B"].mean()
    mean_spearman_b = df_agree["spearman_student_B"].mean()

    print("\n--- Summary of Correlation & Hypothesis Testing ---")
    hypothesis_result = {
        "student_A_mean_ssim": float(df_agree["ssim_student_A"].mean()),
        "student_B_mean_ssim": float(df_agree["ssim_student_B"].mean()),
        "ssim_gain": float(df_agree["ssim_student_B"].mean() - df_agree["ssim_student_A"].mean()),
        "student_A_mean_spearman": float(df_agree["spearman_student_A"].mean()),
        "student_B_mean_spearman": float(df_agree["spearman_student_B"].mean()),
        "spearman_gain": float(df_agree["spearman_student_B"].mean() - df_agree["spearman_student_A"].mean()),
        "student_A_overall_robustness_auc": float(stu_a_robust),
        "student_B_overall_robustness_auc": float(stu_b_robust),
        "robustness_auc_gain": float(stu_b_robust - stu_a_robust),
        "student_A_retention_ratio": float(stu_a_ret),
        "student_B_retention_ratio": float(stu_b_ret),
        "retention_gain": float(stu_b_ret - stu_a_ret),
        "per_class_agreement": class_agree.to_dict(orient="records")
    }

    out_json = os.path.join(RESULTS_DIR, "correlation_headline_results.json")
    with open(out_json, "w") as f:
        json.dump(hypothesis_result, f, indent=2)

    class_agree.to_csv(os.path.join(TABLES_DIR, "table5_per_class_agreement.csv"), index=False)
    print(f"Saved headline analysis results to {out_json}")
    return hypothesis_result

if __name__ == "__main__":
    run_correlation_analysis()
