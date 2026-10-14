import os
import numpy as np
import pandas as pd
from scipy import stats
from skimage.metrics import structural_similarity as ssim

def compute_heatmap_similarity(heatmap_teacher, heatmap_student):
    """
    Computes SSIM and Spearman rank correlation between two normalized 2D heatmaps.
    """
    # SSIM requires float in [-1, 1] or [0, 1]
    ssim_val, _ = ssim(heatmap_teacher, heatmap_student, data_range=1.0, full=True)

    # Flatten for Spearman rank correlation
    flat_t = heatmap_teacher.flatten()
    flat_s = heatmap_student.flatten()

    spearman_corr, p_val = stats.spearmanr(flat_t, flat_s)
    if np.isnan(spearman_corr):
        spearman_corr = 0.0

    return float(ssim_val), float(spearman_corr)

def analyze_teacher_student_agreement(agreement_records):
    """
    agreement_records: list of dicts with keys:
    ['image_path', 'class_name', 'ssim_student_A', 'spearman_student_A',
     'ssim_student_B', 'spearman_student_B']
    """
    df = pd.DataFrame(agreement_records)

    # Overall metrics
    mean_ssim_A = df['ssim_student_A'].mean()
    std_ssim_A = df['ssim_student_A'].std()
    mean_ssim_B = df['ssim_student_B'].mean()
    std_ssim_B = df['ssim_student_B'].std()

    mean_spearman_A = df['spearman_student_A'].mean()
    std_spearman_A = df['spearman_student_A'].std()
    mean_spearman_B = df['spearman_student_B'].mean()
    std_spearman_B = df['spearman_student_B'].std()

    # Paired t-tests
    t_ssim, p_ssim = stats.ttest_rel(df['ssim_student_B'], df['ssim_student_A'])
    t_spearman, p_spearman = stats.ttest_rel(df['spearman_student_B'], df['spearman_student_A'])

    # Per-class summary
    per_class = df.groupby('class_name').agg({
        'ssim_student_A': ['mean', 'std'],
        'ssim_student_B': ['mean', 'std'],
        'spearman_student_A': ['mean', 'std'],
        'spearman_student_B': ['mean', 'std']
    })

    summary = {
        "overall": {
            "Student_A_SSIM_mean": float(mean_ssim_A),
            "Student_A_SSIM_std": float(std_ssim_A),
            "Student_B_SSIM_mean": float(mean_ssim_B),
            "Student_B_SSIM_std": float(std_ssim_B),
            "SSIM_paired_t": float(t_ssim),
            "SSIM_p_value": float(p_ssim),
            "Student_A_Spearman_mean": float(mean_spearman_A),
            "Student_A_Spearman_std": float(std_spearman_A),
            "Student_B_Spearman_mean": float(mean_spearman_B),
            "Student_B_Spearman_std": float(std_spearman_B),
            "Spearman_paired_t": float(t_spearman),
            "Spearman_p_value": float(p_spearman)
        },
        "per_class_df": per_class,
        "full_df": df
    }
    return summary
