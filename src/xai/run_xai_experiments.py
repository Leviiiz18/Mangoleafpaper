import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

from src.config import (
    NUM_CLASSES,
    DEVICE,
    CHECKPOINTS_DIR,
    SPLITS_DIR,
    MASKS_DIR,
    RESULTS_DIR,
    TABLES_DIR,
    IMAGE_SIZE,
    NORM_MEAN,
    NORM_STD,
    CLASS_NAMES
)
from src.train_teacher import build_swin_teacher
from src.train_students import build_mobilenet_student
from src.xai.gradcam import get_gradcam_explainer
from src.xai.faithfulness_metrics import compute_deletion_insertion_auc, compute_mask_iou
from src.xai.explanation_agreement import compute_heatmap_similarity, analyze_teacher_student_agreement

def run_xai_pipeline(max_eval_samples=200):
    print("==========================================")
    print("--- RUNNING QUANTITATIVE XAI PIPELINE ---")
    print("==========================================")

    # 1. Load Models
    teacher = build_swin_teacher(NUM_CLASSES, pretrained=False, freeze_early=False).to(DEVICE)
    ckpt_t = torch.load(os.path.join(CHECKPOINTS_DIR, "teacher_swin.pth"), map_location=DEVICE)
    teacher.load_state_dict(ckpt_t["model_state_dict"])
    teacher.eval()

    student_A = build_mobilenet_student(NUM_CLASSES, pretrained=False).to(DEVICE)
    ckpt_a = torch.load(os.path.join(CHECKPOINTS_DIR, "student_plain.pth"), map_location=DEVICE)
    student_A.load_state_dict(ckpt_a["model_state_dict"])
    student_A.eval()

    student_B = build_mobilenet_student(NUM_CLASSES, pretrained=False).to(DEVICE)
    ckpt_b = torch.load(os.path.join(CHECKPOINTS_DIR, "student_distilled.pth"), map_location=DEVICE)
    student_B.load_state_dict(ckpt_b["model_state_dict"])
    student_B.eval()

    cam_teacher = get_gradcam_explainer(teacher, "swin")
    cam_student_A = get_gradcam_explainer(student_A, "mobilenet")
    cam_student_B = get_gradcam_explainer(student_B, "mobilenet")

    norm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    test_df = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"))
    if max_eval_samples and len(test_df) > max_eval_samples:
        per_class_n = max_eval_samples // NUM_CLASSES
        sample_df = test_df.groupby("class_name", as_index=False).head(per_class_n).reset_index(drop=True)
    else:
        sample_df = test_df

    # Load lesion ground-truth masks if available
    masks_manifest_path = os.path.join(MASKS_DIR, "lesion_masks_manifest.csv")
    gt_masks_dict = {}
    if os.path.exists(masks_manifest_path):
        m_df = pd.read_csv(masks_manifest_path)
        for _, row in m_df.iterrows():
            mask_img = np.array(Image.open(row["mask_path"]).convert("L"))
            gt_masks_dict[row["image_path"]] = (mask_img > 127).astype(np.uint8)

    records_faithfulness = {"Teacher": [], "Student_A": [], "Student_B": []}
    agreement_records = []

    print(f"Evaluating faithfulness and agreement across {len(sample_df)} test images...")

    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        img_path = row["filepath"]
        label = int(row["label"])
        cls_name = row["class_name"]

        img_pil = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        img_tensor = norm_transform(img_pil).unsqueeze(0).to(DEVICE)

        # Grad-CAMs
        heat_t, logits_t = cam_teacher.generate_cam(img_tensor, target_class=label)
        heat_a, logits_a = cam_student_A.generate_cam(img_tensor, target_class=label)
        heat_b, logits_b = cam_student_B.generate_cam(img_tensor, target_class=label)

        pred_t = torch.argmax(logits_t, dim=1).item()
        pred_a = torch.argmax(logits_a, dim=1).item()
        pred_b = torch.argmax(logits_b, dim=1).item()

        gt_mask = gt_masks_dict.get(img_path, None)

        # Deletion and Insertion AUC
        del_t, ins_t, _, _ = compute_deletion_insertion_auc(teacher, img_tensor, heat_t, target_class=label, device=DEVICE)
        del_a, ins_a, _, _ = compute_deletion_insertion_auc(student_A, img_tensor, heat_a, target_class=label, device=DEVICE)
        del_b, ins_b, _, _ = compute_deletion_insertion_auc(student_B, img_tensor, heat_b, target_class=label, device=DEVICE)

        iou_t = compute_mask_iou(heat_t, gt_mask) if gt_mask is not None else np.nan
        iou_a = compute_mask_iou(heat_a, gt_mask) if gt_mask is not None else np.nan
        iou_b = compute_mask_iou(heat_b, gt_mask) if gt_mask is not None else np.nan

        records_faithfulness["Teacher"].append({"del_auc": del_t, "ins_auc": ins_t, "iou": iou_t})
        records_faithfulness["Student_A"].append({"del_auc": del_a, "ins_auc": ins_a, "iou": iou_a})
        records_faithfulness["Student_B"].append({"del_auc": del_b, "ins_auc": ins_b, "iou": iou_b})

        # Agreement (SSIM & Spearman)
        ssim_a, spear_a = compute_heatmap_similarity(heat_t, heat_a)
        ssim_b, spear_b = compute_heatmap_similarity(heat_t, heat_b)

        agreement_records.append({
            "image_path": img_path,
            "class_name": cls_name,
            "label": label,
            "pred_teacher": pred_t,
            "pred_student_A": pred_a,
            "pred_student_B": pred_b,
            "correct_teacher": int(pred_t == label),
            "correct_student_A": int(pred_a == label),
            "correct_student_B": int(pred_b == label),
            "ssim_student_A": ssim_a,
            "spearman_student_A": spear_a,
            "ssim_student_B": ssim_b,
            "spearman_student_B": spear_b
        })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summarize Faithfulness Table
    table2_rows = []
    for m_name in ["Teacher", "Student_A", "Student_B"]:
        del_mean = np.nanmean([r["del_auc"] for r in records_faithfulness[m_name]])
        del_std = np.nanstd([r["del_auc"] for r in records_faithfulness[m_name]])
        ins_mean = np.nanmean([r["ins_auc"] for r in records_faithfulness[m_name]])
        ins_std = np.nanstd([r["ins_auc"] for r in records_faithfulness[m_name]])
        iou_mean = np.nanmean([r["iou"] for r in records_faithfulness[m_name]])
        iou_std = np.nanstd([r["iou"] for r in records_faithfulness[m_name]])

        table2_rows.append({
            "Model": m_name,
            "Deletion AUC (lower=better)": f"{del_mean:.4f} ± {del_std:.4f}",
            "Insertion AUC (higher=better)": f"{ins_mean:.4f} ± {ins_std:.4f}",
            "Lesion Mask IoU": f"{iou_mean:.4f} ± {iou_std:.4f}"
        })

    df_table2 = pd.DataFrame(table2_rows)
    print("\n--- Quantitative XAI Faithfulness Table (Contribution #1) ---")
    print(df_table2.to_string(index=False))

    t2_path = os.path.join(TABLES_DIR, "table2_xai_faithfulness.csv")
    df_table2.to_csv(t2_path, index=False)

    # Summarize Agreement (Contribution #3)
    agreement_summary = analyze_teacher_student_agreement(agreement_records)
    print("\n--- Teacher-Student Explanation Agreement (Contribution #3) ---")
    ov = agreement_summary["overall"]
    print(f"Student A (Plain) SSIM:     {ov['Student_A_SSIM_mean']:.4f} ± {ov['Student_A_SSIM_std']:.4f}")
    print(f"Student B (Distilled) SSIM: {ov['Student_B_SSIM_mean']:.4f} ± {ov['Student_B_SSIM_std']:.4f}")
    print(f"SSIM Paired t-test: t = {ov['SSIM_paired_t']:.3f}, p-value = {ov['SSIM_p_value']:.4e}")
    print(f"Student A Spearman Rho:     {ov['Student_A_Spearman_mean']:.4f} ± {ov['Student_A_Spearman_std']:.4f}")
    print(f"Student B Spearman Rho:     {ov['Student_B_Spearman_mean']:.4f} ± {ov['Student_B_Spearman_std']:.4f}")
    print(f"Spearman Paired t-test: t = {ov['Spearman_paired_t']:.3f}, p-value = {ov['Spearman_p_value']:.4e}")

    agree_path = os.path.join(RESULTS_DIR, "teacher_agreement_results.json")
    with open(agree_path, "w") as f:
        json.dump({
            "overall": ov,
            "num_evaluated": len(agreement_records)
        }, f, indent=2)

    agree_df_path = os.path.join(RESULTS_DIR, "teacher_agreement_records.csv")
    pd.DataFrame(agreement_records).to_csv(agree_df_path, index=False)

    cam_teacher.remove_hooks()
    cam_student_A.remove_hooks()
    cam_student_B.remove_hooks()

    return df_table2, agreement_summary

if __name__ == "__main__":
    run_xai_pipeline()
