# Research Project Specification & Empirical Results Report
## Completed Multi-Set Distillation Benchmark across 4 Architectural Quadrants
**Target Journal**: *Smart Agricultural Technology* (Elsevier)

---

### 1. Executive Summary & Experimental Design
To rigorously evaluate Knowledge Distillation and Explanation Preservation in automated plant pathology, we completed the evaluation of a structured **4-Model System** across **4 distinct vision families** on a group-aware, leak-free mango leaf dataset.

```
                  ┌──────────────────────────────────────────────┐
                  │                 1. TEACHER                   │
                  │             (Heavy Master Model)             │
                  └──────────────────────┬───────────────────────┘
                                         │
         ┌───────────────────────────────┼───────────────────────────────┐
         │                               │                               │
         ▼                               ▼                               ▼
  2. STUDENT PLAIN              3. STUDENT DISTILLED               4. STUDENT XKD
  (Baseline: Cross-Entropy)     (Standard KD: Soft Logits)        (Explanation-Guided: CAM)
```

---

### 2. Completed Benchmark Results Across All 4 Sets

| Set | Role | Model ID | Architecture & Method | Parameters (M) | Test Accuracy (%) | Test Macro-F1 (%) | Macro AUC | Training Time (min) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Set 1 | Teacher (Master) | set1_teacher_swin_t | Set 1 Teacher: Swin-T (CE) | 27.53M | 99.38 | 99.37 | 1.0 | 14.5 |
| Set 1 | Child 3 (XKD Student) | set1_student_xkd_mnv3_s | Set 1: MobileNetV3-Small (XKD from Swin-T) | 1.53M | 98.44 | 98.43 | 0.998 | 15.1 |
| Set 2 | Teacher (Master) | set2_teacher_convnext_t | Set 2 Teacher: ConvNeXt-Tiny (CE) | 27.83M | 99.06 | 99.06 | 1.0 | 13.7 |
| Set 2 | Child 1 (Plain Student) | set2_student_plain_mnv4_s | Set 2 Student: MobileNetV4-Small (Plain CE) | 2.50M | 96.25 | 96.23 | 0.978 | 11.1 |
| Set 2 | Child 2 (Distilled Student) | set2_student_kd_mnv4_s | Set 2 Student: MobileNetV4-Small (KD from ConvNeXt) | 2.50M | 98.12 | 98.12 | 0.996 | 13.3 |
| Set 2 | Child 3 (XKD Student) | set2_student_xkd_mnv4_s | Set 2 Student: MobileNetV4-Small (XKD from ConvNeXt) | 2.50M | 96.88 | 96.85 | 0.991 | 13.3 |
| Set 3 | Teacher (Master) | set3_teacher_effnetv2_s | Set 3 Teacher: EfficientNetV2-Small (CE) | 20.19M | 99.53 | 99.53 | 1.0 | 13.8 |
| Set 3 | Child 1 (Plain Student) | set3_student_plain_effnet_b0 | Set 3 Student: EfficientNet-B0 (Plain CE) | 4.02M | 99.06 | 99.06 | 1.0 | 13.8 |
| Set 3 | Child 2 (Distilled Student) | set3_student_kd_effnet_b0 | Set 3 Student: EfficientNet-B0 (KD from EffNetV2) | 4.02M | 99.22 | 99.22 | 0.999 | 15.8 |
| Set 3 | Child 3 (XKD Student) | set3_student_xkd_effnet_b0 | Set 3 Student: EfficientNet-B0 (XKD from EffNetV2) | 4.02M | 99.06 | 99.07 | 0.998 | 16.0 |
| Set 4 | Teacher (Master) | set4_teacher_maxvit_t | Set 4 Teacher: MaxViT-Tiny (CE) | 30.41M | 98.91 | 98.91 | 1.0 | 22.3 |
| Set 4 | Child 1 (Plain Student) | set4_student_plain_ghostnet | Set 4 Student: GhostNet (Plain CE) | 3.91M | 98.75 | 98.74 | 0.999 | 14.3 |
| Set 4 | Child 2 (Distilled Student) | set4_student_kd_ghostnet | Set 4 Student: GhostNet (KD from MaxViT) | 3.91M | 98.91 | 98.91 | 1.0 | 18.6 |
| Set 4 | Child 3 (XKD Student) | set4_student_xkd_ghostnet | Set 4 Student: GhostNet (XKD from MaxViT) | 3.91M | 99.06 | 99.06 | 0.999 | 18.4 |

---

### 3. Generated Visual Deliverables (300 DPI)
All 39 figures have been rendered and saved in `d:/code/Mangoleaf/all_models_evaluations/figures/`:
1. **14 Confusion Matrices (`cm_*.png`)**: 300 DPI 8x8 heatmaps with per-class classification metrics.
2. **14 Multi-Class ROC Curves (`roc_*.png`)**: One-vs-Rest ROC curves per disease class with macro-average AUC.
3. **14 Learning Dynamics (`curves_*.png`)**: Epoch-by-epoch loss convergence and validation accuracy.
4. **Master Multi-Set Bar Chart (`all_sets_master_comparison.png`)**: Consolidated side-by-side performance chart across all 4 sets.
