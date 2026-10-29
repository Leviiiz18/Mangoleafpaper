"""Script to assemble code, reports, figures, metrics, and scripts into a clean package folder
and generate a convenient zip file for the user and mentor.
"""
import os
import shutil
import zipfile

BASE_DIR = r"d:\code\Mangoleaf"
PACKAGE_DIR = os.path.join(BASE_DIR, "Mango_Leaf_MultiSet_Research_Package")
ZIP_PATH = os.path.join(BASE_DIR, "Mango_Leaf_MultiSet_Research_Package.zip")

# Clean existing
if os.path.exists(PACKAGE_DIR):
    shutil.rmtree(PACKAGE_DIR)
if os.path.exists(ZIP_PATH):
    os.remove(ZIP_PATH)

os.makedirs(PACKAGE_DIR, exist_ok=True)
figures_dest = os.path.join(PACKAGE_DIR, "figures")
metrics_dest = os.path.join(PACKAGE_DIR, "metrics")
code_dest = os.path.join(PACKAGE_DIR, "code")
docs_dest = os.path.join(PACKAGE_DIR, "reports_and_documentation")

for d in [figures_dest, metrics_dest, code_dest, docs_dest]:
    os.makedirs(d, exist_ok=True)

print("Copying documentation and reports...")
for f in ["Model_Sets_and_Training_Plan.docx", "Model_Sets_and_Training_Plan.md"]:
    src_f = os.path.join(BASE_DIR, f)
    if os.path.exists(src_f):
        shutil.copy2(src_f, docs_dest)

csv_src = os.path.join(BASE_DIR, "all_models_evaluations", "master_evaluation_metrics.csv")
if os.path.exists(csv_src):
    shutil.copy2(csv_src, docs_dest)

print("Copying high-resolution 300 DPI figures...")
fig_src_dir = os.path.join(BASE_DIR, "all_models_evaluations", "figures")
if os.path.exists(fig_src_dir):
    for f in os.listdir(fig_src_dir):
        if f.endswith(".png") or f.endswith(".pdf"):
            shutil.copy2(os.path.join(fig_src_dir, f), figures_dest)

print("Copying metrics JSONs...")
metrics_src_dir = os.path.join(BASE_DIR, "all_models_evaluations", "metrics")
if os.path.exists(metrics_src_dir):
    for f in os.listdir(metrics_src_dir):
        if f.endswith(".json"):
            shutil.copy2(os.path.join(metrics_src_dir, f), metrics_dest)

print("Copying code and scripts...")
mld_src = os.path.join(BASE_DIR, "Mango Leaf Diseases", "src")
if os.path.exists(mld_src):
    shutil.copytree(mld_src, os.path.join(code_dest, "src"))

scripts_to_copy = [
    "plot_master_comparison.py",
    "update_mentor_docs.py",
    "generate_mentor_model_sets_doc.py"
]
for sc in scripts_to_copy:
    p = os.path.join(BASE_DIR, sc)
    if os.path.exists(p):
        shutil.copy2(p, code_dest)

mld_scripts = [
    "train_all_sets_master_pipeline.py",
    "auto_train_and_plot.py",
    "finalize_set1_plots.py"
]
for sc in mld_scripts:
    p = os.path.join(BASE_DIR, "Mango Leaf Diseases", sc)
    if os.path.exists(p):
        shutil.copy2(p, code_dest)

# Add a comprehensive README in the package
readme_content = """# Mango Leaf Pathology: Multi-Set Vision Model Benchmark & XKD Package
Target Journal: Smart Agricultural Technology (Elsevier)

This package contains the complete empirical results, 300 DPI figures, evaluation metrics, reports, and code for all 4 Architectural Sets (16 models).

## Directory Structure
- reports_and_documentation/
  - Model_Sets_and_Training_Plan.docx (Official Word document for mentors/collaborators)
  - Model_Sets_and_Training_Plan.md (Markdown specification)
  - master_evaluation_metrics.csv (Complete empirical test results table)
- figures/
  - all_sets_master_comparison.png (Consolidated 300 DPI comparison across all 4 sets)
  - cm_*.png (14 Confusion Matrices for Teachers and Students)
  - roc_*.png (14 Multi-Class ROC Curves with AUC per disease and macro AUC)
  - curves_*.png (14 Training Dynamics Loss and Accuracy curves)
- metrics/
  - *_metrics.json (Per-model detailed test evaluation and training logs)
- code/
  - train_all_sets_master_pipeline.py (Automated training pipeline for all 4 sets)
  - auto_train_and_plot.py (Evaluation and 300 DPI plotting engine)
  - src/ (Core model definitions, KD/XKD loss functions, metrics, and dataloaders)

## Summary of Results
- Set 1 (ViT Family): Swin-T Teacher (99.38% Acc) -> MobileNetV3-S XKD (98.44% Acc, 0.998 AUC)
- Set 2 (Pure CNN Family): ConvNeXt-T Teacher (99.06% Acc) -> MobileNetV4-S KD (98.12% Acc, +1.87% gain)
- Set 3 (Efficiency Family): EffNetV2-S Teacher (99.53% Acc) -> EffNet-B0 KD (99.22% Acc, +0.16% gain)
- Set 4 (Ultra-Edge Family): MaxViT-T Teacher (98.91% Acc) -> GhostNet XKD (99.06% Acc, +0.31% gain, exceeds teacher)
"""
with open(os.path.join(PACKAGE_DIR, "README.md"), "w", encoding="utf-8") as f:
    f.write(readme_content)

print(f"Creating zip archive at: {ZIP_PATH} ...")
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(PACKAGE_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, BASE_DIR)
            zipf.write(full_path, rel_path)

zip_size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
print(f"Zip created successfully! Size: {zip_size_mb:.2f} MB")
