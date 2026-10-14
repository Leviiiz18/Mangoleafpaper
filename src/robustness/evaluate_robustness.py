import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from src.config import (
    NUM_CLASSES,
    DEVICE,
    CHECKPOINTS_DIR,
    SPLITS_DIR,
    RESULTS_DIR,
    TABLES_DIR,
    IMAGE_SIZE,
    NORM_MEAN,
    NORM_STD
)
from src.train_teacher import build_swin_teacher
from src.train_students import build_mobilenet_student
from src.robustness.corruptions import CORRUPTION_TYPES, SEVERITY_LEVELS, apply_corruption

def get_clean_test_items():
    test_csv = os.path.join(SPLITS_DIR, "test.csv")
    df = pd.read_csv(test_csv)
    items = []
    for _, row in df.iterrows():
        items.append((row["filepath"], int(row["label"]), row["class_name"]))
    return items

def evaluate_model_on_corruptions(model, test_items, device=DEVICE):
    model.eval()
    model.to(device)

    norm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    results = {}

    # Clean accuracy
    clean_preds = []
    clean_labels = []
    for path, label, _ in test_items:
        img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        tensor = norm_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = torch.argmax(model(tensor), dim=1).item()
        clean_preds.append(pred)
        clean_labels.append(label)
    clean_acc = accuracy_score(clean_labels, clean_preds)
    results["clean"] = {"accuracy": float(clean_acc)}

    print(f"Clean Test Accuracy: {clean_acc * 100:.2f}%")

    for corr_type in CORRUPTION_TYPES:
        results[corr_type] = {}
        for sev in SEVERITY_LEVELS:
            corrupt_preds = []
            for path, label, _ in test_items:
                img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                img_np = np.array(img)
                corrupt_np = apply_corruption(img_np, corr_type, severity=sev)
                corrupt_pil = Image.fromarray(corrupt_np)
                tensor = norm_transform(corrupt_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = torch.argmax(model(tensor), dim=1).item()
                corrupt_preds.append(pred)

            sev_acc = accuracy_score(clean_labels, corrupt_preds)
            results[corr_type][f"severity_{sev}"] = float(sev_acc)
            print(f"  {corr_type} (Sev {sev}): {sev_acc * 100:.2f}% (Retention: {sev_acc/clean_acc*100:.1f}%)")

        # Compute average corruption accuracy & area under severity curve
        accs = [clean_acc] + [results[corr_type][f"severity_{s}"] for s in SEVERITY_LEVELS]
        # Normalized AUC over severity levels [0, 1, 2, 3] / 3.0
        auc_score = float(np.trapz(accs, [0, 1, 2, 3]) / 3.0)
        results[corr_type]["mean_acc"] = float(np.mean(accs[1:]))
        results[corr_type]["auc_robustness"] = auc_score
        results[corr_type]["retention_ratio"] = float(auc_score / clean_acc if clean_acc > 0 else 0.0)

    return results

def run_robustness_benchmark():
    test_items = get_clean_test_items()
    models = [
        ("Teacher (Swin-T)", "teacher_swin.pth", lambda: build_swin_teacher(NUM_CLASSES, pretrained=False, freeze_early=False)),
        ("Student A (Plain)", "student_plain.pth", lambda: build_mobilenet_student(NUM_CLASSES, pretrained=False)),
        ("Student B (Distilled)", "student_distilled.pth", lambda: build_mobilenet_student(NUM_CLASSES, pretrained=False))
    ]

    all_results = {}
    table_rows = []

    for model_name, ckpt_name, builder in models:
        ckpt_path = os.path.join(CHECKPOINTS_DIR, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"Skipping {model_name}: checkpoint {ckpt_path} not found.")
            continue

        print(f"\n==========================================")
        print(f"Evaluating Robustness for {model_name}...")
        print(f"==========================================")
        model = builder()
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])

        res = evaluate_model_on_corruptions(model, test_items, DEVICE)
        all_results[model_name] = res

        row = {
            "Model": model_name,
            "Clean Acc (%)": round(res["clean"]["accuracy"] * 100, 2)
        }
        for corr in CORRUPTION_TYPES:
            row[f"{corr} (Sev 1)"] = round(res[corr]["severity_1"] * 100, 2)
            row[f"{corr} (Sev 2)"] = round(res[corr]["severity_2"] * 100, 2)
            row[f"{corr} (Sev 3)"] = round(res[corr]["severity_3"] * 100, 2)
            row[f"{corr} AUC"] = round(res[corr]["auc_robustness"] * 100, 2)
            row[f"{corr} Ret (%)"] = round(res[corr]["retention_ratio"] * 100, 2)

        # Average AUC across all 5 corruptions
        mean_auc = np.mean([res[c]["auc_robustness"] for c in CORRUPTION_TYPES])
        mean_ret = np.mean([res[c]["retention_ratio"] for c in CORRUPTION_TYPES])
        row["Overall Robustness AUC (%)"] = round(mean_auc * 100, 2)
        row["Overall Retention (%)"] = round(mean_ret * 100, 2)

        table_rows.append(row)

    df_table = pd.DataFrame(table_rows)
    print("\n--- Robustness Benchmark Summary ---")
    print(df_table[["Model", "Clean Acc (%)", "Overall Robustness AUC (%)", "Overall Retention (%)"]].to_string(index=False))

    results_path = os.path.join(RESULTS_DIR, "robustness_benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    table_path = os.path.join(TABLES_DIR, "table3_robustness_comparison.csv")
    df_table.to_csv(table_path, index=False)
    print(f"Saved robustness results to {results_path} and {table_path}")

    return all_results, df_table

if __name__ == "__main__":
    run_robustness_benchmark()
