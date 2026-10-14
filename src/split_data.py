import os
import random
import pandas as pd
from src.config import (
    PROCESS_DATA_DIR,
    AUGMENTED_DATA_DIR,
    SPLITS_DIR,
    CLASS_NAMES,
    CLASS_TO_IDX,
    SEED
)

def get_class_dir_map(base_dir):
    """Map canonical class name to actual folder name (case-insensitive)."""
    mapping = {}
    if not os.path.exists(base_dir):
        return mapping
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            folder_lower = folder.lower().strip()
            for c in CLASS_NAMES:
                if c.lower().strip() == folder_lower:
                    mapping[c] = folder_path
                    break
    return mapping

def generate_splits(val_per_class=80, test_per_class=80, max_aug_per_class=300, seed=SEED):
    """
    Replicate base paper (Prabhu et al., 2025):
    - 80 val images/class strictly from original process data (640 total)
    - 80 test images/class strictly from original process data (640 total)
    - Train set: Remaining original process data + up to 300 augmented images/class
    """
    random.seed(seed)
    process_map = get_class_dir_map(PROCESS_DATA_DIR)
    augmented_map = get_class_dir_map(AUGMENTED_DATA_DIR)

    train_rows = []
    val_rows = []
    test_rows = []

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    print("--- Partitioning Dataset ---")
    for cls_name in CLASS_NAMES:
        cls_idx = CLASS_TO_IDX[cls_name]
        proc_path = process_map.get(cls_name)
        if not proc_path:
            raise FileNotFoundError(f"Missing process data directory for class '{cls_name}'")

        # Collect original images
        orig_files = [
            os.path.join(proc_path, f)
            for f in sorted(os.listdir(proc_path))
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
        total_orig = len(orig_files)
        if total_orig < (val_per_class + test_per_class):
            raise ValueError(f"Class '{cls_name}' only has {total_orig} original images; need {val_per_class + test_per_class}")

        # Shuffle deterministically
        shuffled_orig = orig_files.copy()
        random.shuffle(shuffled_orig)

        # Allocate test and val
        test_files = shuffled_orig[:test_per_class]
        val_files = shuffled_orig[test_per_class : test_per_class + val_per_class]
        rem_orig_files = shuffled_orig[test_per_class + val_per_class :]

        for f in test_files:
            test_rows.append({"filepath": f, "class_name": cls_name, "label": cls_idx, "is_augmented": 0})
        for f in val_files:
            val_rows.append({"filepath": f, "class_name": cls_name, "label": cls_idx, "is_augmented": 0})
        for f in rem_orig_files:
            train_rows.append({"filepath": f, "class_name": cls_name, "label": cls_idx, "is_augmented": 0})

        # Collect augmented images
        aug_path = augmented_map.get(cls_name)
        aug_files = []
        if aug_path and os.path.exists(aug_path):
            aug_files = [
                os.path.join(aug_path, f)
                for f in sorted(os.listdir(aug_path))
                if os.path.splitext(f)[1].lower() in valid_exts
            ]
            random.shuffle(aug_files)
            selected_aug = aug_files[:max_aug_per_class]
            for f in selected_aug:
                train_rows.append({"filepath": f, "class_name": cls_name, "label": cls_idx, "is_augmented": 1})
        
        print(f"Class '{cls_name}': orig={total_orig} -> test={len(test_files)}, val={len(val_files)}, train_orig={len(rem_orig_files)}, train_aug={min(len(aug_files), max_aug_per_class)}")

    df_train = pd.DataFrame(train_rows)
    df_val = pd.DataFrame(val_rows)
    df_test = pd.DataFrame(test_rows)

    train_csv = os.path.join(SPLITS_DIR, "train.csv")
    val_csv = os.path.join(SPLITS_DIR, "val.csv")
    test_csv = os.path.join(SPLITS_DIR, "test.csv")

    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    print("\n--- Split Summary ---")
    print(f"Train samples: {len(df_train)} (Orig: {(df_train['is_augmented']==0).sum()}, Aug: {(df_train['is_augmented']==1).sum()})")
    print(f"Val samples:   {len(df_val)} (80/class)")
    print(f"Test samples:  {len(df_test)} (80/class)")
    print(f"Saved splits to {SPLITS_DIR}")

if __name__ == "__main__":
    generate_splits()
