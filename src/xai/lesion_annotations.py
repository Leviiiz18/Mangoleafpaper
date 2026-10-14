import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.config import MASKS_DIR, SPLITS_DIR, IMAGE_SIZE

DISEASED_CLASSES = [
    "Anthracnose",
    "Bacterial Canker",
    "Die Back",
    "Gall Midge",
    "Powdery Mildew",
    "Sooty Mould"
]

def segment_pathology_mask(image_rgb, class_name):
    """
    Domain-aware heuristic lesion segmentation for mango leaf diseases.
    Generates binary lesion masks (224, 224) based on plant pathology color/texture signatures.
    """
    img_resized = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    # First isolate the leaf foreground from the background
    # Mango leaves are predominantly green/yellow-green
    # Non-leaf background is often white/neutral or wooden bench
    leaf_mask = (hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 90) & (hsv[:, :, 1] >= 30)
    # Morphological close to fill interior leaf holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    leaf_mask = cv2.morphologyEx(leaf_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_DILATE, kernel)

    if class_name in ["Anthracnose", "Bacterial Canker"]:
        # Dark necrotic brown/black spots: low V, high contrast in LAB L-channel
        l_channel = lab[:, :, 0]
        # Otsu threshold on darker pixels within leaf
        dark_pixels = (l_channel < 110) & (leaf_mask > 0)
        mask = dark_pixels.astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    elif class_name == "Die Back":
        # Necrotic brown discoloration typically at tips and margins
        # Low green hue, high saturation, reddish/brownish
        brown = ((hsv[:, :, 0] < 25) | (hsv[:, :, 0] > 160)) & (hsv[:, :, 1] > 50) & (hsv[:, :, 2] < 180) & (leaf_mask > 0)
        mask = brown.astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    elif class_name == "Gall Midge":
        # Small raised circular blister spots
        # Local adaptive thresholding on gray channel
        adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
        mask = cv2.bitwise_and(adapt, adapt, mask=leaf_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    elif class_name == "Powdery Mildew":
        # White/pale grayish fungal patches: high L, low saturation
        pale = (lab[:, :, 0] > 165) & (hsv[:, :, 1] < 60) & (leaf_mask > 0)
        mask = pale.astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    elif class_name == "Sooty Mould":
        # Dense black velvety fungal mycelium covering: very low V in HSV (< 65)
        black_mould = (hsv[:, :, 2] < 70) & (leaf_mask > 0)
        mask = black_mould.astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    else:
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    # Binary mask: 1 for lesion, 0 for background/healthy leaf
    return (mask > 0).astype(np.uint8)

def generate_annotated_dataset(num_per_class=20):
    """
    Selects test images for the 6 diseased classes and generates ground-truth lesion masks.
    Saves masks to masks/ directory as binary PNGs.
    """
    test_csv = os.path.join(SPLITS_DIR, "test.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test split {test_csv} not found. Run split_data.py first!")

    df_test = pd.read_csv(test_csv)
    manifest = []

    print(f"--- Generating Lesion Annotation Masks ({num_per_class}/class for 6 disease classes) ---")
    for cls_name in DISEASED_CLASSES:
        cls_df = df_test[df_test["class_name"] == cls_name].head(num_per_class)
        print(f"Class '{cls_name}': annotating {len(cls_df)} images...")

        for idx, row in cls_df.iterrows():
            img_path = row["filepath"]
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_filename = f"{cls_name}_{img_name}_mask.png"
            mask_path = os.path.join(MASKS_DIR, mask_filename)

            image = Image.open(img_path).convert("RGB")
            img_np = np.array(image)

            lesion_mask = segment_pathology_mask(img_np, cls_name)
            # Save mask as 0 and 255 PNG
            cv2.imwrite(mask_path, lesion_mask * 255)

            manifest.append({
                "image_path": img_path,
                "mask_path": mask_path,
                "class_name": cls_name,
                "label": row["label"],
                "lesion_pixel_count": int(np.sum(lesion_mask)),
                "lesion_percentage": float(np.mean(lesion_mask) * 100)
            })

    df_manifest = pd.DataFrame(manifest)
    manifest_csv = os.path.join(MASKS_DIR, "lesion_masks_manifest.csv")
    df_manifest.to_csv(manifest_csv, index=False)
    print(f"Generated {len(df_manifest)} ground-truth lesion masks.")
    print(f"Manifest saved to {manifest_csv}")
    return df_manifest

if __name__ == "__main__":
    generate_annotated_dataset()
