import os
import sys
import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2

from src.config import (
    CLASS_NAMES,
    NUM_CLASSES,
    DEVICE,
    CHECKPOINTS_DIR,
    SPLITS_DIR,
    FIGURES_DIR,
    IMAGE_SIZE,
    NORM_MEAN,
    NORM_STD
)
from src.train_teacher import build_swin_teacher
from src.xai.gradcam import get_gradcam_explainer, overlay_cam_on_image

def load_teacher_model():
    ckpt_path = os.path.join(CHECKPOINTS_DIR, "teacher_swin.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    model = build_swin_teacher(num_classes=NUM_CLASSES, pretrained=False, freeze_early=False)
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model

def verify_leaf_presence(img_pil):
    img_np = np.array(img_pil.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    green_mask = (hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] >= 25)
    brown_mask = (hsv[:, :, 0] >= 10) & (hsv[:, :, 0] < 25) & (hsv[:, :, 1] >= 25) & (hsv[:, :, 2] < 210)
    leaf_pixels = (green_mask | brown_mask).sum()
    leaf_ratio = leaf_pixels / (IMAGE_SIZE * IMAGE_SIZE)
    return (leaf_ratio >= 0.18), round(float(leaf_ratio * 100), 1)

def predict_and_explain(image_path, model=None, save_output=True):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    orig_pil = Image.open(image_path).convert("RGB")
    is_valid, leaf_pct = verify_leaf_presence(orig_pil)
    if not is_valid:
        print("\n" + "="*60)
        print(f"REJECTION: {os.path.basename(image_path)} IS NOT A MANGO LEAF")
        print("="*60)
        print(f"Foliage Detection Score: {leaf_pct}% (Threshold: >= 18%)")
        print("RESULT: REJECTED - Out-of-Distribution (OOD) non-leaf image.")
        print("No diagnosis performed.")
        print("="*60)
        return {
            "image_path": image_path,
            "predicted_class": "NOT A MANGO LEAF",
            "confidence": 0.0,
            "is_valid": False
        }

    if model is None:
        model = load_teacher_model()

    cam_explainer = get_gradcam_explainer(model, "swin")

    # Load and preprocess image
    orig_pil = Image.open(image_path).convert("RGB")
    resized_pil = orig_pil.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_np = np.array(resized_pil)

    norm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    input_tensor = norm_transform(resized_pil).unsqueeze(0).to(DEVICE)

    # Grad-CAM and inference
    cam_heatmap, logits = cam_explainer.generate_cam(input_tensor)
    cam_explainer.remove_hooks()

    probs = F.softmax(logits, dim=1).detach().cpu().squeeze().numpy()
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx] * 100.0

    print("\n" + "="*60)
    print(f"MANUAL INFERENCE RESULT FOR: {os.path.basename(image_path)}")
    print("="*60)
    print(f"PREDICTED CLASS:  {pred_class.upper()}")
    print(f"CONFIDENCE:       {confidence:.2f}%")
    print("-" * 60)
    print("All Class Probabilities:")
    sorted_indices = np.argsort(-probs)
    for rank, idx in enumerate(sorted_indices, 1):
        bar = "#" * int(probs[idx] * 30)
        print(f" {rank}. {CLASS_NAMES[idx]:<18}: {probs[idx]*100:6.2f}% {bar}")
    print("="*60)

    # Create visualization
    overlay = overlay_cam_on_image(img_np, cam_heatmap, alpha=0.5)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image (224x224)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(cam_heatmap, cmap="jet")
    axes[1].set_title("Swin-T Attention / Grad-CAM", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Prediction: {pred_class} ({confidence:.1f}%)", fontsize=12, fontweight="bold", color="darkgreen")
    axes[2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "manual_test_result.png")
    if save_output:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"\nSaved visualization with Grad-CAM to: {out_path}")
    plt.close()

    return {
        "image_path": image_path,
        "predicted_class": pred_class,
        "confidence": float(confidence),
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)},
        "visualization_path": out_path
    }

def get_random_test_image():
    test_csv = os.path.join(SPLITS_DIR, "test.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError("test.csv not found!")
    import pandas as pd
    df = pd.read_csv(test_csv)
    sample = df.sample(n=1).iloc[0]
    return sample["filepath"], sample["class_name"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Swin-T Teacher Model on any leaf image.")
    parser.add_argument("--image", type=str, default=None, help="Path to leaf image file")
    parser.add_argument("--random", action="store_true", help="Pick a random image from the test set")
    args = parser.parse_args()

    if args.image:
        predict_and_explain(args.image)
    else:
        # Default: pick a random test image
        img_path, true_cls = get_random_test_image()
        print(f"[Random Sample Selected] True Label: {true_cls}")
        predict_and_explain(img_path)
