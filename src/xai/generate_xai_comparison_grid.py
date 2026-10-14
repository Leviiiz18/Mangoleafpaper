import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

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
from src.train_students import build_mobilenet_student
from src.xai.gradcam import get_gradcam_explainer, overlay_cam_on_image
from src.xai.lime_explainer import explain_lime

def load_models():
    print("Loading models from checkpoints...")
    # 1. Teacher Swin-T
    teacher = build_swin_teacher(NUM_CLASSES, pretrained=False, freeze_early=False).to(DEVICE)
    ckpt_t = torch.load(os.path.join(CHECKPOINTS_DIR, "teacher_swin.pth"), map_location=DEVICE)
    teacher.load_state_dict(ckpt_t["model_state_dict"])
    teacher.eval()

    # 2. Student A (Plain)
    student_a = build_mobilenet_student(NUM_CLASSES, pretrained=False).to(DEVICE)
    ckpt_a = torch.load(os.path.join(CHECKPOINTS_DIR, "student_plain.pth"), map_location=DEVICE)
    student_a.load_state_dict(ckpt_a["model_state_dict"])
    student_a.eval()

    # 3. Student B (Distilled)
    student_b = build_mobilenet_student(NUM_CLASSES, pretrained=False).to(DEVICE)
    ckpt_b = torch.load(os.path.join(CHECKPOINTS_DIR, "student_distilled.pth"), map_location=DEVICE)
    student_b.load_state_dict(ckpt_b["model_state_dict"])
    student_b.eval()

    # Explainers
    cam_teacher = get_gradcam_explainer(teacher, "swin")
    cam_student_a = get_gradcam_explainer(student_a, "mobilenet")
    cam_student_b = get_gradcam_explainer(student_b, "mobilenet")

    return (teacher, student_a, student_b), (cam_teacher, cam_student_a, cam_student_b)

def pick_representative_images():
    """
    Selects 1 clear, high-quality test image for each of the 8 classes.
    """
    test_csv = os.path.join(SPLITS_DIR, "test.csv")
    df = pd.read_csv(test_csv)
    selected = {}

    for cls in CLASS_NAMES:
        cls_df = df[df["class_name"] == cls]
        # Choose a representative sample (e.g. index 3 or 5 for clear presentation)
        sample_path = cls_df.iloc[min(3, len(cls_df)-1)]["filepath"]
        selected[cls] = sample_path

    return selected

def create_xai_comparison_grid():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    models, explainers = load_models()
    teacher, student_a, student_b = models
    cam_t, cam_a, cam_b = explainers

    norm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    representative_images = pick_representative_images()

    cols = [
        "Original Leaf",
        "Teacher Grad-CAM\n(Swin-T)",
        "Student A Grad-CAM\n(Plain Mobile)",
        "Student B Grad-CAM\n(Distilled Mobile)",
        "Teacher LIME\n(Swin-T)",
        "Student A LIME\n(Plain Mobile)",
        "Student B LIME\n(Distilled Mobile)"
    ]

    fig, axes = plt.subplots(8, 7, figsize=(18, 22), dpi=300)

    for row_idx, (cls_name, img_path) in enumerate(representative_images.items()):
        print(f"[{row_idx+1}/8] Processing class: {cls_name} ({os.path.basename(img_path)})...")
        # Load & resize image
        pil_img = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        img_np = np.array(pil_img)
        img_tensor = norm_transform(pil_img).unsqueeze(0).to(DEVICE)
        target_class = CLASS_NAMES.index(cls_name)

        # 1. Grad-CAMs
        cam_map_t, _ = cam_t.generate_cam(img_tensor, target_class=target_class)
        overlay_cam_t = overlay_cam_on_image(img_np, cam_map_t, colormap=cv2.COLORMAP_JET, alpha=0.55)

        cam_map_a, _ = cam_a.generate_cam(img_tensor, target_class=target_class)
        overlay_cam_a = overlay_cam_on_image(img_np, cam_map_a, colormap=cv2.COLORMAP_JET, alpha=0.55)

        cam_map_b, _ = cam_b.generate_cam(img_tensor, target_class=target_class)
        overlay_cam_b = overlay_cam_on_image(img_np, cam_map_b, colormap=cv2.COLORMAP_JET, alpha=0.55)

        # 2. LIMEs
        lime_map_t, _, _ = explain_lime(teacher, img_np, target_class=target_class, num_samples=150, num_features=5, device=DEVICE)
        overlay_lime_t = overlay_cam_on_image(img_np, lime_map_t, colormap=cv2.COLORMAP_JET, alpha=0.55)

        lime_map_a, _, _ = explain_lime(student_a, img_np, target_class=target_class, num_samples=150, num_features=5, device=DEVICE)
        overlay_lime_a = overlay_cam_on_image(img_np, lime_map_a, colormap=cv2.COLORMAP_JET, alpha=0.55)

        lime_map_b, _, _ = explain_lime(student_b, img_np, target_class=target_class, num_samples=150, num_features=5, device=DEVICE)
        overlay_lime_b = overlay_cam_on_image(img_np, lime_map_b, colormap=cv2.COLORMAP_JET, alpha=0.55)

        row_images = [
            img_np,
            overlay_cam_t,
            overlay_cam_a,
            overlay_cam_b,
            overlay_lime_t,
            overlay_lime_a,
            overlay_lime_b
        ]

        for col_idx, im in enumerate(row_images):
            ax = axes[row_idx, col_idx]
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])

            # Column title only on top row
            if row_idx == 0:
                ax.set_title(cols[col_idx], fontsize=11, fontweight="bold", pad=8)

            # Row label on left-most column
            if col_idx == 0:
                ax.set_ylabel(cls_name, fontsize=12, fontweight="bold", rotation=0, labelpad=70, va="center")

    plt.suptitle("Qualitative Explainable AI (XAI) Comparison Grid Across All 8 Mango Leaf Disease Classes\nGrad-CAM and LIME Attribution Overlays (Teacher vs. Plain Student vs. Distilled Student)",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    out_file = os.path.join(FIGURES_DIR, "fig_xai_comparison_grid.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSuccessfully generated Experiment 1 figure at: {out_file}")

if __name__ == "__main__":
    create_xai_comparison_grid()
