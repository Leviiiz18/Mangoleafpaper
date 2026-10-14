import numpy as np
import torch
import torch.nn.functional as F

def compute_deletion_insertion_auc(model, image_tensor, heatmap, target_class=None, steps=20, device="cuda"):
    """
    Computes Deletion AUC and Insertion AUC for an image and its importance heatmap.
    - Deletion: Progressively remove top pixels (e.g. replace with blurred/mean), tracking confidence drop.
      Lower AUC indicates faster drop -> higher explanation faithfulness.
    - Insertion: Progressively insert top pixels into blank/blurred background, tracking confidence rise.
      Higher AUC indicates faster rise -> higher explanation faithfulness.
    """
    model.eval()
    model.to(device)

    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        orig_logits = model(image_tensor)
        orig_probs = F.softmax(orig_logits, dim=1)
        if target_class is None:
            target_class = torch.argmax(orig_probs, dim=1).item()
        base_score = orig_probs[0, target_class].item()

    H, W = heatmap.shape
    num_pixels = H * W
    flat_heatmap = heatmap.flatten()
    sorted_indices = np.argsort(-flat_heatmap) # descending order (most important first)

    # Substrate for baseline: blurred image
    # Simple blur kernel
    blur_kernel = torch.ones(1, 1, 11, 11, device=device) / 121.0
    blurred_img = torch.cat([
        F.conv2d(image_tensor[:, c:c+1], blur_kernel, padding=5)
        for c in range(3)
    ], dim=1)

    # Percentage steps: 0% to 100%
    step_fractions = np.linspace(0.0, 1.0, steps + 1)
    deletion_scores = [base_score]
    insertion_scores = []

    # Initial insertion (0% inserted -> fully blurred image)
    with torch.no_grad():
        init_ins_probs = F.softmax(model(blurred_img), dim=1)
        insertion_scores.append(init_ins_probs[0, target_class].item())

    # Pixel masks
    for frac in step_fractions[1:]:
        k = int(frac * num_pixels)
        top_k_indices = sorted_indices[:k]

        mask = np.zeros(num_pixels, dtype=bool)
        mask[top_k_indices] = True
        mask = mask.reshape(H, W)
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        # Deletion: image where top pixels are replaced by blurred pixels
        del_input = image_tensor * (1.0 - mask_tensor) + blurred_img * mask_tensor
        # Insertion: blurred image where top pixels are restored from original image
        ins_input = blurred_img * (1.0 - mask_tensor) + image_tensor * mask_tensor

        with torch.no_grad():
            del_p = F.softmax(model(del_input), dim=1)[0, target_class].item()
            ins_p = F.softmax(model(ins_input), dim=1)[0, target_class].item()

        deletion_scores.append(del_p)
        insertion_scores.append(ins_p)

    # Compute trapezoidal AUC
    del_auc = float(np.trapz(deletion_scores, step_fractions))
    ins_auc = float(np.trapz(insertion_scores, step_fractions))

    return del_auc, ins_auc, deletion_scores, insertion_scores

def compute_mask_iou(heatmap, ground_truth_mask, threshold_pct=0.30):
    """
    Calculates Intersection-over-Union (IoU) between the binarized explanation heatmap and the ground truth lesion mask.
    threshold_pct: Fraction of top activation considered as explanation region (e.g. top 30%).
    """
    if np.sum(ground_truth_mask) == 0:
        return 0.0

    threshold = np.percentile(heatmap, (1.0 - threshold_pct) * 100)
    binary_cam = heatmap >= threshold

    gt_binary = ground_truth_mask > 0

    intersection = np.logical_and(binary_cam, gt_binary).sum()
    union = np.logical_or(binary_cam, gt_binary).sum()

    if union == 0:
        return 0.0
    return float(intersection / union)
