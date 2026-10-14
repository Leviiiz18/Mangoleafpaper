import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
try:
    from lime import lime_image
    from skimage.segmentation import slic, mark_boundaries
except ImportError:
    lime_image = None
    slic = None

from src.config import NORM_MEAN, NORM_STD, DEVICE

class LIMEModelWrapper:
    def __init__(self, model, device=DEVICE):
        self.model = model
        self.device = device
        self.mean = np.array(NORM_MEAN)
        self.std = np.array(NORM_STD)

    def predict(self, images_np):
        """
        Expects batch of images as np.ndarray with values in [0, 1] or [0, 255].
        Returns class probabilities (N, num_classes).
        """
        self.model.eval()
        if images_np.max() > 1.0:
            images_np = images_np / 255.0

        # Normalize with ImageNet mean and std
        norm_imgs = (images_np - self.mean) / self.std
        # Transpose (N, H, W, C) -> (N, C, H, W)
        tensor_imgs = torch.tensor(norm_imgs, dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_imgs)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()

        return probs

def explain_lime(model, image_np, target_class=None, num_samples=300, num_features=5, device=DEVICE):
    """
    Generates LIME explanation for an image (H, W, 3 uint8 or float).
    Returns heatmap/mask where positive superpixels have importance weights.
    """
    if lime_image is None:
        raise ImportError("lime or scikit-image is not installed.")

    wrapper = LIMEModelWrapper(model, device)
    explainer = lime_image.LimeImageExplainer()

    if image_np.dtype == np.uint8:
        img_input = image_np.astype(np.float64) / 255.0
    else:
        img_input = image_np.astype(np.float64)

    explanation = explainer.explain_instance(
        img_input,
        wrapper.predict,
        top_labels=8,
        hide_color=0,
        num_samples=num_samples,
        segmentation_fn=lambda x: slic(x, n_segments=50, compactness=10, sigma=1)
    )

    if target_class is None:
        target_class = explanation.top_labels[0]

    # Extract weights for segments
    temp, mask = explanation.get_image_and_mask(
        target_class,
        positive_only=True,
        num_features=num_features,
        hide_rest=False
    )

    # Generate continuous heatmap based on segment local weights
    segments = explanation.segments
    dict_weights = dict(explanation.local_exp[target_class])
    heatmap = np.zeros_like(segments, dtype=np.float32)
    for seg_idx, weight in dict_weights.items():
        if weight > 0:
            heatmap[segments == seg_idx] = weight

    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)

    return heatmap, mask, target_class
