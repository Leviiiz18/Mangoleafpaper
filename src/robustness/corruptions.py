import io
import random
import cv2
import numpy as np
from PIL import Image

CORRUPTION_TYPES = [
    "gaussian_blur",
    "brightness_contrast",
    "gaussian_noise",
    "jpeg_compression",
    "occlusion"
]

SEVERITY_LEVELS = [1, 2, 3]

def apply_gaussian_blur(image_np, severity=1):
    """
    Gaussian blur with increasing kernel sizes.
    severity 1: 5x5, severity 2: 9x9, severity 3: 15x15
    """
    ksize_map = {1: (5, 5), 2: (9, 9), 3: (15, 15)}
    sigma_map = {1: 1.5, 2: 3.0, 3: 5.0}
    ksize = ksize_map[severity]
    sigma = sigma_map[severity]
    return cv2.GaussianBlur(image_np, ksize, sigma)

def apply_brightness_contrast(image_np, severity=1):
    """
    Brightness and contrast variation.
    """
    alpha_map = {1: 1.25, 2: 1.5, 3: 1.75} # contrast
    beta_map = {1: 25, 2: 50, 3: 75}       # brightness
    alpha = alpha_map[severity]
    beta = beta_map[severity]
    # randomly increase or decrease
    if severity % 2 == 1:
        corrupted = cv2.convertScaleAbs(image_np, alpha=alpha, beta=beta)
    else:
        corrupted = cv2.convertScaleAbs(image_np, alpha=1.0/alpha, beta=-beta)
    return corrupted

def apply_gaussian_noise(image_np, severity=1):
    """
    Gaussian additive noise.
    severity 1: sigma=15, severity 2: sigma=30, severity 3: sigma=50
    """
    sigma_map = {1: 15, 2: 30, 3: 50}
    sigma = sigma_map[severity]
    noise = np.random.normal(0, sigma, image_np.shape).astype(np.float32)
    corrupted = np.clip(image_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return corrupted

def apply_jpeg_compression(image_np, severity=1):
    """
    JPEG compression artifacts.
    severity 1: Q=50, severity 2: Q=30, severity 3: Q=10
    """
    quality_map = {1: 50, 2: 30, 3: 10}
    quality = quality_map[severity]
    pil_img = Image.fromarray(image_np)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    corrupted_pil = Image.open(buffer)
    return np.array(corrupted_pil)

def apply_occlusion(image_np, severity=1):
    """
    Random square occlusion cutout representing leaves/obstructions in field photos.
    severity 1: 32x32 patch, severity 2: 64x64 patch, severity 3: 96x96 patch
    """
    size_map = {1: 32, 2: 64, 3: 96}
    patch_size = size_map[severity]
    H, W, C = image_np.shape
    corrupted = image_np.copy()

    # Place random square in image
    max_y = max(0, H - patch_size)
    max_x = max(0, W - patch_size)
    y1 = random.randint(0, max_y)
    x1 = random.randint(0, max_x)
    y2 = min(H, y1 + patch_size)
    x2 = min(W, x1 + patch_size)

    # Fill patch with neutral gray (128)
    corrupted[y1:y2, x1:x2, :] = 128
    return corrupted

def apply_corruption(image_np, corruption_type, severity=1):
    if corruption_type == "gaussian_blur":
        return apply_gaussian_blur(image_np, severity)
    elif corruption_type == "brightness_contrast":
        return apply_brightness_contrast(image_np, severity)
    elif corruption_type == "gaussian_noise":
        return apply_gaussian_noise(image_np, severity)
    elif corruption_type == "jpeg_compression":
        return apply_jpeg_compression(image_np, severity)
    elif corruption_type == "occlusion":
        return apply_occlusion(image_np, severity)
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")
