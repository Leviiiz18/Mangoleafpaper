import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2

class BaseGradCAM:
    def __init__(self, model, target_layer, is_transformer=False):
        self.model = model
        self.target_layer = target_layer
        self.is_transformer = is_transformer
        self.activations = None
        self.gradients = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        self.activations = None
        self.gradients = None

        # Forward pass
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        score = logits[0, target_class]

        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=False)

        activations = self.activations.detach()
        gradients = self.gradients.detach()
        self.activations = None
        self.gradients = None
        self.model.zero_grad()

        if activations is None or gradients is None:
            raise RuntimeError("Failed to capture activations/gradients in Grad-CAM.")

        if self.is_transformer:
            # Handle Swin Transformer shape: could be (B, H, W, C) or (B, N, C)
            if len(activations.shape) == 3: # (B, N, C) where N = H*W = 49
                B, N, C = activations.shape
                H = W = int(np.sqrt(N))
                act = activations[0].reshape(H, W, C)
                grad = gradients[0].reshape(H, W, C)
            elif len(activations.shape) == 4: # (B, H, W, C)
                act = activations[0]
                grad = gradients[0]
            else:
                raise ValueError(f"Unexpected transformer activation shape: {activations.shape}")

            # Global average pool over spatial dims (H, W) -> weights per channel C
            weights = torch.mean(grad, dim=(0, 1)) # (C,)
            cam = torch.sum(weights * act, dim=-1) # (H, W)
        else:
            # Standard CNN shape: (B, C, H, W)
            act = activations[0] # (C, H, W)
            grad = gradients[0] # (C, H, W)
            weights = torch.mean(grad, dim=(1, 2), keepdim=True) # (C, 1, 1)
            cam = torch.sum(weights * act, dim=0) # (H, W)

        cam = F.relu(cam)
        cam = cam.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        # Min-Max Normalization to [0, 1]
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, logits.detach()

def get_gradcam_explainer(model, model_name="swin"):
    """
    Factory for Grad-CAM explainer based on model architecture.
    """
    if "swin" in model_name.lower():
        # Target last stage of Swin Transformer
        # In timm swin_tiny: model.layers[-1].blocks[-1].norm2 or model.norm
        target_layer = model.layers[-1].blocks[-1].norm2
        return BaseGradCAM(model, target_layer, is_transformer=True)
    elif "mobilenet" in model_name.lower():
        # Target last conv layer in features
        target_layer = model.features[-1]
        return BaseGradCAM(model, target_layer, is_transformer=False)
    else:
        raise ValueError(f"Unsupported model for Grad-CAM: {model_name}")

def overlay_cam_on_image(img_np, cam, colormap=cv2.COLORMAP_JET, alpha=0.5):
    """
    Overlay Grad-CAM heatmap onto image (np.uint8, HxWx3).
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.float32(heatmap) * alpha + np.float32(img_np) * (1.0 - alpha)
    return np.uint8(np.clip(overlay, 0, 255))
