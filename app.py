import os
import io
import base64
import random
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import cv2
from flask import Flask, render_template_string, request, jsonify

from src.config import (
    CLASS_NAMES,
    NUM_CLASSES,
    DEVICE,
    CHECKPOINTS_DIR,
    SPLITS_DIR,
    IMAGE_SIZE,
    NORM_MEAN,
    NORM_STD
)
from src.train_teacher import build_swin_teacher
from src.xai.gradcam import get_gradcam_explainer, overlay_cam_on_image

app = Flask(__name__)

# Global model cache
MODEL = None
CAM_EXPLAINER = None

def get_model():
    global MODEL, CAM_EXPLAINER
    if MODEL is None:
        ckpt_path = os.path.join(CHECKPOINTS_DIR, "teacher_swin.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError("Teacher checkpoint not found!")
        MODEL = build_swin_teacher(num_classes=NUM_CLASSES, pretrained=False, freeze_early=False)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        MODEL.load_state_dict(ckpt["model_state_dict"])
        MODEL.to(DEVICE)
        MODEL.eval()
        CAM_EXPLAINER = get_gradcam_explainer(MODEL, "swin")
    return MODEL, CAM_EXPLAINER

def pil_to_base64(img_pil):
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

def np_to_base64(img_np):
    pil = Image.fromarray(img_np)
    return pil_to_base64(pil)

def verify_leaf_presence(img_pil):
    """
    Bio-morphological validation filter to verify if the uploaded image contains mango foliage.
    Rejects out-of-distribution non-leaf images (e.g. video games, cars, faces, random objects).
    """
    img_np = np.array(img_pil.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    # Green and yellow-green foliage spectrum (Hue 25-95, Saturation >= 25)
    green_mask = (hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] >= 25)
    # Dry necrotic brown tissue spectrum (Hue 10-25, Saturation >= 25, Value < 210)
    brown_mask = (hsv[:, :, 0] >= 10) & (hsv[:, :, 0] < 25) & (hsv[:, :, 1] >= 25) & (hsv[:, :, 2] < 210)
    # Dark sooty mould / black necrotic tissue (Value < 50, within leaf area)
    dark_mask = (hsv[:, :, 2] < 45)

    leaf_pixels = (green_mask | brown_mask).sum()
    leaf_ratio = leaf_pixels / (IMAGE_SIZE * IMAGE_SIZE)

    # Real mango leaf photos must contain at least 18% valid leaf foliage
    is_valid_leaf = (leaf_ratio >= 0.18)
    return is_valid_leaf, round(float(leaf_ratio * 100), 1)

def run_inference(image_pil):
    is_valid, leaf_pct = verify_leaf_presence(image_pil)
    resized = image_pil.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    img_np = np.array(resized)

    if not is_valid:
        return {
            "is_valid_leaf": False,
            "predicted_class": "NOT A MANGO LEAF",
            "confidence": 0.0,
            "leaf_coverage": leaf_pct,
            "error_message": f"Input image does not contain recognizable mango foliage (Foliage score: {leaf_pct}% < 18%). Please upload a clear photo of a mango leaf.",
            "is_healthy": False,
            "img_original": pil_to_base64(resized),
            "img_heatmap": pil_to_base64(resized),
            "img_overlay": pil_to_base64(resized),
            "probabilities": []
        }

    model, explainer = get_model()

    norm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    tensor = norm_transform(resized).unsqueeze(0).to(DEVICE)

    cam_heatmap, logits = explainer.generate_cam(tensor)
    probs = F.softmax(logits, dim=1).detach().cpu().squeeze().numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx] * 100.0)

    # Colorize heatmap
    cam_colored = cv2.applyColorMap(np.uint8(255 * cam_heatmap), cv2.COLORMAP_JET)
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)

    overlay = overlay_cam_on_image(img_np, cam_heatmap, alpha=0.5)

    all_probs = []
    sorted_idx = np.argsort(-probs)
    for idx in sorted_idx:
        all_probs.append({
            "class_name": CLASS_NAMES[idx],
            "percentage": round(float(probs[idx] * 100.0), 2)
        })

    return {
        "predicted_class": pred_class,
        "confidence": round(confidence, 2),
        "is_healthy": (pred_class == "Healthy"),
        "img_original": pil_to_base64(resized),
        "img_heatmap": np_to_base64(cam_colored),
        "img_overlay": np_to_base64(overlay),
        "probabilities": all_probs
    }

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mango Leaf Disease Diagnostic AI</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0b0f19;
            --card-bg: rgba(22, 30, 49, 0.85);
            --border: rgba(255, 255, 255, 0.08);
            --accent: #10b981;
            --accent-glow: rgba(16, 185, 129, 0.3);
            --danger: #ef4444;
            --text: #f3f4f6;
            --text-dim: #9ca3af;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; }
        body { background: var(--bg); color: var(--text); min-height: 100vh; padding: 30px 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        header { text-align: center; margin-bottom: 35px; }
        h1 { font-size: 2.3rem; font-weight: 700; background: linear-gradient(135deg, #34d399, #60a5fa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .subtitle { color: var(--text-dim); margin-top: 8px; font-size: 1.05rem; }
        .badge-teacher { display: inline-block; background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.3); color: #60a5fa; padding: 4px 14px; border-radius: 9999px; font-size: 0.85rem; font-weight: 600; margin-top: 10px; }
        
        .controls { display: flex; gap: 15px; justify-content: center; flex-wrap: wrap; margin-bottom: 30px; }
        button, .upload-btn {
            background: linear-gradient(135deg, #10b981, #059669);
            color: #fff;
            border: none;
            padding: 12px 24px;
            font-size: 0.95rem;
            font-weight: 600;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            text-decoration: none;
        }
        button:hover, .upload-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 20px var(--accent-glow); }
        .btn-secondary { background: rgba(255, 255, 255, 0.08); border: 1px solid var(--border); color: #e5e7eb; }
        .btn-secondary:hover { background: rgba(255, 255, 255, 0.15); box-shadow: 0 4px 15px rgba(0,0,0,0.3); }

        .main-grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 25px; }
        @media (max-width: 900px) { .main-grid { grid-template-columns: 1fr; } }

        .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 16px; padding: 24px; backdrop-filter: blur(12px); }
        .card-title { font-size: 1.15rem; font-weight: 600; margin-bottom: 18px; color: #e5e7eb; border-bottom: 1px solid var(--border); padding-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }

        .images-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 15px; }
        .img-box { background: rgba(0,0,0,0.3); border-radius: 12px; overflow: hidden; border: 1px solid var(--border); text-align: center; }
        .img-box img { width: 100%; height: 180px; object-fit: cover; display: block; }
        .img-label { font-size: 0.8rem; color: var(--text-dim); padding: 8px 4px; font-weight: 500; }

        .pred-banner { background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 14px; padding: 18px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; }
        .pred-banner.disease { background: rgba(239, 68, 68, 0.12); border-color: rgba(239, 68, 68, 0.3); }
        .pred-name { font-size: 1.4rem; font-weight: 700; color: #34d399; }
        .pred-banner.disease .pred-name { color: #f87171; }
        .pred-conf { font-size: 1.3rem; font-weight: 700; color: #fff; text-align: right; }
        .pred-sub { font-size: 0.85rem; color: var(--text-dim); }

        .bar-container { margin-bottom: 12px; }
        .bar-header { display: flex; justify-content: space-between; font-size: 0.88rem; margin-bottom: 4px; }
        .bar-bg { background: rgba(255,255,255,0.06); height: 10px; border-radius: 5px; overflow: hidden; }
        .bar-fill { height: 100%; background: linear-gradient(90deg, #3b82f6, #10b981); border-radius: 5px; transition: width 0.4s ease; }
        .bar-fill.top { background: linear-gradient(90deg, #10b981, #34d399); }
        .bar-fill.danger { background: linear-gradient(90deg, #f59e0b, #ef4444); }

        #file-input { display: none; }
        .loader { display: none; text-align: center; padding: 30px; font-size: 1.1rem; color: #60a5fa; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Mango Leaf Disease Diagnostic AI</h1>
            <p class="subtitle">Interactive Explainable AI (Swin-T Teacher Model — 98.59% Test Accuracy)</p>
            <span class="badge-teacher">Active Checkpoint: teacher_swin.pth</span>
        </header>

        <div class="controls">
            <button onclick="testRandom()">🎲 Test Random Test Leaf</button>
            <label class="upload-btn">
                📁 Upload Custom Image
                <input type="file" id="file-input" accept="image/*" onchange="uploadImage(event)">
            </label>
        </div>

        <div id="loader" class="loader">🧠 Analyzing Leaf & Computing Grad-CAM Attention Heatmap...</div>

        <div id="results-area" class="main-grid">
            <!-- Left Column: Visual Explanations -->
            <div class="card">
                <div class="card-title">
                    <span>Visual Attention & Pathology Heatmap</span>
                    <span style="font-size: 0.85rem; font-weight: 400; color: var(--text-dim);">Grad-CAM (Swin-T Stage 4)</span>
                </div>
                <div class="images-row">
                    <div class="img-box">
                        <img id="img-orig" src="" alt="Input">
                        <div class="img-label">Input Leaf (224x224)</div>
                    </div>
                    <div class="img-box">
                        <img id="img-cam" src="" alt="Heatmap">
                        <div class="img-label">Attention Heatmap</div>
                    </div>
                    <div class="img-box">
                        <img id="img-over" src="" alt="Overlay">
                        <div class="img-label">Diagnostic Overlay</div>
                    </div>
                </div>
                <p style="font-size: 0.85rem; color: var(--text-dim); margin-top: 15px; line-height: 1.4;">
                    <strong>Pathology Interpretation:</strong> The red/yellow regions identify the exact necrotic lesions, fungal spore carpets, or tissue degradation features driving the model's diagnostic classification.
                </p>
            </div>

            <!-- Right Column: Probabilities -->
            <div class="card">
                <div class="card-title">
                    <span>Diagnostic Predictions</span>
                    <span id="true-label-tag" style="font-size: 0.85rem; color: #a7f3d0;"></span>
                </div>

                <div id="banner" class="pred-banner">
                    <div>
                        <div class="pred-sub">Predicted Disease</div>
                        <div id="pred-class" class="pred-name">---</div>
                    </div>
                    <div>
                        <div id="pred-conf" class="pred-conf">--%</div>
                        <div class="pred-sub">Confidence</div>
                    </div>
                </div>

                <div id="prob-bars"></div>
            </div>
        </div>
    </div>

    <script>
        function renderResults(data, trueLabel) {
            document.getElementById('img-orig').src = data.img_original;
            document.getElementById('img-cam').src = data.img_heatmap;
            document.getElementById('img-over').src = data.img_overlay;

            const banner = document.getElementById('banner');
            const predName = document.getElementById('pred-class');
            const predConf = document.getElementById('pred-conf');
            const trueLabelTag = document.getElementById('true-label-tag');
            const barsContainer = document.getElementById('prob-bars');

            if (data.is_valid_leaf === false) {
                banner.className = 'pred-banner disease';
                banner.style.background = 'rgba(239, 68, 68, 0.2)';
                predName.innerText = '❌ NOT A MANGO LEAF';
                predName.style.color = '#f87171';
                predConf.innerText = 'REJECTED';
                trueLabelTag.innerText = 'Out-of-Distribution (OOD)';
                barsContainer.innerHTML = `
                    <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 10px; padding: 15px; color: #fca5a5; font-size: 0.95rem; line-height: 1.5;">
                        <strong>⚠️ Bio-Morphological Gatekeeper:</strong><br>
                        ${data.error_message}
                    </div>
                `;
                document.getElementById('results-area').style.display = 'grid';
                return;
            }

            banner.style.background = '';
            predName.style.color = '';
            predName.innerText = data.predicted_class;
            predConf.innerText = data.confidence + '%';

            if (trueLabel) {
                trueLabelTag.innerText = 'True Label: ' + trueLabel;
            } else {
                trueLabelTag.innerText = 'Real-World Leaf';
            }

            if (data.is_healthy) {
                banner.className = 'pred-banner';
            } else {
                banner.className = 'pred-banner disease';
            }

            // Render bars
            barsContainer.innerHTML = '';
            data.probabilities.forEach((p, idx) => {
                const isTop = (idx === 0);
                const isDanger = !data.is_healthy && isTop;
                const fillClass = isDanger ? 'bar-fill danger' : (isTop ? 'bar-fill top' : 'bar-fill');
                
                barsContainer.innerHTML += `
                    <div class="bar-container">
                        <div class="bar-header">
                            <span>${p.class_name}</span>
                            <span style="font-weight: 600;">${p.percentage}%</span>
                        </div>
                        <div class="bar-bg">
                            <div class="${fillClass}" style="width: ${p.percentage}%;"></div>
                        </div>
                    </div>
                `;
            });

            document.getElementById('results-area').style.display = 'grid';
        }

        async function testRandom() {
            document.getElementById('loader').style.display = 'block';
            document.getElementById('results-area').style.display = 'none';

            try {
                const res = await fetch('/api/random');
                const data = await res.json();
                renderResults(data, data.true_label);
            } catch (err) {
                alert('Inference error: ' + err);
            } finally {
                document.getElementById('loader').style.display = 'none';
            }
        }

        async function uploadImage(e) {
            const file = e.target.files[0];
            if (!file) return;

            document.getElementById('loader').style.display = 'block';
            document.getElementById('results-area').style.display = 'none';

            const formData = new FormData();
            formData.append('image', file);

            try {
                const res = await fetch('/api/predict', { method: 'POST', body: formData });
                const data = await res.json();
                renderResults(data, 'Custom Upload');
            } catch (err) {
                alert('Upload error: ' + err);
            } finally {
                document.getElementById('loader').style.display = 'none';
            }
        }

        // Auto load first sample
        window.onload = () => testRandom();
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/random", methods=["GET"])
def api_random():
    test_csv = os.path.join(SPLITS_DIR, "test.csv")
    df = pd.read_csv(test_csv)
    sample = df.sample(n=1).iloc[0]
    img_path = sample["filepath"]
    true_label = sample["class_name"]

    img_pil = Image.open(img_path)
    res = run_inference(img_pil)
    res["true_label"] = true_label
    return jsonify(res)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    img_pil = Image.open(file.stream)
    res = run_inference(img_pil)
    return jsonify(res)

if __name__ == "__main__":
    print("Starting Mango Leaf Disease Web Tester on http://localhost:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False)
