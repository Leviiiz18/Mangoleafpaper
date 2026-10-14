import os
import time
import json
import torch
import pandas as pd
import numpy as np
try:
    from thop import profile, clever_format
except ImportError:
    profile = None

from src.config import (
    NUM_CLASSES,
    CHECKPOINTS_DIR,
    RESULTS_DIR,
    TABLES_DIR,
    IMAGE_SIZE
)
from src.train_teacher import build_swin_teacher
from src.train_students import build_mobilenet_student

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6, trainable / 1e6

def compute_flops(model, input_size=(1, 3, IMAGE_SIZE, IMAGE_SIZE), device="cpu"):
    if profile is None:
        return -1.0
    model.eval()
    model.to(device)
    dummy_input = torch.randn(*input_size, device=device)
    try:
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        return flops / 1e9 # GFLOPs
    except Exception as e:
        print(f"FLOP profiling warning: {e}")
        return -1.0

def measure_latency(model, device, num_runs=200, warmup=30):
    model.eval()
    model.to(device)
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timing
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0) # milliseconds

    mean_latency = float(np.mean(timings))
    std_latency = float(np.std(timings))
    throughput = 1000.0 / mean_latency
    return mean_latency, std_latency, throughput

def benchmark_all_models():
    models_info = [
        ("Teacher (Swin-T)", "teacher_swin.pth", lambda: build_swin_teacher(NUM_CLASSES, pretrained=False, freeze_early=False)),
        ("Student A (Plain MobileNetV3)", "student_plain.pth", lambda: build_mobilenet_student(NUM_CLASSES, pretrained=False)),
        ("Student B (Distilled MobileNetV3)", "student_distilled.pth", lambda: build_mobilenet_student(NUM_CLASSES, pretrained=False))
    ]

    records = []
    has_gpu = torch.cuda.is_available()

    for name, ckpt_name, builder in models_info:
        model = builder()
        ckpt_path = os.path.join(CHECKPOINTS_DIR, ckpt_name)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
            ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
        else:
            ckpt_size_mb = 0.0

        total_params, trainable_params = count_parameters(model)
        flops = compute_flops(model, device="cpu")

        # CPU Latency
        cpu_mean, cpu_std, cpu_fps = measure_latency(model, torch.device("cpu"))

        # GPU Latency
        if has_gpu:
            gpu_mean, gpu_std, gpu_fps = measure_latency(model, torch.device("cuda"))
        else:
            gpu_mean, gpu_std, gpu_fps = -1.0, -1.0, -1.0

        records.append({
            "Model": name,
            "Total Params (M)": round(total_params, 2),
            "FLOPs (G)": round(flops, 3) if flops > 0 else "N/A",
            "Checkpoint Size (MB)": round(ckpt_size_mb, 2),
            "GPU Latency (ms)": round(gpu_mean, 2) if has_gpu else "N/A",
            "GPU FPS": round(gpu_fps, 1) if has_gpu else "N/A",
            "CPU Latency (ms)": round(cpu_mean, 2),
            "CPU FPS": round(cpu_fps, 1)
        })

    df = pd.DataFrame(records)
    print("\n--- Efficiency & Computational Benchmarks ---")
    print(df.to_string(index=False))

    # Save to CSV and JSON
    csv_path = os.path.join(TABLES_DIR, "table1_efficiency_comparison.csv")
    df.to_csv(csv_path, index=False)
    json_path = os.path.join(RESULTS_DIR, "efficiency_benchmarks.json")
    df.to_json(json_path, orient="records", indent=2)
    print(f"\nSaved efficiency benchmarks to {csv_path} and {json_path}")
    return df

if __name__ == "__main__":
    benchmark_all_models()
