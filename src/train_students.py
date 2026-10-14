import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.config import (
    NUM_CLASSES,
    DEVICE,
    CHECKPOINTS_DIR,
    RESULTS_DIR,
    STUDENT_LR,
    STEP_SIZE,
    GAMMA,
    WEIGHT_DECAY,
    KD_ALPHA,
    KD_TEMPERATURE,
    SEED
)
from src.dataset import get_dataloaders
from src.train_teacher import build_swin_teacher

def set_seed(seed=SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_mobilenet_student(num_classes=NUM_CLASSES, pretrained=True):
    """
    MobileNetV3-Small model:
    Replaces final classifier layer with Linear(1024, num_classes).
    """
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model

class DistillationLoss(nn.Module):
    def __init__(self, alpha=KD_ALPHA, temperature=KD_TEMPERATURE):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels):
        # Hard label loss
        loss_ce = self.ce_loss(student_logits, labels)
        # Soft label distillation loss
        p_s = F.log_softmax(student_logits / self.temperature, dim=1)
        p_t = F.softmax(teacher_logits / self.temperature, dim=1)
        loss_kd = self.kl_div(p_s, p_t) * (self.temperature ** 2)

        return self.alpha * loss_ce + (1.0 - self.alpha) * loss_kd, loss_ce, loss_kd

def train_epoch_plain(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels, _ in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        if device.type == "cuda":
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def train_epoch_distilled(student, teacher, dataloader, kd_criterion, optimizer, scaler, device):
    student.train()
    teacher.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels, _ in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.no_grad():
            if device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    teacher_outputs = teacher(images)
            else:
                teacher_outputs = teacher(images)

        if device.type == "cuda":
            with torch.amp.autocast('cuda'):
                student_outputs = student(images)
                loss, _, _ = kd_criterion(student_outputs, teacher_outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            student_outputs = student(images)
            loss, _, _ = kd_criterion(student_outputs, teacher_outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(student_outputs, dim=1).detach().cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate_student(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels, _ in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if device.type == "cuda":
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(dataloader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    metrics = {
        "loss": float(val_loss),
        "accuracy": float(val_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }
    return metrics, all_preds, all_labels

def run_student_training(mode="plain", epochs=35, lr=STUDENT_LR, seed=SEED):
    set_seed(seed)
    train_loader, val_loader, test_loader = get_dataloaders()
    criterion = nn.CrossEntropyLoss()

    student = build_mobilenet_student(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    optimizer = optim.Adam(student.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == "cuda"))

    teacher = None
    kd_criterion = None
    if mode == "distilled":
        teacher_ckpt_path = os.path.join(CHECKPOINTS_DIR, "teacher_swin.pth")
        if not os.path.exists(teacher_ckpt_path):
            raise FileNotFoundError(f"Teacher checkpoint not found at {teacher_ckpt_path}. Train teacher first!")
        teacher = build_swin_teacher(num_classes=NUM_CLASSES, pretrained=False, freeze_early=False).to(DEVICE)
        checkpoint = torch.load(teacher_ckpt_path, map_location=DEVICE)
        teacher.load_state_dict(checkpoint["model_state_dict"])
        teacher.eval()
        kd_criterion = DistillationLoss(alpha=KD_ALPHA, temperature=KD_TEMPERATURE)

    tag = "Student A (Plain)" if mode == "plain" else "Student B (Distilled)"
    ckpt_name = f"student_{mode}.pth" if seed == SEED else f"student_{mode}_seed{seed}.pth"
    best_checkpoint_path = os.path.join(CHECKPOINTS_DIR, ckpt_name)

    print(f"\n--- Starting {tag} Training (Seed: {seed}) ---")
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        if mode == "plain":
            train_loss, train_acc = train_epoch_plain(student, train_loader, criterion, optimizer, scaler, DEVICE)
        else:
            train_loss, train_acc = train_epoch_distilled(student, teacher, train_loader, kd_criterion, optimizer, scaler, DEVICE)

        val_metrics, _, _ = evaluate_student(student, val_loader, criterion, DEVICE)
        scheduler.step()

        val_acc = val_metrics["accuracy"]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        curr_lr = scheduler.get_last_lr()[0]
        print(f"[{tag}] Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | LR: {curr_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_acc*100:.2f}% F1: {val_metrics['f1']:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "val_acc": val_acc,
                "history": history
            }, best_checkpoint_path)
            print(f" -> Saved new best {tag} model (Val Acc: {val_acc*100:.2f}%)", flush=True)

    total_time = time.time() - start_time
    print(f"\n{tag} Training Completed in {total_time/60:.2f} minutes.", flush=True)

    # Evaluate best checkpoint on Test Set
    checkpoint = torch.load(best_checkpoint_path, map_location=DEVICE)
    student.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, preds, labels = evaluate_student(student, test_loader, criterion, DEVICE)

    print("\n==========================================")
    print(f"--- {tag.upper()} TEST SET RESULTS ---")
    print(f"Test Accuracy:  {test_metrics['accuracy']*100:.2f}%")
    print(f"Macro Precision:{test_metrics['precision']*100:.2f}%")
    print(f"Macro Recall:   {test_metrics['recall']*100:.2f}%")
    print(f"Macro F1 Score: {test_metrics['f1']*100:.2f}%")
    print("==========================================")

    results_file = f"student_{mode}_metrics.json" if seed == SEED else f"student_{mode}_seed{seed}_metrics.json"
    results_path = os.path.join(RESULTS_DIR, results_file)
    with open(results_path, "w") as f:
        json.dump({
            "test_metrics": test_metrics,
            "best_val_acc": float(best_val_acc),
            "history": history,
            "training_time_sec": float(total_time)
        }, f, indent=2)

    return student, test_metrics

if __name__ == "__main__":
    print("Training Student A (Plain)...")
    run_student_training(mode="plain", epochs=35)
    print("\nTraining Student B (Distilled)...")
    run_student_training(mode="distilled", epochs=35)
