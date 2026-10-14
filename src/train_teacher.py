import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import timm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from src.config import (
    NUM_CLASSES,
    DEVICE,
    CHECKPOINTS_DIR,
    CACHE_DIR,
    RESULTS_DIR,
    TEACHER_LR,
    STEP_SIZE,
    GAMMA,
    WEIGHT_DECAY,
    SEED
)
from src.dataset import get_dataloaders

def set_seed(seed=SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_swin_teacher(num_classes=NUM_CLASSES, pretrained=True, freeze_early=True):
    """
    Swin Transformer Tiny model matching base paper:
    'swin_tiny_patch4_window7_224' pretrained on ImageNet.
    Optionally freeze first stages, fine-tuning final stages and classification head.
    """
    model = timm.create_model(
        'swin_tiny_patch4_window7_224',
        pretrained=pretrained,
        num_classes=num_classes
    )
    if freeze_early:
        # Freeze patch embedding and first 2 layers, fine-tune layer 3 and 4 + head
        for name, param in model.named_parameters():
            if "patch_embed" in name or "layers.0" in name or "layers.1" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    return model

def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels, _ in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        if device.type == "cuda":
            with torch.cuda.amp.autocast():
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

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_logits = []

    for images, labels, _ in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        all_logits.append(outputs.detach().cpu())
        preds = torch.argmax(outputs, dim=1).detach().cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(dataloader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    all_logits = torch.cat(all_logits, dim=0)
    metrics = {
        "loss": float(val_loss),
        "accuracy": float(val_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }
    return metrics, all_logits, all_preds, all_labels

def cache_teacher_logits(model, train_loader, val_loader, test_loader, device, output_path):
    print("Extracting and caching teacher logits across all splits...")
    model.eval()
    
    def extract(loader):
        logits_list = []
        paths_list = []
        labels_list = []
        with torch.no_grad():
            for images, labels, paths in loader:
                images = images.to(device, non_blocking=True)
                if device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        out = model(images)
                else:
                    out = model(images)
                logits_list.append(out.detach().cpu())
                paths_list.extend(paths)
                labels_list.extend(labels.numpy())
        return {
            "logits": torch.cat(logits_list, dim=0),
            "labels": torch.tensor(labels_list),
            "paths": paths_list
        }

    cached_data = {
        "train": extract(train_loader),
        "val": extract(val_loader),
        "test": extract(test_loader)
    }
    torch.save(cached_data, output_path)
    print(f"Cached teacher logits saved to {output_path}")

def run_teacher_training(epochs=35, lr=TEACHER_LR):
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    train_loader, val_loader, test_loader = get_dataloaders()
    model = build_swin_teacher(num_classes=NUM_CLASSES, pretrained=True, freeze_early=True)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_val_acc = 0.0
    best_checkpoint_path = os.path.join(CHECKPOINTS_DIR, "teacher_swin.pth")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print("\n--- Starting Swin-T Teacher Training ---")
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE)
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        val_acc = val_metrics["accuracy"]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        curr_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | LR: {curr_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_acc*100:.2f}% F1: {val_metrics['f1']:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "history": history
            }, best_checkpoint_path)
            print(f" -> Saved new best teacher model (Val Acc: {val_acc*100:.2f}%)")

    total_time = time.time() - start_time
    print(f"\nTeacher Training Completed in {total_time/60:.2f} minutes.")

    # Load best checkpoint and evaluate on Test Set
    checkpoint = torch.load(best_checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, _, _, _ = evaluate(model, test_loader, criterion, DEVICE)

    print("\n==========================================")
    print("--- TEACHER TEST SET RESULTS (Swin-T) ---")
    print(f"Test Accuracy:  {test_metrics['accuracy']*100:.2f}%")
    print(f"Macro Precision:{test_metrics['precision']*100:.2f}%")
    print(f"Macro Recall:   {test_metrics['recall']*100:.2f}%")
    print(f"Macro F1 Score: {test_metrics['f1']*100:.2f}%")
    print("==========================================")

    # Save metrics and history
    results_path = os.path.join(RESULTS_DIR, "teacher_swin_metrics.json")
    with open(results_path, "w") as f:
        json.dump({
            "test_metrics": test_metrics,
            "best_val_acc": float(best_val_acc),
            "history": history,
            "training_time_sec": float(total_time)
        }, f, indent=2)

    # Cache logits
    cache_path = os.path.join(CACHE_DIR, "teacher_logits.pt")
    cache_teacher_logits(model, train_loader, val_loader, test_loader, DEVICE, cache_path)

    return model, test_metrics

if __name__ == "__main__":
    run_teacher_training(epochs=35)
