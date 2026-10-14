import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from src.config import (
    IMAGE_SIZE,
    NORM_MEAN,
    NORM_STD,
    BATCH_SIZE,
    SPLITS_DIR
)

class MangoDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        label = int(row["label"])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback for corrupt files if any
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)

        return image, label, img_path

def get_transforms():
    """
    Standard training and evaluation transforms matching base paper Table 5.
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    return train_transform, eval_transform

def get_dataloaders(batch_size=BATCH_SIZE, num_workers=0, pin_memory=True):
    train_transform, eval_transform = get_transforms()

    train_csv = os.path.join(SPLITS_DIR, "train.csv")
    val_csv = os.path.join(SPLITS_DIR, "val.csv")
    test_csv = os.path.join(SPLITS_DIR, "test.csv")

    train_dataset = MangoDataset(train_csv, transform=train_transform)
    val_dataset = MangoDataset(val_csv, transform=eval_transform)
    test_dataset = MangoDataset(test_csv, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader
