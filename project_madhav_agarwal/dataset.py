import os
import random
from typing import List, Tuple
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from config import (RESIZE_X, RESIZE_Y, MEAN, STD, FULL_DATASET_PATH,
                    SAMPLES_PER_CLASS, BATCH_SIZE, NUM_WORKERS, RANDOM_SEED)

# ── Transforms ────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((RESIZE_Y, RESIZE_X)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1,
                           saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((RESIZE_Y, RESIZE_X)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


# ── Dataset ───────────────────────────────────────────────────────────────────
class IDCDataset(Dataset):
    """
    Loads IDC patches from the nested Kaggle folder layout.
    Accepts an optional pre-selected list of (path, label) pairs so
    train / val / test subsets each get the correct transform.
    """

    VALID_EXTS = {".png", ".jpg", ".jpeg"}

    def __init__(self, root_dir: str, transform=None,
                 items: List[Tuple[str, int]] = None):
        self.transform = transform

        if items is not None:
            self.image_paths = [x[0] for x in items]
            self.labels      = [x[1] for x in items]
        else:
            self.image_paths: List[str] = []
            self.labels:      List[int] = []
            self._walk(root_dir)

    def _walk(self, root_dir: str):
        for patient_id in sorted(os.listdir(root_dir)):
            patient_dir = os.path.join(root_dir, patient_id)
            if not os.path.isdir(patient_dir):
                continue
            for class_label in [0, 1]:
                class_dir = os.path.join(patient_dir, str(class_label))
                if not os.path.isdir(class_dir):
                    continue
                for fname in sorted(os.listdir(class_dir)):
                    if os.path.splitext(fname)[1].lower() in self.VALID_EXTS:
                        self.image_paths.append(os.path.join(class_dir, fname))
                        self.labels.append(class_label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

    @property
    def class_counts(self) -> dict:
        return {0: self.labels.count(0), 1: self.labels.count(1)}


# ── DataLoader factory ────────────────────────────────────────────────────────
def the_dataloader(root_dir=FULL_DATASET_PATH,
                   samples_per_class=SAMPLES_PER_CLASS,
                   batch_size=BATCH_SIZE,
                   num_workers=NUM_WORKERS,
                   seed=RANDOM_SEED):
    """
    Returns train_loader, val_loader, test_loader, test_ds.
    Stratified 70/15/15 split on a balanced 12,000-image subset.
    """
    print("Scanning dataset...")
    full = IDCDataset(root_dir=root_dir, transform=None)
    print(f"Total images found : {len(full)}")
    print(f"Class counts       : {full.class_counts}")

    # Balanced subsample
    random.seed(seed)
    neg = random.sample([i for i, l in enumerate(full.labels) if l == 0],
                        min(samples_per_class, full.labels.count(0)))
    pos = random.sample([i for i, l in enumerate(full.labels) if l == 1],
                        min(samples_per_class, full.labels.count(1)))
    sampled  = neg + pos
    s_labels = [full.labels[i] for i in sampled]
    print(f"Subsampled         : {len(sampled)} ({len(neg)} neg | {len(pos)} pos)")

    # Stratified 70 / 30 then 50 / 50
    tr_idx, vt_idx, _, vt_lab = train_test_split(
        sampled, s_labels, test_size=0.30,
        stratify=s_labels, random_state=seed)
    v_idx, t_idx = train_test_split(
        vt_idx, test_size=0.50,
        stratify=vt_lab, random_state=seed)

    print(f"Split              : train={len(tr_idx)} | val={len(v_idx)} | test={len(t_idx)}")

    def make_items(indices):
        return [(full.image_paths[i], full.labels[i]) for i in indices]

    train_ds = IDCDataset(root_dir, transform=train_transform, items=make_items(tr_idx))
    val_ds   = IDCDataset(root_dir, transform=eval_transform,  items=make_items(v_idx))
    test_ds  = IDCDataset(root_dir, transform=eval_transform,  items=make_items(t_idx))

    pin = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)

    imgs, lbls = next(iter(train_loader))
    print(f"\nBatch shape  : {tuple(imgs.shape)}")
    print(f"Label sample : {lbls[:8].tolist()}")
    print(f"Pixel range  : [{imgs.min():.2f}, {imgs.max():.2f}]")

    return train_loader, val_loader, test_loader, test_ds
