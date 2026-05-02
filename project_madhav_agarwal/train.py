import os
import time
import torch
import torch.nn as nn

from config import (DEVICE, LEARNING_RATE, WEIGHT_DECAY, LR_SCHEDULER_FACTOR,
                    LR_SCHEDULER_PATIENCE, EARLY_STOPPING_PATIENCE,
                    CHECKPOINT_DIR, WEIGHTS_PATH, NUM_EPOCHS)


class _EarlyStopping:
    def __init__(self, patience=EARLY_STOPPING_PATIENCE, min_delta=1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = float("inf")
        self.counter   = 0

    def step(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best    = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def _metrics(logits, labels):
    """Accuracy + recall for positive class from raw logits."""
    preds = (torch.sigmoid(logits) > 0.5).int()
    lbls  = labels.int()
    acc   = (preds == lbls).float().mean().item()
    tp    = ((preds == 1) & (lbls == 1)).sum().item()
    fn    = ((preds == 0) & (lbls == 1)).sum().item()
    recall = tp / (tp + fn + 1e-8)
    return acc, recall


def train_model(model, num_epochs, train_loader, val_loader,
                loss_fn=None, optimizer=None):
    """
    Trains IDCClassifier. Saves best weights to WEIGHTS_PATH.
    Returns history dict with per-epoch lists.
    """
    model = model.to(DEVICE)

    if loss_fn is None:
        loss_fn = nn.BCEWithLogitsLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE)

    stopper = _EarlyStopping()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    history  = {"train_loss": [], "val_loss": [],
                "train_acc":  [], "val_acc":  [], "val_recall": []}
    best_val = float("inf")

    print(f"\n{'='*68}")
    print(f"  Device : {DEVICE}  |  Epochs : {num_epochs}  "
          f"|  Batches/ep : {len(train_loader)}")
    print(f"{'='*68}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────────────────
        model.train()
        tr_loss, tr_logits, tr_labels = 0., [], []

        for imgs, lbls in train_loader:
            imgs = imgs.to(DEVICE)
            lbls = lbls.float().unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()
            logits = model(imgs)
            loss   = loss_fn(logits, lbls)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            tr_logits.append(logits.detach())
            tr_labels.append(lbls.detach())

        tr_loss /= len(train_loader)
        tr_acc, _ = _metrics(torch.cat(tr_logits), torch.cat(tr_labels))

        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        vl_loss, vl_logits, vl_labels = 0., [], []

        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(DEVICE)
                lbls = lbls.float().unsqueeze(1).to(DEVICE)
                logits = model(imgs)
                vl_loss += loss_fn(logits, lbls).item()
                vl_logits.append(logits)
                vl_labels.append(lbls)

        vl_loss /= len(val_loader)
        vl_acc, vl_recall = _metrics(torch.cat(vl_logits), torch.cat(vl_labels))

        scheduler.step(vl_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        # ── Save best ──────────────────────────────────────────────────
        tag = ""
        if vl_loss < best_val:
            best_val = vl_loss
            torch.save(model.state_dict(), WEIGHTS_PATH)
            tag = "  ✓ saved"

        for k, v in zip(history, [tr_loss, vl_loss, tr_acc, vl_acc, vl_recall]):
            history[k].append(v)

        print(f"Ep [{epoch:>3}/{num_epochs}] "
              f"tr_loss:{tr_loss:.4f} acc:{tr_acc:.3f} | "
              f"val_loss:{vl_loss:.4f} acc:{vl_acc:.3f} recall:{vl_recall:.3f} | "
              f"lr:{lr_now:.1e} | {time.time()-t0:.1f}s{tag}")

        if stopper.step(vl_loss):
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print(f"\nBest val loss : {best_val:.4f}")
    print(f"Weights saved : {WEIGHTS_PATH}")
    return history


if __name__ == "__main__":
    from dataset import the_dataloader
    from model import IDCClassifier

    train_loader, val_loader, test_loader, test_ds = the_dataloader()
    model = IDCClassifier().to(DEVICE)
    print(f"Parameters : {model.count_parameters():,}")
    history = train_model(model, NUM_EPOCHS, train_loader, val_loader)
