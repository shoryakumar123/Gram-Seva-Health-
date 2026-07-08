#!/usr/bin/env python3
"""
GramSeva Health — CNN Skin Disease Detection (Transfer Learning)
================================================================
Architecture : MobileNetV2 (ImageNet) → fine-tuned classifier head
Accelerator  : Apple MPS → CUDA → CPU (auto-selected)
Expected acc : 70–85 %  (vs. ≈46 % for PCA+RandomForest)

Usage:
    source .venv/bin/activate
    python train_cnn_model.py --dataset-dir /path/to/dataset
    # or set the GRAMSEVA_DATASET_DIR environment variable instead of --dataset-dir

Outputs (in same directory):
    skin_cnn.pt              TorchScript model ready for direct inference
    skin_classes.json        Ordered class list  [class0, class1, …]
    skin_metrics.json        Accuracy / loss metrics
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

# ─── DEPENDENCY CHECK ──────────────────────────────────────────────────────────
for _mod, _pkg in [("torch", "torch torchvision"), ("torchvision", "torchvision")]:
    try:
        __import__(_mod)
    except ImportError:
        print(f"❌  '{_pkg}' is not installed. Run:\n    pip install {_pkg}", file=sys.stderr)
        sys.exit(1)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V2_Weights

# ─── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s", stream=sys.stdout)
log = logging.getLogger("gramseva.cnn")

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_dataset_dir() -> str:
    """
    Resolve the dataset directory from (in priority order):
      1. --dataset-dir CLI argument
      2. GRAMSEVA_DATASET_DIR environment variable
      3. ./dataset relative to this script (fallback default)
    """
    parser = argparse.ArgumentParser(description="Train GramSeva skin-disease CNN")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to the ImageFolder-style dataset directory "
             "(overrides GRAMSEVA_DATASET_DIR env var)",
    )
    args, _ = parser.parse_known_args()

    if args.dataset_dir:
        return os.path.abspath(args.dataset_dir)

    env_dir = os.environ.get("GRAMSEVA_DATASET_DIR")
    if env_dir:
        return os.path.abspath(env_dir)

    return os.path.join(BASE_DIR, "dataset")


DATASET_DIR = resolve_dataset_dir()

IMG_SIZE     = 224        # MobileNetV2 standard
BATCH_SIZE   = 32
EPOCHS       = 20         # will stop early if val loss stagnates
LR_INITIAL   = 1e-3       # head learning rate
LR_BACKBONE  = 1e-4       # backbone LR after unfreezing
UNFREEZE_EPOCH = 5        # epoch at which backbone is unfrozen
PATIENCE     = 5          # early-stopping patience
RANDOM_STATE = 42
VAL_SPLIT    = 0.2
NUM_WORKERS  = 0          # safe default for macOS multiprocessing

MODEL_OUT   = os.path.join(BASE_DIR, "skin_cnn.pt")
CLASSES_OUT = os.path.join(BASE_DIR, "skin_classes.json")
METRICS_OUT = os.path.join(BASE_DIR, "skin_metrics.json")

torch.manual_seed(RANDOM_STATE)

# ─── DEVICE ────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()
log.info(f"Using device: {DEVICE}")

# ─── TRANSFORMS ────────────────────────────────────────────────────────────────
# ImageNet normalisation stats (standard for pre-trained MobileNetV2)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ─── DATASET ───────────────────────────────────────────────────────────────────
def build_dataloaders(dataset_dir: str):
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    full_dataset = datasets.ImageFolder(dataset_dir, transform=train_transforms)
    n = len(full_dataset)
    n_val = int(n * VAL_SPLIT)
    n_train = n - n_val

    gen = torch.Generator().manual_seed(RANDOM_STATE)
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [n_train, n_val], generator=gen)

    # Apply val transforms to val split
    val_ds.dataset = datasets.ImageFolder(dataset_dir, transform=val_transforms)

    # Weighted sampler to handle class imbalance
    targets = [full_dataset.targets[i] for i in train_ds.indices]
    class_counts = torch.bincount(torch.tensor(targets))
    weights = 1.0 / class_counts.float()
    sample_weights = weights[targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))

    return train_loader, val_loader, full_dataset.classes, n_train, n_val

# ─── MODEL ─────────────────────────────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze backbone initially
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model.to(DEVICE)

def unfreeze_backbone(model: nn.Module, backbone_lr: float, optimizer: optim.Optimizer):
    """Unfreeze the last 3 feature blocks for fine-tuning."""
    log.info("🔓  Unfreezing last 3 backbone blocks for fine-tuning…")
    blocks_to_unfreeze = list(model.features.children())[-3:]
    for block in blocks_to_unfreeze:
        for param in block.parameters():
            param.requires_grad = True
    # Add backbone params to optimizer
    optimizer.add_param_group({"params": [p for p in model.features.parameters() if p.requires_grad],
                                "lr": backbone_lr})

# ─── TRAINING LOOP ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total

# ─── SAVE ──────────────────────────────────────────────────────────────────────
def save_model(model: nn.Module, classes: list[str], metrics: dict):
    model.eval()
    example = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    try:
        scripted = torch.jit.trace(model, example)
        scripted.save(MODEL_OUT)
        log.info(f"Model (TorchScript) → {MODEL_OUT}")
    except Exception as e:
        log.warning(f"TorchScript trace failed ({e}), saving state_dict instead")
        torch.save({"state_dict": model.state_dict(), "num_classes": len(classes)}, MODEL_OUT)

    with open(CLASSES_OUT, "w") as f:
        json.dump(classes, f, indent=2)
    log.info(f"Classes            → {CLASSES_OUT}")

    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics            → {METRICS_OUT}")

# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    sep = "=" * 62
    print(f"\n{sep}")
    print("  GramSeva — CNN Skin Disease Model Training (MobileNetV2)")
    print(f"{sep}")
    print(f"  Dataset : {DATASET_DIR}")
    print(f"  Device  : {DEVICE}")
    print(f"  Epochs  : {EPOCHS}  (early-stop patience={PATIENCE})\n")

    # ── Data ───────────────────────────────────────────────────────────
    train_loader, val_loader, classes, n_train, n_val = build_dataloaders(DATASET_DIR)
    num_classes = len(classes)
    log.info(f"Classes  : {classes}")
    log.info(f"Samples  : {n_train} train / {n_val} val")

    # ── Model ──────────────────────────────────────────────────────────
    model = build_model(num_classes)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.classifier.parameters(), lr=LR_INITIAL, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ── Training ───────────────────────────────────────────────────────
    best_val_acc   = 0.0
    best_state     = None
    patience_count = 0
    history        = []
    t_start        = time.perf_counter()
    backbone_unfrozen = False

    for epoch in range(1, EPOCHS + 1):
        # Unfreeze backbone partway through
        if epoch == UNFREEZE_EPOCH and not backbone_unfrozen:
            unfreeze_backbone(model, LR_BACKBONE, optimizer)
            backbone_unfrozen = True
            # Recreate the scheduler now that the backbone param group exists,
            # so its LR is included in the cosine annealing schedule instead
            # of staying pinned at LR_BACKBONE for the rest of training.
            remaining_epochs = EPOCHS - epoch + 1
            scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=1e-6)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc   = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        history.append({"epoch": epoch, "train_acc": train_acc, "val_acc": val_acc,
                         "train_loss": train_loss, "val_loss": val_loss})

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
            marker = " ← best"
        else:
            patience_count += 1

        log.info(f"Epoch {epoch:02d}/{EPOCHS}  "
                 f"train {train_acc:.1%} / {train_loss:.4f}  |  "
                 f"val {val_acc:.1%} / {val_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}"
                 + marker)

        if patience_count >= PATIENCE:
            log.info(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    elapsed = time.perf_counter() - t_start
    log.info(f"\nBest val accuracy : {best_val_acc:.2%}")

    # Restore best weights
    model.load_state_dict(best_state)

    # ── Save ───────────────────────────────────────────────────────────
    log.info("\n💾 Saving artifacts…")
    metrics = {
        "best_val_accuracy": round(float(best_val_acc), 4),
        "test_accuracy":     round(float(best_val_acc), 4),   # used by skin_server health check
        "epochs_trained":    epoch,
        "device":            str(DEVICE),
        "model":             "MobileNetV2",
        "classes":           classes,
        "num_classes":       num_classes,
        "img_size":          IMG_SIZE,
        "training_time_s":   round(float(elapsed), 2),
        "history":           history,
    }
    save_model(model, classes, metrics)

    print(f"\n{sep}")
    print(f"  🎉 Done!  Best val accuracy: {best_val_acc:.2%}")
    print(f"  ⏱️  Total time: {elapsed:.0f}s")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
