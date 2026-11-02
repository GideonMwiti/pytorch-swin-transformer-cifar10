"""
Project 40: Swin Transformer Mini Project (Hierarchical Vision Transformer Transfer Learning)
=============================================================================================
Description:
Transfer learning and fine-tuning of the hierarchical Swin Transformer (Liu et al., ICCV 2021)
architecture on CIFAR-10 in PyTorch. Explores linear computational complexity via local
Window-based Self-Attention (W-MSA), cross-window connections via Shifted Window Partitioning
(SW-MSA), hierarchical spatial downsampling with patch merging, and discriminative fine-tuning.
"""

import copy
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

# -----------------------------------------------------------------------------
# Configuration & Seed Initialization
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
TORCH_CACHE_DIR = PROJECT_DIR / "torch_cache"
DATA_DIR = PROJECT_DIR / "data"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["TORCH_HOME"] = str(TORCH_CACHE_DIR)

RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 4
NUM_CLASSES = 10
NUM_TRAIN_SAMPLES = 1200
NUM_VAL_SAMPLES = 400
INPUT_SIZE = 224  # Standard Swin-T window-aligned spatial dimension (7x7 windows)

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)


def set_seed(seed: int = 42) -> None:
    """Ensure strict deterministic reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Dataset Preparation
# -----------------------------------------------------------------------------
def get_dataloaders(
    data_dir: Path,
    num_train: int = NUM_TRAIN_SAMPLES,
    num_val: int = NUM_VAL_SAMPLES,
    batch_size: int = BATCH_SIZE,
    input_size: int = INPUT_SIZE,
) -> Tuple[DataLoader, DataLoader, datasets.CIFAR10]:
    """
    Constructs train and test dataloaders with standard ImageNet normalization and
    bicubic upsampling to 224x224 to match Swin-T's 7x7 local window partition geometry.
    """
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_train = datasets.CIFAR10(root=str(data_dir), train=True, download=False, transform=train_transform)
    full_test = datasets.CIFAR10(root=str(data_dir), train=False, download=False, transform=test_transform)

    train_subset = Subset(full_train, list(range(min(num_train, len(full_train)))))
    test_subset = Subset(full_test, list(range(min(num_val, len(full_test)))))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader, full_test


# -----------------------------------------------------------------------------
# Pretrained Swin Transformer Model & Discriminative Optimizer
# -----------------------------------------------------------------------------
def build_swin_model(
    num_classes: int = NUM_CLASSES,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """
    Loads pretrained Swin-T (Tiny), substitutes the 10-class linear classification head,
    and configures discriminative differential learning rates across transformer stages.
    """
    weights = models.Swin_T_Weights.IMAGENET1K_V1
    model = models.swin_t(weights=weights)

    # Substitute classification head
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    nn.init.kaiming_normal_(model.head.weight, nonlinearity="relu")
    nn.init.constant_(model.head.bias, 0.0)

    model = model.to(device)

    # Differential Learning Rates:
    # Backbone: conservative update (2e-5) to preserve hierarchical spatial priors
    # Classification Head: standard update (1e-3) for rapid adaptation
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if "head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 2e-5, "weight_decay": 1e-2},
            {"params": head_params, "lr": 1e-3, "weight_decay": 1e-2},
        ]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    return model, optimizer, scheduler


# -----------------------------------------------------------------------------
# Training and Evaluation Engine
# -----------------------------------------------------------------------------
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    return running_loss / total, (correct / total) * 100.0


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluates the model on test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

    return running_loss / total, (correct / total) * 100.0


# -----------------------------------------------------------------------------
# Diagnostic Plotting Dashboard
# -----------------------------------------------------------------------------
def plot_swin_dashboard(
    history: Dict[str, List[float]],
    model: nn.Module,
    test_dataset: datasets.CIFAR10,
    device: torch.device,
    save_path: Path,
) -> None:
    """Generates a publication-grade 4-panel evaluation dashboard."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)

    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. Cross-Entropy Loss Trajectory
    ax_loss = axes[0, 0]
    ax_loss.plot(epochs, history["train_loss"], "o-", color="#1f77b4", linewidth=2.2, label="Train Loss")
    ax_loss.plot(epochs, history["val_loss"], "s--", color="#d62728", linewidth=2.2, label="Validation Loss")
    ax_loss.set_title("(a) Cross-Entropy Loss Trajectory", fontsize=13, fontweight="bold", pad=10)
    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Loss", fontsize=11)
    ax_loss.set_xticks(list(epochs))
    ax_loss.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    ax_loss.grid(True, linestyle=":", alpha=0.6)

    # 2. Classification Top-1 Accuracy Trajectory
    ax_acc = axes[0, 1]
    ax_acc.plot(epochs, history["train_acc"], "o-", color="#2ca02c", linewidth=2.2, label="Train Top-1 Acc (%)")
    ax_acc.plot(epochs, history["val_acc"], "s--", color="#ff7f0e", linewidth=2.2, label="Val Top-1 Acc (%)")
    ax_acc.set_title("(b) Classification Top-1 Accuracy (%)", fontsize=13, fontweight="bold", pad=10)
    ax_acc.set_xlabel("Epoch", fontsize=11)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=11)
    ax_acc.set_xticks(list(epochs))
    ax_acc.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    ax_acc.grid(True, linestyle=":", alpha=0.6)

    # 3. Swin Shifted Window Mechanism Concept Diagram
    ax_diag = axes[1, 0]
    ax_diag.axis("off")
    ax_diag.set_title("(c) Swin Transformer: Shifted Window Partitioning (W-MSA vs. SW-MSA)", fontsize=12, fontweight="bold", pad=10)

    sub_spec = axes[1, 0].get_subplotspec()
    inner_grid = fig.add_subfigure(sub_spec)
    sub_axes = inner_grid.subplots(1, 2)

    # Regular Window Partitioning (W-MSA)
    grid1 = np.zeros((8, 8, 3))
    grid1[::2, ::2] = [0.27, 0.45, 0.77]
    grid1[1::2, 1::2] = [0.27, 0.45, 0.77]
    grid1[::2, 1::2] = [0.85, 0.88, 0.95]
    grid1[1::2, ::2] = [0.85, 0.88, 0.95]
    sub_axes[0].imshow(grid1)
    sub_axes[0].set_title("Layer l: Regular Windows (W-MSA)", fontsize=9, fontweight="semibold")
    sub_axes[0].set_xticks([1.5, 3.5, 5.5])
    sub_axes[0].set_yticks([1.5, 3.5, 5.5])
    sub_axes[0].set_xticklabels([])
    sub_axes[0].set_yticklabels([])
    sub_axes[0].grid(color="black", linewidth=2.0)

    # Shifted Window Partitioning (SW-MSA) shifted by (M/2, M/2)
    grid2 = np.zeros((8, 8, 3))
    grid2[:] = [0.95, 0.90, 0.85]
    # Highlight cross-window shifted boundaries
    grid2[1:5, 1:5] = [0.90, 0.35, 0.25]
    grid2[5:8, 1:5] = [0.95, 0.65, 0.35]
    grid2[1:5, 5:8] = [0.95, 0.65, 0.35]
    grid2[5:8, 5:8] = [0.75, 0.25, 0.20]
    sub_axes[1].imshow(grid2)
    sub_axes[1].set_title("Layer l+1: Shifted Windows (SW-MSA)", fontsize=9, fontweight="semibold")
    sub_axes[1].set_xticks([0.5, 4.5])
    sub_axes[1].set_yticks([0.5, 4.5])
    sub_axes[1].set_xticklabels([])
    sub_axes[1].set_yticklabels([])
    sub_axes[1].grid(color="black", linewidth=2.0)

    # 4. CIFAR-10 Sample Predictions Grid (8 Samples)
    ax_grid = axes[1, 1]
    ax_grid.axis("off")
    ax_grid.set_title("(d) Fine-Tuned Swin-T Test Predictions (Sampled)", fontsize=13, fontweight="bold", pad=10)

    inner_spec = axes[1, 1].get_subplotspec()
    pred_grid = fig.add_subfigure(inner_spec)
    pred_axes = pred_grid.subplots(2, 4)

    eval_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    sample_indices = [3, 10, 15, 23, 31, 44, 52, 60]
    model.eval()

    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            raw_img, label = test_dataset.data[idx], test_dataset.targets[idx]
            tensor_img = eval_transform(transforms.ToPILImage()(raw_img)).unsqueeze(0).to(device)
            logits = model(tensor_img)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            pred_conf, pred_idx = probs.max(0)

            pred_label = CLASSES[pred_idx.item()]
            true_label = CLASSES[label]
            is_correct = (pred_label == true_label)

            sub_ax = pred_axes[i // 4, i % 4]
            sub_ax.imshow(raw_img)
            sub_ax.axis("off")

            color_tag = "#2ca02c" if is_correct else "#d62728"
            title_text = f"P: {pred_label} ({pred_conf*100:.0f}%)\nT: {true_label}"
            sub_ax.set_title(title_text, fontsize=8.5, color=color_tag, fontweight="bold", pad=3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Artifact] Saved Swin Transformer dashboard to: {save_path}")


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("Project 40: Swin Transformer Mini Project — Transfer Learning on CIFAR-10")
    print("=" * 75)
    print(f"Compute Device        : {device}")
    print(f"Pretrained Weights    : Swin_T_Weights.IMAGENET1K_V1 (Hierarchical ViT)")
    print(f"Input Resolution      : {INPUT_SIZE}x{INPUT_SIZE} pixels (Bicubic Upsampling)")
    print(f"Window Size           : 7x7 (Local W-MSA & Shifted SW-MSA)")
    print(f"Dataset               : CIFAR-10 ({NUM_CLASSES} classes)")
    print(f"Training Samples      : {NUM_TRAIN_SAMPLES} | Validation Samples: {NUM_VAL_SAMPLES}")
    print(f"Epochs                : {NUM_EPOCHS} | Batch Size: {BATCH_SIZE}")
    print(f"Optimization          : AdamW with Differential LRs + Cosine Annealing")
    print("-" * 75)

    # 1. Dataloaders
    train_loader, test_loader, full_test = get_dataloaders(DATA_DIR)

    # 2. Build Model
    model, optimizer, scheduler = build_swin_model(num_classes=NUM_CLASSES, device=device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters      : {total_params:,}")
    print("-" * 75)

    # 3. Initial Baseline Evaluation
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    init_val_loss, init_val_acc = evaluate(model, test_loader, criterion, device)
    print(f"Pre-training Baseline -> Loss: {init_val_loss:.4f} | Top-1 Acc: {init_val_acc:.2f}%")
    print("-" * 75)

    # 4. Fine-Tuning Loop
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start_time = time.perf_counter()

    for epoch in range(1, NUM_EPOCHS + 1):
        ep_start = time.perf_counter()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        ep_duration = time.perf_counter() - ep_start
        print(
            f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:6.2f}% | "
            f"Time: {ep_duration:.2f}s"
        )

    total_time = time.perf_counter() - start_time
    print("-" * 75)
    print(f"Fine-Tuning Completed in {total_time:.2f} seconds.")
    print(f"Final Validation Top-1 Accuracy: {history['val_acc'][-1]:.2f}% (Gain: +{history['val_acc'][-1] - init_val_acc:.2f}%)")

    # 5. Save Checkpoint
    checkpoint_path = CHECKPOINT_DIR / "swin_cifar10_finetuned.pt"
    torch.save(
        {
            "epoch": NUM_EPOCHS,
            "model_state_dict": model.state_dict(),
            "val_acc": history["val_acc"][-1],
            "val_loss": history["val_loss"][-1],
        },
        checkpoint_path,
    )
    print(f"[Checkpoint] Model saved to: {checkpoint_path}")

    # 6. Generate Publication Dashboard
    dashboard_path = PROJECT_DIR / "swin_transformer_cifar10_results.png"
    plot_swin_dashboard(history, model, full_test, device, dashboard_path)

    print("=" * 75)
    print("Swin Transformer Transfer Learning Completed Successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()

