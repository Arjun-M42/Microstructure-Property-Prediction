from __future__ import annotations

import os
import platform
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import MicrostructureDataset, TargetNormalizer, load_structure_targets
from model import get_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
TRAIN_IMAGE_DIR = PROJECT_ROOT / "processed_data" / "train_images"
TRAIN_LABEL_FILE = PROJECT_ROOT / "data" / "kappa_train.mat"
BEST_CHECKPOINT = MODEL_DIR / "vit_best.pth"
LAST_CHECKPOINT = MODEL_DIR / "vit_last.pth"
TARGET_MODE = "scalar"

SEED = 42
VAL_SPLIT = 0.1
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 5
RESUME_TRAINING = False


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_structure_splits(num_structures: int, val_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    structure_ids = np.arange(1, num_structures + 1, dtype=np.int32)
    rng.shuffle(structure_ids)
    val_count = max(1, int(num_structures * val_ratio))
    val_ids = np.sort(structure_ids[:val_count])
    train_ids = np.sort(structure_ids[val_count:])
    return train_ids, val_ids


def build_loader(dataset: MicrostructureDataset, batch_size: int, shuffle: bool) -> DataLoader:
    cpu_count = os.cpu_count() or 1
    is_windows = platform.system() == "Windows"
    worker_count = min(2 if is_windows else 8, cpu_count)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": worker_count,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": worker_count > 0 and not is_windows,
    }
    if worker_count > 0 and not is_windows:
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(**loader_kwargs)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    normalizer: TargetNormalizer,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    predictions: dict[int, list[torch.Tensor]] = {}
    targets: dict[int, torch.Tensor] = {}
    autocast_context = (
        lambda: torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )

    with torch.no_grad():
        for images, labels, structure_ids in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast_context():
                outputs = model(pixel_values=images).logits.view(-1, 1)
                loss = criterion(outputs, labels)

            total_loss += loss.item()

            denorm_predictions = normalizer.denormalize(outputs).cpu().view(-1)
            denorm_targets = normalizer.denormalize(labels).cpu().view(-1)

            for prediction, target, structure_id in zip(
                denorm_predictions, denorm_targets, structure_ids.tolist()
            ):
                predictions.setdefault(structure_id, []).append(prediction)
                targets[structure_id] = target

    structure_preds = []
    structure_targets = []
    for structure_id, prediction_list in predictions.items():
        structure_preds.append(torch.stack(prediction_list).mean())
        structure_targets.append(targets[structure_id])

    pred_tensor = torch.stack(structure_preds)
    target_tensor = torch.stack(structure_targets)
    structure_mae = torch.mean(torch.abs(pred_tensor - target_tensor)).item()

    return total_loss / len(loader), structure_mae


def maybe_resume(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, float, int]:
    if not RESUME_TRAINING or not LAST_CHECKPOINT.exists():
        return 0, float("inf"), 0

    checkpoint = torch.load(LAST_CHECKPOINT, map_location=device)
    if checkpoint.get("target_mode") != TARGET_MODE:
        print("Ignoring old checkpoint because target mode changed.")
        return 0, float("inf"), 0

    print(f"Resuming from checkpoint: {LAST_CHECKPOINT}")
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler_state is not None and device.type == "cuda":
        scaler.load_state_dict(scaler_state)

    start_epoch = int(checkpoint.get("epoch", 0))
    best_val_mae = float(checkpoint.get("best_val_mae", checkpoint.get("val_structure_mae", float("inf"))))
    epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
    print(f"Resumed at epoch {start_epoch} with best_val_mae={best_val_mae:.4f}")
    return start_epoch, best_val_mae, epochs_without_improvement


def main() -> None:
    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print("Using device:", device)
    print(f"Training target mode: {TARGET_MODE}")

    MODEL_DIR.mkdir(exist_ok=True)

    target_matrix = load_structure_targets(TRAIN_LABEL_FILE, target_mode=TARGET_MODE)
    train_ids, val_ids = build_structure_splits(target_matrix.shape[0], VAL_SPLIT)
    normalizer = TargetNormalizer.from_targets(target_matrix[train_ids - 1])

    train_dataset = MicrostructureDataset(
        TRAIN_IMAGE_DIR,
        TRAIN_LABEL_FILE,
        structure_ids=train_ids,
        normalizer=normalizer,
        target_mode=TARGET_MODE,
    )
    val_dataset = MicrostructureDataset(
        TRAIN_IMAGE_DIR,
        TRAIN_LABEL_FILE,
        structure_ids=val_ids,
        normalizer=normalizer,
        target_mode=TARGET_MODE,
    )

    train_loader = build_loader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = build_loader(val_dataset, BATCH_SIZE, shuffle=False)

    model = get_model(num_labels=1).to(device)
    criterion = nn.SmoothL1Loss(beta=0.25)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    autocast_context = (
        lambda: torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )

    start_epoch, best_val_mae, epochs_without_improvement = maybe_resume(
        model, optimizer, scheduler, scaler, device
    )

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_index, (images, labels, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast_context():
                outputs = model(pixel_values=images).logits.view(-1, 1)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if batch_index % 100 == 0 or batch_index == len(train_loader):
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} "
                    f"Batch {batch_index}/{len(train_loader)} "
                    f"Loss {loss.item():.4f}"
                )

        train_loss = total_loss / len(train_loader)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device, normalizer)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_structure_mae={val_mae:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        improved = val_mae < best_val_mae
        if improved:
            best_val_mae = val_mae
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "normalizer": normalizer.to_state_dict(),
            "epoch": epoch + 1,
            "val_structure_mae": val_mae,
            "best_val_mae": best_val_mae,
            "epochs_without_improvement": epochs_without_improvement,
            "target_mode": TARGET_MODE,
        }
        torch.save(checkpoint, LAST_CHECKPOINT)
        print(f"Saved last checkpoint to {LAST_CHECKPOINT}")

        if improved:
            torch.save(checkpoint, BEST_CHECKPOINT)
            print(f"Saved new best checkpoint to {BEST_CHECKPOINT}")
        else:
            print(
                f"Best checkpoint unchanged. Current val_structure_mae={val_mae:.4f}, "
                f"best_val_mae={best_val_mae:.4f}"
            )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping triggered after {epoch + 1} epochs. "
                f"Best validation MAE: {best_val_mae:.4f}"
            )
            break


if __name__ == "__main__":
    main()
