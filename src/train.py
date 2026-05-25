from __future__ import annotations

import platform
import json
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import (
    MicrostructureDataset,
    TargetNormalizer,
    kappa_scalar_from_tensor9,
    load_structure_targets,
    structure_collate_fn,
)
from model import get_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
TRAIN_IMAGE_DIR = PROJECT_ROOT / "processed_data" / "train_images"
TRAIN_LABEL_FILE = PROJECT_ROOT / "data" / "kappa_train.mat"
BEST_CHECKPOINT = MODEL_DIR / "vit_best.pth"
LAST_CHECKPOINT = MODEL_DIR / "vit_last.pth"
METRICS_PATH = MODEL_DIR / "training_metrics.json"
TARGET_MODE = "tensor9"

SEED = 42
VAL_SPLIT = 0.1
BATCH_SIZE = 1
GRAD_ACCUMULATION_STEPS = 4
EPOCHS = 20
BACKBONE_LEARNING_RATE = 1e-5
HEAD_LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 6
FREEZE_BACKBONE_EPOCHS = 3
HIGH_VALUE_LOSS_WEIGHT = 8.0
SAMPLER_HIGH_VALUE_WEIGHT = 8.0
RESUME_TRAINING = True


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


def build_sampler(structure_ids: np.ndarray, targets: np.ndarray) -> WeightedRandomSampler:
    target_rows = targets[structure_ids - 1]
    scalar_targets = kappa_scalar_from_tensor9(target_rows)
    rare_score = np.log1p(np.max(target_rows, axis=1)) + np.log1p(scalar_targets)
    rare_score = rare_score - rare_score.min()
    weights = 1.0 + SAMPLER_HIGH_VALUE_WEIGHT * (rare_score / (rare_score.max() + 1e-6))
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def build_loader(
    dataset: MicrostructureDataset,
    batch_size: int,
    shuffle: bool,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    cpu_count = os.cpu_count() or 1
    is_windows = platform.system() == "Windows"
    worker_count = 0 if is_windows else min(8, cpu_count)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": worker_count,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": worker_count > 0 and not is_windows,
        "collate_fn": structure_collate_fn
    }
    if worker_count > 0 and not is_windows:
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(**loader_kwargs)


class WeightedLogMSELoss(nn.Module):
    def __init__(
        self,
        normalizer: TargetNormalizer,
        max_log_target: float,
        high_value_weight: float = HIGH_VALUE_LOSS_WEIGHT,
    ) -> None:
        super().__init__()
        self.normalizer = normalizer
        self.max_log_target = max(max_log_target, 1e-6)
        self.high_value_weight = high_value_weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        per_element_loss = (predictions - targets).pow(2)
        with torch.no_grad():
            target_values = self.normalizer.denormalize(targets).clamp_min(0.0)
            log_strength = torch.log1p(target_values) / self.max_log_target
            weights = 1.0 + self.high_value_weight * log_strength.pow(1.5)
        return (per_element_loss * weights).sum() / weights.sum().clamp_min(1e-6)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    normalizer: TargetNormalizer,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    autocast_context = (
        lambda: torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )

    with torch.no_grad():
        for images, labels, _, masks in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast_context():
                outputs = model(pixel_values=images, masks=masks)
                loss = criterion(outputs, labels)

            total_loss += loss.item()

            denorm_predictions = normalizer.denormalize(outputs).cpu()
            denorm_targets = normalizer.denormalize(labels).cpu()
            
            all_preds.append(denorm_predictions)
            all_targets.append(denorm_targets)

    pred_tensor = torch.cat(all_preds, dim=0).float()
    target_tensor = torch.cat(all_targets, dim=0).float()
    pred_log = torch.log1p(pred_tensor.clamp_min(0.0))
    target_log = torch.log1p(target_tensor.clamp_min(0.0))
    log_errors = pred_log - target_log
    log_mae = log_errors.abs().mean().item()
    log_rmse = torch.sqrt((log_errors**2).mean()).item()
    errors = pred_tensor - target_tensor
    abs_errors = errors.abs()
    mae = abs_errors.mean().item()
    rmse = torch.sqrt((errors**2).mean()).item()
    mape = (abs_errors / target_tensor.abs().clamp_min(1e-6)).mean().item() * 100.0
    ss_res = (errors**2).sum()
    ss_tot = ((target_tensor - target_tensor.mean()) ** 2).sum().clamp_min(1e-12)
    r2 = (1.0 - ss_res / ss_tot).item()

    pred_scalar = kappa_scalar_from_tensor9(pred_tensor)
    target_scalar = kappa_scalar_from_tensor9(target_tensor)
    scalar_abs_errors = (pred_scalar - target_scalar).abs()
    scalar_mae = scalar_abs_errors.mean().item()
    scalar_rmse = torch.sqrt(((pred_scalar - target_scalar) ** 2).mean()).item()
    scalar_mape = (
        scalar_abs_errors / target_scalar.abs().clamp_min(1e-6)
    ).mean().item() * 100.0

    zero_normalized = torch.zeros_like(target_tensor)
    baseline_prediction = normalizer.denormalize(zero_normalized)
    baseline_abs_errors = (baseline_prediction - target_tensor).abs()
    baseline_mae = baseline_abs_errors.mean().item()
    baseline_scalar = kappa_scalar_from_tensor9(baseline_prediction)
    baseline_scalar_mae = (baseline_scalar - target_scalar).abs().mean().item()

    return {
        "loss": total_loss / len(loader),
        "log_mae": log_mae,
        "log_rmse": log_rmse,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "accuracy": max(0.0, 100.0 - mape),
        "r2": r2,
        "scalar_mae": scalar_mae,
        "scalar_rmse": scalar_rmse,
        "scalar_mape": scalar_mape,
        "scalar_accuracy": max(0.0, 100.0 - scalar_mape),
        "baseline_mae": baseline_mae,
        "baseline_scalar_mae": baseline_scalar_mae,
        "pred_mean": pred_tensor.mean().item(),
        "pred_min": pred_tensor.min().item(),
        "pred_max": pred_tensor.max().item(),
        "target_mean": target_tensor.mean().item(),
        "target_min": target_tensor.min().item(),
        "target_max": target_tensor.max().item(),
    }


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    if hasattr(model, "set_backbone_trainable"):
        model.set_backbone_trainable(trainable)


def maybe_resume(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, float, int]:
    if not RESUME_TRAINING:
        print("Checkpoint resume is disabled; starting from epoch 0.")
        return 0, float("inf"), 0
    if not LAST_CHECKPOINT.exists():
        print(f"No checkpoint found at {LAST_CHECKPOINT}; starting from epoch 0.")
        return 0, float("inf"), 0

    checkpoint = torch.load(LAST_CHECKPOINT, map_location=device)
    if checkpoint.get("target_mode") != TARGET_MODE:
        print("Ignoring old checkpoint because target mode changed.")
        return 0, float("inf"), 0

    print(f"Resuming from checkpoint: {LAST_CHECKPOINT}")
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        print(f"Ignoring old checkpoint because the model architecture changed: {exc}")
        return 0, float("inf"), 0

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
        except ValueError as exc:
            print(f"Could not load optimizer state; continuing with a fresh optimizer: {exc}")

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler_state is not None:
        try:
            scheduler.load_state_dict(scheduler_state)
        except Exception as exc:
            print(f"Could not load scheduler state; continuing with a fresh scheduler: {exc}")

    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler_state is not None and device.type == "cuda":
        try:
            scaler.load_state_dict(scaler_state)
        except Exception as exc:
            print(f"Could not load AMP scaler state; continuing with a fresh scaler: {exc}")

    start_epoch = int(checkpoint.get("epoch", 0))
    best_val_score = float(
        checkpoint.get(
            "best_val_score",
            checkpoint.get("val_metrics", {}).get("loss", checkpoint.get("best_val_mae", float("inf"))),
        )
    )
    epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
    print(f"Resumed at epoch {start_epoch} with best_val_score={best_val_score:.4f}")
    return start_epoch, best_val_score, epochs_without_improvement


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
    max_log_target = float(np.log1p(target_matrix[train_ids - 1]).max())

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

    train_sampler = build_sampler(train_ids, target_matrix)
    train_loader = build_loader(train_dataset, BATCH_SIZE, shuffle=True, sampler=train_sampler)
    val_loader = build_loader(val_dataset, BATCH_SIZE, shuffle=False)

    num_labels = target_matrix.shape[1]
    model = get_model(num_labels=num_labels).to(device)
    criterion = WeightedLogMSELoss(normalizer, max_log_target=max_log_target)
    optimizer = optim.AdamW(
        [
            {
                "params": model.vit.parameters(),
                "lr": BACKBONE_LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            },
            {
                "params": model.regressor.parameters(),
                "lr": HEAD_LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            },
        ]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.7,
        patience=4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    autocast_context = (
        lambda: torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )

    start_epoch, best_val_score, epochs_without_improvement = maybe_resume(
        model, optimizer, scheduler, scaler, device
    )

    if start_epoch >= EPOCHS:
        print(
            f"Checkpoint is already at epoch {start_epoch}, which is >= EPOCHS={EPOCHS}. "
            "Increase EPOCHS or delete/rename vit_last.pth to train more."
        )
        return

    for epoch in range(start_epoch, EPOCHS):
        backbone_trainable = epoch >= FREEZE_BACKBONE_EPOCHS
        set_backbone_trainable(model, backbone_trainable)
        model.train()
        if not backbone_trainable and hasattr(model, "vit"):
            model.vit.eval()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for batch_index, (images, labels, _, masks) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast_context():
                outputs = model(pixel_values=images, masks=masks)
                loss = criterion(outputs, labels)
                # Scale loss for gradient accumulation
                loss = loss / GRAD_ACCUMULATION_STEPS

            scaler.scale(loss).backward()
            
            if batch_index % GRAD_ACCUMULATION_STEPS == 0 or batch_index == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * GRAD_ACCUMULATION_STEPS

            if batch_index % 100 == 0 or batch_index == len(train_loader):
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} "
                    f"Batch {batch_index}/{len(train_loader)} "
                    f"Loss {loss.item():.4f}"
                )

        train_loss = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, criterion, device, normalizer)
        val_loss = val_metrics["loss"]
        val_mae = val_metrics["mae"]
        val_score = val_loss
        scheduler.step(val_score)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_log_rmse={val_metrics['log_rmse']:.4f} "
            f"val_mae={val_metrics['mae']:.4f} "
            f"baseline_mae={val_metrics['baseline_mae']:.4f} "
            f"val_rmse={val_metrics['rmse']:.4f} "
            f"val_accuracy={val_metrics['accuracy']:.2f}% "
            f"val_scalar_mae={val_metrics['scalar_mae']:.4f} "
            f"pred_range=[{val_metrics['pred_min']:.2f}, {val_metrics['pred_max']:.2f}] "
            f"backbone={'train' if backbone_trainable else 'frozen'} "
            f"vit_lr={optimizer.param_groups[0]['lr']:.6f} "
            f"head_lr={optimizer.param_groups[1]['lr']:.6f}"
        )

        improved = val_score < best_val_score
        if improved:
            best_val_score = val_score
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
            "val_score": val_score,
            "best_val_score": best_val_score,
            "val_metrics": val_metrics,
            "epochs_without_improvement": epochs_without_improvement,
            "target_mode": TARGET_MODE,
            "num_labels": num_labels,
            "max_log_target": max_log_target,
            "train_config": {
                "backbone_learning_rate": BACKBONE_LEARNING_RATE,
                "head_learning_rate": HEAD_LEARNING_RATE,
                "high_value_loss_weight": HIGH_VALUE_LOSS_WEIGHT,
                "sampler_high_value_weight": SAMPLER_HIGH_VALUE_WEIGHT,
                "freeze_backbone_epochs": FREEZE_BACKBONE_EPOCHS,
                "gradient_accumulation_steps": GRAD_ACCUMULATION_STEPS,
            },
        }
        torch.save(checkpoint, LAST_CHECKPOINT)
        METRICS_PATH.write_text(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "target_mode": TARGET_MODE,
                    "train_loss": train_loss,
                    "validation": val_metrics,
                    "best_val_score": best_val_score,
                    "selection_metric": "weighted_log_mse_loss",
                },
                indent=2,
            )
        )
        print(f"Saved last checkpoint to {LAST_CHECKPOINT}")
        print(f"Saved latest metrics to {METRICS_PATH}")

        if improved:
            torch.save(checkpoint, BEST_CHECKPOINT)
            print(f"Saved new best checkpoint to {BEST_CHECKPOINT}")
        else:
            print(
                f"Best checkpoint unchanged. Current val_score={val_score:.4f}, "
                f"best_val_score={best_val_score:.4f}"
            )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping triggered after {epoch + 1} epochs. "
                f"Best validation score: {best_val_score:.4f}"
            )
            break


if __name__ == "__main__":
    main()
