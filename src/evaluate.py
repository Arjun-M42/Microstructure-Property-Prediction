from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    MicrostructureDataset,
    kappa_scalar_from_tensor9,
    load_structure_targets,
    structure_collate_fn,
)
from predict import CHECKPOINT_PATH, TRAIN_IMAGE_DIR, load_checkpoint_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "models" / "evaluation_results.csv"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "models" / "evaluation_metrics.json"


def regression_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    errors = predictions - targets
    abs_errors = errors.abs()
    mae = abs_errors.mean().item()
    rmse = torch.sqrt((errors**2).mean()).item()
    mape = (abs_errors / targets.abs().clamp_min(1e-6)).mean().item() * 100.0
    ss_res = (errors**2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum().clamp_min(1e-12)
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "accuracy": max(0.0, 100.0 - mape),
        "r2": (1.0 - ss_res / ss_tot).item(),
    }


def evaluate_model(
    image_dir: str | Path,
    label_file: str | Path,
    checkpoint_path: str | Path = CHECKPOINT_PATH,
    limit: int | None = None,
    batch_size: int = 1,
) -> tuple[dict, list[dict]]:
    model, normalizer, checkpoint, device = load_checkpoint_model(checkpoint_path)
    target_mode = checkpoint.get("target_mode", "tensor9")
    target_matrix = load_structure_targets(label_file, target_mode=target_mode)
    structure_ids = list(range(1, target_matrix.shape[0] + 1))
    if limit is not None:
        structure_ids = structure_ids[:limit]

    dataset = MicrostructureDataset(
        image_dir,
        label_file=label_file,
        structure_ids=structure_ids,
        normalizer=normalizer,
        target_mode=target_mode,
        inference_mode=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=structure_collate_fn)

    rows = []
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, labels, ids, masks in tqdm(loader, total=len(loader), desc="Evaluation"):
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            outputs = model(pixel_values=images, masks=masks)
            predictions = normalizer.denormalize(outputs).cpu()
            targets = normalizer.denormalize(labels).cpu()

            all_preds.append(predictions)
            all_targets.append(targets)
            pred_scalar = kappa_scalar_from_tensor9(predictions)
            target_scalar = kappa_scalar_from_tensor9(targets)

            for row_index, structure_id in enumerate(ids.tolist()):
                row = {
                    "structure_id": structure_id,
                    "predicted_scalar_kappa": float(pred_scalar[row_index]),
                    "actual_scalar_kappa": float(target_scalar[row_index]),
                    "scalar_abs_error": float(
                        abs(pred_scalar[row_index] - target_scalar[row_index])
                    ),
                }
                for value_index, value in enumerate(predictions[row_index].tolist()):
                    row[f"pred_{value_index}"] = value
                for value_index, value in enumerate(targets[row_index].tolist()):
                    row[f"actual_{value_index}"] = value
                rows.append(row)

    pred_tensor = torch.cat(all_preds, dim=0)
    target_tensor = torch.cat(all_targets, dim=0)
    metrics = regression_metrics(pred_tensor, target_tensor)
    scalar_metrics = regression_metrics(
        kappa_scalar_from_tensor9(pred_tensor).view(-1, 1),
        kappa_scalar_from_tensor9(target_tensor).view(-1, 1),
    )
    metrics["scalar"] = scalar_metrics
    metrics["target_mode"] = target_mode
    metrics["num_structures"] = len(rows)
    return metrics, rows


def write_rows(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model results on labeled structures")
    parser.add_argument("--image_dir", type=str, default=str(TRAIN_IMAGE_DIR))
    parser.add_argument("--label_file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--results_csv", type=str, default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--metrics_json", type=str, default=str(DEFAULT_METRICS_PATH))
    args = parser.parse_args()

    metrics, rows = evaluate_model(
        args.image_dir,
        args.label_file,
        checkpoint_path=args.checkpoint,
        limit=args.limit,
        batch_size=args.batch_size,
    )
    write_rows(args.results_csv, rows)
    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_json).write_text(json.dumps(metrics, indent=2))

    print("\nModel results")
    print(f"  Structures evaluated : {metrics['num_structures']}")
    print(f"  Target mode          : {metrics['target_mode']}")
    print(f"  MAE                  : {metrics['mae']:.6f}")
    print(f"  RMSE                 : {metrics['rmse']:.6f}")
    # print(f"  Accuracy (100-MAPE)  : {metrics['accuracy']:.2f}%")
    print(f"  R2                   : {metrics['r2']:.4f}")
    print(f"  Scalar MAE           : {metrics['scalar']['mae']:.6f}")
    print(f"  Scalar Accuracy      : {metrics['scalar']['accuracy']:.2f}%")
    print(f"  Wrote predictions    : {args.results_csv}")
    print(f"  Wrote metrics        : {args.metrics_json}")


if __name__ == "__main__":
    main()
