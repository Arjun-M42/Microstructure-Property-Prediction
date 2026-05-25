from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (
    MicrostructureDataset,
    TargetNormalizer,
    kappa_scalar_from_tensor9,
    load_structure_targets,
    preprocess_grayscale_image,
    structure_collate_fn,
)
from model import get_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "vit_best.pth"
TRAIN_IMAGE_DIR = PROJECT_ROOT / "processed_data" / "train_images"


def load_checkpoint_model(
    checkpoint_path: str | Path = CHECKPOINT_PATH,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, TargetNormalizer, dict, torch.device]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    normalizer = TargetNormalizer.from_state_dict(checkpoint["normalizer"])
    num_labels = int(checkpoint.get("num_labels", normalizer.mean.numel()))
    model = get_model(num_labels=num_labels).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            "This checkpoint does not match the current model architecture. "
            "Retrain with `python src\\train.py` to create a fresh checkpoint."
        ) from exc
    model.eval()
    return model, normalizer, checkpoint, device


def _result_payload(
    prediction: torch.Tensor,
    target: torch.Tensor | None = None,
    structure_id: int | None = None,
) -> dict:
    pred_values = prediction.detach().cpu().view(-1)
    payload = {
        "structure_id": structure_id,
        "predicted_values": pred_values.tolist(),
        "predicted_scalar_kappa": float(kappa_scalar_from_tensor9(pred_values.unsqueeze(0))[0]),
    }

    if pred_values.numel() == 9:
        payload["predicted_tensor"] = np.asarray(pred_values.tolist()).reshape(3, 3).tolist()

    if target is not None:
        target_values = target.detach().cpu().view(-1)
        abs_errors = (pred_values - target_values).abs()
        payload.update(
            {
                "actual_values": target_values.tolist(),
                "actual_scalar_kappa": float(
                    kappa_scalar_from_tensor9(target_values.unsqueeze(0))[0]
                ),
                "mae": float(abs_errors.mean()),
                "rmse": float(torch.sqrt(((pred_values - target_values) ** 2).mean())),
                "mape": float(
                    (abs_errors / target_values.abs().clamp_min(1e-6)).mean() * 100.0
                ),
            }
        )
        payload["accuracy"] = max(0.0, 100.0 - payload["mape"])
        if target_values.numel() == 9:
            payload["actual_tensor"] = np.asarray(target_values.tolist()).reshape(3, 3).tolist()

    return payload


def predict_structure(
    structure_id: int,
    image_dir: str | Path = TRAIN_IMAGE_DIR,
    label_file: str | Path | None = None,
    checkpoint_path: str | Path = CHECKPOINT_PATH,
) -> dict:
    model, normalizer, checkpoint, device = load_checkpoint_model(checkpoint_path)
    target_mode = checkpoint.get("target_mode", "tensor9")
    is_inference_mode = label_file is None

    dataset = MicrostructureDataset(
        image_dir,
        label_file=label_file,
        structure_ids=[structure_id],
        normalizer=normalizer,
        target_mode=target_mode,
        inference_mode=is_inference_mode,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=structure_collate_fn)

    with torch.no_grad():
        for images, labels, structure_ids, masks in loader:
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            outputs = model(pixel_values=images, masks=masks)
            prediction = normalizer.denormalize(outputs)[0]
            target = None if is_inference_mode else normalizer.denormalize(labels)[0]
            return _result_payload(prediction, target, int(structure_ids[0]))

    raise RuntimeError(f"No prediction was made for structure {structure_id}")


def predict_image_files(
    image_paths: list[str | Path],
    checkpoint_path: str | Path = CHECKPOINT_PATH,
) -> dict:
    if not image_paths:
        raise ValueError("At least one image is required for custom-image inference.")

    model, normalizer, _, device = load_checkpoint_model(checkpoint_path)
    tensors = []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image {path}")
        tensors.append(preprocess_grayscale_image(image))

    images = torch.stack(tensors).unsqueeze(0).to(device)
    masks = torch.ones((1, len(tensors)), dtype=torch.bool, device=device)
    with torch.no_grad():
        outputs = model(pixel_values=images, masks=masks)
        prediction = normalizer.denormalize(outputs)[0]
    return _result_payload(prediction)


def predict_image_arrays(
    images: list[np.ndarray],
    checkpoint_path: str | Path = CHECKPOINT_PATH,
) -> dict:
    if not images:
        raise ValueError("At least one image is required for custom-image inference.")

    model, normalizer, _, device = load_checkpoint_model(checkpoint_path)
    tensors = [preprocess_grayscale_image(image) for image in images]
    batch = torch.stack(tensors).unsqueeze(0).to(device)
    masks = torch.ones((1, len(tensors)), dtype=torch.bool, device=device)
    with torch.no_grad():
        outputs = model(pixel_values=batch, masks=masks)
        prediction = normalizer.denormalize(outputs)[0]
    return _result_payload(prediction)


def print_result(result: dict) -> None:
    label = (
        f"structure_{result['structure_id']}"
        if result.get("structure_id") is not None
        else "custom images"
    )
    print(f"\nResults for {label}:")
    print(f"  Predicted scalar kappa : {result['predicted_scalar_kappa']:.6f}")

    if "actual_scalar_kappa" in result:
        print(f"  Actual scalar kappa    : {result['actual_scalar_kappa']:.6f}")
        print(f"  MAE                    : {result['mae']:.6f}")
        print(f"  RMSE                   : {result['rmse']:.6f}")
        print(f"  Accuracy (100 - MAPE)  : {result['accuracy']:.2f}%")

    if "predicted_tensor" in result:
        print("\n  Predicted kappa tensor:")
        print(np.array2string(np.asarray(result["predicted_tensor"]), precision=4))
    elif result["predicted_values"]:
        print(f"  Predicted value        : {result['predicted_values'][0]:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict kappa for microstructures")
    parser.add_argument("--structure_id", type=int, help="Structure ID to predict")
    parser.add_argument("--image_paths", nargs="+", help="Custom image paths for inference")
    parser.add_argument("--image_dir", type=str, default=str(TRAIN_IMAGE_DIR))
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH))
    parser.add_argument(
        "--label_file",
        type=str,
        default=None,
        help="Optional MAT file containing true labels for comparison.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    if args.structure_id is None and not args.image_paths:
        parser.error("Provide --structure_id or --image_paths.")
    if args.structure_id is not None and args.image_paths:
        parser.error("Use either --structure_id or --image_paths, not both.")

    if args.structure_id is not None:
        result = predict_structure(
            args.structure_id,
            image_dir=args.image_dir,
            label_file=args.label_file,
            checkpoint_path=args.checkpoint,
        )
    else:
        result = predict_image_files(args.image_paths, checkpoint_path=args.checkpoint)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_result(result)


if __name__ == "__main__":
    main()
