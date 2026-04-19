from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import MicrostructureDataset, TargetNormalizer
from model import get_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "vit_best.pth"
TRAIN_IMAGE_DIR = PROJECT_ROOT / "processed_data" / "train_images"
TRAIN_LABEL_FILE = PROJECT_ROOT / "data" / "kappa_train.mat"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    target_mode = checkpoint.get("target_mode", "scalar")
    normalizer = TargetNormalizer.from_state_dict(checkpoint["normalizer"])

    dataset = MicrostructureDataset(
        TRAIN_IMAGE_DIR,
        TRAIN_LABEL_FILE,
        normalizer=normalizer,
        target_mode=target_mode,
    )
    print("Dataset size:", len(dataset))

    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model = get_model(num_labels=1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions: dict[int, list[torch.Tensor]] = {}
    targets: dict[int, torch.Tensor] = {}

    with torch.no_grad():
        for images, labels, structure_ids in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(pixel_values=images).logits.view(-1, 1)
            denorm_predictions = normalizer.denormalize(outputs).cpu().view(-1)
            denorm_targets = normalizer.denormalize(labels).cpu().view(-1)

            for prediction, target, structure_id in zip(
                denorm_predictions, denorm_targets, structure_ids.tolist()
            ):
                predictions.setdefault(structure_id, []).append(prediction)
                targets[structure_id] = target

    sorted_structure_ids = sorted(predictions)
    print("\nFirst 10 structure-level kappa predictions:")
    for structure_id in sorted_structure_ids[:10]:
        mean_prediction = torch.stack(predictions[structure_id]).mean().item()
        target = targets[structure_id].item()
        print(f"structure_{structure_id}: pred={mean_prediction:.6f}, true={target:.6f}")


if __name__ == "__main__":
    main()
