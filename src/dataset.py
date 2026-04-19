from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_structure_targets(label_file: str | Path, target_mode: str = "scalar") -> np.ndarray:
    with h5py.File(label_file, "r") as handle:
        key = next(iter(handle.keys()))
        labels = np.array(handle[key], dtype=np.float32).reshape(-1)

    if labels.size % 9 != 0:
        raise ValueError(
            f"Expected label count in {label_file} to be divisible by 9, got {labels.size}."
        )

    tensor_targets = labels.reshape(-1, 3, 3)
    if target_mode == "tensor9":
        return tensor_targets.reshape(-1, 9)
    if target_mode == "scalar":
        scalar_targets = np.trace(tensor_targets, axis1=1, axis2=2) / 3.0
        return scalar_targets.reshape(-1, 1)

    raise ValueError(f"Unsupported target_mode={target_mode!r}")


@dataclass(frozen=True)
class TargetNormalizer:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def from_targets(cls, targets: np.ndarray) -> "TargetNormalizer":
        log_targets = np.log1p(np.asarray(targets, dtype=np.float32))
        if log_targets.ndim == 1:
            log_targets = log_targets[:, None]
        mean = torch.from_numpy(log_targets.mean(axis=0))
        std = torch.from_numpy(log_targets.std(axis=0).clip(min=1e-6))
        return cls(mean=mean, std=std)

    def normalize(self, targets: np.ndarray) -> torch.Tensor:
        array = np.log1p(np.asarray(targets, dtype=np.float32))
        if array.ndim == 1:
            array = array[:, None]
        tensor = torch.from_numpy(array)
        return (tensor - self.mean) / self.std

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.expm1((tensor * self.std.to(tensor.device)) + self.mean.to(tensor.device))

    def to_state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean.clone(), "std": self.std.clone()}

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> "TargetNormalizer":
        return cls(mean=state_dict["mean"].float(), std=state_dict["std"].float())


class MicrostructureDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        label_file: str | Path,
        structure_ids: Iterable[int] | None = None,
        normalizer: TargetNormalizer | None = None,
        target_mode: str = "scalar",
    ) -> None:
        self.image_dir = Path(image_dir)
        targets = load_structure_targets(label_file, target_mode=target_mode)

        all_structure_ids = np.arange(1, targets.shape[0] + 1, dtype=np.int32)
        if structure_ids is None:
            allowed_ids = set(all_structure_ids.tolist())
        else:
            allowed_ids = {int(structure_id) for structure_id in structure_ids}

        self.files = sorted(
            file
            for file in self.image_dir.iterdir()
            if file.suffix == ".png" and self._parse_structure_id(file.name) in allowed_ids
        )

        if not self.files:
            raise FileNotFoundError(
                f"No PNG files found in {self.image_dir} for the requested structure ids."
            )

        self.structure_ids = np.array(
            [self._parse_structure_id(file.name) for file in self.files], dtype=np.int32
        )
        self.targets = torch.from_numpy(np.asarray(targets, dtype=np.float32))
        self.normalizer = normalizer
        log_targets = torch.log1p(self.targets)
        self.target_tensors = (
            normalizer.normalize(targets) if normalizer is not None else log_targets
        )

    @staticmethod
    def _parse_structure_id(filename: str) -> int:
        return int(filename.split("_")[1])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        file_path = self.files[idx]
        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image {file_path}")

        image = image.astype(np.float32) / 255.0
        image = np.repeat(image[None, :, :], 3, axis=0)
        image = (image - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        image_tensor = torch.from_numpy(np.ascontiguousarray(image))

        structure_id = self.structure_ids[idx]
        target_tensor = self.target_tensors[structure_id - 1]

        return image_tensor, target_tensor, int(structure_id)
