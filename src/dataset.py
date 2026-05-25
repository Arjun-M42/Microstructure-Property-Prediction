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
    if target_mode.startswith("component"):
        try:
            component_index = int(target_mode.replace("component", ""))
        except ValueError as exc:
            raise ValueError(
                "Component target modes must look like component0 through component8."
            ) from exc
        if component_index < 0 or component_index > 8:
            raise ValueError("Component target index must be between 0 and 8.")
        return tensor_targets.reshape(-1, 9)[:, component_index : component_index + 1]

    raise ValueError(f"Unsupported target_mode={target_mode!r}")


def kappa_scalar_from_tensor9(values: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Return the mean diagonal kappa from flattened 3x3 tensors."""
    if isinstance(values, torch.Tensor):
        if values.ndim == 1 and values.numel() > 1 and values.numel() % 9 == 0:
            values = values.reshape(-1, 9)
        if values.shape[-1] == 1:
            return values[..., 0]
        tensor = values.reshape(*values.shape[:-1], 3, 3)
        return torch.diagonal(tensor, dim1=-2, dim2=-1).mean(dim=-1)

    array = np.asarray(values)
    if array.ndim == 1 and array.size > 1 and array.size % 9 == 0:
        array = array.reshape(-1, 9)
    if array.shape[-1] == 1:
        return array[..., 0]
    tensor = array.reshape(*array.shape[:-1], 3, 3)
    return np.diagonal(tensor, axis1=-2, axis2=-1).mean(axis=-1)


def preprocess_grayscale_image(image: np.ndarray) -> torch.Tensor:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    image = np.repeat(image[None, :, :], 3, axis=0)
    image = (image - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return torch.from_numpy(np.ascontiguousarray(image))


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
        tensor = torch.from_numpy(array).to(self.mean.device)
        return (tensor - self.mean) / self.std

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.expm1((tensor * self.std.to(tensor.device)) + self.mean.to(tensor.device))

    def to_state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean.cpu().clone(), "std": self.std.cpu().clone()}

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> "TargetNormalizer":
        return cls(mean=state_dict["mean"].cpu().float(), std=state_dict["std"].cpu().float())


class MicrostructureDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        label_file: str | Path | None = None,
        structure_ids: Iterable[int] | None = None,
        normalizer: TargetNormalizer | None = None,
        target_mode: str = "scalar",
        inference_mode: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.inference_mode = inference_mode
        self.normalizer = normalizer

        if not self.inference_mode:
            if label_file is None:
                raise ValueError("label_file must be provided when inference_mode is False")
            targets = load_structure_targets(label_file, target_mode=target_mode)
            all_structure_ids = np.arange(1, targets.shape[0] + 1, dtype=np.int32)
            self.targets = torch.from_numpy(np.asarray(targets, dtype=np.float32))
            log_targets = torch.log1p(self.targets)
            self.target_tensors = (
                normalizer.normalize(targets) if normalizer is not None else log_targets
            )
        else:
            all_structure_ids = None

        if structure_ids is None:
            if all_structure_ids is not None:
                allowed_ids = set(all_structure_ids.tolist())
            else:
                allowed_ids = None
        else:
            allowed_ids = {int(structure_id) for structure_id in structure_ids}

        self.structure_to_files = {}
        for file in self.image_dir.iterdir():
            if file.suffix == ".png":
                sid = self._parse_structure_id(file.name)
                if allowed_ids is None or sid in allowed_ids:
                    self.structure_to_files.setdefault(sid, []).append(file)

        if not self.structure_to_files:
            raise FileNotFoundError(
                f"No PNG files found in {self.image_dir} for the requested structure ids."
            )

        self.structure_ids = sorted(self.structure_to_files.keys())

    @staticmethod
    def _parse_structure_id(filename: str) -> int:
        return int(filename.split("_")[1])

    def __len__(self) -> int:
        return len(self.structure_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        structure_id = self.structure_ids[idx]
        file_paths = sorted(self.structure_to_files[structure_id])
        
        image_tensors = []
        for file_path in file_paths:
            image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Could not read image {file_path}")

            image_tensors.append(preprocess_grayscale_image(image))

        stacked_images = torch.stack(image_tensors)  # Shape: (num_slices, 3, 224, 224)

        if self.inference_mode:
            target_tensor = torch.zeros(1)
        else:
            target_tensor = self.target_tensors[structure_id - 1]

        return stacked_images, target_tensor, int(structure_id)


def structure_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collates a batch of variable-slice structures.
    Returns:
        padded_images: [B, max_slices, 3, 224, 224]
        targets: [B, num_labels]
        structure_ids: [B]
        masks: [B, max_slices] boolean tensor where True means real image, False means padding.
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    structure_ids = [item[2] for item in batch]

    max_slices = max(img.shape[0] for img in images)
    
    padded_images = []
    masks = []
    
    for img in images:
        num_slices = img.shape[0]
        if num_slices < max_slices:
            pad = torch.zeros((max_slices - num_slices,) + img.shape[1:], dtype=img.dtype)
            padded_images.append(torch.cat([img, pad], dim=0))
        else:
            padded_images.append(img)
            
        mask = torch.cat([
            torch.ones(num_slices, dtype=torch.bool),
            torch.zeros(max_slices - num_slices, dtype=torch.bool)
        ])
        masks.append(mask)

    return torch.stack(padded_images), torch.stack(targets), torch.tensor(structure_ids, dtype=torch.long), torch.stack(masks)

