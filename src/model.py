from __future__ import annotations

import torch
import torch.nn as nn
from transformers import ViTModel


class MicrostructureViT(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str = "google/vit-base-patch16-224",
        num_labels: int = 9,
    ) -> None:
        super().__init__()
        try:
            self.vit = ViTModel.from_pretrained(
                pretrained_model_name,
                add_pooling_layer=False,
                local_files_only=True,
            )
        except OSError:
            self.vit = ViTModel.from_pretrained(pretrained_model_name, add_pooling_layer=False)

        hidden_size = self.vit.config.hidden_size
        self.slice_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.slice_attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, num_labels),
        )
        nn.init.normal_(self.regressor[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.regressor[-1].bias)
        self.num_labels = num_labels
        self.backbone_trainable = True

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.backbone_trainable = trainable
        for parameter in self.vit.parameters():
            parameter.requires_grad = trainable

    def _encode_slices(self, flat_pixels: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.backbone_trainable and torch.is_grad_enabled()):
            outputs = self.vit(pixel_values=flat_pixels)
        cls_tokens = outputs.last_hidden_state[:, 0, :]
        return self.slice_projection(cls_tokens)

    def forward(self, pixel_values: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, slice_count, channels, height, width = pixel_values.shape
        flat_pixels = pixel_values.reshape(batch_size * slice_count, channels, height, width)

        if masks is None:
            slice_features = self._encode_slices(flat_pixels).reshape(batch_size, slice_count, -1)
            pooled = slice_features.mean(dim=1)
        else:
            flat_masks = masks.reshape(-1)
            valid_pixels = flat_pixels[flat_masks]
            valid_features = self._encode_slices(valid_pixels)

            feature_dim = valid_features.shape[-1]
            slice_features = torch.zeros(
                (batch_size * slice_count, feature_dim),
                device=pixel_values.device,
                dtype=valid_features.dtype,
            )
            slice_features[flat_masks] = valid_features
            slice_features = slice_features.reshape(batch_size, slice_count, feature_dim)

            attention_scores = self.slice_attention(slice_features).squeeze(-1)
            attention_scores = attention_scores.masked_fill(~masks, torch.finfo(slice_features.dtype).min)
            attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)
            pooled = (slice_features * attention_weights).sum(dim=1)

        return self.regressor(pooled)


def get_model(num_labels: int = 9) -> MicrostructureViT:
    return MicrostructureViT(num_labels=num_labels)
