"""Shared helpers for MNIST encoders."""

from __future__ import annotations

import torch
import torch.nn as nn


def validate_mnist_images(
    images: torch.Tensor,
    *,
    image_size: int,
    in_channels: int,
    caller: str,
) -> None:
    if images.dim() != 4:
        raise ValueError(f"{caller} expects (batch, channels, height, width).")
    if images.size(1) != in_channels:
        raise ValueError(f"Expected {in_channels} channel(s), got {images.size(1)}.")
    if images.size(2) != image_size or images.size(3) != image_size:
        raise ValueError(
            f"Expected {image_size}x{image_size} images, got "
            f"{images.size(2)}x{images.size(3)}."
        )


def init_pos_embed(num_tokens: int, embed_dim: int) -> nn.Parameter:
    pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
    nn.init.trunc_normal_(pos_embed, std=0.02)
    return pos_embed


def flatten_feature_map(features: torch.Tensor) -> torch.Tensor:
    return features.flatten(2).transpose(1, 2)
