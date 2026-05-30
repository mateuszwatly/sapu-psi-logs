"""Row-wise MNIST encoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import init_pos_embed, validate_mnist_images


class MNISTRowEncoder(nn.Module):
    """Encode each MNIST row as one sequence token."""

    def __init__(
        self,
        image_size: int = 28,
        in_channels: int = 1,
        embed_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError("MNISTRowEncoder currently expects one input channel.")

        self.image_size = image_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = image_size
        self.proj = nn.Sequential(
            nn.LayerNorm(image_size),
            nn.Linear(image_size, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_embed = init_pos_embed(image_size, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        validate_mnist_images(
            images,
            image_size=self.image_size,
            in_channels=self.in_channels,
            caller=self.__class__.__name__,
        )
        return self.proj(images.squeeze(1)) + self.pos_embed
