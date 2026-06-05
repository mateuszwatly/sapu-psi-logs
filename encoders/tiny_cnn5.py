"""Five-layer tiny CNN image encoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import flatten_feature_map, init_pos_embed, validate_mnist_images


class TinyCNN5Encoder(nn.Module):
    """Five convolution layers, returning a quarter-resolution token grid."""

    def __init__(
        self,
        image_size: int = 28,
        in_channels: int = 1,
        embed_dim: int = 128,
        hidden_channels: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        token_grid = ((image_size + 1) // 2 + 1) // 2
        self.num_patches = token_grid * token_grid
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.pos_embed = init_pos_embed(self.num_patches, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        validate_mnist_images(
            images,
            image_size=self.image_size,
            in_channels=self.in_channels,
            caller=self.__class__.__name__,
        )
        return flatten_feature_map(self.net(images)) + self.pos_embed
