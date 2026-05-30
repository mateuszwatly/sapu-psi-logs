"""Residual tiny CNN encoder for MNIST."""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import flatten_feature_map, init_pos_embed, validate_mnist_images


class ResidualTinyCNNEncoder(nn.Module):
    """
    Residual CNN encoder exposing both second- and third-layer token grids.

    The output sequence is ``[layer2 tokens, residual layer3 tokens]``.
    """

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
        self.num_patches = 98
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
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
        layer2 = self.conv2(self.conv1(images))
        layer3 = layer2 + self.conv3(layer2)
        tokens = torch.cat(
            [flatten_feature_map(layer2), flatten_feature_map(layer3)],
            dim=1,
        )
        return tokens + self.pos_embed
