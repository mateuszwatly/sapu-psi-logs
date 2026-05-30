"""MLP patch projection encoder for MNIST."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import init_pos_embed, validate_mnist_images


class MLPPatchEncoder(nn.Module):
    """Patchify MNIST images and project each patch with a small MLP."""

    def __init__(
        self,
        image_size: int = 28,
        patch_size: int = 7,
        in_channels: int = 1,
        embed_dim: int = 128,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        hidden_dim = hidden_dim or max(embed_dim, patch_dim * 2)
        self.proj = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.pos_embed = init_pos_embed(self.num_patches, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        validate_mnist_images(
            images,
            image_size=self.image_size,
            in_channels=self.in_channels,
            caller=self.__class__.__name__,
        )
        patches = F.unfold(
            images,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).transpose(1, 2)
        return self.proj(patches) + self.pos_embed
