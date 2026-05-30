"""2x2 intensity-driven LIF encoder for MNIST."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import init_pos_embed, validate_mnist_images
from .spiking import functional, neuron, require_spikingjelly, surrogate


class LIF2x2Encoder(nn.Module):
    """
    Encode each 2x2 patch with LIF neurons driven by pixel intensity.

    Patches with high mean intensity receive an immediate voltage boost so they
    can spike on the same timestep.
    """

    def __init__(
        self,
        image_size: int = 28,
        in_channels: int = 1,
        embed_dim: int = 128,
        white_threshold: float = 0.6,
        intensity_gain: float = 1.2,
        immediate_boost: float = 1.0,
    ) -> None:
        super().__init__()
        require_spikingjelly(self.__class__.__name__)
        if image_size % 2 != 0:
            raise ValueError("image_size must be divisible by 2.")
        if in_channels != 1:
            raise ValueError("LIF2x2Encoder currently expects one input channel.")

        self.image_size = image_size
        self.patch_size = 2
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.white_threshold = white_threshold
        self.intensity_gain = intensity_gain
        self.immediate_boost = immediate_boost
        self.num_patches = (image_size // 2) ** 2
        self.patch_proj = nn.Linear(4, embed_dim, bias=False)
        self.intensity_proj = nn.Linear(1, embed_dim)
        self.lif = neuron.LIFNode(
            tau=2.0,
            v_threshold=1.0,
            v_reset=0.0,
            detach_reset=True,
            surrogate_function=surrogate.ATan(),
        )
        self.pos_embed = init_pos_embed(self.num_patches, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        validate_mnist_images(
            images,
            image_size=self.image_size,
            in_channels=self.in_channels,
            caller=self.__class__.__name__,
        )
        images = (images * 0.3081 + 0.1307).clamp(0.0, 1.0)
        patches = F.unfold(images, kernel_size=2, stride=2).transpose(1, 2)
        intensity = patches.mean(dim=-1, keepdim=True)
        white_area = (intensity >= self.white_threshold).to(dtype=patches.dtype)
        base_drive = intensity * self.intensity_gain + white_area * self.immediate_boost
        learned_detail = self.patch_proj(patches) + self.intensity_proj(intensity)
        currents = base_drive.expand(-1, -1, self.embed_dim) + 0.1 * learned_detail

        functional.reset_net(self.lif)
        spikes = []
        for step in range(currents.size(1)):
            spikes.append(self.lif(currents[:, step, :]))
        return torch.stack(spikes, dim=1) + self.pos_embed
