"""ImageNet entry point for the SAPU training pipeline.

The data directory must use torchvision ImageFolder layout:

    imagenet/Data/CLS-LOC/train/<class_name>/*.jpg
    imagenet/Data/CLS-LOC/val/<class_name>/*.jpg
"""

from __future__ import annotations

import sys

import train_pipeline


DEFAULT_ARGS = [
    "--dataset",
    "imagenet",
    "--data-dir",
    "imagenet/Data/CLS-LOC",
    "--image-size",
    "224",
    "--in-channels",
    "3",
    "--num-classes",
    "1000",
    "--patch-size",
    "16",
    "--batch-size",
    "64",
    "--checkpoint-out",
    "checkpoints/tpsapu_imagenet.pt",
]


def main() -> None:
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    train_pipeline.main()


if __name__ == "__main__":
    main()
