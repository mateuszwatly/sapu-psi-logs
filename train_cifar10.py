"""CIFAR-10 entry point for the SAPU training pipeline."""

from __future__ import annotations

import sys

import train_pipeline


DEFAULT_ARGS = [
    "--dataset",
    "cifar10",
    "--image-size",
    "32",
    "--in-channels",
    "3",
    "--num-classes",
    "10",
    "--patch-size",
    "4",
    "--checkpoint-out",
    "checkpoints/tpsapu_cifar10.pt",
]


def main() -> None:
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    train_pipeline.main()


if __name__ == "__main__":
    main()
