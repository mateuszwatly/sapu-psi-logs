"""Tiny ImageNet-200 entry point for the SAPU training pipeline.

Uses the Hugging Face dataset:

    slegroux/tiny-imagenet-200-clean

The dataset is cached under data/tiny-imagenet-200-clean by default.
"""

from __future__ import annotations

import sys

import train_pipeline


DEFAULT_ARGS = [
    "--dataset",
    "tiny_imagenet",
    "--data-dir",
    "data/tiny-imagenet-200-clean",
    "--encoder",
    "cnn3",
    "--decoder",
    "membrane_transformer",
    "--pooling",
    "none",
    "--image-size",
    "64",
    "--in-channels",
    "3",
    "--num-classes",
    "200",
    "--taus",
    "1.1,2.0,4.0,8.0,16.0,32.0,64.0,128.0",
    "--patch-size",
    "8",
    "--decoder-max-steps",
    "256",
    "--batch-size",
    "64",
    "--cosine-epochs",
    "80",
    "--prune-epochs",
    "20",
    "--target-sparsity",
    "0.5",
    "--num-workers",
    "0",
    "--no-download",
    "--continuous",
    "--validation-source",
    "train_split",
    "--checkpoint-out",
    "checkpoints/tpsapu_tiny_imagenet/latest.pt",
]


def main() -> None:
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    train_pipeline.main()


if __name__ == "__main__":
    main()
