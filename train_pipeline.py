"""
Training pipeline for encoder -> TPSAPU backbone -> decoder experiments.

Default schedule:
    5 warmup epochs
    35 cosine-annealing training epochs
    20 L2-pruning epochs on the TPSAPU shared recurrent topology

Example:
    python train_pipeline.py --encoder linear_patch --decoder membrane_mlp
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

from decoders import (
    AllStateMLPDecoder,
    AllStateTransformerDecoder,
    ClassLIFSpikeCountDecoder,
    LinearDecoder,
    MembraneMLPDecoder,
    MembraneSpikeMLPDecoder,
    MembraneSpikeTransformerDecoder,
    MembraneTransformerDecoder,
    SpikeMLPDecoder,
    SpikeTransformerDecoder,
)
from encoders import (
    LIF2x2Encoder,
    LinearPatchEncoder,
    MLPPatchEncoder,
    MNISTRowEncoder,
    ResidualTinyCNNEncoder,
    TinyCNN2Encoder,
    TinyCNN3Encoder,
)
from tpsapu import TPSAPUBackbone


class EncoderBackboneDecoder(nn.Module):
    """Composable model: raw input -> encoder -> backbone -> decoder."""

    def __init__(
        self,
        encoder: nn.Module,
        backbone: TPSAPUBackbone,
        decoder: nn.Module,
        *,
        pooling: str,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.backbone = backbone
        self.decoder = decoder
        self.pooling = pooling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(x)
        features = self.backbone(
            tokens,
            pooling="none" if self.decoder.needs_sequence else self.pooling,
            state=self.decoder.input_state,
            reset_state=True,
        )
        return self.decoder(features)


class RecurrentMagnitudePruner:
    """Persistent magnitude mask for the TPSAPU shared recurrent matrix."""

    def __init__(
        self,
        weight: torch.Tensor,
        *,
        threshold: float,
        target_sparsity: float,
    ) -> None:
        self.weight = weight
        self.threshold = threshold
        self.target_sparsity = target_sparsity
        self.mask = torch.ones_like(weight, dtype=torch.bool)

    @torch.no_grad()
    def apply(self) -> None:
        self.weight.mul_(self.mask.to(dtype=self.weight.dtype))

    @torch.no_grad()
    def update(self, target_sparsity: float | None = None) -> None:
        target = self.target_sparsity if target_sparsity is None else target_sparsity
        target = min(max(target, 0.0), 1.0)
        if self.threshold > 0:
            self.mask &= self.weight.abs() >= self.threshold
        if target > 0:
            self.mask &= self._topk_mask(target)
        self.apply()

    def sparsity(self) -> float:
        return 1.0 - self.mask.float().mean().item()

    @torch.no_grad()
    def _topk_mask(self, target_sparsity: float) -> torch.Tensor:
        numel = self.weight.numel()
        keep = max(1, int(round(numel * (1.0 - target_sparsity))))
        keep = min(keep, numel)
        flat_abs = self.weight.detach().abs().flatten()
        if keep >= numel:
            return torch.ones_like(self.mask)
        threshold = torch.topk(flat_abs, keep, largest=True, sorted=False).values.min()
        return self.weight.abs() >= threshold


class NeuronActivityPruner:
    """Persistent channel mask for reservoir neurons with little activity."""

    def __init__(
        self,
        backbone: TPSAPUBackbone,
        *,
        spike_threshold: float,
        membrane_std_threshold: float,
        target_sparsity: float,
        min_keep: int,
    ) -> None:
        self.backbone = backbone
        self.spike_threshold = spike_threshold
        self.membrane_std_threshold = membrane_std_threshold
        self.target_sparsity = target_sparsity
        self.min_keep = min_keep
        self.mask = backbone.neuron_mask.detach().clone().bool()
        self.reset_activity()

    def reset_activity(self) -> None:
        device = self.backbone.neuron_mask.device
        dim = self.backbone.reservoir_dim
        self.spike_sum = torch.zeros(dim, device=device)
        self.membrane_sum = torch.zeros(dim, device=device)
        self.membrane_sq_sum = torch.zeros(dim, device=device)
        self.count = 0

    @torch.no_grad()
    def record(self, states: dict[str, torch.Tensor] | None) -> None:
        if states is None:
            return
        spikes = states["spike"].detach()
        membranes = states["membrane"].detach()
        reservoir_dim = self.backbone.reservoir_dim
        tau_count = len(self.backbone.taus)
        spikes = spikes.view(*spikes.shape[:2], tau_count, reservoir_dim)
        membranes = membranes.view(*membranes.shape[:2], tau_count, reservoir_dim)
        reduce_dims = (0, 1, 2)
        self.spike_sum += spikes.sum(dim=reduce_dims).to(self.spike_sum.device)
        self.membrane_sum += membranes.sum(dim=reduce_dims).to(self.membrane_sum.device)
        self.membrane_sq_sum += membranes.pow(2).sum(dim=reduce_dims).to(
            self.membrane_sq_sum.device
        )
        self.count += spikes.numel() // reservoir_dim

    @torch.no_grad()
    def update(self) -> None:
        if self.count <= 0:
            return
        spike_rate, membrane_std = self.activity()
        inactive = (spike_rate <= self.spike_threshold) & (
            membrane_std <= self.membrane_std_threshold
        )
        candidate = self.mask & ~inactive

        target = min(max(self.target_sparsity, 0.0), 1.0)
        if target > 0:
            target_keep = max(self.min_keep, int(round(self.mask.numel() * (1.0 - target))))
            target_keep = min(target_keep, self.mask.numel())
            if int(candidate.sum().item()) > target_keep:
                score = spike_rate + membrane_std
                keep_score = score.masked_fill(~self.mask, float("-inf"))
                keep_indices = torch.topk(
                    keep_score,
                    k=target_keep,
                    largest=True,
                    sorted=False,
                ).indices
                target_mask = torch.zeros_like(self.mask)
                target_mask[keep_indices] = True
                candidate &= target_mask

        if int(candidate.sum().item()) < self.min_keep:
            score = spike_rate + membrane_std
            keep_score = score.masked_fill(~self.mask, float("-inf"))
            keep_count = min(self.min_keep, self.mask.numel())
            keep_indices = torch.topk(
                keep_score,
                k=keep_count,
                largest=True,
                sorted=False,
            ).indices
            candidate = torch.zeros_like(self.mask)
            candidate[keep_indices] = True

        self.mask &= candidate
        self.apply()

    @torch.no_grad()
    def apply(self) -> None:
        self.backbone.set_neuron_mask(self.mask.to(self.backbone.neuron_mask.device))
        mask = self.backbone.neuron_mask.to(
            device=self.backbone.shared_input_proj.weight.device,
            dtype=self.backbone.shared_input_proj.weight.dtype,
        )
        self.backbone.shared_input_proj.weight.mul_(mask.view(-1, 1))
        self.backbone.shared_recurrent.weight.mul_(mask.view(-1, 1))
        self.backbone.shared_recurrent.weight.mul_(mask.view(1, -1))

    def sparsity(self) -> float:
        return 1.0 - self.mask.float().mean().item()

    def activity(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count <= 0:
            zeros = torch.zeros_like(self.mask, dtype=torch.float32)
            return zeros, zeros
        spike_rate = self.spike_sum / self.count
        membrane_mean = self.membrane_sum / self.count
        membrane_var = (self.membrane_sq_sum / self.count) - membrane_mean.pow(2)
        membrane_std = membrane_var.clamp_min(0.0).sqrt()
        return spike_rate, membrane_std


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["mnist", "cifar10", "imagenet", "tiny_imagenet"],
        default="mnist",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--encoder",
        choices=[
            "linear_patch",
            "mlp_patch",
            "lif_2x2",
            "cnn2",
            "cnn3",
            "res_cnn",
            "patch",
            "rows",
        ],
        default="linear_patch",
    )
    parser.add_argument(
        "--decoder",
        choices=[
            "linear",
            "membrane_mlp",
            "spike_mlp",
            "both_mlp",
            "all_state_mlp",
            "membrane_transformer",
            "spike_transformer",
            "both_transformer",
            "all_state_transformer",
            "lif_count",
            "mlp",
        ],
        default="membrane_mlp",
    )
    parser.add_argument("--pooling", choices=["last", "mean"], default="last")

    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--reservoir-dim", type=int, default=64)
    parser.add_argument("--taus", default="1.1,8.0,64.0")
    parser.add_argument("--input-hidden-dim", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=7)
    parser.add_argument("--encoder-hidden-dim", type=int, default=0)
    parser.add_argument("--cnn-channels", type=int, default=64)
    parser.add_argument("--lif-white-threshold", type=float, default=0.6)
    parser.add_argument("--encoder-dropout", type=float, default=0.05)
    parser.add_argument("--decoder-hidden-dim", type=int, default=128)
    parser.add_argument("--decoder-dropout", type=float, default=0.1)
    parser.add_argument("--decoder-transformer-layers", type=int, default=2)
    parser.add_argument("--decoder-transformer-heads", type=int, default=4)
    parser.add_argument("--decoder-transformer-ff-mult", type=float, default=4.0)
    parser.add_argument("--decoder-max-steps", type=int, default=256)
    parser.add_argument("--recurrent-drop", type=float, default=0.1)

    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--cosine-epochs", type=int, default=35)
    parser.add_argument(
        "--cosine-cycles",
        type=float,
        default=1.0,
        help="Number of cosine annealing descents; 1.0 decays once from base LR to min LR.",
    )
    parser.add_argument("--prune-epochs", type=int, default=30)
    parser.add_argument(
        "--prune-cycles",
        type=float,
        default=1.0,
        help="Number of pruning-phase cosine annealing descents.",
    )
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--prune-lr-scale", type=float, default=0.2)
    parser.add_argument("--l2-prune-start-lambda", type=float, default=1e-7)
    parser.add_argument("--l2-prune-growth-epochs", type=int, default=4)
    parser.add_argument("--prune-threshold", type=float, default=0.0)
    parser.add_argument("--target-sparsity", type=float, default=0.95)
    parser.add_argument("--prune-start-sparsity", type=float, default=0.0)
    parser.add_argument("--prune-ramp-epochs", type=int, default=0)
    parser.add_argument("--prune-stabilize-epochs", type=int, default=6)
    parser.add_argument("--keep-dropout-during-prune", action="store_true")
    parser.add_argument("--neuron-prune", action="store_true")
    parser.add_argument("--neuron-prune-target-sparsity", type=float, default=0.0)
    parser.add_argument("--neuron-spike-threshold", type=float, default=1e-4)
    parser.add_argument("--neuron-membrane-std-threshold", type=float, default=1e-3)
    parser.add_argument("--neuron-prune-min-keep", type=int, default=1)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "DataLoader worker processes. Default 0 keeps dataset decoding in "
            "the main process and avoids hidden worker-subprocess crashes."
        ),
    )
    parser.add_argument(
        "--log-batches",
        type=int,
        default=0,
        help="Print running loss/accuracy every N batches; disabled by default.",
    )
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument(
        "--validation-source",
        choices=["dataset", "train_split"],
        default="dataset",
        help=(
            "Use the dataset validation/test split for per-epoch validation, "
            "or hold out part of the training split and reserve the dataset "
            "validation/test split for final testing."
        ),
    )
    parser.add_argument(
        "--train-val-fraction",
        type=float,
        default=0.1,
        help=(
            "Fraction of the training split to hold out when "
            "--validation-source=train_split."
        ),
    )
    parser.add_argument(
        "--train-val-samples",
        type=int,
        default=0,
        help=(
            "Fixed training-split validation size; overrides "
            "--train-val-fraction when positive."
        ),
    )
    parser.add_argument(
        "--eval-test-after-training",
        action="store_true",
        help=(
            "Evaluate the reserved dataset validation/test split once after "
            "training completes."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use only already-cached datasets; fail fast instead of contacting remote hosts.",
    )
    parser.add_argument(
        "--download",
        action="store_false",
        dest="no_download",
        default=False,
        help="Allow dataset downloads when an entry point defaults to --no-download.",
    )
    parser.add_argument("--checkpoint-out", default="checkpoints/tpsapu_mnist.pt")
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--resume", default="")
    return parser.parse_args()


def parse_taus(value: str) -> tuple[float, ...]:
    taus = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not taus:
        raise ValueError("--taus must contain at least one numeric value.")
    return taus


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def optimizer_to_device(optimizer: optim.Optimizer, device: torch.device) -> None:
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if isinstance(value, torch.Tensor):
                optimizer_state[key] = value.to(device)


def load_training_checkpoint(path: str) -> dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def merge_resume_args(
    cli_args: argparse.Namespace,
    checkpoint: dict[str, object] | None,
) -> argparse.Namespace:
    if checkpoint is None or not isinstance(checkpoint.get("args"), dict):
        return cli_args

    saved = vars(cli_args).copy()
    saved.update(checkpoint["args"])
    cli = vars(cli_args)
    operational_overrides = {
        "checkpoint_out",
        "data_dir",
        "log_dir",
        "log_batches",
        "no_download",
        "num_workers",
        "eval_test_after_training",
        "resume",
        "train_samples",
        "test_samples",
    }
    for key in operational_overrides:
        saved[key] = cli[key]
    return argparse.Namespace(**saved)


class Tee:
    """Mirror writes to a stream and a log file."""

    def __init__(self, stream, path: Path, mode: str) -> None:
        self.stream = stream
        self.file = path.open(mode, buffering=1, encoding="utf-8")

    def write(self, message: str) -> int:
        self.stream.write(message)
        self.file.write(message)
        return len(message)

    def flush(self) -> None:
        self.stream.flush()
        self.file.flush()


class TrainingLogger:
    """Persist run configuration, console output, and epoch metrics."""

    fieldnames = [
        "timestamp",
        "global_epoch",
        "phase",
        "phase_epoch",
        "phase_total_epochs",
        "completed_train_epochs",
        "completed_prune_epochs",
        "lr",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "best_val_acc",
        "best_epoch",
        "sparsity",
        "target_sparsity",
        "neuron_sparsity",
        "l2_lambda",
    ]

    def __init__(self, args: argparse.Namespace, *, append: bool) -> None:
        self.log_dir = training_log_dir(args)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        self.console_path = self.log_dir / "train.log"
        self.jsonl_path = self.log_dir / "metrics.jsonl"
        self.csv_path = self.log_dir / "metrics.csv"
        self.args_path = self.log_dir / "args.json"
        if not append:
            self.jsonl_path.write_text("", encoding="utf-8")
            self.csv_path.write_text("", encoding="utf-8")
        self._csv_header_written = append and self.csv_path.exists()

        sys.stdout = Tee(sys.stdout, self.console_path, mode)
        sys.stderr = Tee(sys.stderr, self.console_path, "a")
        self.args_path.write_text(
            json.dumps(vars(args), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Logging to: {self.log_dir}")

    def log_metric(self, record: dict[str, object]) -> None:
        record = {"timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z", **record}
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow({key: record.get(key, "") for key in self.fieldnames})


def training_log_dir(args: argparse.Namespace) -> Path:
    if args.log_dir:
        return Path(args.log_dir)
    if args.checkpoint_out:
        return Path(args.checkpoint_out).parent
    return Path("logs")


def subset_dataset(dataset, sample_count: int, *, seed: int):
    if sample_count <= 0:
        return dataset
    count = min(sample_count, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    return Subset(dataset, indices)


def train_validation_subsets(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    *,
    validation_fraction: float,
    validation_samples: int,
    seed: int,
) -> tuple[Subset, Subset]:
    if len(train_dataset) != len(validation_dataset):
        raise ValueError(
            "Train and validation-view datasets must have matching lengths."
        )
    length = len(train_dataset)
    if length < 2:
        raise ValueError(
            "At least two training samples are required for a train/validation split."
        )

    if validation_samples > 0:
        validation_count = validation_samples
    else:
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("--train-val-fraction must be between 0 and 1.")
        validation_count = int(round(length * validation_fraction))

    validation_count = min(max(1, validation_count), length - 1)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(length, generator=generator).tolist()
    validation_indices = indices[:validation_count]
    train_indices = indices[validation_count:]
    return Subset(train_dataset, train_indices), Subset(
        validation_dataset,
        validation_indices,
    )


def build_mnist_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_ds = datasets.MNIST(
        args.data_dir,
        train=True,
        download=not args.no_download,
        transform=transform,
    )
    test_ds = datasets.MNIST(
        args.data_dir,
        train=False,
        download=not args.no_download,
        transform=transform,
    )

    train_ds = subset_dataset(train_ds, args.train_samples, seed=args.seed)
    test_ds = subset_dataset(test_ds, args.test_samples, seed=args.seed + 1)

    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_ds, shuffle=True, **common),
        DataLoader(test_ds, shuffle=False, **common),
    )


def build_cifar10_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(args.image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ]
    )
    train_ds = datasets.CIFAR10(
        args.data_dir,
        train=True,
        download=not args.no_download,
        transform=train_transform,
    )
    test_ds = datasets.CIFAR10(
        args.data_dir,
        train=False,
        download=not args.no_download,
        transform=test_transform,
    )

    train_ds = subset_dataset(train_ds, args.train_samples, seed=args.seed)
    test_ds = subset_dataset(test_ds, args.test_samples, seed=args.seed + 1)

    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_ds, shuffle=True, **common),
        DataLoader(test_ds, shuffle=False, **common),
    )


class HuggingFaceImageDataset(Dataset):
    """Thin Torch dataset wrapper for Hugging Face image-classification splits."""

    def __init__(self, hf_dataset, transform: Callable | None = None) -> None:
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        item = self.hf_dataset[index]
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(item["label"])


def load_hf_split(
    dataset_name: str,
    *,
    split: str,
    cache_dir: str,
    no_download: bool,
):
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as exc:
        raise ImportError(
            "Tiny ImageNet training requires the Hugging Face datasets package. "
            "Install it with `pip install datasets`."
        ) from exc

    mode = "local cache only" if no_download else "download allowed"
    print(f"Loading Hugging Face dataset split: {dataset_name}/{split} ({mode})")
    kwargs = {
        "cache_dir": cache_dir,
        "split": split,
    }
    if no_download:
        kwargs["download_config"] = DownloadConfig(local_files_only=True)
        kwargs["download_mode"] = "reuse_dataset_if_exists"
    try:
        return load_dataset(dataset_name, **kwargs)
    except Exception as exc:
        if not no_download:
            raise
        raise RuntimeError(
            "Tiny ImageNet is not available in the local Hugging Face cache. "
            "Run once with `--download` to populate the cache, or set HF_TOKEN "
            "if the Hugging Face Hub is rate-limiting unauthenticated requests."
        ) from exc


def build_tiny_imagenet_loaders(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    dataset_name = "slegroux/tiny-imagenet-200-clean"
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ]
    )
    val_resize = max(args.image_size, int(round(args.image_size * 256 / 224)))
    val_transform = transforms.Compose(
        [
            transforms.Resize(val_resize),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ]
    )
    train_hf = load_hf_split(
        dataset_name,
        split="train",
        cache_dir=args.data_dir,
        no_download=args.no_download,
    )
    val_hf = load_hf_split(
        dataset_name,
        split="validation",
        cache_dir=args.data_dir,
        no_download=args.no_download,
    )
    if args.num_classes <= 0:
        args.num_classes = 200

    train_base_ds = HuggingFaceImageDataset(train_hf, transform=train_transform)
    train_validation_base_ds = HuggingFaceImageDataset(
        train_hf,
        transform=val_transform,
    )
    dataset_validation_ds = HuggingFaceImageDataset(val_hf, transform=val_transform)
    test_ds = None

    if args.validation_source == "train_split":
        train_ds, val_ds = train_validation_subsets(
            train_base_ds,
            train_validation_base_ds,
            validation_fraction=args.train_val_fraction,
            validation_samples=args.train_val_samples,
            seed=args.seed + 17,
        )
        train_ds = subset_dataset(train_ds, args.train_samples, seed=args.seed)
        test_ds = subset_dataset(
            dataset_validation_ds,
            args.test_samples,
            seed=args.seed + 1,
        )
    else:
        train_ds = subset_dataset(train_base_ds, args.train_samples, seed=args.seed)
        val_ds = subset_dataset(
            dataset_validation_ds,
            args.test_samples,
            seed=args.seed + 1,
        )

    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_ds, shuffle=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
        None if test_ds is None else DataLoader(test_ds, shuffle=False, **common),
    )


class ImageNetLOCValDataset(Dataset):
    """Validation dataset for flat ILSVRC LOC image directories."""

    def __init__(
        self,
        val_root: Path,
        solution_csv: Path,
        class_to_idx: dict[str, int],
        transform: Callable | None = None,
    ) -> None:
        self.val_root = val_root
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        with solution_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_id = row["ImageId"]
                synset = row["PredictionString"].split()[0]
                if synset not in class_to_idx:
                    raise ValueError(
                        f"Validation class {synset} from {solution_csv} "
                        "does not exist in the ImageNet class mapping."
                    )
                self.samples.append((val_root / f"{image_id}.JPEG", class_to_idx[synset]))
        if not self.samples:
            raise ValueError(f"No validation samples found in {solution_csv}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, target = self.samples[index]
        image = default_loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def load_imagenet_class_to_idx(mapping_path: Path = Path("LOC_synset_mapping.txt")) -> dict[str, int]:
    if not mapping_path.is_file():
        return {}
    class_to_idx: dict[str, int] = {}
    with mapping_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            synset = stripped.split(maxsplit=1)[0]
            class_to_idx[synset] = index
    return class_to_idx


def remap_imagenet_train_targets(
    train_ds: datasets.ImageFolder,
    class_to_idx: dict[str, int],
) -> None:
    local_to_global: dict[int, int] = {}
    for class_name, local_index in train_ds.class_to_idx.items():
        if class_name not in class_to_idx:
            raise ValueError(
                f"Training class folder {class_name} does not exist in "
                "LOC_synset_mapping.txt."
            )
        local_to_global[local_index] = class_to_idx[class_name]
    train_ds.target_transform = local_to_global.__getitem__


def build_imagenet_val_dataset(
    val_root: Path,
    transform: Callable,
    class_to_idx: dict[str, int],
) -> Dataset:
    class_dirs = [path for path in val_root.iterdir() if path.is_dir()]
    if class_dirs:
        return datasets.ImageFolder(val_root, transform=transform)
    solution_csv = Path("LOC_val_solution.csv")
    if not solution_csv.is_file():
        raise FileNotFoundError(
            "ImageNet validation is a flat LOC directory, so LOC_val_solution.csv "
            f"is required next to the training script. Missing: {solution_csv}"
        )
    return ImageNetLOCValDataset(
        val_root=val_root,
        solution_csv=solution_csv,
        class_to_idx=class_to_idx,
        transform=transform,
    )


def build_imagenet_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    data_root = Path(args.data_dir)
    train_root = data_root / "train"
    val_root = data_root / "val"
    if not train_root.is_dir() or not val_root.is_dir():
        raise FileNotFoundError(
            "ImageNet expects train and val directories under "
            f"{train_root} and {val_root} directories."
        )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ]
    )
    val_resize = max(args.image_size, int(round(args.image_size * 256 / 224)))
    val_transform = transforms.Compose(
        [
            transforms.Resize(val_resize),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ]
    )
    train_ds = datasets.ImageFolder(train_root, transform=train_transform)
    class_to_idx = load_imagenet_class_to_idx() or train_ds.class_to_idx
    if class_to_idx is not train_ds.class_to_idx:
        remap_imagenet_train_targets(train_ds, class_to_idx)
    val_ds = build_imagenet_val_dataset(
        val_root,
        val_transform,
        class_to_idx,
    )
    if args.num_classes <= 0:
        args.num_classes = len(class_to_idx)

    train_ds = subset_dataset(train_ds, args.train_samples, seed=args.seed)
    val_ds = subset_dataset(val_ds, args.test_samples, seed=args.seed + 1)

    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_ds, shuffle=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
    )


def build_classification_loaders(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    if args.dataset == "mnist":
        train_loader, val_loader = build_mnist_loaders(args)
        return train_loader, val_loader, None
    if args.dataset == "cifar10":
        train_loader, val_loader = build_cifar10_loaders(args)
        return train_loader, val_loader, None
    if args.dataset == "tiny_imagenet":
        return build_tiny_imagenet_loaders(args)
    if args.dataset == "imagenet":
        train_loader, val_loader = build_imagenet_loaders(args)
        return train_loader, val_loader, None
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def build_encoder(args: argparse.Namespace) -> nn.Module:
    builders: dict[str, Callable[[argparse.Namespace], nn.Module]] = {
        "linear_patch": lambda cfg: LinearPatchEncoder(
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            in_channels=cfg.in_channels,
            embed_dim=cfg.embed_dim,
            dropout=cfg.encoder_dropout,
        ),
        "patch": lambda cfg: LinearPatchEncoder(
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            in_channels=cfg.in_channels,
            embed_dim=cfg.embed_dim,
            dropout=cfg.encoder_dropout,
        ),
        "mlp_patch": lambda cfg: MLPPatchEncoder(
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            in_channels=cfg.in_channels,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.encoder_hidden_dim or None,
            dropout=cfg.encoder_dropout,
        ),
        "lif_2x2": lambda cfg: LIF2x2Encoder(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            embed_dim=cfg.embed_dim,
            white_threshold=cfg.lif_white_threshold,
        ),
        "cnn2": lambda cfg: TinyCNN2Encoder(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            embed_dim=cfg.embed_dim,
            hidden_channels=cfg.cnn_channels,
            dropout=cfg.encoder_dropout,
        ),
        "cnn3": lambda cfg: TinyCNN3Encoder(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            embed_dim=cfg.embed_dim,
            hidden_channels=cfg.cnn_channels,
            dropout=cfg.encoder_dropout,
        ),
        "res_cnn": lambda cfg: ResidualTinyCNNEncoder(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            embed_dim=cfg.embed_dim,
            hidden_channels=cfg.cnn_channels,
            dropout=cfg.encoder_dropout,
        ),
        "rows": lambda cfg: MNISTRowEncoder(
            image_size=cfg.image_size,
            in_channels=cfg.in_channels,
            embed_dim=cfg.embed_dim,
            dropout=cfg.encoder_dropout,
        ),
    }
    return builders[args.encoder](args)


def build_decoder(args: argparse.Namespace, input_dim: int) -> nn.Module:
    def transformer_kwargs(cfg: argparse.Namespace) -> dict[str, object]:
        return {
            "model_dim": cfg.decoder_hidden_dim,
            "num_heads": cfg.decoder_transformer_heads,
            "num_layers": cfg.decoder_transformer_layers,
            "ff_multiplier": cfg.decoder_transformer_ff_mult,
            "dropout": cfg.decoder_dropout,
            "max_steps": cfg.decoder_max_steps,
        }

    builders: dict[str, Callable[[argparse.Namespace, int], nn.Module]] = {
        "linear": lambda cfg, dim: LinearDecoder(input_dim=dim, num_classes=cfg.num_classes),
        "membrane_mlp": lambda cfg, dim: MembraneMLPDecoder(
            input_dim=dim,
            num_classes=cfg.num_classes,
            hidden_dim=cfg.decoder_hidden_dim,
            dropout=cfg.decoder_dropout,
        ),
        "mlp": lambda cfg, dim: MembraneMLPDecoder(
            input_dim=dim,
            num_classes=cfg.num_classes,
            hidden_dim=cfg.decoder_hidden_dim,
            dropout=cfg.decoder_dropout,
        ),
        "spike_mlp": lambda cfg, dim: SpikeMLPDecoder(
            input_dim=dim,
            num_classes=cfg.num_classes,
            hidden_dim=cfg.decoder_hidden_dim,
            dropout=cfg.decoder_dropout,
        ),
        "both_mlp": lambda cfg, dim: MembraneSpikeMLPDecoder(
            input_dim=dim * 2,
            num_classes=cfg.num_classes,
            hidden_dim=cfg.decoder_hidden_dim,
            dropout=cfg.decoder_dropout,
        ),
        "all_state_mlp": lambda cfg, dim: AllStateMLPDecoder(
            input_dim=dim * 4,
            num_classes=cfg.num_classes,
            hidden_dim=cfg.decoder_hidden_dim,
            dropout=cfg.decoder_dropout,
        ),
        "membrane_transformer": lambda cfg, dim: MembraneTransformerDecoder(
            input_dim=dim,
            num_classes=cfg.num_classes,
            **transformer_kwargs(cfg),
        ),
        "spike_transformer": lambda cfg, dim: SpikeTransformerDecoder(
            input_dim=dim,
            num_classes=cfg.num_classes,
            **transformer_kwargs(cfg),
        ),
        "both_transformer": lambda cfg, dim: MembraneSpikeTransformerDecoder(
            input_dim=dim * 2,
            num_classes=cfg.num_classes,
            **transformer_kwargs(cfg),
        ),
        "all_state_transformer": lambda cfg, dim: AllStateTransformerDecoder(
            input_dim=dim * 4,
            num_classes=cfg.num_classes,
            **transformer_kwargs(cfg),
        ),
        "lif_count": lambda cfg, dim: ClassLIFSpikeCountDecoder(
            input_dim=dim * 2,
            num_classes=cfg.num_classes,
        ),
    }
    return builders[args.decoder](args, input_dim)


def build_model(args: argparse.Namespace) -> EncoderBackboneDecoder:
    taus = parse_taus(args.taus)
    encoder = build_encoder(args)
    backbone = TPSAPUBackbone(
        input_dim=args.embed_dim,
        reservoir_dim=args.reservoir_dim,
        taus=taus,
        recurrent_drop_p=args.recurrent_drop,
        input_hidden_dim=args.input_hidden_dim or None,
    )
    decoder = build_decoder(args, backbone.out_features)
    return EncoderBackboneDecoder(
        encoder=encoder,
        backbone=backbone,
        decoder=decoder,
        pooling=args.pooling,
    )


def lr_multiplier(
    epoch_index: int,
    *,
    warmup_epochs: int,
    cosine_epochs: int,
    cycles: float,
    min_lr_ratio: float,
) -> float:
    if warmup_epochs > 0 and epoch_index < warmup_epochs:
        return float(epoch_index + 1) / warmup_epochs

    if cosine_epochs <= 0:
        return 1.0

    cosine_index = max(0, epoch_index - warmup_epochs)
    denominator = max(1, cosine_epochs - 1)
    progress = min(1.0, cosine_index / denominator)
    cosine = 0.5 * (1.0 + math.cos(math.pi * cycles * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def cosine_only_multiplier(
    epoch_index: int,
    *,
    total_epochs: int,
    cycles: float,
    min_lr_ratio: float,
) -> float:
    if total_epochs <= 0:
        return 1.0
    denominator = max(1, total_epochs - 1)
    progress = min(1.0, epoch_index / denominator)
    cosine = 0.5 * (1.0 + math.cos(math.pi * cycles * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def set_optimizer_lr(optimizer: optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def shared_recurrent_l2(model: EncoderBackboneDecoder) -> torch.Tensor:
    return model.backbone.shared_recurrent.weight.pow(2).sum()


def l2_prune_lambda_for_epoch(
    epoch_index: int,
    *,
    start_lambda: float,
    growth_epochs: int,
) -> float:
    if start_lambda < 0:
        raise ValueError("start_lambda must be non-negative.")
    if growth_epochs <= 0:
        raise ValueError("growth_epochs must be positive.")
    return start_lambda * (10 ** (epoch_index // growth_epochs))


def target_sparsity_for_epoch(
    epoch_index: int,
    *,
    start_sparsity: float,
    target_sparsity: float,
    ramp_epochs: int,
    total_epochs: int,
) -> float:
    if not 0.0 <= start_sparsity < 1.0:
        raise ValueError("start_sparsity must be in [0.0, 1.0).")
    if not 0.0 <= target_sparsity < 1.0:
        raise ValueError("target_sparsity must be in [0.0, 1.0).")
    if target_sparsity < start_sparsity:
        raise ValueError("target_sparsity must be >= start_sparsity.")
    if total_epochs <= 0:
        return target_sparsity

    effective_ramp = min(max(1, ramp_epochs), total_epochs)
    progress = min(1.0, float(epoch_index + 1) / effective_ramp)
    smooth = progress * progress * (3.0 - 2.0 * progress)
    return start_sparsity + (target_sparsity - start_sparsity) * smooth


def effective_prune_ramp_epochs(args: argparse.Namespace) -> int:
    if args.prune_ramp_epochs > 0:
        return min(args.prune_ramp_epochs, args.prune_epochs)
    return max(1, args.prune_epochs - max(0, args.prune_stabilize_epochs))


def run_epoch(
    model: EncoderBackboneDecoder,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: optim.Optimizer | None = None,
    grad_clip: float = 1.0,
    l2_prune_lambda: float = 0.0,
    pruner: RecurrentMagnitudePruner | None = None,
    neuron_pruner: NeuronActivityPruner | None = None,
    phase: str = "",
    epoch: int = 0,
    total_epochs: int = 0,
    lr: float | None = None,
    log_batches: int = 0,
) -> tuple[float, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad() if train else torch.no_grad()

    total_batches = len(loader)
    mode = "train" if train else "val"

    with context:
        for batch_index, (images, targets) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            model.backbone.reset_state()
            logits = model(images)
            if neuron_pruner is not None:
                neuron_pruner.record(model.backbone.last_states)
            task_loss = criterion(logits, targets)
            loss = task_loss
            if train and l2_prune_lambda > 0:
                loss = loss + l2_prune_lambda * shared_recurrent_l2(model)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                if pruner is not None:
                    pruner.apply()
                if neuron_pruner is not None:
                    neuron_pruner.apply()
            model.backbone.reset_state()

            batch_size = targets.size(0)
            batch_loss = task_loss.item()
            batch_correct = (logits.argmax(dim=1) == targets).sum().item()
            total_loss += batch_loss * batch_size
            correct += batch_correct
            total += batch_size
            if log_batches > 0 and (
                batch_index % log_batches == 0 or batch_index == total_batches
            ):
                batch_acc = batch_correct / batch_size
                running_loss = total_loss / total
                running_acc = correct / total
                batch_width = len(str(total_batches))
                prefix = phase or mode
                epoch_part = (
                    f" {epoch:03d}/{total_epochs:03d}"
                    if epoch > 0 and total_epochs > 0
                    else ""
                )
                lr_part = "" if lr is None else f" | lr {lr:.3e}"
                l2_part = (
                    ""
                    if l2_prune_lambda <= 0 or not train
                    else f" | l2 lambda {l2_prune_lambda:.1e}"
                )
                print(
                    f"{prefix}{epoch_part} {mode} batch "
                    f"{batch_index:0{batch_width}d}/{total_batches} | "
                    f"batch loss {batch_loss:.4f} acc {batch_acc:.2%} | "
                    f"running loss {running_loss:.4f} acc {running_acc:.2%}"
                    f"{lr_part}{l2_part}"
                )

    return total_loss / total, correct / total


def print_epoch(
    *,
    phase: str,
    epoch: int,
    total_epochs: int,
    lr: float,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    sparsity: float | None = None,
    target_sparsity: float | None = None,
    neuron_sparsity: float | None = None,
    l2_lambda: float | None = None,
) -> None:
    suffix = "" if sparsity is None else f" | recurrent sparsity {sparsity:.2%}"
    if target_sparsity is not None:
        suffix += f" target {target_sparsity:.2%}"
    if neuron_sparsity is not None:
        suffix += f" | neuron sparsity {neuron_sparsity:.2%}"
    if l2_lambda is not None:
        suffix = f" | l2 lambda {l2_lambda:.1e}" + suffix
    print(
        f"{phase} {epoch:03d}/{total_epochs:03d} | "
        f"lr {lr:.3e} | "
        f"train loss {train_loss:.4f} acc {train_acc:.2%} | "
        f"val loss {val_loss:.4f} acc {val_acc:.2%}"
        f"{suffix}"
    )


def save_checkpoint(
    path: str,
    *,
    model: EncoderBackboneDecoder,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    completed_train_epochs: int,
    completed_prune_epochs: int,
    phase: str,
    val_loss: float,
    val_acc: float,
    best_val_acc: float,
    best_epoch: int,
    pruner: RecurrentMagnitudePruner | None = None,
    neuron_pruner: NeuronActivityPruner | None = None,
) -> None:
    if not path:
        return
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 2,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "completed_train_epochs": completed_train_epochs,
            "completed_prune_epochs": completed_prune_epochs,
            "epoch": completed_train_epochs + completed_prune_epochs,
            "phase": phase,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "log_dir": str(training_log_dir(args)),
            "pruner": None
            if pruner is None
            else {
                "threshold": pruner.threshold,
                "target_sparsity": pruner.target_sparsity,
                "mask": pruner.mask.detach().cpu(),
            },
            "neuron_pruner": None
            if neuron_pruner is None
            else {
                "spike_threshold": neuron_pruner.spike_threshold,
                "membrane_std_threshold": neuron_pruner.membrane_std_threshold,
                "target_sparsity": neuron_pruner.target_sparsity,
                "min_keep": neuron_pruner.min_keep,
                "mask": neuron_pruner.mask.detach().cpu(),
            },
            "rng_state": rng_state(),
        },
        checkpoint_path,
    )


def best_checkpoint_path(path: str) -> str:
    checkpoint_path = Path(path)
    return str(checkpoint_path.with_name("best.pt"))


def print_checkpoints(path: str, best_epoch: int, best_val_acc: float) -> None:
    if path:
        print(f"Latest checkpoint: {path}")
    if path and best_epoch > 0:
        print(
            f"Best checkpoint: {best_checkpoint_path(path)} "
            f"(epoch {best_epoch}, val acc {best_val_acc:.2%})"
        )


def print_log_paths(args: argparse.Namespace) -> None:
    log_dir = training_log_dir(args)
    print(f"Training log: {log_dir / 'train.log'}")
    print(f"Metrics JSONL: {log_dir / 'metrics.jsonl'}")
    print(f"Metrics CSV: {log_dir / 'metrics.csv'}")
    print(f"Run args: {log_dir / 'args.json'}")


def maybe_run_final_test(
    *,
    args: argparse.Namespace,
    model: EncoderBackboneDecoder,
    test_loader: DataLoader | None,
    criterion: nn.Module,
    device: torch.device,
) -> None:
    if test_loader is None:
        return
    if not args.eval_test_after_training:
        print(
            "Reserved test split was not evaluated. "
            "Pass --eval-test-after-training when you are ready for one final test."
        )
        return

    if args.checkpoint_out:
        best_path = Path(best_checkpoint_path(args.checkpoint_out))
        if best_path.is_file():
            checkpoint = load_training_checkpoint(str(best_path))
            load_result = model.load_state_dict(
                checkpoint["model_state"],
                strict=False,
            )
            if load_result.missing_keys:
                print(
                    "Best checkpoint missing keys initialized from current model: "
                    f"{load_result.missing_keys}"
                )
            if load_result.unexpected_keys:
                print(
                    "Best checkpoint ignored unexpected keys: "
                    f"{load_result.unexpected_keys}"
                )
            model.to(device)
            print(f"Loaded best checkpoint for final test: {best_path}")

    test_loss, test_acc = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        phase="test",
    )
    print(f"final test | loss {test_loss:.4f} acc {test_acc:.2%}")


def count_parameters(module: nn.Module, *, trainable_only: bool) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad or not trainable_only
    )


def print_parameter_summary(model: EncoderBackboneDecoder) -> None:
    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model, trainable_only=False)
    frozen = total - trainable

    print("Parameter summary:")
    print(f"  trainable : {trainable:,}")
    print(f"  frozen    : {frozen:,}")
    print(f"  total     : {total:,}")
    print("  by module:")
    for name, module in (
        ("encoder", model.encoder),
        ("backbone", model.backbone),
        ("decoder", model.decoder),
    ):
        module_trainable = count_parameters(module, trainable_only=True)
        module_total = count_parameters(module, trainable_only=False)
        print(
            f"    {name:<8} trainable {module_trainable:>12,} | "
            f"total {module_total:>12,}"
        )


def main() -> None:
    args = parse_args()
    if args.neuron_prune_min_keep <= 0:
        raise ValueError("--neuron-prune-min-keep must be positive.")
    if args.log_batches < 0:
        raise ValueError("--log-batches must be non-negative.")
    if args.train_val_samples < 0:
        raise ValueError("--train-val-samples must be non-negative.")
    resume_checkpoint = load_training_checkpoint(args.resume) if args.resume else None
    args = merge_resume_args(args, resume_checkpoint)
    if args.validation_source == "train_split" and args.dataset != "tiny_imagenet":
        raise ValueError(
            "--validation-source=train_split is currently supported for Tiny ImageNet only."
        )
    set_seed(args.seed)
    logger = TrainingLogger(args, append=bool(args.resume))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = build_classification_loaders(args)
    model = build_model(args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    completed_train_epochs = 0
    completed_prune_epochs = 0
    best_val_acc = -1.0
    best_epoch = 0
    pruner = None
    neuron_pruner = None

    if resume_checkpoint is not None:
        load_result = model.load_state_dict(resume_checkpoint["model_state"], strict=False)
        if load_result.missing_keys:
            print(f"Checkpoint missing keys initialized from defaults: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"Checkpoint ignored unexpected keys: {load_result.unexpected_keys}")
        if "optimizer_state" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
            optimizer_to_device(optimizer, device)
        completed_train_epochs = int(resume_checkpoint.get("completed_train_epochs", 0))
        completed_prune_epochs = int(resume_checkpoint.get("completed_prune_epochs", 0))
        best_val_acc = float(resume_checkpoint.get("best_val_acc", -1.0))
        best_epoch = int(resume_checkpoint.get("best_epoch", 0))
        restore_rng_state(resume_checkpoint.get("rng_state"))
        print(f"Resumed checkpoint: {args.resume}")

    print_parameter_summary(model)

    train_epochs = args.warmup_epochs + args.cosine_epochs
    prune_ramp_epochs = effective_prune_ramp_epochs(args)
    print(f"Device: {device}")
    print(
        "Pipeline: "
        f"dataset={args.dataset}, encoder={args.encoder}, backbone=tpsapu, "
        f"decoder={args.decoder}, pooling={args.pooling}"
    )
    if test_loader is None:
        print(f"Validation source: {args.validation_source}")
    else:
        print(
            "Validation source: train_split; "
            "dataset validation/test split reserved for final test"
        )
    print(
        "Schedule: "
        f"warmup={args.warmup_epochs}, cosine={args.cosine_epochs}, "
        f"cosine_cycles={args.cosine_cycles}, prune={args.prune_epochs}, "
        f"prune_cycles={args.prune_cycles}, "
        f"l2_prune_start={args.l2_prune_start_lambda:.1e}, "
        f"l2_prune_growth_epochs={args.l2_prune_growth_epochs}, "
        f"target_sparsity={args.target_sparsity:.1%}, "
        f"prune_ramp_epochs={prune_ramp_epochs}, "
        f"prune_stabilize_epochs={args.prune_epochs - prune_ramp_epochs}"
    )
    if args.neuron_prune:
        print(
            "Neuron pruning: "
            f"target_sparsity={args.neuron_prune_target_sparsity:.1%}, "
            f"spike_threshold={args.neuron_spike_threshold:.1e}, "
            f"membrane_std_threshold={args.neuron_membrane_std_threshold:.1e}, "
            f"min_keep={args.neuron_prune_min_keep}"
        )

    for epoch_index in range(completed_train_epochs, train_epochs):
        phase = "warmup" if epoch_index < args.warmup_epochs else "cosine"
        multiplier = lr_multiplier(
            epoch_index,
            warmup_epochs=args.warmup_epochs,
            cosine_epochs=args.cosine_epochs,
            cycles=args.cosine_cycles,
            min_lr_ratio=args.min_lr_ratio,
        )
        lr = args.lr * multiplier
        set_optimizer_lr(optimizer, lr)

        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
            phase=phase,
            epoch=epoch_index + 1,
            total_epochs=train_epochs,
            lr=lr,
            log_batches=args.log_batches,
        )
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            phase=phase,
            epoch=epoch_index + 1,
            total_epochs=train_epochs,
            lr=lr,
            log_batches=args.log_batches,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch_index + 1
            if args.checkpoint_out:
                save_checkpoint(
                    best_checkpoint_path(args.checkpoint_out),
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    completed_train_epochs=epoch_index + 1,
                    completed_prune_epochs=0,
                    phase=phase,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    best_val_acc=best_val_acc,
                    best_epoch=best_epoch,
                    pruner=None,
                )
        print_epoch(
            phase=phase,
            epoch=epoch_index + 1,
            total_epochs=train_epochs,
            lr=lr,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
        )
        logger.log_metric(
            {
                "global_epoch": epoch_index + 1,
                "phase": phase,
                "phase_epoch": epoch_index + 1,
                "phase_total_epochs": train_epochs,
                "completed_train_epochs": epoch_index + 1,
                "completed_prune_epochs": 0,
                "lr": lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "sparsity": None,
                "target_sparsity": None,
                "neuron_sparsity": None,
                "l2_lambda": None,
            }
        )
        save_checkpoint(
            args.checkpoint_out,
            model=model,
            optimizer=optimizer,
            args=args,
            completed_train_epochs=epoch_index + 1,
            completed_prune_epochs=0,
            phase=phase,
            val_loss=val_loss,
            val_acc=val_acc,
            best_val_acc=best_val_acc,
            best_epoch=best_epoch,
            pruner=None,
        )

    if args.prune_epochs <= 0:
        maybe_run_final_test(
            args=args,
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
        )
        print_checkpoints(args.checkpoint_out, best_epoch, best_val_acc)
        print_log_paths(args)
        return

    if not args.keep_dropout_during_prune:
        model.backbone.set_recurrent_dropout(0.0)

    pruner = RecurrentMagnitudePruner(
        model.backbone.shared_recurrent.weight,
        threshold=args.prune_threshold,
        target_sparsity=args.target_sparsity,
    )
    if resume_checkpoint is not None and resume_checkpoint.get("pruner") is not None:
        saved_pruner = resume_checkpoint["pruner"]
        pruner.threshold = float(saved_pruner.get("threshold", pruner.threshold))
        pruner.target_sparsity = float(
            saved_pruner.get("target_sparsity", pruner.target_sparsity)
        )
        pruner.mask = saved_pruner["mask"].to(
            device=model.backbone.shared_recurrent.weight.device,
            dtype=torch.bool,
        )
        pruner.apply()
    if args.neuron_prune:
        neuron_pruner = NeuronActivityPruner(
            model.backbone,
            spike_threshold=args.neuron_spike_threshold,
            membrane_std_threshold=args.neuron_membrane_std_threshold,
            target_sparsity=args.neuron_prune_target_sparsity,
            min_keep=args.neuron_prune_min_keep,
        )
        if (
            resume_checkpoint is not None
            and resume_checkpoint.get("neuron_pruner") is not None
        ):
            saved_neuron_pruner = resume_checkpoint["neuron_pruner"]
            neuron_pruner.spike_threshold = float(
                saved_neuron_pruner.get(
                    "spike_threshold", neuron_pruner.spike_threshold
                )
            )
            neuron_pruner.membrane_std_threshold = float(
                saved_neuron_pruner.get(
                    "membrane_std_threshold",
                    neuron_pruner.membrane_std_threshold,
                )
            )
            neuron_pruner.target_sparsity = float(
                saved_neuron_pruner.get(
                    "target_sparsity", neuron_pruner.target_sparsity
                )
            )
            neuron_pruner.min_keep = int(
                saved_neuron_pruner.get("min_keep", neuron_pruner.min_keep)
            )
            neuron_pruner.mask = saved_neuron_pruner["mask"].to(
                device=model.backbone.neuron_mask.device,
                dtype=torch.bool,
            )
            neuron_pruner.apply()
    prune_base_lr = args.lr * args.prune_lr_scale

    for epoch_index in range(completed_prune_epochs, args.prune_epochs):
        if neuron_pruner is not None:
            neuron_pruner.reset_activity()
        l2_lambda = l2_prune_lambda_for_epoch(
            epoch_index,
            start_lambda=args.l2_prune_start_lambda,
            growth_epochs=args.l2_prune_growth_epochs,
        )
        current_target_sparsity = target_sparsity_for_epoch(
            epoch_index,
            start_sparsity=args.prune_start_sparsity,
            target_sparsity=args.target_sparsity,
            ramp_epochs=prune_ramp_epochs,
            total_epochs=args.prune_epochs,
        )
        multiplier = cosine_only_multiplier(
            epoch_index,
            total_epochs=args.prune_epochs,
            cycles=args.prune_cycles,
            min_lr_ratio=args.min_lr_ratio,
        )
        lr = prune_base_lr * multiplier
        set_optimizer_lr(optimizer, lr)

        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
            l2_prune_lambda=l2_lambda,
            pruner=pruner,
            neuron_pruner=neuron_pruner,
            phase="l2-prune",
            epoch=epoch_index + 1,
            total_epochs=args.prune_epochs,
            lr=lr,
            log_batches=args.log_batches,
        )
        pruner.update(current_target_sparsity)
        if neuron_pruner is not None:
            neuron_pruner.update()
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            phase="l2-prune",
            epoch=epoch_index + 1,
            total_epochs=args.prune_epochs,
            lr=lr,
            log_batches=args.log_batches,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = train_epochs + epoch_index + 1
            if args.checkpoint_out:
                save_checkpoint(
                    best_checkpoint_path(args.checkpoint_out),
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    completed_train_epochs=train_epochs,
                    completed_prune_epochs=epoch_index + 1,
                    phase="l2-prune",
                    val_loss=val_loss,
                    val_acc=val_acc,
                    best_val_acc=best_val_acc,
                    best_epoch=best_epoch,
                    pruner=pruner,
                    neuron_pruner=neuron_pruner,
                )
        sparsity = pruner.sparsity()
        neuron_sparsity = (
            None if neuron_pruner is None else neuron_pruner.sparsity()
        )
        print_epoch(
            phase="l2-prune",
            epoch=epoch_index + 1,
            total_epochs=args.prune_epochs,
            lr=lr,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            sparsity=sparsity,
            target_sparsity=current_target_sparsity,
            neuron_sparsity=neuron_sparsity,
            l2_lambda=l2_lambda,
        )
        logger.log_metric(
            {
                "global_epoch": train_epochs + epoch_index + 1,
                "phase": "l2-prune",
                "phase_epoch": epoch_index + 1,
                "phase_total_epochs": args.prune_epochs,
                "completed_train_epochs": train_epochs,
                "completed_prune_epochs": epoch_index + 1,
                "lr": lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "sparsity": sparsity,
                "target_sparsity": current_target_sparsity,
                "neuron_sparsity": neuron_sparsity,
                "l2_lambda": l2_lambda,
            }
        )
        save_checkpoint(
            args.checkpoint_out,
            model=model,
            optimizer=optimizer,
            args=args,
            completed_train_epochs=train_epochs,
            completed_prune_epochs=epoch_index + 1,
            phase="l2-prune",
            val_loss=val_loss,
            val_acc=val_acc,
            best_val_acc=best_val_acc,
            best_epoch=best_epoch,
            pruner=pruner,
            neuron_pruner=neuron_pruner,
        )

    maybe_run_final_test(
        args=args,
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
    )
    print_checkpoints(args.checkpoint_out, best_epoch, best_val_acc)
    print_log_paths(args)


if __name__ == "__main__":
    main()
