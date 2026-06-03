"""
Training pipeline for encoder -> decoder baseline experiments without TPSAPU.

Default schedule:
    5 warmup epochs
    35 cosine-annealing training epochs
    optional finetuning epochs using the pruning-stage LR schedule

Example:
    python train_no_backbone.py --encoder linear_patch --decoder membrane_mlp
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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

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
class EncoderDecoder(nn.Module):
    """Composable no-backbone model: raw input -> encoder -> decoder."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        *,
        pooling: str,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pooling = pooling
        self.last_states: dict[str, torch.Tensor] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(x)
        states = self.forward_states(tokens)
        sequence = self._state_sequence(states, self.decoder.input_state)
        features = sequence if self.decoder.needs_sequence else self._pool(sequence)
        return self.decoder(features)

    def forward_states(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Build decoder-compatible state views directly from encoder tokens."""

        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(1)
        if tokens.dim() != 3:
            raise ValueError(
                "No-backbone encoder output must have shape (batch, steps, features)."
            )

        membranes = tokens
        spikes = torch.sigmoid(tokens)
        dynamics = torch.cat(
            [membranes[:, :1, :], membranes[:, 1:, :] - membranes[:, :-1, :]],
            dim=1,
        )
        spike_history = spikes.cumsum(dim=1)
        states = {
            "membrane": membranes,
            "spike": spikes,
            "dynamics": dynamics,
            "spike_history": spike_history,
        }
        self.last_states = {key: value.detach() for key, value in states.items()}
        return states

    def _state_sequence(
        self,
        states: dict[str, torch.Tensor],
        state: str,
    ) -> torch.Tensor:
        if state == "membrane":
            return states["membrane"]
        if state == "spike":
            return states["spike"]
        if state == "both":
            return torch.cat([states["membrane"], states["spike"]], dim=-1)
        if state == "all":
            return torch.cat(
                [
                    states["membrane"],
                    states["spike"],
                    states["dynamics"],
                    states["spike_history"],
                ],
                dim=-1,
            )
        raise ValueError('state must be one of "membrane", "spike", "both", or "all".')

    def _pool(self, sequence: torch.Tensor) -> torch.Tensor:
        if self.pooling == "last":
            return sequence[:, -1, :]
        if self.pooling == "mean":
            return sequence.mean(dim=1)
        raise ValueError('pooling must be one of "last" or "mean".')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "imagenet"], default="mnist")
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
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--checkpoint-out", default="checkpoints/no_backbone_mnist.pt")
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
        "no_download",
        "num_workers",
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


def subset_dataset(dataset, sample_count: int):
    if sample_count <= 0:
        return dataset
    return Subset(dataset, range(min(sample_count, len(dataset))))


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

    train_ds = subset_dataset(train_ds, args.train_samples)
    test_ds = subset_dataset(test_ds, args.test_samples)

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

    train_ds = subset_dataset(train_ds, args.train_samples)
    test_ds = subset_dataset(test_ds, args.test_samples)

    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_ds, shuffle=True, **common),
        DataLoader(test_ds, shuffle=False, **common),
    )


def build_imagenet_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    data_root = Path(args.data_dir)
    train_root = data_root / "train"
    val_root = data_root / "val"
    if not train_root.is_dir() or not val_root.is_dir():
        raise FileNotFoundError(
            "ImageNet expects an ImageFolder layout with "
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
    val_ds = datasets.ImageFolder(val_root, transform=val_transform)
    if args.num_classes <= 0:
        args.num_classes = len(train_ds.classes)

    train_ds = subset_dataset(train_ds, args.train_samples)
    val_ds = subset_dataset(val_ds, args.test_samples)

    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_ds, shuffle=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
    )


def build_classification_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    if args.dataset == "mnist":
        return build_mnist_loaders(args)
    if args.dataset == "cifar10":
        return build_cifar10_loaders(args)
    if args.dataset == "imagenet":
        return build_imagenet_loaders(args)
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


def build_model(args: argparse.Namespace) -> EncoderDecoder:
    encoder = build_encoder(args)
    decoder = build_decoder(args, args.embed_dim)
    return EncoderDecoder(
        encoder=encoder,
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
    model: EncoderDecoder,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: optim.Optimizer | None = None,
    grad_clip: float = 1.0,
) -> tuple[float, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += batch_size

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
    model: EncoderDecoder,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    completed_train_epochs: int,
    completed_prune_epochs: int,
    phase: str,
    val_loss: float,
    val_acc: float,
    best_val_acc: float,
    best_epoch: int,
    pruner: object | None = None,
    neuron_pruner: object | None = None,
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
            "pruner": None,
            "neuron_pruner": None,
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


def count_parameters(module: nn.Module, *, trainable_only: bool) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad or not trainable_only
    )


def print_parameter_summary(model: EncoderDecoder) -> None:
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
    resume_checkpoint = load_training_checkpoint(args.resume) if args.resume else None
    args = merge_resume_args(args, resume_checkpoint)
    set_seed(args.seed)
    logger = TrainingLogger(args, append=bool(args.resume))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_classification_loaders(args)
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
    print(f"Device: {device}")
    print(
        "Pipeline: "
        f"dataset={args.dataset}, encoder={args.encoder}, backbone=none, "
        f"decoder={args.decoder}, pooling={args.pooling}"
    )
    print(
        "Schedule: "
        f"warmup={args.warmup_epochs}, cosine={args.cosine_epochs}, "
        f"cosine_cycles={args.cosine_cycles}, "
        f"finetune={args.prune_epochs}, finetune_cycles={args.prune_cycles}, "
        f"finetune_lr_scale={args.prune_lr_scale}"
    )
    if args.neuron_prune:
        print("Neuron pruning: ignored because the no-backbone baseline has no reservoir neurons.")

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
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
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
        print_checkpoints(args.checkpoint_out, best_epoch, best_val_acc)
        print_log_paths(args)
        return

    finetune_base_lr = args.lr * args.prune_lr_scale

    for epoch_index in range(completed_prune_epochs, args.prune_epochs):
        multiplier = cosine_only_multiplier(
            epoch_index,
            total_epochs=args.prune_epochs,
            cycles=args.prune_cycles,
            min_lr_ratio=args.min_lr_ratio,
        )
        lr = finetune_base_lr * multiplier
        set_optimizer_lr(optimizer, lr)

        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
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
                    phase="finetune",
                    val_loss=val_loss,
                    val_acc=val_acc,
                    best_val_acc=best_val_acc,
                    best_epoch=best_epoch,
                    pruner=None,
                    neuron_pruner=None,
                )
        print_epoch(
            phase="finetune",
            epoch=epoch_index + 1,
            total_epochs=args.prune_epochs,
            lr=lr,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
        )
        logger.log_metric(
            {
                "global_epoch": train_epochs + epoch_index + 1,
                "phase": "finetune",
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
            completed_train_epochs=train_epochs,
            completed_prune_epochs=epoch_index + 1,
            phase="finetune",
            val_loss=val_loss,
            val_acc=val_acc,
            best_val_acc=best_val_acc,
            best_epoch=best_epoch,
            pruner=None,
            neuron_pruner=None,
        )

    print_checkpoints(args.checkpoint_out, best_epoch, best_val_acc)
    print_log_paths(args)


if __name__ == "__main__":
    main()
