"""Train ResNet-18 from scratch on Tiny ImageNet.

This script intentionally reuses ``train_pipeline.build_classification_loaders``
so the Tiny ImageNet train/internal-validation split is identical to the SAPU
Tiny ImageNet entry point when the same seed and split arguments are used.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

import train_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/tiny-imagenet-200-clean")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers; default 0 avoids Hugging Face worker crashes.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        default=True,
        help="Use only cached Hugging Face data by default.",
    )
    parser.add_argument(
        "--download",
        action="store_false",
        dest="no_download",
        help="Allow Hugging Face dataset download.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument(
        "--validation-source",
        choices=["train_split", "dataset"],
        default="train_split",
        help=(
            "Default train_split matches train_tiny_imagenet.py and reserves "
            "the official validation split for final testing."
        ),
    )
    parser.add_argument("--train-val-fraction", type=float, default=0.1)
    parser.add_argument("--train-val-samples", type=int, default=0)
    parser.add_argument("--eval-test-after-training", action="store_true")

    parser.add_argument(
        "--stem",
        choices=["tiny", "standard"],
        default="tiny",
        help="tiny uses a 3x3 stride-1 stem without maxpool for 64x64 images.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "sgd"],
        default="adamw",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--cosine-epochs", type=int, default=35)
    parser.add_argument("--cosine-cycles", type=float, default=1.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--log-batches", type=int, default=0)

    parser.add_argument(
        "--checkpoint-out",
        default="sweep_runs_tiny_imagenet/resnet18_scratch/latest.pt",
    )
    parser.add_argument(
        "--log-dir",
        default="sweep_runs_tiny_imagenet/resnet18_scratch",
    )
    parser.add_argument("--resume", default="")
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> nn.Module:
    model = models.resnet18(weights=None)
    if args.stem == "tiny":
        model.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    return model


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> optim.Optimizer:
    if args.optimizer == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    return optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


def best_checkpoint_path(path: str) -> str:
    checkpoint_path = Path(path)
    return str(checkpoint_path.with_name("best.pt"))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: optim.Optimizer | None = None,
    grad_clip: float = 0.0,
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
    total_batches = len(loader)
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for batch_index, (images, targets) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += batch_size

            if log_batches > 0 and (
                batch_index % log_batches == 0 or batch_index == total_batches
            ):
                batch_width = len(str(total_batches))
                epoch_part = (
                    f" {epoch:03d}/{total_epochs:03d}"
                    if epoch > 0 and total_epochs > 0
                    else ""
                )
                lr_part = "" if lr is None else f" | lr {lr:.3e}"
                print(
                    f"{phase}{epoch_part} batch "
                    f"{batch_index:0{batch_width}d}/{total_batches} | "
                    f"loss {total_loss / total:.4f} acc {correct / total:.2%}"
                    f"{lr_part}",
                    flush=True,
                )

    return total_loss / total, correct / total


def save_checkpoint(
    path: str,
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    completed_epochs: int,
    phase: str,
    val_loss: float,
    val_acc: float,
    best_val_acc: float,
    best_epoch: int,
) -> None:
    if not path:
        return
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "model": "resnet18_scratch",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "completed_epochs": completed_epochs,
            "epoch": completed_epochs,
            "phase": phase,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "rng_state": train_pipeline.rng_state(),
        },
        checkpoint_path,
    )


def log_metric(log_dir: Path, record: dict[str, object]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        **record,
    }
    with (log_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")

    csv_path = log_dir / "metrics.csv"
    fieldnames = [
        "timestamp",
        "epoch",
        "phase",
        "lr",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "best_val_acc",
        "best_epoch",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: record.get(key, "") for key in fieldnames})


def maybe_run_final_test(
    *,
    args: argparse.Namespace,
    model: nn.Module,
    test_loader,
    criterion: nn.Module,
    device: torch.device,
) -> None:
    if test_loader is None:
        return
    if not args.eval_test_after_training:
        print(
            "Reserved official validation split was not evaluated. "
            "Pass --eval-test-after-training for one final test.",
            flush=True,
        )
        return

    if args.checkpoint_out:
        best_path = Path(best_checkpoint_path(args.checkpoint_out))
        if best_path.is_file():
            checkpoint = train_pipeline.load_training_checkpoint(str(best_path))
            model.load_state_dict(checkpoint["model_state"])
            model.to(device)
            print(f"Loaded best checkpoint for final test: {best_path}", flush=True)

    test_loss, test_acc = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        phase="test",
    )
    print(f"final test | loss {test_loss:.4f} acc {test_acc:.2%}", flush=True)


def main() -> None:
    args = parse_args()
    if args.validation_source == "train_split" and args.train_val_samples < 0:
        raise ValueError("--train-val-samples must be non-negative.")
    if args.log_batches < 0:
        raise ValueError("--log-batches must be non-negative.")

    train_pipeline.set_seed(args.seed)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    loader_args = argparse.Namespace(
        dataset="tiny_imagenet",
        data_dir=args.data_dir,
        image_size=args.image_size,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        no_download=args.no_download,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        seed=args.seed,
        validation_source=args.validation_source,
        train_val_fraction=args.train_val_fraction,
        train_val_samples=args.train_val_samples,
    )
    train_loader, val_loader, test_loader = train_pipeline.build_classification_loaders(
        loader_args
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args, model)

    completed_epochs = 0
    best_val_acc = -1.0
    best_epoch = 0
    if args.resume:
        checkpoint = train_pipeline.load_training_checkpoint(args.resume)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        train_pipeline.optimizer_to_device(optimizer, device)
        completed_epochs = int(checkpoint.get("completed_epochs", 0))
        best_val_acc = float(checkpoint.get("best_val_acc", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        train_pipeline.restore_rng_state(checkpoint.get("rng_state"))
        print(f"Resumed checkpoint: {args.resume}", flush=True)

    total_epochs = args.warmup_epochs + args.cosine_epochs
    print(f"Device: {device}", flush=True)
    print(
        "Pipeline: dataset=tiny_imagenet, model=resnet18, "
        f"stem={args.stem}, weights=None, optimizer={args.optimizer}",
        flush=True,
    )
    print(f"Parameters: {count_parameters(model):,}", flush=True)
    if test_loader is None:
        print(f"Validation source: {args.validation_source}", flush=True)
    else:
        print(
            "Validation source: train_split; official validation split reserved",
            flush=True,
        )
    print(
        "Schedule: "
        f"warmup={args.warmup_epochs}, cosine={args.cosine_epochs}, "
        f"cosine_cycles={args.cosine_cycles}, min_lr_ratio={args.min_lr_ratio}",
        flush=True,
    )

    for epoch_index in range(completed_epochs, total_epochs):
        phase = "warmup" if epoch_index < args.warmup_epochs else "cosine"
        multiplier = train_pipeline.lr_multiplier(
            epoch_index,
            warmup_epochs=args.warmup_epochs,
            cosine_epochs=args.cosine_epochs,
            cycles=args.cosine_cycles,
            min_lr_ratio=args.min_lr_ratio,
        )
        lr = args.lr * multiplier
        train_pipeline.set_optimizer_lr(optimizer, lr)

        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
            phase=phase,
            epoch=epoch_index + 1,
            total_epochs=total_epochs,
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
            total_epochs=total_epochs,
            lr=lr,
            log_batches=args.log_batches,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch_index + 1
            save_checkpoint(
                best_checkpoint_path(args.checkpoint_out),
                model=model,
                optimizer=optimizer,
                args=args,
                completed_epochs=epoch_index + 1,
                phase=phase,
                val_loss=val_loss,
                val_acc=val_acc,
                best_val_acc=best_val_acc,
                best_epoch=best_epoch,
            )

        print(
            f"{phase} {epoch_index + 1:03d}/{total_epochs:03d} | "
            f"lr {lr:.3e} | "
            f"train loss {train_loss:.4f} acc {train_acc:.2%} | "
            f"val loss {val_loss:.4f} acc {val_acc:.2%}",
            flush=True,
        )
        log_metric(
            log_dir,
            {
                "epoch": epoch_index + 1,
                "phase": phase,
                "lr": lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
            },
        )
        save_checkpoint(
            args.checkpoint_out,
            model=model,
            optimizer=optimizer,
            args=args,
            completed_epochs=epoch_index + 1,
            phase=phase,
            val_loss=val_loss,
            val_acc=val_acc,
            best_val_acc=best_val_acc,
            best_epoch=best_epoch,
        )

    maybe_run_final_test(
        args=args,
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
    )
    print(f"Latest checkpoint: {args.checkpoint_out}", flush=True)
    print(
        f"Best checkpoint: {best_checkpoint_path(args.checkpoint_out)} "
        f"(epoch {best_epoch}, val acc {best_val_acc:.2%})",
        flush=True,
    )
    print(f"Training log: {log_dir / 'metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
