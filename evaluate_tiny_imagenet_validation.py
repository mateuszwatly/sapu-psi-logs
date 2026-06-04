"""Evaluate a TPSAPU Tiny ImageNet checkpoint on a full split.

The default split is the official Hugging Face validation split, which is the
unseen split when ``train_tiny_imagenet.py`` is used with its default
``--validation-source train_split``.

Example:
    python evaluate_tiny_imagenet_validation.py --checkpoint latest.pt

Outputs:
    metrics.json
    per_class_metrics.csv
    worst_classes.csv
    best_classes.csv
    confusion_matrix.csv
    confusion_pairs.csv
    misclassified_samples.csv
    confusion_matrix.png
    per_class_accuracy.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

import train_pipeline


TINY_IMAGENET_MEAN = (0.485, 0.456, 0.406)
TINY_IMAGENET_STD = (0.229, 0.224, 0.225)

ARCHITECTURE_KEYS = {
    "dataset",
    "backbone",
    "encoder",
    "decoder",
    "pooling",
    "image_size",
    "in_channels",
    "num_classes",
    "embed_dim",
    "reservoir_dim",
    "taus",
    "input_hidden_dim",
    "cross_rank",
    "cross_gain",
    "patch_size",
    "encoder_hidden_dim",
    "cnn_channels",
    "lif_white_threshold",
    "encoder_dropout",
    "decoder_hidden_dim",
    "decoder_dropout",
    "decoder_transformer_layers",
    "decoder_transformer_heads",
    "decoder_transformer_ff_mult",
    "decoder_max_steps",
    "recurrent_drop",
}


class IndexedHuggingFaceImageDataset(Dataset):
    """Hugging Face image split returning dataset index for diagnostics."""

    def __init__(self, hf_dataset, transform) -> None:
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        item = self.hf_dataset[index]
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image), int(item["label"]), index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="latest.pt")
    parser.add_argument("--data-dir", default="data/tiny-imagenet-200-clean")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--output-dir", default="eval_tiny_imagenet_validation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", default="1,5")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Evaluate only the first N split samples; 0 means full split.",
    )
    parser.add_argument("--max-confusion-pairs", type=int, default=200)
    parser.add_argument("--allow-random-weights", action="store_true")
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
    return parser.parse_args()


def parse_topk(value: str) -> list[int]:
    topk = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not topk or topk[0] <= 0:
        raise ValueError("--topk must contain positive integer values.")
    return topk


def default_tiny_imagenet_architecture() -> argparse.Namespace:
    return argparse.Namespace(
        dataset="tiny_imagenet",
        backbone="tpsapu",
        encoder="cnn3",
        decoder="membrane_transformer",
        pooling="last",
        image_size=64,
        in_channels=3,
        num_classes=200,
        embed_dim=128,
        reservoir_dim=64,
        taus="1.1,2.0,4.0,8.0,16.0,32.0,64.0,128.0",
        input_hidden_dim=0,
        cross_rank=16,
        cross_gain=0.1,
        patch_size=8,
        encoder_hidden_dim=0,
        cnn_channels=64,
        lif_white_threshold=0.6,
        encoder_dropout=0.05,
        decoder_hidden_dim=128,
        decoder_dropout=0.1,
        decoder_transformer_layers=2,
        decoder_transformer_heads=4,
        decoder_transformer_ff_mult=4.0,
        decoder_max_steps=256,
        recurrent_drop=0.1,
    )


def architecture_from_checkpoint(
    checkpoint: dict[str, object] | None,
) -> argparse.Namespace:
    architecture = vars(default_tiny_imagenet_architecture()).copy()
    if checkpoint is not None and isinstance(checkpoint.get("args"), dict):
        saved_args = checkpoint["args"]
        architecture.update(
            {key: saved_args[key] for key in ARCHITECTURE_KEYS if key in saved_args}
        )
    architecture["dataset"] = "tiny_imagenet"
    architecture["in_channels"] = 3
    architecture["num_classes"] = int(architecture.get("num_classes") or 200)
    return argparse.Namespace(**architecture)


def load_checkpoint(path: str, device: torch.device) -> dict[str, object] | None:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return None
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def validation_transform(image_size: int) -> transforms.Compose:
    resize = max(image_size, int(round(image_size * 256 / 224)))
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
        ]
    )


def load_human_names(mapping_path: Path = Path("LOC_synset_mapping.txt")) -> dict[str, str]:
    if not mapping_path.is_file():
        return {}
    names: dict[str, str] = {}
    with mapping_path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            synset, name = stripped.split(maxsplit=1)
            names[synset] = name
    return names


def class_display_names(label_names: list[str]) -> list[str]:
    human_by_synset = load_human_names()
    return [human_by_synset.get(synset, synset) for synset in label_names]


def build_loader(
    args: argparse.Namespace,
    architecture_args: argparse.Namespace,
) -> tuple[DataLoader, list[str], list[str]]:
    hf_dataset = train_pipeline.load_hf_split(
        "slegroux/tiny-imagenet-200-clean",
        split=args.split,
        cache_dir=args.data_dir,
        no_download=args.no_download,
    )
    label_names = list(hf_dataset.features["label"].names)
    display_names = class_display_names(label_names)
    dataset = IndexedHuggingFaceImageDataset(
        hf_dataset,
        transform=validation_transform(architecture_args.image_size),
    )
    if args.max_samples > 0:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, label_names, display_names


def update_confusion(
    confusion: np.ndarray,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    *,
    num_classes: int,
) -> None:
    encoded = targets.cpu() * num_classes + predictions.cpu()
    counts = torch.bincount(encoded, minlength=num_classes * num_classes)
    confusion += counts.reshape(num_classes, num_classes).numpy()


def topk_correct(topk_labels: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    k = min(k, topk_labels.size(1))
    return int((topk_labels[:, :k] == targets.view(-1, 1)).any(dim=1).sum().item())


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    num_classes: int,
    topk_values: list[int],
    label_names: list[str],
    display_names: list[str],
) -> dict[str, object]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    max_topk = min(max(topk_values), num_classes)

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    topk_totals = {k: 0 for k in topk_values}
    topk_class_correct = {
        k: np.zeros(num_classes, dtype=np.int64) for k in topk_values
    }
    total_loss = 0.0
    total = 0
    sample_rows: list[dict[str, object]] = []

    for batch_index, (images, targets, indices) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        top_probs, top_labels = torch.topk(probabilities, k=max_topk, dim=1)
        predictions = top_labels[:, 0]

        total_loss += float(criterion(logits, targets).item())
        batch_size = targets.size(0)
        total += batch_size
        update_confusion(
            confusion,
            targets,
            predictions,
            num_classes=num_classes,
        )
        for k in topk_values:
            k_correct = (top_labels[:, : min(k, top_labels.size(1))] == targets.view(-1, 1)).any(
                dim=1
            )
            topk_totals[k] += int(k_correct.sum().item())
            class_hits = torch.bincount(
                targets[k_correct].cpu(),
                minlength=num_classes,
            )
            topk_class_correct[k] += class_hits.numpy()

        true_probs = probabilities.gather(1, targets.view(-1, 1)).squeeze(1)
        correct_top1 = predictions.eq(targets)
        correct_top5 = (top_labels[:, : min(5, max_topk)] == targets.view(-1, 1)).any(
            dim=1
        )

        for row in range(batch_size):
            true_label = int(targets[row].item())
            pred_label = int(predictions[row].item())
            top_label_values = [int(value) for value in top_labels[row].cpu().tolist()]
            top_prob_values = [float(value) for value in top_probs[row].cpu().tolist()]
            sample_rows.append(
                {
                    "index": int(indices[row].item()),
                    "true_label": true_label,
                    "true_synset": label_names[true_label],
                    "true_name": display_names[true_label],
                    "pred_label": pred_label,
                    "pred_synset": label_names[pred_label],
                    "pred_name": display_names[pred_label],
                    "top1_confidence": float(top_probs[row, 0].item()),
                    "true_class_probability": float(true_probs[row].item()),
                    "top1_correct": bool(correct_top1[row].item()),
                    "top5_correct": bool(correct_top5[row].item()),
                    "top_labels": " ".join(str(value) for value in top_label_values),
                    "top_probabilities": " ".join(
                        f"{value:.8f}" for value in top_prob_values
                    ),
                }
            )

        if batch_index % 20 == 0:
            top1 = topk_totals.get(1, 0) / total
            print(
                f"batch {batch_index:04d}/{len(loader):04d} | "
                f"seen {total} | top1 {top1:.2%}",
                flush=True,
            )

    metrics = {
        "num_samples": total,
        "loss": total_loss / total,
        "topk": {str(k): topk_totals[k] / total for k in topk_values},
    }
    return {
        "metrics": metrics,
        "confusion": confusion,
        "topk_class_correct": topk_class_correct,
        "sample_rows": sample_rows,
    }


def per_class_rows(
    confusion: np.ndarray,
    *,
    topk_class_correct: dict[int, np.ndarray],
    label_names: list[str],
    display_names: list[str],
) -> list[dict[str, object]]:
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    correct = np.diag(confusion)
    rows = []
    for label in range(confusion.shape[0]):
        class_support = int(support[label])
        top1_correct = int(correct[label])
        false_negative = int(class_support - top1_correct)
        false_positive = int(predicted[label] - top1_correct)
        row_confusions = confusion[label].copy()
        row_confusions[label] = 0
        most_confused_pred = int(row_confusions.argmax())
        most_confused_count = int(row_confusions[most_confused_pred])
        rows.append(
            {
                "label": label,
                "synset": label_names[label],
                "name": display_names[label],
                "support": class_support,
                "top1_correct": top1_correct,
                "top1_accuracy": (
                    top1_correct / class_support if class_support > 0 else 0.0
                ),
                "top5_correct": (
                    int(topk_class_correct[5][label])
                    if 5 in topk_class_correct
                    else ""
                ),
                "top5_accuracy": (
                    float(topk_class_correct[5][label] / class_support)
                    if 5 in topk_class_correct and class_support > 0
                    else ""
                ),
                "missed_count": false_negative,
                "missed_rate": (
                    false_negative / class_support if class_support > 0 else 0.0
                ),
                "false_positive_count": false_positive,
                "precision_when_predicted": (
                    top1_correct / int(predicted[label]) if predicted[label] > 0 else 0.0
                ),
                "most_confused_pred_label": most_confused_pred,
                "most_confused_pred_synset": label_names[most_confused_pred],
                "most_confused_pred_name": display_names[most_confused_pred],
                "most_confused_pred_count": most_confused_count,
                "most_confused_pred_rate": (
                    most_confused_count / class_support if class_support > 0 else 0.0
                ),
            }
        )
    return rows


def confusion_pair_rows(
    confusion: np.ndarray,
    *,
    label_names: list[str],
    display_names: list[str],
    limit: int,
) -> list[dict[str, object]]:
    support = confusion.sum(axis=1)
    pairs = []
    for true_label in range(confusion.shape[0]):
        for pred_label in range(confusion.shape[1]):
            if true_label == pred_label:
                continue
            count = int(confusion[true_label, pred_label])
            if count <= 0:
                continue
            pairs.append(
                {
                    "true_label": true_label,
                    "true_synset": label_names[true_label],
                    "true_name": display_names[true_label],
                    "pred_label": pred_label,
                    "pred_synset": label_names[pred_label],
                    "pred_name": display_names[pred_label],
                    "count": count,
                    "true_class_rate": (
                        count / int(support[true_label])
                        if support[true_label] > 0
                        else 0.0
                    ),
                }
            )
    pairs.sort(key=lambda row: (row["count"], row["true_class_rate"]), reverse=True)
    return pairs[:limit]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_confusion_csv(
    path: Path,
    confusion: np.ndarray,
    *,
    label_names: list[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label", "true_synset", *range(confusion.shape[1])])
        for label, row in enumerate(confusion):
            writer.writerow([label, label_names[label], *row.tolist()])


def save_confusion_png(path: Path, confusion: np.ndarray) -> None:
    support = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion,
        support,
        out=np.zeros_like(confusion, dtype=np.float32),
        where=support > 0,
    )
    fig, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(normalized, cmap="magma", vmin=0.0, vmax=1.0)
    axis.set_title("Confusion matrix, row-normalized")
    axis.set_xlabel("predicted label")
    axis.set_ylabel("true label")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_per_class_accuracy_png(
    path: Path,
    rows: list[dict[str, object]],
    *,
    worst_count: int = 40,
) -> None:
    sorted_rows = sorted(rows, key=lambda row: row["top1_accuracy"])
    shown = sorted_rows[: min(worst_count, len(sorted_rows))]
    labels = [f"{row['label']} {row['name'].split(',')[0]}" for row in shown]
    values = [row["top1_accuracy"] for row in shown]

    fig, axis = plt.subplots(figsize=(12, max(6, len(shown) * 0.24)))
    y = np.arange(len(shown))
    axis.barh(y, values, color="#b84a62")
    axis.set_yticks(y)
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("top-1 accuracy")
    axis.set_title(f"Worst {len(shown)} classes")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_outputs(
    output_dir: Path,
    *,
    metrics: dict[str, object],
    confusion: np.ndarray,
    per_class: list[dict[str, object]],
    sample_rows: list[dict[str, object]],
    label_names: list[str],
    display_names: list[str],
    max_confusion_pairs: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    per_class_fields = [
        "label",
        "synset",
        "name",
        "support",
        "top1_correct",
        "top1_accuracy",
        "top5_correct",
        "top5_accuracy",
        "missed_count",
        "missed_rate",
        "false_positive_count",
        "precision_when_predicted",
        "most_confused_pred_label",
        "most_confused_pred_synset",
        "most_confused_pred_name",
        "most_confused_pred_count",
        "most_confused_pred_rate",
    ]
    write_csv(output_dir / "per_class_metrics.csv", per_class, per_class_fields)
    write_csv(
        output_dir / "worst_classes.csv",
        sorted(per_class, key=lambda row: row["top1_accuracy"])[:40],
        per_class_fields,
    )
    write_csv(
        output_dir / "best_classes.csv",
        sorted(per_class, key=lambda row: row["top1_accuracy"], reverse=True)[:40],
        per_class_fields,
    )

    pair_fields = [
        "true_label",
        "true_synset",
        "true_name",
        "pred_label",
        "pred_synset",
        "pred_name",
        "count",
        "true_class_rate",
    ]
    write_csv(
        output_dir / "confusion_pairs.csv",
        confusion_pair_rows(
            confusion,
            label_names=label_names,
            display_names=display_names,
            limit=max_confusion_pairs,
        ),
        pair_fields,
    )

    sample_fields = [
        "index",
        "true_label",
        "true_synset",
        "true_name",
        "pred_label",
        "pred_synset",
        "pred_name",
        "top1_confidence",
        "true_class_probability",
        "top1_correct",
        "top5_correct",
        "top_labels",
        "top_probabilities",
    ]
    missed_rows = [row for row in sample_rows if not row["top1_correct"]]
    write_csv(output_dir / "misclassified_samples.csv", missed_rows, sample_fields)
    write_csv(output_dir / "all_samples.csv", sample_rows, sample_fields)
    write_confusion_csv(output_dir / "confusion_matrix.csv", confusion, label_names=label_names)
    np.save(output_dir / "confusion_matrix.npy", confusion)
    save_confusion_png(output_dir / "confusion_matrix.png", confusion)
    save_per_class_accuracy_png(output_dir / "per_class_accuracy.png", per_class)


def main() -> None:
    args = parse_args()
    topk_values = parse_topk(args.topk)
    train_pipeline.set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint = load_checkpoint(args.checkpoint, device)
    if checkpoint is None and not args.allow_random_weights:
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. "
            "Pass --allow-random-weights only for a pipeline smoke test."
        )

    architecture_args = architecture_from_checkpoint(checkpoint)
    model = train_pipeline.build_model(architecture_args).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint loaded; evaluating random weights.")

    loader, label_names, display_names = build_loader(args, architecture_args)
    num_classes = len(label_names)
    if architecture_args.num_classes != num_classes:
        raise ValueError(
            f"Checkpoint expects num_classes={architecture_args.num_classes}, "
            f"but dataset has {num_classes} labels."
        )

    print(
        f"Evaluating split={args.split} samples={len(loader.dataset)} "
        f"classes={num_classes} device={device}",
        flush=True,
    )
    result = evaluate(
        model,
        loader,
        device=device,
        num_classes=num_classes,
        topk_values=topk_values,
        label_names=label_names,
        display_names=display_names,
    )
    metrics = {
        **result["metrics"],
        "checkpoint": args.checkpoint,
        "split": args.split,
        "architecture": vars(architecture_args),
    }
    confusion = result["confusion"]
    per_class = per_class_rows(
        confusion,
        topk_class_correct=result["topk_class_correct"],
        label_names=label_names,
        display_names=display_names,
    )

    output_dir = Path(args.output_dir)
    save_outputs(
        output_dir,
        metrics=metrics,
        confusion=confusion,
        per_class=per_class,
        sample_rows=result["sample_rows"],
        label_names=label_names,
        display_names=display_names,
        max_confusion_pairs=args.max_confusion_pairs,
    )

    print(f"loss: {metrics['loss']:.4f}")
    for k, value in metrics["topk"].items():
        print(f"top-{k}: {value:.2%}")
    print(f"Saved evaluation report to: {output_dir}")


if __name__ == "__main__":
    main()
