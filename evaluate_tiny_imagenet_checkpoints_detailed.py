"""Detailed matched evaluation of two Tiny ImageNet checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as torch_functional
from torch.utils.data import DataLoader, Dataset, Subset

import train_pipeline
from evaluate_tiny_imagenet_validation import (
    IndexedHuggingFaceImageDataset,
    architecture_from_checkpoint,
    class_display_names,
    validation_transform,
)


TOPK_VALUES = [1, 2, 3, 5, 10, 20, 50]
CALIBRATION_BINS = 15
SELECTIVE_COVERAGES = [1.0, 0.9, 0.75, 0.5, 0.25, 0.1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--small-checkpoint",
        default="best_702k_cnn3_membrane_transformer.pt",
    )
    parser.add_argument("--large-checkpoint", default="best.pt")
    parser.add_argument("--data-dir", default="data/tiny-imagenet-200-clean")
    parser.add_argument("--splits", default="validation,test")
    parser.add_argument(
        "--output-dir",
        default="visualizations/tiny_imagenet_official_evaluation",
    )
    parser.add_argument("--small-batch-size", type=int, default=64)
    parser.add_argument("--large-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional per-split development limit; zero evaluates every image.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-download", action="store_true", default=True)
    parser.add_argument(
        "--download",
        action="store_false",
        dest="no_download",
    )
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    return torch.load(path, map_location=device, weights_only=False)


def load_split(
    split: str,
    *,
    data_dir: str,
    image_size: int,
    no_download: bool,
) -> tuple[IndexedHuggingFaceImageDataset, list[str], list[str]]:
    hf_dataset = train_pipeline.load_hf_split(
        "slegroux/tiny-imagenet-200-clean",
        split=split,
        cache_dir=data_dir,
        no_download=no_download,
    )
    label_names = list(hf_dataset.features["label"].names)
    display_names = class_display_names(label_names)
    dataset = IndexedHuggingFaceImageDataset(
        hf_dataset,
        validation_transform(image_size),
    )
    return dataset, label_names, display_names


@torch.no_grad()
def collect_logits(
    checkpoint: dict[str, object],
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    architecture = architecture_from_checkpoint(checkpoint)
    model = train_pipeline.build_model(architecture).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    logits_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    indices_all: list[np.ndarray] = []
    start = time.perf_counter()
    for batch_index, (images, targets, indices) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        logits_all.append(logits.cpu().numpy())
        targets_all.append(targets.numpy())
        indices_all.append(indices.numpy())
        if batch_index % 20 == 0 or batch_index == len(loader):
            seen = sum(len(values) for values in targets_all)
            elapsed = time.perf_counter() - start
            print(
                f"{model_name}: batch {batch_index}/{len(loader)} | "
                f"seen {seen}/{len(dataset)} | {seen / max(elapsed, 1e-9):.1f} img/s",
                flush=True,
            )
    elapsed = time.perf_counter() - start
    return (
        np.concatenate(logits_all),
        np.concatenate(targets_all),
        np.concatenate(indices_all),
        elapsed,
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def calibration_metrics(
    confidence: np.ndarray,
    correct: np.ndarray,
    bins: int = CALIBRATION_BINS,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    maximum_gap = 0.0
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        selected = (
            (confidence >= lower) & (confidence < upper)
            if index < bins - 1
            else (confidence >= lower) & (confidence <= upper)
        )
        count = int(selected.sum())
        accuracy = float(correct[selected].mean()) if count else 0.0
        mean_confidence = float(confidence[selected].mean()) if count else 0.0
        gap = abs(accuracy - mean_confidence)
        ece += count / len(confidence) * gap
        maximum_gap = max(maximum_gap, gap if count else 0.0)
        rows.append(
            {
                "bin": index,
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
                "gap": gap,
            }
        )
    return {
        "ece_15": float(ece),
        "maximum_calibration_gap": float(maximum_gap),
    }, rows


def adaptive_ece(
    confidence: np.ndarray,
    correct: np.ndarray,
    bins: int = CALIBRATION_BINS,
) -> float:
    order = np.argsort(confidence)
    partitions = np.array_split(order, bins)
    return float(
        sum(
            len(partition)
            / len(confidence)
            * abs(
                float(correct[partition].mean())
                - float(confidence[partition].mean())
            )
            for partition in partitions
            if len(partition)
        )
    )


def selective_metrics(
    confidence: np.ndarray,
    correct: np.ndarray,
) -> list[dict[str, float]]:
    order = np.argsort(confidence)[::-1]
    rows = []
    for coverage in SELECTIVE_COVERAGES:
        count = max(1, int(round(len(order) * coverage)))
        selected = order[:count]
        rows.append(
            {
                "coverage": coverage,
                "num_samples": count,
                "accuracy": float(correct[selected].mean()),
                "mean_confidence": float(confidence[selected].mean()),
                "risk": float(1.0 - correct[selected].mean()),
                "minimum_confidence": float(confidence[selected].min()),
            }
        )
    return rows


def per_class_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    ranks: np.ndarray,
    losses: np.ndarray,
    label_names: list[str],
    display_names: list[str],
) -> list[dict[str, object]]:
    num_classes = probabilities.shape[1]
    rows = []
    predicted_counts = np.bincount(predictions, minlength=num_classes)
    for label in range(num_classes):
        selected = targets == label
        support = int(selected.sum())
        correct = int(np.logical_and(selected, predictions == label).sum())
        false_positive = int(
            np.logical_and(targets != label, predictions == label).sum()
        )
        precision = (
            correct / predicted_counts[label] if predicted_counts[label] else 0.0
        )
        recall = correct / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        class_ranks = ranks[selected]
        class_losses = losses[selected]
        rows.append(
            {
                "label": label,
                "synset": label_names[label],
                "name": display_names[label],
                "support": support,
                "top1_accuracy": recall,
                "top5_accuracy": (
                    float((class_ranks <= 5).mean()) if support else 0.0
                ),
                "top10_accuracy": (
                    float((class_ranks <= 10).mean()) if support else 0.0
                ),
                "mean_loss": float(class_losses.mean()) if support else 0.0,
                "mean_true_probability": float(
                    probabilities[selected, label].mean()
                )
                if support
                else 0.0,
                "mean_true_rank": float(class_ranks.mean()) if support else 0.0,
                "median_true_rank": (
                    float(np.median(class_ranks)) if support else 0.0
                ),
                "predicted_count": int(predicted_counts[label]),
                "false_positive_count": false_positive,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    encoded = targets * num_classes + predictions
    return np.bincount(
        encoded,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def confusion_pairs(
    confusion: np.ndarray,
    label_names: list[str],
    display_names: list[str],
    limit: int = 300,
) -> list[dict[str, object]]:
    support = confusion.sum(axis=1)
    rows = []
    for true_label in range(len(confusion)):
        for predicted_label in range(len(confusion)):
            if true_label == predicted_label:
                continue
            count = int(confusion[true_label, predicted_label])
            if count == 0:
                continue
            rows.append(
                {
                    "true_label": true_label,
                    "true_synset": label_names[true_label],
                    "true_name": display_names[true_label],
                    "predicted_label": predicted_label,
                    "predicted_synset": label_names[predicted_label],
                    "predicted_name": display_names[predicted_label],
                    "count": count,
                    "true_class_fraction": float(count / support[true_label]),
                }
            )
    rows.sort(
        key=lambda row: (row["count"], row["true_class_fraction"]),
        reverse=True,
    )
    return rows[:limit]


def evaluate_logits(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    label_names: list[str],
    display_names: list[str],
) -> dict[str, object]:
    probabilities = softmax(logits)
    sorted_labels = np.argsort(-probabilities, axis=1)
    predictions = sorted_labels[:, 0]
    ranks = (
        np.argmax(sorted_labels == targets[:, None], axis=1).astype(np.int64) + 1
    )
    true_probabilities = probabilities[np.arange(len(targets)), targets]
    losses = -np.log(np.maximum(true_probabilities, 1e-300))
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, 1e-300)),
        axis=1,
    )
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(targets)), targets] = 1.0
    brier = np.sum((probabilities - one_hot) ** 2, axis=1)
    calibration, calibration_rows = calibration_metrics(confidence, correct)
    calibration["adaptive_ece_15"] = adaptive_ece(confidence, correct)
    topk = {
        str(k): float((ranks <= min(k, probabilities.shape[1])).mean())
        for k in TOPK_VALUES
    }
    per_class = per_class_metrics(
        probabilities,
        targets,
        predictions,
        ranks,
        losses,
        label_names,
        display_names,
    )
    confusion = confusion_matrix(targets, predictions, probabilities.shape[1])
    macro_topk = {
        str(k): float(
            np.mean(
                [
                    float((ranks[targets == label] <= k).mean())
                    for label in range(probabilities.shape[1])
                    if np.any(targets == label)
                ]
            )
        )
        for k in [1, 5, 10]
    }
    metrics = {
        "num_samples": len(targets),
        "loss_nll": float(losses.mean()),
        "perplexity": float(np.exp(min(float(losses.mean()), 700.0))),
        "topk": topk,
        "macro_topk": macro_topk,
        "mean_true_rank": float(ranks.mean()),
        "median_true_rank": float(np.median(ranks)),
        "true_rank_p90": float(np.percentile(ranks, 90)),
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
        "brier_score": float(brier.mean()),
        "mean_confidence": float(confidence.mean()),
        "mean_confidence_correct": float(confidence[correct].mean()),
        "mean_confidence_incorrect": float(confidence[~correct].mean()),
        "mean_true_class_probability": float(true_probabilities.mean()),
        "mean_entropy": float(entropy.mean()),
        "normalized_entropy": float(
            entropy.mean() / math.log(probabilities.shape[1])
        ),
        "calibration": calibration,
    }
    return {
        "metrics": metrics,
        "probabilities": probabilities,
        "predictions": predictions,
        "ranks": ranks,
        "losses": losses,
        "confidence": confidence,
        "correct": correct,
        "entropy": entropy,
        "per_class": per_class,
        "calibration_rows": calibration_rows,
        "selective_rows": selective_metrics(confidence, correct),
        "confusion": confusion,
        "confusion_pairs": confusion_pairs(
            confusion,
            label_names,
            display_names,
        ),
    }


def paired_metrics(
    small: dict[str, object],
    large: dict[str, object],
    targets: np.ndarray,
) -> dict[str, object]:
    small_predictions = np.asarray(small["predictions"])
    large_predictions = np.asarray(large["predictions"])
    small_correct = small_predictions == targets
    large_correct = large_predictions == targets
    both_correct = np.logical_and(small_correct, large_correct)
    small_only = np.logical_and(small_correct, ~large_correct)
    large_only = np.logical_and(~small_correct, large_correct)
    neither = np.logical_and(~small_correct, ~large_correct)
    both_wrong = np.logical_and(~small_correct, ~large_correct)
    same_wrong = np.logical_and(both_wrong, small_predictions == large_predictions)
    union_errors = np.logical_or(~small_correct, ~large_correct)
    intersection_errors = both_wrong
    return {
        "prediction_agreement": float(
            (small_predictions == large_predictions).mean()
        ),
        "both_correct": float(both_correct.mean()),
        "small_only_correct": float(small_only.mean()),
        "large_only_correct": float(large_only.mean()),
        "neither_correct": float(neither.mean()),
        "oracle_top1": float(np.logical_or(small_correct, large_correct).mean()),
        "same_wrong_prediction": float(same_wrong.mean()),
        "same_wrong_given_both_wrong": float(
            same_wrong.sum() / max(int(both_wrong.sum()), 1)
        ),
        "error_jaccard": float(
            intersection_errors.sum() / max(int(union_errors.sum()), 1)
        ),
        "confidence_correlation": float(
            np.corrcoef(small["confidence"], large["confidence"])[0, 1]
        ),
        "loss_correlation": float(
            np.corrcoef(small["losses"], large["losses"])[0, 1]
        ),
        "mean_probability_l1": float(
            np.abs(small["probabilities"] - large["probabilities"]).sum(axis=1).mean()
        ),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_confusion(path: Path, confusion: np.ndarray, title: str) -> None:
    support = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion,
        support,
        out=np.zeros_like(confusion, dtype=np.float64),
        where=support > 0,
    )
    fig, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(normalized, cmap="magma", vmin=0.0, vmax=1.0)
    axis.set_title(title)
    axis.set_xlabel("predicted class")
    axis.set_ylabel("true class")
    fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_calibration(
    path: Path,
    evaluations: dict[str, dict[str, object]],
    title: str,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 7))
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", label="perfect")
    for name, result in evaluations.items():
        rows = [row for row in result["calibration_rows"] if row["count"]]
        axis.plot(
            [row["mean_confidence"] for row in rows],
            [row["accuracy"] for row in rows],
            marker="o",
            linewidth=2,
            label=name,
        )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("mean confidence")
    axis.set_ylabel("empirical accuracy")
    axis.set_title(title)
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_rank_cdf(
    path: Path,
    evaluations: dict[str, dict[str, object]],
    title: str,
) -> None:
    fig, axis = plt.subplots(figsize=(9, 6))
    for name, result in evaluations.items():
        ranks = np.asarray(result["ranks"])
        values = np.arange(1, 51)
        axis.plot(
            values,
            [(ranks <= value).mean() for value in values],
            linewidth=2,
            label=name,
        )
    for value in [1, 5, 10]:
        axis.axvline(value, color="gray", alpha=0.25)
    axis.set_xlim(1, 50)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("top-k")
    axis.set_ylabel("fraction with true class in top-k")
    axis.set_title(title)
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_selective_accuracy(
    path: Path,
    evaluations: dict[str, dict[str, object]],
    title: str,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 6))
    for name, result in evaluations.items():
        rows = result["selective_rows"]
        axis.plot(
            [row["coverage"] for row in rows],
            [row["accuracy"] for row in rows],
            marker="o",
            linewidth=2,
            label=name,
        )
    axis.set_xlim(1.0, 0.1)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("coverage retained by confidence")
    axis.set_ylabel("top-1 accuracy")
    axis.set_title(title)
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_per_class_comparison(
    path: Path,
    small: dict[str, object],
    large: dict[str, object],
    display_names: list[str],
    title: str,
) -> None:
    small_accuracy = np.asarray(
        [row["top1_accuracy"] for row in small["per_class"]]
    )
    large_accuracy = np.asarray(
        [row["top1_accuracy"] for row in large["per_class"]]
    )
    difference = large_accuracy - small_accuracy
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].scatter(small_accuracy, large_accuracy, s=20, alpha=0.7)
    axes[0].plot([0, 1], [0, 1], color="black", linestyle="--")
    axes[0].set_xlabel("702K class accuracy")
    axes[0].set_ylabel("2048-unit class accuracy")
    axes[0].set_title("Per-class top-1 comparison")
    axes[0].grid(alpha=0.2)

    count = 20
    selected = np.argsort(np.abs(difference))[-count:]
    order = selected[np.argsort(difference[selected])]
    labels = [display_names[index].split(",")[0] for index in order]
    colors = ["tab:red" if difference[index] < 0 else "tab:blue" for index in order]
    positions = np.arange(len(order))
    axes[1].barh(positions, difference[order], color=colors)
    axes[1].set_yticks(positions)
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("2048-unit minus 702K class accuracy")
    axes[1].set_title("Largest per-class differences")
    axes[1].grid(axis="x", alpha=0.2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def sample_rows(
    indices: np.ndarray,
    targets: np.ndarray,
    evaluation: dict[str, object],
    label_names: list[str],
    display_names: list[str],
) -> list[dict[str, object]]:
    rows = []
    probabilities = np.asarray(evaluation["probabilities"])
    predictions = np.asarray(evaluation["predictions"])
    for row in range(len(targets)):
        true_label = int(targets[row])
        predicted_label = int(predictions[row])
        top_labels = np.argsort(-probabilities[row])[:10]
        rows.append(
            {
                "index": int(indices[row]),
                "true_label": true_label,
                "true_synset": label_names[true_label],
                "true_name": display_names[true_label],
                "predicted_label": predicted_label,
                "predicted_synset": label_names[predicted_label],
                "predicted_name": display_names[predicted_label],
                "true_rank": int(evaluation["ranks"][row]),
                "loss": float(evaluation["losses"][row]),
                "confidence": float(evaluation["confidence"][row]),
                "true_probability": float(probabilities[row, true_label]),
                "entropy": float(evaluation["entropy"][row]),
                "top1_correct": bool(evaluation["correct"][row]),
                "top5_correct": bool(evaluation["ranks"][row] <= 5),
                "top10_correct": bool(evaluation["ranks"][row] <= 10),
                "top10_labels": " ".join(str(int(value)) for value in top_labels),
                "top10_probabilities": " ".join(
                    f"{probabilities[row, value]:.8f}" for value in top_labels
                ),
            }
        )
    return rows


def save_split_outputs(
    output_dir: Path,
    split: str,
    evaluations: dict[str, dict[str, object]],
    paired: dict[str, object],
    logits: dict[str, np.ndarray],
    targets: np.ndarray,
    indices: np.ndarray,
    label_names: list[str],
    display_names: list[str],
) -> None:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        name: result["metrics"] for name, result in evaluations.items()
    }
    metrics["paired"] = paired
    (split_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        split_dir / "logits_and_targets.npz",
        targets=targets,
        indices=indices,
        small_logits=logits["702K"],
        large_logits=logits["2048-unit"],
        ensemble_probabilities=evaluations["ensemble"]["probabilities"],
    )
    for name, result in evaluations.items():
        slug = name.lower().replace("-", "").replace(" ", "_")
        write_csv(split_dir / f"{slug}_per_class.csv", result["per_class"])
        write_csv(
            split_dir / f"{slug}_calibration.csv",
            result["calibration_rows"],
        )
        write_csv(
            split_dir / f"{slug}_selective_accuracy.csv",
            result["selective_rows"],
        )
        write_csv(
            split_dir / f"{slug}_confusion_pairs.csv",
            result["confusion_pairs"],
        )
        rows = sample_rows(
            indices,
            targets,
            result,
            label_names,
            display_names,
        )
        write_csv(split_dir / f"{slug}_all_samples.csv", rows)
        write_csv(
            split_dir / f"{slug}_misclassified.csv",
            [row for row in rows if not row["top1_correct"]],
        )
        np.save(split_dir / f"{slug}_confusion.npy", result["confusion"])
        save_confusion(
            split_dir / f"{slug}_confusion.png",
            result["confusion"],
            f"{split}: {name} row-normalized confusion",
        )
    save_calibration(
        split_dir / "calibration_comparison.png",
        evaluations,
        f"{split}: reliability diagram",
    )
    save_rank_cdf(
        split_dir / "topk_rank_cdf.png",
        evaluations,
        f"{split}: true-label rank CDF",
    )
    save_selective_accuracy(
        split_dir / "selective_accuracy.png",
        evaluations,
        f"{split}: accuracy when retaining most-confident samples",
    )
    save_per_class_comparison(
        split_dir / "per_class_model_comparison.png",
        evaluations["702K"],
        evaluations["2048-unit"],
        display_names,
        f"{split}: per-class checkpoint comparison",
    )


def combined_metrics(
    split_results: dict[str, dict[str, object]],
    model_name: str,
) -> dict[str, object]:
    targets = np.concatenate(
        [np.asarray(result["targets"]) for result in split_results.values()]
    )
    logits = np.concatenate(
        [np.asarray(result["logits"][model_name]) for result in split_results.values()]
    )
    first = next(iter(split_results.values()))
    return evaluate_logits(
        logits,
        targets,
        label_names=first["label_names"],
        display_names=first["display_names"],
    )


def write_report(
    output_dir: Path,
    split_results: dict[str, dict[str, object]],
    combined: dict[str, dict[str, object]],
    combined_paired: dict[str, object],
    checkpoint_metadata: dict[str, dict[str, object]],
) -> None:
    lines = [
        "# Tiny ImageNet Official-Split Evaluation",
        "",
        "Both checkpoints were evaluated with identical deterministic preprocessing:",
        "`Resize(73) -> CenterCrop(64) -> ImageNet normalization`.",
        "",
        "| Checkpoint | Parameters | Epoch | Training-split best validation |",
        "|---|---:|---:|---:|",
    ]
    for name in ["702K", "2048-unit"]:
        metadata = checkpoint_metadata[name]
        lines.append(
            f"| {name} | {metadata['parameters']:,} | {metadata['epoch']} | "
            f"{metadata['best_val_accuracy']:.2%} |"
        )

    for split, result in split_results.items():
        lines.extend(
            [
                "",
                f"## {split.title()} ({len(result['targets']):,} images)",
                "",
                "| Model | Loss | Top-1 | Top-5 | Top-10 | MRR | Mean rank | ECE | Brier |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in ["702K", "2048-unit", "ensemble"]:
            metrics = result["evaluations"][name]["metrics"]
            lines.append(
                f"| {name} | {metrics['loss_nll']:.4f} | "
                f"{metrics['topk']['1']:.2%} | {metrics['topk']['5']:.2%} | "
                f"{metrics['topk']['10']:.2%} | "
                f"{metrics['mean_reciprocal_rank']:.3f} | "
                f"{metrics['mean_true_rank']:.2f} | "
                f"{metrics['calibration']['ece_15']:.3f} | "
                f"{metrics['brier_score']:.3f} |"
            )
        paired = result["paired"]
        lines.extend(
            [
                "",
                f"- prediction agreement: **{paired['prediction_agreement']:.2%}**",
                f"- correct only for 702K: **{paired['small_only_correct']:.2%}**",
                f"- correct only for 2048-unit: **{paired['large_only_correct']:.2%}**",
                f"- oracle accuracy if either model is correct: **{paired['oracle_top1']:.2%}**",
            ]
        )

    lines.extend(
        [
            "",
            f"## Combined Unseen Splits ({sum(len(result['targets']) for result in split_results.values()):,} images)",
            "",
            "| Model | Loss | Top-1 | Top-5 | Top-10 | Macro Top-1 | MRR | Mean rank | ECE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ["702K", "2048-unit", "ensemble"]:
        metrics = combined[name]["metrics"]
        lines.append(
            f"| {name} | {metrics['loss_nll']:.4f} | "
            f"{metrics['topk']['1']:.2%} | {metrics['topk']['5']:.2%} | "
            f"{metrics['topk']['10']:.2%} | "
            f"{metrics['macro_topk']['1']:.2%} | "
            f"{metrics['mean_reciprocal_rank']:.3f} | "
            f"{metrics['mean_true_rank']:.2f} | "
            f"{metrics['calibration']['ece_15']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Combined prediction agreement is "
            f"**{combined_paired['prediction_agreement']:.2%}**. The models' "
            f"complementary correct predictions produce an oracle top-1 ceiling of "
            f"**{combined_paired['oracle_top1']:.2%}**.",
            "",
            "The ensemble is the arithmetic mean of the two softmax probability vectors; "
            "it is not trained or calibrated on these splits.",
        ]
    )
    (output_dir / "REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    train_pipeline.set_seed(args.seed)
    splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    if not splits:
        raise ValueError("--splits must contain at least one split.")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    paths = {
        "702K": Path(args.small_checkpoint),
        "2048-unit": Path(args.large_checkpoint),
    }
    checkpoints = {
        name: load_checkpoint(path, device) for name, path in paths.items()
    }
    architectures = {
        name: architecture_from_checkpoint(checkpoint)
        for name, checkpoint in checkpoints.items()
    }
    image_sizes = {architecture.image_size for architecture in architectures.values()}
    if len(image_sizes) != 1:
        raise ValueError("Matched evaluation requires identical input image sizes.")
    image_size = image_sizes.pop()
    checkpoint_metadata = {}
    for name, checkpoint in checkpoints.items():
        state = checkpoint["model_state"]
        checkpoint_metadata[name] = {
            "path": str(paths[name].resolve()),
            "parameters": int(sum(value.numel() for value in state.values())),
            "epoch": int(checkpoint.get("epoch", -1)),
            "best_val_accuracy": float(
                checkpoint.get("best_val_acc", float("nan"))
            ),
            "architecture": vars(architectures[name]),
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_results: dict[str, dict[str, object]] = {}
    batch_sizes = {
        "702K": args.small_batch_size,
        "2048-unit": args.large_batch_size,
    }

    for split in splits:
        print(f"\n=== split: {split} ===", flush=True)
        dataset, label_names, display_names = load_split(
            split,
            data_dir=args.data_dir,
            image_size=image_size,
            no_download=args.no_download,
        )
        if args.max_samples > 0:
            dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
        logits: dict[str, np.ndarray] = {}
        targets_reference = None
        indices_reference = None
        runtimes = {}
        for name in ["702K", "2048-unit"]:
            values, targets, indices, elapsed = collect_logits(
                checkpoints[name],
                dataset,
                batch_size=batch_sizes[name],
                num_workers=args.num_workers,
                device=device,
                model_name=f"{split}/{name}",
            )
            logits[name] = values
            runtimes[name] = {
                "seconds": elapsed,
                "images_per_second": len(targets) / elapsed,
                "batch_size": batch_sizes[name],
            }
            if targets_reference is None:
                targets_reference = targets
                indices_reference = indices
            elif not np.array_equal(targets_reference, targets) or not np.array_equal(
                indices_reference,
                indices,
            ):
                raise RuntimeError("Model evaluations did not preserve sample ordering.")

        targets = np.asarray(targets_reference)
        indices = np.asarray(indices_reference)
        evaluations = {
            name: evaluate_logits(
                values,
                targets,
                label_names=label_names,
                display_names=display_names,
            )
            for name, values in logits.items()
        }
        ensemble_probabilities = (
            evaluations["702K"]["probabilities"]
            + evaluations["2048-unit"]["probabilities"]
        ) / 2.0
        ensemble_logits = np.log(np.maximum(ensemble_probabilities, 1e-300))
        evaluations["ensemble"] = evaluate_logits(
            ensemble_logits,
            targets,
            label_names=label_names,
            display_names=display_names,
        )
        paired = paired_metrics(
            evaluations["702K"],
            evaluations["2048-unit"],
            targets,
        )
        split_results[split] = {
            "targets": targets,
            "indices": indices,
            "logits": logits,
            "evaluations": evaluations,
            "paired": paired,
            "label_names": label_names,
            "display_names": display_names,
            "runtimes": runtimes,
        }
        save_split_outputs(
            output_dir,
            split,
            evaluations,
            paired,
            logits,
            targets,
            indices,
            label_names,
            display_names,
        )

    combined = {
        name: combined_metrics(split_results, name)
        for name in ["702K", "2048-unit"]
    }
    combined_probabilities = np.concatenate(
        [
            result["evaluations"]["ensemble"]["probabilities"]
            for result in split_results.values()
        ]
    )
    combined_targets = np.concatenate(
        [result["targets"] for result in split_results.values()]
    )
    first_result = next(iter(split_results.values()))
    combined["ensemble"] = evaluate_logits(
        np.log(np.maximum(combined_probabilities, 1e-300)),
        combined_targets,
        label_names=first_result["label_names"],
        display_names=first_result["display_names"],
    )
    combined_paired = paired_metrics(
        combined["702K"],
        combined["2048-unit"],
        combined_targets,
    )
    combined_dir = output_dir / "combined"
    combined_dir.mkdir(exist_ok=True)
    combined_metrics_json = {
        name: evaluation["metrics"] for name, evaluation in combined.items()
    }
    combined_metrics_json["paired"] = combined_paired
    (combined_dir / "metrics.json").write_text(
        json.dumps(combined_metrics_json, indent=2),
        encoding="utf-8",
    )
    for name, evaluation in combined.items():
        slug = name.lower().replace("-", "").replace(" ", "_")
        write_csv(
            combined_dir / f"{slug}_per_class.csv",
            evaluation["per_class"],
        )
    save_calibration(
        combined_dir / "calibration_comparison.png",
        combined,
        "Combined unseen splits: reliability diagram",
    )
    save_rank_cdf(
        combined_dir / "topk_rank_cdf.png",
        combined,
        "Combined unseen splits: true-label rank CDF",
    )
    save_selective_accuracy(
        combined_dir / "selective_accuracy.png",
        combined,
        "Combined unseen splits: selective accuracy",
    )
    save_per_class_comparison(
        combined_dir / "per_class_model_comparison.png",
        combined["702K"],
        combined["2048-unit"],
        first_result["display_names"],
        "Combined unseen splits: per-class comparison",
    )
    metadata = {
        "device": str(device),
        "splits": {
            split: {
                "num_samples": len(result["targets"]),
                "runtimes": result["runtimes"],
            }
            for split, result in split_results.items()
        },
        "checkpoints": checkpoint_metadata,
        "preprocessing": {
            "resize": 73,
            "center_crop": 64,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    write_report(
        output_dir,
        split_results,
        combined,
        combined_paired,
        checkpoint_metadata,
    )
    print(f"\nSaved detailed evaluation to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
