"""Comprehensive comparison of three Tiny ImageNet models:

  1. ResNet-18 (scratch)              – sweep_runs_tiny_imagenet/resnet18_scratch
  2. CNN3 + TPSAPU + membrane_xfmr   – checkpoints/best_702k_cnn3_membrane_transformer.pt
  3. ResCNN + TPSAPU + both_xfmr     – checkpoints/res_cnn_cross_both_transformer/best.pt

Produces:
  visualizations/comparison/01_training_curves.png
  visualizations/comparison/02_model_summary.png
  visualizations/comparison/03_sparsity_analysis.png
  visualizations/comparison/04_per_class_accuracy.png   (requires eval outputs)
  visualizations/comparison/05_confusion_comparison.png  (requires eval outputs)
  visualizations/comparison/06_efficiency.png
  visualizations/comparison/07_combined_summary.png

Eval outputs are also written under:
  visualizations/comparison/eval_cnn3_702k/
  visualizations/comparison/eval_rescnn_2048/
  (ResNet-18 eval is run inline here)
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

import evaluate_tiny_imagenet_validation as eval_script
import train_pipeline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent

RESNET_METRICS = BASE / "sweep_runs_tiny_imagenet/resnet18_scratch/metrics.csv"
CNN3_METRICS = BASE / "sweep_runs_tiny_imagenet/cnn3__membrane_transformer/metrics.csv"
RESCNN_METRICS = BASE / "runs/res_cnn_cross_both_transformer/metrics.csv"

RESNET_CKPT = BASE / "sweep_runs_tiny_imagenet/resnet18_scratch/best.pt"
CNN3_CKPT = BASE / "checkpoints/best_702k_cnn3_membrane_transformer.pt"
RESCNN_CKPT = BASE / "checkpoints/res_cnn_cross_both_transformer/best.pt"

OUT = BASE / "visualizations/comparison"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Colours / labels
# ---------------------------------------------------------------------------
COLORS = {
    "ResNet-18": "#e07b39",
    "CNN3+TPSAPU (702k)": "#4c9be8",
    "ResCNN+TPSAPU (2048 neurons)": "#6bbf59",
}
LABELS = list(COLORS.keys())

PHASE_COLORS = {
    "warmup": "#aaaaaa",
    "cosine": "#4c9be8",
    "prune": "#e07b39",
    "l2-prune": "#e07b39",
    "l2_prune": "#e07b39",
}

# ---------------------------------------------------------------------------
# Model metadata (static facts from checkpoints / logs)
# ---------------------------------------------------------------------------
MODEL_META = {
    "ResNet-18": {
        "params": 11_281_052,
        "best_val_acc": 0.5646,
        "best_epoch": 36,
        "total_epochs": 40,
        "architecture": "ResNet-18 (tiny stem)",
        "reservoir": "—",
        "sparsity": 0.0,
        "pruning": False,
        "batch_size": 64,
    },
    "CNN3+TPSAPU (702k)": {
        "params": 701_576,
        "best_val_acc": 0.4145,
        "best_epoch": 65,
        "total_epochs": 104,
        "architecture": "CNN3 → TPSAPU → Membrane-Xfmr",
        "reservoir": "512 neurons (64×8τ)",
        "sparsity": 0.95,
        "pruning": True,
        "batch_size": 64,
    },
    "ResCNN+TPSAPU (2048 neurons)": {
        "params": 1_650_568,
        "best_val_acc": 0.3939,
        "best_epoch": 61,
        "total_epochs": 63,  # training interrupted at 63/85
        "architecture": "ResCNN → TPSAPU-CrossRes → Both-Xfmr",
        "reservoir": "2048 neurons (256×8τ)",
        "sparsity": 0.0,
        "pruning": False,
        "batch_size": 8,
    },
}

# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------


def load_resnet_metrics(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "global_epoch": int(row["epoch"]),
                    "phase": row["phase"],
                    "lr": float(row["lr"]),
                    "train_loss": float(row["train_loss"]),
                    "train_acc": float(row["train_acc"]),
                    "val_loss": float(row["val_loss"]),
                    "val_acc": float(row["val_acc"]),
                    "best_val_acc": float(row["best_val_acc"]),
                    "sparsity": None,
                }
            )
    return rows


def load_tpsapu_metrics(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            sparsity = row.get("sparsity", "") or None
            rows.append(
                {
                    "global_epoch": int(row["global_epoch"]),
                    "phase": row["phase"],
                    "lr": float(row["lr"]),
                    "train_loss": float(row["train_loss"]),
                    "train_acc": float(row["train_acc"]),
                    "val_loss": float(row["val_loss"]),
                    "val_acc": float(row["val_acc"]),
                    "best_val_acc": float(row["best_val_acc"]),
                    "sparsity": float(sparsity) if sparsity else None,
                    "completed_train_epochs": int(
                        row.get("completed_train_epochs", 0) or 0
                    ),
                    "completed_prune_epochs": int(
                        row.get("completed_prune_epochs", 0) or 0
                    ),
                }
            )
    return rows


def _col(rows, key):
    return [r[key] for r in rows]


# ---------------------------------------------------------------------------
# ResNet-18 evaluation
# ---------------------------------------------------------------------------


def evaluate_resnet(args) -> dict | None:
    """Run ResNet-18 on the HF validation split and return per-class accuracy."""
    out_dir = OUT / "eval_resnet18"
    metrics_path = out_dir / "metrics.json"
    per_class_path = out_dir / "per_class_metrics.csv"
    if metrics_path.exists() and per_class_path.exists():
        print(f"[ResNet-18 eval] Using cached results in {out_dir}")
        with open(metrics_path) as f:
            metrics = json.load(f)
        per_class = _load_per_class(per_class_path)
        return {"metrics": metrics, "per_class": per_class}

    print("[ResNet-18 eval] Running evaluation …")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model identical to training script
    ck = torch.load(RESNET_CKPT, map_location=device, weights_only=False)
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 200)
    model.load_state_dict(ck["model_state"])
    model = model.to(device)
    model.eval()

    # Build dataset using the same HF loader
    transform = eval_script.validation_transform(64)
    hf_val = train_pipeline.load_hf_split(
        "slegroux/tiny-imagenet-200-clean",
        split="validation",
        cache_dir=str(BASE / "data/tiny-imagenet-200-clean"),
        no_download=True,
    )
    dataset = eval_script.IndexedHuggingFaceImageDataset(hf_val, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Load class names using the same helper
    synsets = hf_val.features["label"].names  # list of synset strings
    display_names = eval_script.class_display_names(synsets)
    num_classes = len(synsets)

    result = eval_script.evaluate(
        model,
        loader,
        device=device,
        num_classes=num_classes,
        topk_values=[1, 5],
        label_names=synsets,
        display_names=display_names,
    )

    metrics_out = {
        **result["metrics"],
        "checkpoint": str(RESNET_CKPT),
        "split": "validation",
        "architecture": {"encoder": "resnet18"},
    }
    confusion = result["confusion"]
    per_class_data = eval_script.per_class_rows(
        confusion,
        topk_class_correct=result["topk_class_correct"],
        label_names=synsets,
        display_names=display_names,
    )
    eval_script.save_outputs(
        out_dir,
        metrics=metrics_out,
        confusion=confusion,
        per_class=per_class_data,
        sample_rows=result["sample_rows"],
        label_names=synsets,
        display_names=display_names,
        max_confusion_pairs=200,
    )
    print(f"[ResNet-18 eval] top-1={result['metrics']['topk']['1']:.2%}")
    with open(metrics_path) as f:
        metrics = json.load(f)
    return {"metrics": metrics, "per_class": per_class_data}


def evaluate_tpsapu(ckpt_path: Path, out_dir: Path, args) -> dict | None:
    """Run a TPSAPU checkpoint through evaluate_tiny_imagenet_validation logic."""
    metrics_path = out_dir / "metrics.json"
    per_class_path = out_dir / "per_class_metrics.csv"
    if metrics_path.exists() and per_class_path.exists():
        print(f"[TPSAPU eval] Using cached results in {out_dir}")
        with open(metrics_path) as f:
            metrics = json.load(f)
        per_class = _load_per_class(per_class_path)
        return {"metrics": metrics, "per_class": per_class}

    print(f"[TPSAPU eval] Evaluating {ckpt_path.name} …")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = eval_script.load_checkpoint(str(ckpt_path), device)
    architecture_args = eval_script.architecture_from_checkpoint(checkpoint)
    model = train_pipeline.build_model(architecture_args).to(device)
    model.load_state_dict(checkpoint["model_state"])

    transform = eval_script.validation_transform(64)
    hf_val = train_pipeline.load_hf_split(
        "slegroux/tiny-imagenet-200-clean",
        split="validation",
        cache_dir=str(BASE / "data/tiny-imagenet-200-clean"),
        no_download=True,
    )
    dataset = eval_script.IndexedHuggingFaceImageDataset(hf_val, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    synsets = hf_val.features["label"].names
    display_names = eval_script.class_display_names(synsets)
    num_classes = len(synsets)

    result = eval_script.evaluate(
        model,
        loader,
        device=device,
        num_classes=num_classes,
        topk_values=[1, 5],
        label_names=synsets,
        display_names=display_names,
    )
    metrics_out = {
        **result["metrics"],
        "checkpoint": str(ckpt_path),
        "split": "validation",
        "architecture": vars(architecture_args),
    }
    confusion = result["confusion"]
    per_class_data = eval_script.per_class_rows(
        confusion,
        topk_class_correct=result["topk_class_correct"],
        label_names=synsets,
        display_names=display_names,
    )
    eval_script.save_outputs(
        out_dir,
        metrics=metrics_out,
        confusion=confusion,
        per_class=per_class_data,
        sample_rows=result["sample_rows"],
        label_names=synsets,
        display_names=display_names,
        max_confusion_pairs=200,
    )
    print(f"[TPSAPU eval] top-1={result['metrics']['topk']['1']:.2%}")
    with open(metrics_path) as f:
        metrics = json.load(f)
    return {"metrics": metrics, "per_class": per_class_data}


def _load_per_class(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "label": int(row["label"]),
                    "name": row["name"],
                    "support": int(row["support"]),
                    "top1_accuracy": float(row["top1_accuracy"]),
                    "top5_accuracy": float(row["top5_accuracy"]),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Figure 1 – Training curves
# ---------------------------------------------------------------------------


def fig_training_curves(rn_rows, cnn3_rows, rescnn_rows):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Training Dynamics — Tiny ImageNet (200 classes, 64×64)",
        fontsize=14,
        fontweight="bold",
    )

    data = [
        ("ResNet-18", rn_rows),
        ("CNN3+TPSAPU (702k)", cnn3_rows),
        ("ResCNN+TPSAPU (2048 neurons)", rescnn_rows),
    ]

    for label, rows in data:
        col = COLORS[label]
        epochs = _col(rows, "global_epoch")
        val_acc = [v * 100 for v in _col(rows, "val_acc")]
        train_acc = [v * 100 for v in _col(rows, "train_acc")]
        val_loss = _col(rows, "val_loss")
        train_loss = _col(rows, "train_loss")
        best_acc = MODEL_META[label]["best_val_acc"] * 100

        axes[0, 0].plot(
            epochs,
            val_acc,
            color=col,
            label=f"{label} (best {best_acc:.1f}%)",
            linewidth=1.8,
        )
        axes[0, 1].plot(epochs, train_acc, color=col, label=label, linewidth=1.8)
        axes[1, 0].plot(epochs, val_loss, color=col, label=label, linewidth=1.8)
        axes[1, 1].plot(epochs, train_loss, color=col, label=label, linewidth=1.8)

    # Phase background shading for CNN3 model
    for label, rows in [("CNN3+TPSAPU (702k)", cnn3_rows)]:
        ax_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
        phase_changes = []
        prev = None
        for r in rows:
            ph = r["phase"]
            if ph != prev:
                phase_changes.append((r["global_epoch"], ph))
                prev = ph
        phase_changes.append((rows[-1]["global_epoch"] + 1, None))
        for i, (start_ep, ph) in enumerate(phase_changes[:-1]):
            end_ep = phase_changes[i + 1][0]
            pc = PHASE_COLORS.get(ph, "#cccccc")
            for ax in ax_list:
                ax.axvspan(start_ep, end_ep, alpha=0.07, color=pc, zorder=0)

    titles = [
        "Validation Accuracy (%)",
        "Train Accuracy (%)",
        "Validation Loss",
        "Train Loss",
    ]
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Phase legend patch
    phase_patches = [
        mpatches.Patch(color=PHASE_COLORS["warmup"], alpha=0.5, label="warmup"),
        mpatches.Patch(color=PHASE_COLORS["cosine"], alpha=0.5, label="cosine"),
        mpatches.Patch(color=PHASE_COLORS["l2-prune"], alpha=0.5, label="l2-prune"),
    ]
    fig.legend(
        handles=phase_patches,
        title="CNN3 phase",
        loc="lower right",
        bbox_to_anchor=(0.99, 0.01),
        fontsize=8,
        ncol=3,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    path = OUT / "01_training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 2 – Model summary bar charts
# ---------------------------------------------------------------------------


def fig_model_summary():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Model Comparison Summary — Tiny ImageNet", fontsize=14, fontweight="bold"
    )

    labels = LABELS
    colors = [COLORS[l] for l in labels]
    meta = [MODEL_META[l] for l in labels]

    # Panel 1: best validation accuracy
    accs = [m["best_val_acc"] * 100 for m in meta]
    bars = axes[0].bar(labels, accs, color=colors, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, accs):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.5,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    axes[0].set_ylim(0, max(accs) * 1.2)
    axes[0].set_ylabel("Top-1 Accuracy (%)")
    axes[0].set_title("Best Validation Accuracy")
    axes[0].set_xticklabels(labels, rotation=12, ha="right", fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: parameters (log scale)
    params = [m["params"] / 1e6 for m in meta]
    bars2 = axes[1].bar(labels, params, color=colors, edgecolor="white", linewidth=0.8)
    for bar, v, m in zip(bars2, params, meta):
        sp_note = f"\n({m['sparsity'] * 100:.0f}% sparse)" if m["pruning"] else ""
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            v + max(params) * 0.02,
            f"{v:.2f}M{sp_note}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    axes[1].set_ylabel("Parameters (M)")
    axes[1].set_title("Total Parameters")
    axes[1].set_xticklabels(labels, rotation=12, ha="right", fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    # Panel 3: accuracy / million params (efficiency)
    eff = [m["best_val_acc"] * 100 / (m["params"] / 1e6) for m in meta]
    bars3 = axes[2].bar(labels, eff, color=colors, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars3, eff):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            v + max(eff) * 0.02,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    axes[2].set_ylabel("Top-1 Accuracy % / Million Params")
    axes[2].set_title("Parameter Efficiency")
    axes[2].set_xticklabels(labels, rotation=12, ha="right", fontsize=8)
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = OUT / "02_model_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 3 – Sparsity / pruning analysis (CNN3 model)
# ---------------------------------------------------------------------------


def fig_sparsity_analysis(cnn3_rows):
    prune_rows = [r for r in cnn3_rows if r.get("sparsity") is not None]
    if not prune_rows:
        print("[sparsity] No sparsity data found; skipping figure 3.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("CNN3+TPSAPU: Pruning Analysis", fontsize=13, fontweight="bold")

    epochs = [r["global_epoch"] for r in prune_rows]
    sparsity = [r["sparsity"] * 100 for r in prune_rows]
    val_acc = [r["val_acc"] * 100 for r in prune_rows]
    val_loss = [r["val_loss"] for r in prune_rows]

    # All epochs for reference
    all_epochs = _col(cnn3_rows, "global_epoch")
    all_val_acc = [v * 100 for v in _col(cnn3_rows, "val_acc")]

    # Panel 1: sparsity vs epoch
    axes[0].plot(epochs, sparsity, color="#e07b39", linewidth=2)
    axes[0].axhline(95, color="gray", linestyle="--", alpha=0.6, label="target 95%")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Sparsity (%)")
    axes[0].set_title("Weight Sparsity During Pruning")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: val acc vs sparsity
    sc = axes[1].scatter(sparsity, val_acc, c=epochs, cmap="viridis", s=25, zorder=3)
    fig.colorbar(sc, ax=axes[1], label="Epoch")
    axes[1].set_xlabel("Sparsity (%)")
    axes[1].set_ylabel("Val Accuracy (%)")
    axes[1].set_title("Accuracy vs Sparsity")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: full training with phase shading
    c_warmup = [r["global_epoch"] for r in cnn3_rows if r["phase"] == "warmup"]
    c_cosine = [r["global_epoch"] for r in cnn3_rows if r["phase"] == "cosine"]
    c_prune = [r["global_epoch"] for r in cnn3_rows if "prune" in r["phase"]]

    axes[2].plot(
        all_epochs, all_val_acc, color=COLORS["CNN3+TPSAPU (702k)"], linewidth=1.8
    )
    if c_warmup:
        axes[2].axvspan(
            min(c_warmup),
            max(c_warmup),
            alpha=0.1,
            color=PHASE_COLORS["warmup"],
            label="warmup",
        )
    if c_cosine:
        axes[2].axvspan(
            min(c_cosine),
            max(c_cosine),
            alpha=0.1,
            color=PHASE_COLORS["cosine"],
            label="cosine",
        )
    if c_prune:
        axes[2].axvspan(
            min(c_prune),
            max(c_prune),
            alpha=0.1,
            color=PHASE_COLORS["l2-prune"],
            label="l2-prune",
        )
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Val Accuracy (%)")
    axes[2].set_title("CNN3 Full Training — Val Accuracy")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUT / "03_sparsity_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 4 – Per-class accuracy comparison
# ---------------------------------------------------------------------------


def fig_per_class(eval_results: dict[str, dict]):
    """Compare per-class top-1 accuracy across all three models (sorted by ResNet)."""
    available = {k: v for k, v in eval_results.items() if v is not None}
    if len(available) < 2:
        print("[per-class] Need ≥2 eval results; skipping figure 4.")
        return

    # Build {label_index -> {model: acc}}
    ref_key = "ResNet-18" if "ResNet-18" in available else list(available.keys())[0]
    ref_per_class = {r["label"]: r for r in available[ref_key]["per_class"]}

    # Sort classes by ResNet accuracy descending
    sorted_labels = sorted(
        ref_per_class.keys(),
        key=lambda l: ref_per_class[l]["top1_accuracy"],
        reverse=True,
    )
    x = np.arange(len(sorted_labels))
    name_map = {r["label"]: r["name"] for r in available[ref_key]["per_class"]}

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(
        "Per-Class Top-1 Accuracy — Tiny ImageNet (sorted by ResNet-18)",
        fontsize=13,
        fontweight="bold",
    )

    for model_key, result in available.items():
        pc_map = {r["label"]: r["top1_accuracy"] for r in result["per_class"]}
        accs = [pc_map.get(l, 0) * 100 for l in sorted_labels]
        axes[0].plot(
            x,
            accs,
            alpha=0.6,
            linewidth=0.7,
            color=COLORS.get(model_key, "#888888"),
            label=model_key,
        )

    # Panel 2: per-class accuracy gap  (ResNet vs CNN3)
    if "ResNet-18" in available and "CNN3+TPSAPU (702k)" in available:
        rn_map = {
            r["label"]: r["top1_accuracy"] for r in available["ResNet-18"]["per_class"]
        }
        cnn3_map = {
            r["label"]: r["top1_accuracy"]
            for r in available["CNN3+TPSAPU (702k)"]["per_class"]
        }
        gap = [(rn_map.get(l, 0) - cnn3_map.get(l, 0)) * 100 for l in sorted_labels]
        pos = [max(0, g) for g in gap]
        neg = [min(0, g) for g in gap]
        axes[1].bar(x, pos, color="#e07b39", alpha=0.7, label="ResNet better")
        axes[1].bar(x, neg, color="#4c9be8", alpha=0.7, label="CNN3 better")
        axes[1].axhline(0, color="black", linewidth=0.5)
        axes[1].set_ylabel("Accuracy Gap (pp)")
        axes[1].set_title("Per-Class Accuracy Gap: ResNet-18 minus CNN3+TPSAPU")
        axes[1].legend(fontsize=8)
        axes[1].grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Top-1 Accuracy (%)")
    axes[0].set_title("Per-Class Accuracy (200 classes)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Class (sorted by ResNet-18 accuracy, high → low)")
    axes[-1].set_xticks([])

    plt.tight_layout()
    path = OUT / "04_per_class_accuracy.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 5 – Confusion matrix comparison (top/bottom classes)
# ---------------------------------------------------------------------------


def fig_class_breakdown(eval_results: dict[str, dict]):
    """Side-by-side top-20 / bottom-20 class accuracy bar charts."""
    available = {k: v for k, v in eval_results.items() if v is not None}
    if not available:
        print("[class breakdown] No eval results; skipping figure 5.")
        return

    n_models = len(available)
    fig, axes = plt.subplots(n_models, 2, figsize=(16, 5 * n_models))
    if n_models == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(
        "Top-20 and Bottom-20 Classes by Model", fontsize=13, fontweight="bold"
    )

    for row_idx, (model_key, result) in enumerate(available.items()):
        pc = sorted(result["per_class"], key=lambda r: r["top1_accuracy"])
        worst20 = pc[:20]
        best20 = pc[-20:][::-1]

        col = COLORS.get(model_key, "#888888")

        for col_idx, (classes, title) in enumerate(
            [
                (worst20, f"{model_key}: 20 Worst Classes"),
                (best20, f"{model_key}: 20 Best Classes"),
            ]
        ):
            ax = axes[row_idx, col_idx]
            names = [c["name"][:25] for c in classes]
            accs = [c["top1_accuracy"] * 100 for c in classes]
            bars = ax.barh(names, accs, color=col, alpha=0.8)
            for bar, v in zip(bars, accs):
                ax.text(
                    v + 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.0f}%",
                    va="center",
                    fontsize=7,
                )
            ax.set_xlim(0, 105)
            ax.set_title(title, fontsize=9)
            ax.grid(axis="x", alpha=0.3)
            ax.tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    path = OUT / "05_best_worst_classes.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 6 – Accuracy / params efficiency scatter
# ---------------------------------------------------------------------------


def fig_efficiency(eval_results: dict[str, dict]):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Model Efficiency — Tiny ImageNet", fontsize=13, fontweight="bold")

    # Gather top-1 and top-5 from eval results when available, else from META
    top1 = {}
    top5 = {}
    for lbl, meta in MODEL_META.items():
        top1[lbl] = meta["best_val_acc"] * 100
        top5[lbl] = None
    for lbl, res in eval_results.items():
        if res is not None:
            top1[lbl] = res["metrics"]["topk"]["1"] * 100
            top5[lbl] = res["metrics"]["topk"].get("5", None)
            if top5[lbl]:
                top5[lbl] *= 100

    # Panel 1: params vs top-1 accuracy
    for lbl in LABELS:
        meta = MODEL_META[lbl]
        col = COLORS[lbl]
        pM = meta["params"] / 1e6
        acc = top1[lbl]
        axes[0].scatter(pM, acc, color=col, s=200, zorder=5, label=lbl)
        axes[0].annotate(
            lbl,
            xy=(pM, acc),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=8,
            color=col,
        )

    axes[0].set_xlabel("Parameters (M)")
    axes[0].set_ylabel("Top-1 Accuracy (%)")
    axes[0].set_title("Params vs Accuracy")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: top-1 vs top-5 (when both available)
    has_top5 = {lbl: v for lbl, v in top5.items() if v is not None}
    if len(has_top5) >= 2:
        for lbl in LABELS:
            if lbl not in has_top5:
                continue
            col = COLORS[lbl]
            axes[1].scatter(
                top1[lbl], has_top5[lbl], color=col, s=200, zorder=5, label=lbl
            )
            axes[1].annotate(
                lbl,
                xy=(top1[lbl], has_top5[lbl]),
                xytext=(6, 3),
                textcoords="offset points",
                fontsize=8,
                color=col,
            )
        axes[1].set_xlabel("Top-1 Accuracy (%)")
        axes[1].set_ylabel("Top-5 Accuracy (%)")
        axes[1].set_title("Top-1 vs Top-5 Accuracy")
        axes[1].grid(True, alpha=0.3)
    else:
        # Show efficiency ratio bar chart instead
        labels = LABELS
        eff = [
            MODEL_META[l]["best_val_acc"] * 100 / (MODEL_META[l]["params"] / 1e6)
            for l in labels
        ]
        colors = [COLORS[l] for l in labels]
        bars = axes[1].bar(labels, eff, color=colors, edgecolor="white")
        for bar, v in zip(bars, eff):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.3,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        axes[1].set_ylabel("Accuracy% / Million Params")
        axes[1].set_title("Parameter Efficiency")
        axes[1].set_xticklabels(labels, rotation=12, ha="right", fontsize=8)
        axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = OUT / "06_efficiency.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 7 – Combined one-page summary
# ---------------------------------------------------------------------------


def fig_combined_summary(
    rn_rows, cnn3_rows, rescnn_rows, eval_results: dict[str, dict]
):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "Tiny ImageNet Model Comparison — Summary",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.35)

    ax_curve = fig.add_subplot(gs[0, :2])
    ax_loss = fig.add_subplot(gs[0, 2:])
    ax_bar_acc = fig.add_subplot(gs[1, 0])
    ax_bar_par = fig.add_subplot(gs[1, 1])
    ax_eff = fig.add_subplot(gs[1, 2])
    ax_sparse = fig.add_subplot(gs[1, 3])
    ax_table = fig.add_subplot(gs[2, :])

    data = [
        ("ResNet-18", rn_rows),
        ("CNN3+TPSAPU (702k)", cnn3_rows),
        ("ResCNN+TPSAPU (2048 neurons)", rescnn_rows),
    ]

    # Val acc curve
    for label, rows in data:
        col = COLORS[label]
        epochs = _col(rows, "global_epoch")
        val_acc = [v * 100 for v in _col(rows, "val_acc")]
        best = MODEL_META[label]["best_val_acc"] * 100
        ax_curve.plot(
            epochs, val_acc, color=col, label=f"{label} ({best:.1f}%)", linewidth=1.6
        )
    ax_curve.set_title("Validation Accuracy")
    ax_curve.set_xlabel("Epoch")
    ax_curve.set_ylabel("Top-1 Acc (%)")
    ax_curve.legend(fontsize=7.5)
    ax_curve.grid(True, alpha=0.3)

    # Val loss curve
    for label, rows in data:
        col = COLORS[label]
        epochs = _col(rows, "global_epoch")
        val_loss = _col(rows, "val_loss")
        ax_loss.plot(epochs, val_loss, color=col, label=label, linewidth=1.6)
    ax_loss.set_title("Validation Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=7.5)
    ax_loss.grid(True, alpha=0.3)

    # Bar: best acc
    accs = [MODEL_META[l]["best_val_acc"] * 100 for l in LABELS]
    colors = [COLORS[l] for l in LABELS]
    short = ["ResNet-18", "CNN3 702k", "ResCNN 2048"]
    bars = ax_bar_acc.bar(short, accs, color=colors, edgecolor="white")
    for bar, v in zip(bars, accs):
        ax_bar_acc.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.5,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    ax_bar_acc.set_title("Best Val Accuracy")
    ax_bar_acc.set_ylim(0, max(accs) * 1.25)
    ax_bar_acc.set_ylabel("Top-1 Acc (%)")
    ax_bar_acc.set_xticklabels(short, rotation=10, ha="right", fontsize=8)
    ax_bar_acc.grid(axis="y", alpha=0.3)

    # Bar: params
    params = [MODEL_META[l]["params"] / 1e6 for l in LABELS]
    bars2 = ax_bar_par.bar(short, params, color=colors, edgecolor="white")
    for bar, v, l in zip(bars2, params, LABELS):
        sp = MODEL_META[l]["sparsity"]
        note = f"\n{sp * 100:.0f}%↓" if sp > 0 else ""
        ax_bar_par.text(
            bar.get_x() + bar.get_width() / 2,
            v + max(params) * 0.02,
            f"{v:.1f}M{note}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
        )
    ax_bar_par.set_title("Parameters")
    ax_bar_par.set_ylabel("Params (M)")
    ax_bar_par.set_xticklabels(short, rotation=10, ha="right", fontsize=8)
    ax_bar_par.grid(axis="y", alpha=0.3)

    # Scatter: params vs acc
    for lbl in LABELS:
        pM = MODEL_META[lbl]["params"] / 1e6
        acc = MODEL_META[lbl]["best_val_acc"] * 100
        col = COLORS[lbl]
        ax_eff.scatter(pM, acc, color=col, s=120, zorder=5)
        ax_eff.annotate(
            short[LABELS.index(lbl)],
            xy=(pM, acc),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=7,
            color=col,
        )
    ax_eff.set_title("Params vs Accuracy")
    ax_eff.set_xlabel("Params (M)")
    ax_eff.set_ylabel("Top-1 Acc (%)")
    ax_eff.grid(True, alpha=0.3)

    # Sparsity / pruning panel for CNN3
    cnn3_prune = [r for r in cnn3_rows if r.get("sparsity") is not None]
    if cnn3_prune:
        ep = [r["global_epoch"] for r in cnn3_prune]
        sp = [r["sparsity"] * 100 for r in cnn3_prune]
        ac = [r["val_acc"] * 100 for r in cnn3_prune]
        ax2 = ax_sparse.twinx()
        ax_sparse.plot(ep, sp, color="#e07b39", linewidth=1.5, label="Sparsity")
        ax2.plot(
            ep, ac, color="#4c9be8", linewidth=1.5, linestyle="--", label="Val Acc"
        )
        ax_sparse.set_xlabel("Epoch")
        ax_sparse.set_ylabel("Sparsity (%)", color="#e07b39")
        ax2.set_ylabel("Val Acc (%)", color="#4c9be8")
        ax_sparse.set_title("CNN3: Sparsity vs Acc")
        ax_sparse.grid(True, alpha=0.3)
        lines1, labels1 = ax_sparse.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax_sparse.legend(
            lines1 + lines2, labels1 + labels2, fontsize=7, loc="lower right"
        )
    else:
        ax_sparse.axis("off")
        ax_sparse.text(
            0.5,
            0.5,
            "No pruning\ndata available",
            ha="center",
            va="center",
            transform=ax_sparse.transAxes,
            fontsize=9,
        )

    # Summary table
    ax_table.axis("off")
    table_data = []
    header = [
        "Model",
        "Params",
        "Best Val Acc",
        "Efficiency\n(Acc%/M)",
        "Reservoir",
        "Pruning",
        "Sparsity",
        "Epochs\n(done/total)",
        "Batch",
    ]
    for lbl in LABELS:
        m = MODEL_META[lbl]
        ep = f"{m['best_epoch']}/{m['total_epochs']}"
        eff = m["best_val_acc"] * 100 / (m["params"] / 1e6)
        table_data.append(
            [
                lbl,
                f"{m['params']:,}",
                f"{m['best_val_acc'] * 100:.2f}%",
                f"{eff:.1f}",
                m["reservoir"],
                "Yes (95%)" if m["pruning"] else "No",
                f"{m['sparsity'] * 100:.0f}%",
                ep,
                str(m["batch_size"]),
            ]
        )
    tbl = ax_table.table(
        cellText=table_data,
        colLabels=header,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c2c3e")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f0f5")
        cell.set_linewidth(0.5)

    path = OUT / "07_combined_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 8 – LR schedule comparison
# ---------------------------------------------------------------------------


def fig_lr_schedule(rn_rows, cnn3_rows, rescnn_rows):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle("Learning Rate Schedule Comparison", fontsize=13, fontweight="bold")

    for label, rows in [
        ("ResNet-18", rn_rows),
        ("CNN3+TPSAPU (702k)", cnn3_rows),
        ("ResCNN+TPSAPU (2048 neurons)", rescnn_rows),
    ]:
        epochs = _col(rows, "global_epoch")
        lrs = _col(rows, "lr")
        ax.plot(epochs, lrs, color=COLORS[label], label=label, linewidth=1.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("LR over Epochs (log scale)")

    plt.tight_layout()
    path = OUT / "08_lr_schedule.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 9 – Train/Val acc gap (overfitting check)
# ---------------------------------------------------------------------------


def fig_overfit(rn_rows, cnn3_rows, rescnn_rows):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    fig.suptitle(
        "Train vs Validation Accuracy (Overfitting)", fontsize=13, fontweight="bold"
    )

    for ax, (label, rows) in zip(
        axes,
        [
            ("ResNet-18", rn_rows),
            ("CNN3+TPSAPU (702k)", cnn3_rows),
            ("ResCNN+TPSAPU (2048 neurons)", rescnn_rows),
        ],
    ):
        col = COLORS[label]
        epochs = _col(rows, "global_epoch")
        t_acc = [v * 100 for v in _col(rows, "train_acc")]
        v_acc = [v * 100 for v in _col(rows, "val_acc")]
        gap = [t - v for t, v in zip(t_acc, v_acc)]

        ax.plot(epochs, t_acc, color=col, linewidth=1.8, label="train")
        ax.plot(
            epochs,
            v_acc,
            color=col,
            linewidth=1.8,
            linestyle="--",
            alpha=0.7,
            label="val",
        )
        ax.fill_between(epochs, v_acc, t_acc, alpha=0.15, color=col, label="gap")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUT / "09_overfit_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip running model evaluations (use cached or skip per-class figs).",
    )
    p.add_argument("--batch-size", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    train_pipeline.set_seed(42)

    print("Loading metrics CSVs …")
    rn_rows = load_resnet_metrics(RESNET_METRICS)
    cnn3_rows = load_tpsapu_metrics(CNN3_METRICS)
    rescnn_rows = load_tpsapu_metrics(RESCNN_METRICS)

    print(
        f"  ResNet-18  : {len(rn_rows)} epochs, best={max(_col(rn_rows, 'best_val_acc')):.4f}"
    )
    print(
        f"  CNN3+702k  : {len(cnn3_rows)} epochs, best={max(_col(cnn3_rows, 'best_val_acc')):.4f}"
    )
    print(
        f"  ResCNN+2048: {len(rescnn_rows)} epochs, best={max(_col(rescnn_rows, 'best_val_acc')):.4f}"
    )

    # ------------------------------------------------------------------
    # Run evaluations
    # ------------------------------------------------------------------
    eval_results: dict[str, dict | None] = {
        "ResNet-18": None,
        "CNN3+TPSAPU (702k)": None,
        "ResCNN+TPSAPU (2048 neurons)": None,
    }

    if not args.skip_eval:
        eval_results["ResNet-18"] = evaluate_resnet(args)
        eval_results["CNN3+TPSAPU (702k)"] = evaluate_tpsapu(
            CNN3_CKPT, OUT / "eval_cnn3_702k", args
        )
        eval_results["ResCNN+TPSAPU (2048 neurons)"] = evaluate_tpsapu(
            RESCNN_CKPT, OUT / "eval_rescnn_2048", args
        )
    else:
        # Try to load cached eval outputs
        for lbl, out_subdir in [
            ("ResNet-18", OUT / "eval_resnet18"),
            ("CNN3+TPSAPU (702k)", OUT / "eval_cnn3_702k"),
            ("ResCNN+TPSAPU (2048 neurons)", OUT / "eval_rescnn_2048"),
        ]:
            mp = out_subdir / "metrics.json"
            pp = out_subdir / "per_class_metrics.csv"
            if mp.exists() and pp.exists():
                with open(mp) as f:
                    metrics = json.load(f)
                eval_results[lbl] = {
                    "metrics": metrics,
                    "per_class": _load_per_class(pp),
                }

    # Also try existing eval_latest_full_validation for cnn3 model
    if eval_results["CNN3+TPSAPU (702k)"] is None:
        legacy = BASE / "eval_latest_full_validation"
        mp = legacy / "metrics.json"
        pp = legacy / "per_class_metrics.csv"
        if mp.exists() and pp.exists():
            with open(mp) as f:
                metrics = json.load(f)
            eval_results["CNN3+TPSAPU (702k)"] = {
                "metrics": metrics,
                "per_class": _load_per_class(pp),
            }
            print("[eval] Using existing eval_latest_full_validation for CNN3+702k.")

    # ------------------------------------------------------------------
    # Generate all figures
    # ------------------------------------------------------------------
    print("\nGenerating figures …")
    fig_training_curves(rn_rows, cnn3_rows, rescnn_rows)
    fig_model_summary()
    fig_sparsity_analysis(cnn3_rows)
    fig_per_class(eval_results)
    fig_class_breakdown(eval_results)
    fig_efficiency(eval_results)
    fig_combined_summary(rn_rows, cnn3_rows, rescnn_rows, eval_results)
    fig_lr_schedule(rn_rows, cnn3_rows, rescnn_rows)
    fig_overfit(rn_rows, cnn3_rows, rescnn_rows)

    # ------------------------------------------------------------------
    # Print final summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"{'Model':<35} {'Params':>12} {'BestAcc':>9} {'Top5':>8} {'Eff':>7}")
    print("-" * 72)
    for lbl in LABELS:
        m = MODEL_META[lbl]
        res = eval_results.get(lbl)
        t1 = res["metrics"]["topk"]["1"] * 100 if res else m["best_val_acc"] * 100
        t5_raw = res["metrics"]["topk"].get("5") if res else None
        t5 = f"{t5_raw * 100:.1f}%" if t5_raw else "—"
        eff = t1 / (m["params"] / 1e6)
        print(f"{lbl:<35} {m['params']:>12,} {t1:>8.2f}% {t5:>8} {eff:>6.1f}")
    print("=" * 72)
    print(f"\nAll outputs saved to: {OUT}")


if __name__ == "__main__":
    main()
