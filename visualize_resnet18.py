"""Per-sample and aggregate visualizations for the ResNet-18 checkpoint.

Produces the same folder layout as visualize_cnn3_voltages.py but adapted
to ResNet-18 layers instead of TPSAPU neurons.  All spectrum metrics
(effective rank, stable rank, top-k energy) use the same definition as
analyze_best_checkpoint_structure.py.

Per-sample outputs  (visualizations/resnet18/<split>_<idx>/):
  input_and_layers.png      – image + per-layer spatial norm maps
  top_channels.png          – strongest feature channels per layer
  layer_activation_matrix.png – channel×space heatmap (analog of neuron voltage matrix)
  arrays.npz                – raw activation tensors

Aggregate outputs (visualizations/resnet18/):
  effective_rank_per_layer.png   – eff-rank, stable-rank, top-1-energy vs depth
  singular_value_spectra.png     – singular-value spectrum per layer
  activation_statistics.png      – mean-abs, std, sparsity per layer
  summary.json                   – all aggregate metrics as JSON
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import svdvals
from torch.utils.data import DataLoader
from torchvision import models, transforms

import train_pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TINY_IMAGENET_MEAN = (0.485, 0.456, 0.406)
TINY_IMAGENET_STD = (0.229, 0.224, 0.225)

# ResNet-18 layer names in forward order
LAYER_NAMES = ["stem", "layer1", "layer2", "layer3", "layer4", "avgpool"]

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint", default="sweep_runs_tiny_imagenet/resnet18_scratch/best.pt"
    )
    p.add_argument("--data-dir", default="data/tiny-imagenet-200-clean")
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--output-dir", default="visualizations/resnet18")
    p.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size used for aggregate spectrum analysis.",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of validation samples to use for aggregate stats (0 = all).",
    )
    p.add_argument(
        "--vis-indices",
        default="0,1,2,3,4,5,6,7,8,9",
        help="Comma-separated validation indices to produce per-sample visualizations for.",
    )
    p.add_argument(
        "--num-feature-channels",
        type=int,
        default=16,
        help="How many strongest channels to show in top_channels.png.",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-download",
        action="store_true",
        default=True,
        help="Use cached HF data only.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model building + hooking
# ---------------------------------------------------------------------------


def build_resnet18(checkpoint_path: str, device: torch.device) -> nn.Module:
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 200)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model.to(device)


class ActivationCapture:
    """Register forward hooks on named ResNet submodules and collect outputs."""

    def __init__(self, model: nn.Module) -> None:
        self.activations: dict[str, torch.Tensor] = {}
        self._hooks = []
        targets = {
            "stem": model.layer1,  # output right before layer1 = after initial conv+bn+relu
            "layer1": model.layer1,
            "layer2": model.layer2,
            "layer3": model.layer3,
            "layer4": model.layer4,
            "avgpool": model.avgpool,
        }
        # We hook layer1..layer4 and avgpool directly.
        # For "stem" we hook the output of the first conv block (before layer1) by hooking
        # bn1 + relu via a sequential pre-hook isn't easy so we capture after relu1 instead.
        # Practical approach: hook each layer directly.
        targets2 = {
            "layer1": model.layer1,
            "layer2": model.layer2,
            "layer3": model.layer3,
            "layer4": model.layer4,
            "avgpool": model.avgpool,
        }
        for name, mod in targets2.items():
            h = mod.register_forward_hook(self._make_hook(name))
            self._hooks.append(h)
        # Stem: hook on relu (nn.ReLU inside model directly)
        # Easiest: register on the full model and intercept before layer1
        # We do it by wrapping layer1's pre-hook instead
        h_stem = model.layer1.register_forward_pre_hook(self._stem_hook())
        self._hooks.append(h_stem)

    def _make_hook(self, name: str):
        def hook(module, inp, output):
            self.activations[name] = output.detach().cpu()

        return hook

    def _stem_hook(self):
        # The input to layer1 is the output of stem (conv1→bn1→relu1)
        def hook(module, inp):
            self.activations["stem"] = inp[0].detach().cpu()

        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()


# ---------------------------------------------------------------------------
# Spectrum metrics  (same definition as analyze_best_checkpoint_structure.py)
# ---------------------------------------------------------------------------


def spectrum_metrics(matrix: np.ndarray) -> dict[str, float]:
    """SVD-based spectrum metrics on a 2-D matrix."""
    singular = svdvals(matrix, overwrite_a=False, check_finite=False)
    energy = singular**2
    prob = energy / max(float(energy.sum()), 1e-12)
    entropy = -float(np.sum(prob * np.log(np.maximum(prob, 1e-300))))
    max_rank = min(matrix.shape)
    return {
        "dimension": int(max_rank),
        "spectral_norm": float(singular[0]),
        "stable_rank": float(energy.sum() / max(float(energy[0]), 1e-12)),
        "effective_rank": float(np.exp(entropy)),
        "spectral_entropy": float(entropy / np.log(max(len(singular), 2))),
        "top1_energy": float(prob[0]),
        "top5_energy": float(prob[:5].sum()),
        "top16_energy": float(prob[:16].sum()),
        "singular_values": singular.tolist(),
    }


def layer_spectrum(act: torch.Tensor) -> dict[str, float]:
    """Compute spectrum metrics on layer activations.

    act : (N, C, H, W) or (N, C, 1, 1) or (N, C)
    We reshape to (N, C*H*W) and run SVD.
    """
    flat = act.reshape(act.shape[0], -1).numpy().astype(np.float32)
    # Mean-center across samples (so we analyse the covariance structure)
    flat = flat - flat.mean(axis=0, keepdims=True)
    return spectrum_metrics(flat)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_validation_split(args: argparse.Namespace):
    hf_dataset = train_pipeline.load_hf_split(
        "slegroux/tiny-imagenet-200-clean",
        split=args.split,
        cache_dir=args.data_dir,
        no_download=args.no_download,
    )
    transform = transforms.Compose(
        [
            transforms.Resize(int(round(64 * 256 / 224))),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
        ]
    )

    class _DS(torch.utils.data.Dataset):
        def __init__(self, hf):
            self.hf = hf

        def __len__(self):
            return len(self.hf)

        def __getitem__(self, i):
            item = self.hf[i]
            img = item["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            return transform(img), item["label"], i

    return _DS(hf_dataset), list(hf_dataset.features["label"].names)


def denormalize(t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(TINY_IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(TINY_IMAGENET_STD).view(3, 1, 1)
    img = (t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def load_human_names(data_dir: str) -> dict[str, str]:
    p = Path(data_dir).parent.parent / "LOC_synset_mapping.txt"
    if not p.exists():
        p = Path("LOC_synset_mapping.txt")
    if not p.exists():
        return {}
    names = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        synset, rest = line.split(maxsplit=1)
        names[synset] = rest.split(",")[0].strip()
    return names


# ---------------------------------------------------------------------------
# Per-sample visualizations
# ---------------------------------------------------------------------------


def normalize_for_display(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi <= lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def save_input_and_layers(
    path: Path,
    image: np.ndarray,
    activations: dict[str, torch.Tensor],
    label: int,
    class_names: list[str],
    logits: torch.Tensor,
) -> None:
    probs = torch.softmax(logits, dim=-1)
    pred = int(probs.argmax())
    conf = float(probs[pred])
    gt = class_names[label] if label < len(class_names) else str(label)
    pn = class_names[pred] if pred < len(class_names) else str(pred)
    result_text = "CORRECT" if pred == label else "WRONG"

    layer_order = [l for l in LAYER_NAMES if l in activations]
    n_layers = len(layer_order)
    fig, axes = plt.subplots(1, n_layers + 1, figsize=(4 * (n_layers + 1), 4))

    axes[0].imshow(image)
    axes[0].set_title(
        f"truth: {gt} ({label})\npred: {pn} ({pred})\nconf: {conf:.1%}  {result_text}",
        fontsize=9,
    )
    axes[0].axis("off")

    for ax, lname in zip(axes[1:], layer_order):
        act = activations[lname]  # (1, C, H, W) or (1, C)
        if act.dim() == 4:
            spatial_norm = act[0].norm(dim=0).numpy()  # (H, W)
        else:
            spatial_norm = act[0].numpy().reshape(1, -1)  # (1, C) → flat
        im = ax.imshow(normalize_for_display(spatial_norm), cmap="magma")
        ax.set_title(f"{lname}\n{list(act.shape[1:])} feat", fontsize=8)
        ax.set_xlabel("feature x")
        ax.set_ylabel("feature y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_top_channels(
    path: Path,
    activations: dict[str, torch.Tensor],
    num_channels: int,
) -> None:
    layer_order = [
        l
        for l in LAYER_NAMES
        if l in activations
        and activations[l].dim() == 4
        and activations[l].shape[1] >= 4
    ]
    if not layer_order:
        return

    cols = min(4, num_channels)
    rows = math.ceil(num_channels / cols)
    n_layers = len(layer_order)
    fig, big_axes = plt.subplots(
        n_layers, 1, figsize=(cols * 2.5, n_layers * rows * 2.3 + 0.5)
    )
    if n_layers == 1:
        big_axes = [big_axes]
    fig.suptitle("Top channels by mean absolute activation", fontsize=11)

    for big_ax, lname in zip(big_axes, layer_order):
        big_ax.axis("off")
        big_ax.set_title(lname, fontsize=9, fontweight="bold", loc="left")

    # Re-draw with gridspecs
    fig.clf()
    gs_outer = fig.add_gridspec(n_layers, 1, hspace=0.7)
    for layer_idx, lname in enumerate(layer_order):
        act = activations[lname][0]  # (C, H, W)
        nc = min(num_channels, act.shape[0])
        scores = act.abs().mean(dim=(1, 2))
        selected = torch.topk(scores, k=nc).indices.tolist()
        cols_l = min(4, nc)
        rows_l = math.ceil(nc / cols_l)
        gs_inner = gs_outer[layer_idx].subgridspec(
            rows_l, cols_l, hspace=0.3, wspace=0.15
        )
        for panel_idx in range(nc):
            r, c = divmod(panel_idx, cols_l)
            ax = fig.add_subplot(gs_inner[r, c])
            ch = selected[panel_idx]
            ax.imshow(normalize_for_display(act[ch].numpy()), cmap="viridis")
            ax.set_title(f"{lname}[{ch}]", fontsize=6)
            ax.axis("off")
        # Title row label
        title_ax = fig.add_subplot(gs_outer[layer_idx])
        title_ax.set_visible(False)
        title_ax.set_title(lname, fontsize=9, fontweight="bold")

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_layer_activation_matrix(
    path: Path,
    activations: dict[str, torch.Tensor],
    label: int,
    class_names: list[str],
    logits: torch.Tensor,
) -> None:
    """Analog of tpsapu_all_neuron_voltage_matrix.png.

    Rows = channels, columns = spatial positions (or just 1 for avgpool).
    One panel per layer.
    """
    layer_order = [l for l in LAYER_NAMES if l in activations]
    n_layers = len(layer_order)
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 7))
    if n_layers == 1:
        axes = [axes]

    probs = torch.softmax(logits, dim=-1)
    pred = int(probs.argmax())
    gt = class_names[label] if label < len(class_names) else str(label)
    pn = class_names[pred] if pred < len(class_names) else str(pred)
    correct = "✓" if pred == label else "✗"
    fig.suptitle(
        f"Layer activation matrices  |  truth: {gt}  pred: {pn} {correct}",
        fontsize=10,
    )

    for ax, lname in zip(axes, layer_order):
        act = activations[lname][0]  # (C, H, W) or (C,)
        if act.dim() == 3:
            C, H, W = act.shape
            mat = act.reshape(C, H * W).numpy()  # (C, H*W)
        else:
            mat = act.unsqueeze(-1).numpy()  # (C, 1)

        lim = float(np.percentile(np.abs(mat), 99))
        lim = lim if lim > 0 else 1.0
        im = ax.imshow(
            mat,
            aspect="auto",
            cmap="coolwarm",
            vmin=-lim,
            vmax=lim,
            interpolation="nearest",
        )
        ax.set_title(f"{lname}\n({C}ch × {mat.shape[1]}sp)", fontsize=8)
        ax.set_xlabel("spatial position")
        ax.set_ylabel("channel")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_arrays(
    path: Path,
    image_tensor: torch.Tensor,
    activations: dict[str, torch.Tensor],
    label: int,
    logits: torch.Tensor,
) -> None:
    probs = torch.softmax(logits, dim=-1)
    arrays = {
        "label": np.array(label, dtype=np.int64),
        "image": image_tensor.cpu().numpy(),
        "logits": logits.cpu().numpy(),
        "probabilities": probs.cpu().numpy(),
    }
    for lname, act in activations.items():
        arrays[f"act_{lname}"] = act[0].numpy()
    np.savez_compressed(path, **arrays)


# ---------------------------------------------------------------------------
# Aggregate spectrum analysis
# ---------------------------------------------------------------------------


def collect_aggregate_activations(
    model: nn.Module,
    dataset,
    device: torch.device,
    num_samples: int,
    batch_size: int,
) -> dict[str, list[torch.Tensor]]:
    """Return a dict layer_name → list of per-batch activation tensors (N, C, H, W)."""
    if num_samples > 0:
        indices = list(range(min(num_samples, len(dataset))))
        subset = torch.utils.data.Subset(dataset, indices)
    else:
        subset = dataset

    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    capture = ActivationCapture(model)
    collected: dict[str, list[torch.Tensor]] = {ln: [] for ln in LAYER_NAMES}

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            _ = model(images)
            for lname in LAYER_NAMES:
                if lname in capture.activations:
                    collected[lname].append(capture.activations[lname].cpu())

    capture.remove()

    # Concatenate along batch dimension
    result = {}
    for lname, batches in collected.items():
        if batches:
            result[lname] = torch.cat(batches, dim=0)
    return result


def compute_aggregate_spectra(
    all_acts: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    spectra = {}
    for lname, act in all_acts.items():
        print(f"  Computing spectrum for {lname}  {tuple(act.shape)} …", flush=True)
        metrics = layer_spectrum(act)
        del metrics["singular_values"]  # too large to keep in dict
        spectra[lname] = metrics
    return spectra


def compute_per_layer_singular_spectra(
    all_acts: dict[str, torch.Tensor],
    n_singular: int = 64,
) -> dict[str, np.ndarray]:
    """Return top-n_singular normalised singular values per layer."""
    result = {}
    for lname, act in all_acts.items():
        flat = act.reshape(act.shape[0], -1).numpy().astype(np.float32)
        flat = flat - flat.mean(axis=0, keepdims=True)
        sv = svdvals(flat, overwrite_a=False, check_finite=False)
        sv = sv / (sv[0] + 1e-12)  # normalise to leading SV = 1
        result[lname] = sv[:n_singular]
    return result


def compute_activation_statistics(
    all_acts: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    stats = {}
    for lname, act in all_acts.items():
        flat = act.reshape(act.shape[0], -1).float().numpy()
        stats[lname] = {
            "mean_abs": float(np.abs(flat).mean()),
            "std": float(flat.std()),
            "sparsity": float((np.abs(flat) < 1e-6).mean()),
            "max_abs": float(np.abs(flat).max()),
            "shape": list(act.shape[1:]),
        }
    return stats


# ---------------------------------------------------------------------------
# Aggregate figure functions
# ---------------------------------------------------------------------------


def fig_effective_rank_per_layer(
    spectra: dict[str, dict[str, float]],
    out_dir: Path,
) -> None:
    layer_order = [l for l in LAYER_NAMES if l in spectra]
    x = np.arange(len(layer_order))
    er = [spectra[l]["effective_rank"] for l in layer_order]
    sr = [spectra[l]["stable_rank"] for l in layer_order]
    t1 = [spectra[l]["top1_energy"] for l in layer_order]
    t5 = [spectra[l]["top5_energy"] for l in layer_order]
    t16 = [spectra[l]["top16_energy"] for l in layer_order]
    se = [spectra[l]["spectral_entropy"] for l in layer_order]
    d = [spectra[l]["dimension"] for l in layer_order]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        "ResNet-18: Spectral Properties per Layer", fontsize=13, fontweight="bold"
    )

    kw = dict(marker="o", linewidth=2)

    axes[0, 0].plot(x, er, color="#e07b39", **kw)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(layer_order, rotation=15)
    axes[0, 0].set_ylabel("Effective Rank")
    axes[0, 0].set_title("Effective Rank (exp H of SV² dist)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(x, sr, color="#4c9be8", **kw)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(layer_order, rotation=15)
    axes[0, 1].set_ylabel("Stable Rank")
    axes[0, 1].set_title("Stable Rank (||A||_F² / ||A||²)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(x, t1, label="top-1", color="#e07b39", **kw)
    axes[0, 2].plot(x, t5, label="top-5", color="#4c9be8", **kw)
    axes[0, 2].plot(x, t16, label="top-16", color="#6bbf59", **kw)
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(layer_order, rotation=15)
    axes[0, 2].set_ylabel("Cumulative energy fraction")
    axes[0, 2].set_title("Singular Value Energy Concentration")
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(x, se, color="#9b59b6", **kw)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(layer_order, rotation=15)
    axes[1, 0].set_ylabel("Spectral Entropy (normalised)")
    axes[1, 0].set_title("Spectral Entropy")
    axes[1, 0].grid(True, alpha=0.3)

    # Effective rank as fraction of dimension
    er_frac = [e / d_ for e, d_ in zip(er, d)]
    axes[1, 1].plot(x, er_frac, color="#e74c3c", **kw)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(layer_order, rotation=15)
    axes[1, 1].set_ylabel("Eff-rank / dimension")
    axes[1, 1].set_title("Fractional Effective Rank")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].set_visible(False)

    plt.tight_layout()
    path = out_dir / "effective_rank_per_layer.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_singular_value_spectra(
    sv_spectra: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    layer_order = [l for l in LAYER_NAMES if l in sv_spectra]
    n_layers = len(layer_order)
    cols = min(3, n_layers)
    rows = math.ceil(n_layers / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    fig.suptitle(
        "ResNet-18: Normalised Singular Value Spectra per Layer",
        fontsize=12,
        fontweight="bold",
    )

    cmap = cm.get_cmap("tab10")
    for idx, lname in enumerate(layer_order):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        sv = sv_spectra[lname]
        x = np.arange(1, len(sv) + 1)
        ax.plot(x, sv, color=cmap(idx / max(n_layers - 1, 1)), linewidth=1.8)
        ax.set_xlabel("Singular value rank")
        ax.set_ylabel("Normalised σᵢ / σ₁")
        ax.set_title(lname)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    # Hide unused panels
    for idx in range(len(layer_order), rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    plt.tight_layout()
    path = out_dir / "singular_value_spectra.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_activation_statistics(
    act_stats: dict[str, dict[str, float]],
    out_dir: Path,
) -> None:
    layer_order = [l for l in LAYER_NAMES if l in act_stats]
    x = np.arange(len(layer_order))
    mean_abs = [act_stats[l]["mean_abs"] for l in layer_order]
    std_vals = [act_stats[l]["std"] for l in layer_order]
    sparsity = [act_stats[l]["sparsity"] * 100 for l in layer_order]
    max_abs = [act_stats[l]["max_abs"] for l in layer_order]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        "ResNet-18: Activation Statistics per Layer", fontsize=12, fontweight="bold"
    )

    for ax, vals, ylabel, title, color in [
        (axes[0], mean_abs, "Mean |activation|", "Mean Absolute Activation", "#e07b39"),
        (axes[1], std_vals, "Std dev", "Activation Std", "#4c9be8"),
        (axes[2], sparsity, "Near-zero fraction (%)", "Sparsity (|a|<1e-6)", "#6bbf59"),
        (axes[3], max_abs, "Max |activation|", "Max Absolute Activation", "#9b59b6"),
    ]:
        ax.bar(x, vals, color=color, edgecolor="white", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(layer_order, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(
                i,
                v * 1.02 + max(vals) * 0.01,
                f"{v:.2f}" if v < 100 else f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    path = out_dir / "activation_statistics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_effective_rank_comparison(
    spectra: dict[str, dict[str, float]],
    out_dir: Path,
) -> None:
    """Two-panel summary: effective rank bar + rank-fraction scatter."""
    layer_order = [l for l in LAYER_NAMES if l in spectra]
    er = [spectra[l]["effective_rank"] for l in layer_order]
    sr = [spectra[l]["stable_rank"] for l in layer_order]
    d = [spectra[l]["dimension"] for l in layer_order]
    t1 = [spectra[l]["top1_energy"] * 100 for l in layer_order]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(layer_order)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("ResNet-18: Effective Rank Summary", fontsize=13, fontweight="bold")

    bars = axes[0].bar(layer_order, er, color=colors, edgecolor="white")
    for bar, v, d_ in zip(bars, er, d):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            v + max(er) * 0.02,
            f"{v:.1f}\n/{d_}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[0].set_ylabel("Effective Rank")
    axes[0].set_title("Effective Rank per Layer\n(annotated with /max_rank)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_xticklabels(layer_order, rotation=15, ha="right")

    # Scatter: stable rank vs effective rank (colour = layer depth)
    sc = axes[1].scatter(
        er, sr, c=np.arange(len(layer_order)), cmap="viridis", s=200, zorder=5
    )
    for i, lname in enumerate(layer_order):
        axes[1].annotate(
            lname,
            xy=(er[i], sr[i]),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1].set_xlabel("Effective Rank")
    axes[1].set_ylabel("Stable Rank")
    axes[1].set_title("Effective vs Stable Rank\n(deeper = lighter)")
    axes[1].grid(True, alpha=0.3)
    fig.colorbar(sc, ax=axes[1], label="layer depth")

    plt.tight_layout()
    path = out_dir / "effective_rank_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    train_pipeline.set_seed(args.seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ResNet-18 from {args.checkpoint} …")
    model = build_resnet18(args.checkpoint, device)

    print(f"Loading dataset split={args.split} …")
    dataset, class_names = load_validation_split(args)
    human_names = load_human_names(args.data_dir)
    # Replace synset labels with human-readable names
    display_names = [
        human_names.get(syn, syn).split(",")[0].strip() for syn in class_names
    ]

    # ------------------------------------------------------------------
    # Per-sample visualizations
    # ------------------------------------------------------------------
    vis_indices = [int(s.strip()) for s in args.vis_indices.split(",") if s.strip()]
    print(f"\nGenerating per-sample visualizations for indices: {vis_indices}")

    capture = ActivationCapture(model)
    for sample_idx in vis_indices:
        if sample_idx >= len(dataset):
            print(f"  Index {sample_idx} out of range, skipping.")
            continue

        image_tensor, label, _ = dataset[sample_idx]
        image_np = denormalize(image_tensor)
        batch = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(batch).squeeze(0).cpu()

        # Trim activations to single-sample (squeeze batch dim for storage)
        sample_acts: dict[str, torch.Tensor] = {}
        for lname in LAYER_NAMES:
            if lname in capture.activations:
                sample_acts[lname] = capture.activations[lname][:1]

        sample_dir = out_dir / f"{args.split}_{sample_idx}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        save_input_and_layers(
            sample_dir / "input_and_layers.png",
            image_np,
            sample_acts,
            label,
            display_names,
            logits,
        )
        save_top_channels(
            sample_dir / "top_channels.png",
            sample_acts,
            args.num_feature_channels,
        )
        save_layer_activation_matrix(
            sample_dir / "layer_activation_matrix.png",
            sample_acts,
            label,
            display_names,
            logits,
        )
        save_arrays(
            sample_dir / "arrays.npz",
            image_tensor,
            sample_acts,
            label,
            logits,
        )
        probs = torch.softmax(logits, dim=-1)
        pred = int(probs.argmax())
        print(
            f"  [{sample_idx}] truth={display_names[label]} "
            f"pred={display_names[pred]} conf={float(probs[pred]):.1%} "
            f"{'✓' if pred == label else '✗'}"
        )

    capture.remove()

    # ------------------------------------------------------------------
    # Aggregate spectrum analysis
    # ------------------------------------------------------------------
    print(f"\nCollecting activations over {args.num_samples or 'all'} samples …")
    all_acts = collect_aggregate_activations(
        model,
        dataset,
        device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )

    print("Computing spectral metrics …")
    spectra = compute_aggregate_spectra(all_acts)
    sv_spectra = compute_per_layer_singular_spectra(all_acts, n_singular=64)
    act_stats = compute_activation_statistics(all_acts)

    # ------------------------------------------------------------------
    # Aggregate figures
    # ------------------------------------------------------------------
    print("\nGenerating aggregate figures …")
    fig_effective_rank_per_layer(spectra, out_dir)
    fig_singular_value_spectra(sv_spectra, out_dir)
    fig_activation_statistics(act_stats, out_dir)
    fig_effective_rank_comparison(spectra, out_dir)

    # ------------------------------------------------------------------
    # Print + save summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(
        f"{'Layer':<12} {'EffRank':>10} {'StableRk':>10} "
        f"{'Top1E%':>8} {'SpEntropy':>10} {'Dim':>6}"
    )
    print("-" * 60)
    for lname in LAYER_NAMES:
        if lname not in spectra:
            continue
        s = spectra[lname]
        print(
            f"{lname:<12} {s['effective_rank']:>10.2f} "
            f"{s['stable_rank']:>10.2f} "
            f"{s['top1_energy'] * 100:>7.1f}% "
            f"{s['spectral_entropy']:>10.4f} "
            f"{s['dimension']:>6d}"
        )
    print("=" * 60)

    summary = {
        "model": "resnet18",
        "checkpoint": args.checkpoint,
        "num_samples": args.num_samples,
        "spectra": spectra,
        "activation_statistics": act_stats,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nAll outputs saved to: {out_dir}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
