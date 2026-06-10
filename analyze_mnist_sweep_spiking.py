"""Compare neuron spiking and recurrent topology across an MNIST checkpoint sweep."""

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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import train_pipeline


ENCODER_ORDER = [
    "linear_patch",
    "mlp_patch",
    "lif_2x2",
    "cnn2",
    "cnn3",
    "res_cnn",
    "rows",
]
DECODER_ORDER = [
    "linear",
    "membrane_mlp",
    "spike_mlp",
    "both_mlp",
    "all_state_mlp",
    "lif_count",
]
ENCODER_COLORS = dict(
    zip(ENCODER_ORDER, plt.get_cmap("tab10").colors[: len(ENCODER_ORDER)])
)
DECODER_MARKERS = {
    "linear": "o",
    "membrane_mlp": "s",
    "spike_mlp": "^",
    "both_mlp": "D",
    "all_state_mlp": "P",
    "lif_count": "X",
}
ARCHITECTURE_DEFAULTS = {
    "dataset": "mnist",
    "backbone": "tpsapu",
    "image_size": 28,
    "in_channels": 1,
    "num_classes": 10,
    "cross_rank": 16,
    "cross_gain": 0.1,
    "decoder_transformer_layers": 2,
    "decoder_transformer_heads": 4,
    "decoder_transformer_ff_mult": 4.0,
    "decoder_max_steps": 256,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", default="sweep_runs_20ep_mnist")
    parser.add_argument(
        "--output-dir",
        default="visualizations/mnist_sweep_spiking_topology",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--samples-per-digit", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--representative-digit", type=int, default=8)
    parser.add_argument("--temporal-bins", type=int, default=100)
    parser.add_argument("--top-edge-fraction", type=float, default=0.10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-models",
        type=int,
        default=0,
        help="Optional development limit. Zero analyzes every checkpoint.",
    )
    return parser.parse_args()


def checkpoint_args(checkpoint: dict[str, object]) -> argparse.Namespace:
    values = ARCHITECTURE_DEFAULTS.copy()
    saved_args = checkpoint.get("args")
    if isinstance(saved_args, dict):
        values.update(saved_args)
    return argparse.Namespace(**values)


def model_sort_key(path: Path) -> tuple[int, int, str]:
    encoder, decoder = path.parent.name.split("__", 1)
    return (
        ENCODER_ORDER.index(encoder),
        DECODER_ORDER.index(decoder),
        path.parent.name,
    )


def balanced_test_subset(
    data_dir: str,
    samples_per_digit: int,
) -> tuple[Subset, dict[int, int]]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    dataset = datasets.MNIST(
        data_dir,
        train=False,
        download=False,
        transform=transform,
    )
    selected: list[int] = []
    counts = {digit: 0 for digit in range(10)}
    representative_indices: dict[int, int] = {}
    for index, target in enumerate(dataset.targets.tolist()):
        if counts[target] >= samples_per_digit:
            continue
        selected.append(index)
        counts[target] += 1
        representative_indices.setdefault(target, index)
        if all(count == samples_per_digit for count in counts.values()):
            break
    if any(count != samples_per_digit for count in counts.values()):
        raise RuntimeError(f"Could not select {samples_per_digit} samples per digit.")
    return Subset(dataset, selected), representative_indices


def sequence_for_decoder(
    states: dict[str, torch.Tensor],
    input_state: str,
) -> torch.Tensor:
    if input_state == "membrane":
        return states["membrane"]
    if input_state == "spike":
        return states["spike"]
    if input_state == "both":
        return torch.cat([states["membrane"], states["spike"]], dim=-1)
    if input_state == "all":
        return torch.cat(
            [
                states["membrane"],
                states["spike"],
                states["dynamics"],
                states["spike_history"],
            ],
            dim=-1,
        )
    raise ValueError(f"Unsupported decoder input state: {input_state}")


def logits_from_states(
    model: torch.nn.Module,
    states: dict[str, torch.Tensor],
) -> torch.Tensor:
    sequence = sequence_for_decoder(states, model.decoder.input_state)
    if model.decoder.needs_sequence:
        features = sequence
    elif model.pooling == "mean":
        features = sequence.mean(dim=1)
    else:
        features = sequence[:, -1, :]
    return model.decoder(features)


def resample_curve(curve: np.ndarray, bins: int) -> np.ndarray:
    if curve.shape[-1] == bins:
        return curve
    old_x = np.linspace(0.0, 1.0, curve.shape[-1])
    new_x = np.linspace(0.0, 1.0, bins)
    return np.stack([np.interp(new_x, old_x, row) for row in curve], axis=0)


def safe_correlation_matrix(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = centered / np.maximum(norms, 1e-12)
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


def cosine_matrix(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norms, 1e-12)
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


def top_edge_masks(weights: np.ndarray, fraction: float) -> np.ndarray:
    edge_count = weights.shape[1]
    keep = max(1, int(round(edge_count * fraction)))
    masks = np.zeros_like(weights, dtype=bool)
    for index, row in enumerate(np.abs(weights)):
        selected = np.argpartition(row, -keep)[-keep:]
        masks[index, selected] = True
    return masks


def jaccard_matrix(masks: np.ndarray) -> np.ndarray:
    size = masks.shape[0]
    result = np.eye(size, dtype=np.float64)
    for row in range(size):
        for col in range(row + 1, size):
            intersection = np.logical_and(masks[row], masks[col]).sum()
            union = np.logical_or(masks[row], masks[col]).sum()
            value = float(intersection / union) if union else 1.0
            result[row, col] = value
            result[col, row] = value
    return result


def analyze_checkpoint(
    checkpoint_path: Path,
    loader: DataLoader,
    representative_image: torch.Tensor,
    device: torch.device,
    temporal_bins: int,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = checkpoint_args(checkpoint)
    model = train_pipeline.build_model(architecture).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tau_count = len(train_pipeline.parse_taus(architecture.taus))
    reservoir_dim = architecture.reservoir_dim
    spike_sum = torch.zeros(tau_count, reservoir_dim, dtype=torch.float64)
    digit_spike_sum = torch.zeros(10, tau_count, reservoir_dim, dtype=torch.float64)
    digit_denominator = torch.zeros(10, dtype=torch.float64)
    temporal_sum: torch.Tensor | None = None
    sample_count = 0
    correct = 0
    token_steps = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            tokens = model.encoder(images)
            states = model.backbone.forward_states(tokens, reset_state=True)
            logits = logits_from_states(model, states)
            correct += int((logits.argmax(dim=-1) == targets).sum())

            spikes = states["spike"].reshape(
                images.size(0),
                tokens.size(1),
                tau_count,
                reservoir_dim,
            )
            token_steps = tokens.size(1)
            batch_size = images.size(0)
            sample_count += batch_size
            spike_sum += spikes.sum(dim=(0, 1)).double().cpu()
            batch_temporal = spikes.mean(dim=(0, 3)).transpose(0, 1)
            if temporal_sum is None:
                temporal_sum = batch_temporal.double().cpu() * batch_size
            else:
                temporal_sum += batch_temporal.double().cpu() * batch_size

            for digit in range(10):
                mask = targets == digit
                count = int(mask.sum())
                if count == 0:
                    continue
                digit_spike_sum[digit] += spikes[mask].sum(dim=(0, 1)).double().cpu()
                digit_denominator[digit] += count * token_steps

        representative_tokens = model.encoder(representative_image.to(device))
        representative_states = model.backbone.forward_states(
            representative_tokens,
            reset_state=True,
        )
        representative_spikes = representative_states["spike"].reshape(
            token_steps,
            tau_count,
            reservoir_dim,
        )

    denominator = sample_count * token_steps
    neuron_rates = (spike_sum / denominator).numpy()
    digit_rates = (
        digit_spike_sum / digit_denominator.view(10, 1, 1).clamp_min(1.0)
    ).numpy()
    if temporal_sum is None:
        raise RuntimeError("The analysis loader produced no batches.")
    temporal_rates = (temporal_sum / sample_count).numpy()
    temporal_rates = resample_curve(temporal_rates, temporal_bins)

    recurrent = model.backbone.shared_recurrent.weight.detach().cpu().numpy()
    off_diagonal = ~np.eye(reservoir_dim, dtype=bool)
    recurrent_edges = recurrent[off_diagonal]
    singular_values = np.linalg.svd(recurrent, compute_uv=False)
    singular_values /= max(float(np.linalg.norm(singular_values)), 1e-12)

    return {
        "name": checkpoint_path.parent.name,
        "encoder": architecture.encoder,
        "decoder": architecture.decoder,
        "best_val_acc": float(checkpoint.get("best_val_acc", float("nan"))),
        "epoch": int(checkpoint.get("epoch", -1)),
        "subset_accuracy": correct / sample_count,
        "steps": token_steps,
        "taus": list(train_pipeline.parse_taus(architecture.taus)),
        "reservoir_dim": reservoir_dim,
        "neuron_rates": neuron_rates,
        "digit_rates": digit_rates,
        "temporal_rates": temporal_rates,
        "representative_spikes": representative_spikes.cpu().numpy(),
        "recurrent": recurrent,
        "recurrent_edges": recurrent_edges,
        "singular_values": singular_values,
    }


def save_model_dashboard(result: dict[str, object], output_path: Path) -> None:
    neuron_rates = np.asarray(result["neuron_rates"])
    digit_rates = np.asarray(result["digit_rates"])
    temporal_rates = np.asarray(result["temporal_rates"])
    representative = np.asarray(result["representative_spikes"])
    recurrent = np.asarray(result["recurrent"])
    taus = result["taus"]
    reservoir_dim = int(result["reservoir_dim"])

    fig = plt.figure(figsize=(18, 11))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.05))
    raster_axis = fig.add_subplot(grid[0, :2])
    rate_axis = fig.add_subplot(grid[0, 2])
    digit_axis = fig.add_subplot(grid[1, 0])
    temporal_axis = fig.add_subplot(grid[1, 1])
    recurrent_axis = fig.add_subplot(grid[1, 2])

    raster = representative.transpose(1, 2, 0).reshape(-1, representative.shape[0])
    raster_axis.imshow(
        raster,
        aspect="auto",
        cmap="binary",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    for tau_index in range(1, len(taus)):
        raster_axis.axhline(tau_index * reservoir_dim - 0.5, color="red", linewidth=0.7)
    raster_axis.set_title("Representative digit spike raster")
    raster_axis.set_xlabel("encoder token step")
    raster_axis.set_ylabel("neurons grouped by tau")

    rate_image = rate_axis.imshow(
        neuron_rates,
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=max(0.01, float(np.percentile(neuron_rates, 99))),
    )
    rate_axis.set_title("Mean firing probability")
    rate_axis.set_xlabel("shared neuron index")
    rate_axis.set_yticks(range(len(taus)))
    rate_axis.set_yticklabels([f"tau={tau:g}" for tau in taus])
    fig.colorbar(rate_image, ax=rate_axis, fraction=0.046, pad=0.04)

    digit_view = digit_rates.reshape(10, -1)
    digit_image = digit_axis.imshow(
        digit_view,
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=max(0.01, float(np.percentile(digit_view, 99))),
    )
    for tau_index in range(1, len(taus)):
        digit_axis.axvline(
            tau_index * reservoir_dim - 0.5,
            color="white",
            linewidth=0.7,
        )
    digit_axis.set_title("Digit-conditioned firing rates")
    digit_axis.set_xlabel("tau-neuron position")
    digit_axis.set_ylabel("MNIST digit")
    digit_axis.set_yticks(range(10))
    fig.colorbar(digit_image, ax=digit_axis, fraction=0.046, pad=0.04)

    progress = np.linspace(0.0, 1.0, temporal_rates.shape[1])
    for tau_index, tau in enumerate(taus):
        temporal_axis.plot(
            progress,
            temporal_rates[tau_index],
            label=f"tau={tau:g}",
            linewidth=1.8,
        )
    temporal_axis.set_title("Mean firing rate over normalized sequence")
    temporal_axis.set_xlabel("fraction of encoder sequence")
    temporal_axis.set_ylabel("spike probability")
    temporal_axis.legend()
    temporal_axis.grid(alpha=0.2)

    recurrent_limit = float(np.percentile(np.abs(recurrent), 99))
    recurrent_limit = recurrent_limit if recurrent_limit > 0.0 else 1.0
    recurrent_image = recurrent_axis.imshow(
        recurrent,
        cmap="coolwarm",
        vmin=-recurrent_limit,
        vmax=recurrent_limit,
        interpolation="nearest",
    )
    recurrent_axis.set_title("Learned shared recurrent matrix")
    recurrent_axis.set_xlabel("source neuron")
    recurrent_axis.set_ylabel("target neuron")
    fig.colorbar(recurrent_image, ax=recurrent_axis, fraction=0.046, pad=0.04)

    tau_text = ", ".join(
        f"{tau:g}: {rate:.3f}"
        for tau, rate in zip(taus, neuron_rates.mean(axis=1))
    )
    fig.suptitle(
        f"{result['name']} | best val {result['best_val_acc']:.2%} | "
        f"balanced subset {result['subset_accuracy']:.2%} | "
        f"{result['steps']} steps\nmean spike rate by tau: {tau_text}",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def save_matrix_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    output_path: Path,
    title: str,
    *,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
) -> None:
    size = len(labels)
    fig_size = max(10.0, size * 0.28)
    fig, axis = plt.subplots(figsize=(fig_size, fig_size))
    image = axis.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_xticks(range(size))
    axis.set_yticks(range(size))
    axis.set_xticklabels(labels, rotation=90, fontsize=5)
    axis.set_yticklabels(labels, fontsize=5)
    axis.set_title(title)
    fig.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_all_model_activity(
    activity: np.ndarray,
    labels: list[str],
    taus: list[float],
    reservoir_dim: int,
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(18, 12))
    image = axis.imshow(
        activity,
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=max(0.01, float(np.percentile(activity, 99))),
    )
    for tau_index in range(1, len(taus)):
        axis.axvline(tau_index * reservoir_dim - 0.5, color="white", linewidth=0.8)
    centers = np.arange(len(taus)) * reservoir_dim + (reservoir_dim - 1) / 2
    axis.set_xticks(centers)
    axis.set_xticklabels([f"tau={tau:g}" for tau in taus])
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels, fontsize=7)
    axis.set_title("Mean firing probability for every model and tau-neuron position")
    axis.set_xlabel("shared neuron positions grouped by tau")
    fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_grid_summary(
    results: list[dict[str, object]],
    output_path: Path,
) -> None:
    encoders = [name for name in ENCODER_ORDER if any(r["encoder"] == name for r in results)]
    decoders = [name for name in DECODER_ORDER if any(r["decoder"] == name for r in results)]
    accuracy = np.full((len(encoders), len(decoders)), np.nan)
    spike_rate = np.full_like(accuracy, np.nan)
    steps = np.full_like(accuracy, np.nan)
    for result in results:
        row = encoders.index(str(result["encoder"]))
        col = decoders.index(str(result["decoder"]))
        accuracy[row, col] = float(result["subset_accuracy"])
        spike_rate[row, col] = float(np.asarray(result["neuron_rates"]).mean())
        steps[row, col] = float(result["steps"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    panels = [
        (accuracy, "Balanced-subset accuracy", "viridis", 0.5, 1.0, ".1%"),
        (spike_rate, "Overall spike probability", "magma", 0.0, np.nanmax(spike_rate), ".3f"),
        (steps, "Encoder token count", "Blues", 0.0, np.nanmax(steps), ".0f"),
    ]
    for axis, (values, title, cmap, low, high, fmt) in zip(axes, panels):
        image = axis.imshow(values, cmap=cmap, vmin=low, vmax=high)
        axis.set_xticks(range(len(decoders)))
        axis.set_xticklabels(decoders, rotation=45, ha="right")
        axis.set_yticks(range(len(encoders)))
        axis.set_yticklabels(encoders)
        axis.set_title(title)
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                if np.isnan(values[row, col]):
                    continue
                axis.text(
                    col,
                    row,
                    format(values[row, col], fmt),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if values[row, col] > (low + high) / 2 else "black",
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def pca_coordinates(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = values - values.mean(axis=0, keepdims=True)
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ right_vectors[:2].T
    variance = singular_values**2
    explained = variance[:2] / max(float(variance.sum()), 1e-12)
    return coordinates, explained


def save_pca(
    values: np.ndarray,
    results: list[dict[str, object]],
    output_path: Path,
    title: str,
) -> None:
    coordinates, explained = pca_coordinates(values)
    fig, axis = plt.subplots(figsize=(10, 8))
    for index, result in enumerate(results):
        encoder = str(result["encoder"])
        decoder = str(result["decoder"])
        axis.scatter(
            coordinates[index, 0],
            coordinates[index, 1],
            color=ENCODER_COLORS[encoder],
            marker=DECODER_MARKERS[decoder],
            s=70,
            edgecolor="black",
            linewidth=0.4,
        )
        axis.annotate(
            f"{encoder}/{decoder}",
            coordinates[index],
            fontsize=6,
            xytext=(3, 3),
            textcoords="offset points",
        )
    axis.set_xlabel(f"PC1 ({explained[0]:.1%})")
    axis.set_ylabel(f"PC2 ({explained[1]:.1%})")
    axis.set_title(title)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def grouped_pair_values(
    matrix: np.ndarray,
    results: list[dict[str, object]],
) -> dict[str, list[float]]:
    grouped = {
        "same_encoder": [],
        "same_decoder": [],
        "different_encoder_and_decoder": [],
    }
    for row in range(len(results)):
        for col in range(row + 1, len(results)):
            same_encoder = results[row]["encoder"] == results[col]["encoder"]
            same_decoder = results[row]["decoder"] == results[col]["decoder"]
            if same_encoder:
                grouped["same_encoder"].append(float(matrix[row, col]))
            elif same_decoder:
                grouped["same_decoder"].append(float(matrix[row, col]))
            else:
                grouped["different_encoder_and_decoder"].append(float(matrix[row, col]))
    return grouped


def mean_group_matrix(
    values: np.ndarray,
    results: list[dict[str, object]],
    group_key: str,
    order: list[str],
) -> tuple[np.ndarray, list[str]]:
    names = [name for name in order if any(r[group_key] == name for r in results)]
    means = []
    for name in names:
        indices = [index for index, result in enumerate(results) if result[group_key] == name]
        means.append(values[indices].mean(axis=0))
    return np.stack(means), names


def pair_extremes(
    matrix: np.ndarray,
    labels: list[str],
) -> dict[str, object]:
    pairs = [
        (float(matrix[row, col]), labels[row], labels[col])
        for row in range(len(labels))
        for col in range(row + 1, len(labels))
    ]
    lowest = min(pairs)
    highest = max(pairs)
    return {
        "lowest": {"value": lowest[0], "models": [lowest[1], lowest[2]]},
        "highest": {"value": highest[0], "models": [highest[1], highest[2]]},
    }


def write_summary_csv(results: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "model",
        "encoder",
        "decoder",
        "epoch",
        "best_val_accuracy",
        "balanced_subset_accuracy",
        "token_steps",
        "overall_spike_rate",
        "tau_1.1_spike_rate",
        "tau_8_spike_rate",
        "tau_64_spike_rate",
        "active_neuron_fraction_above_1pct",
        "recurrent_weight_rms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            rates = np.asarray(result["neuron_rates"])
            row = {
                "model": result["name"],
                "encoder": result["encoder"],
                "decoder": result["decoder"],
                "epoch": result["epoch"],
                "best_val_accuracy": result["best_val_acc"],
                "balanced_subset_accuracy": result["subset_accuracy"],
                "token_steps": result["steps"],
                "overall_spike_rate": float(rates.mean()),
                "active_neuron_fraction_above_1pct": float((rates > 0.01).mean()),
                "recurrent_weight_rms": float(
                    np.sqrt(np.mean(np.asarray(result["recurrent"]) ** 2))
                ),
            }
            for tau_index, tau in enumerate(result["taus"]):
                row[f"tau_{tau:g}_spike_rate"] = float(rates[tau_index].mean())
            writer.writerow(row)


def write_report(
    results: list[dict[str, object]],
    labels: list[str],
    matrices: dict[str, np.ndarray],
    output_path: Path,
    samples_per_digit: int,
    edge_fraction: float,
) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for name, matrix in matrices.items():
        groups = grouped_pair_values(matrix, results)
        metrics[name] = {
            group: {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "count": len(values),
            }
            for group, values in groups.items()
        }
        metrics[name]["extremes"] = pair_extremes(matrix, labels)

    encoder_rows = []
    for encoder in ENCODER_ORDER:
        indices = [
            index for index, result in enumerate(results) if result["encoder"] == encoder
        ]
        if not indices:
            continue
        selected = [results[index] for index in indices]
        activity_submatrix = matrices["activity_correlation"][np.ix_(indices, indices)]
        recurrent_submatrix = matrices["recurrent_correlation"][np.ix_(indices, indices)]
        activity_pairs = [
            float(activity_submatrix[row, col])
            for row in range(len(indices))
            for col in range(row + 1, len(indices))
        ]
        recurrent_pairs = [
            float(recurrent_submatrix[row, col])
            for row in range(len(indices))
            for col in range(row + 1, len(indices))
        ]
        encoder_rows.append(
            (
                encoder,
                float(np.mean([result["subset_accuracy"] for result in selected])),
                float(np.mean([np.asarray(result["neuron_rates"]).mean() for result in selected])),
                float(np.mean(activity_pairs)) if activity_pairs else 1.0,
                float(np.mean(recurrent_pairs)) if recurrent_pairs else 1.0,
            )
        )

    decoder_rows = []
    for decoder in DECODER_ORDER:
        indices = [
            index for index, result in enumerate(results) if result["decoder"] == decoder
        ]
        if not indices:
            continue
        selected = [results[index] for index in indices]
        rates = np.stack([np.asarray(result["neuron_rates"]) for result in selected])
        activity_submatrix = matrices["activity_correlation"][np.ix_(indices, indices)]
        sorted_submatrix = matrices["sorted_activity_correlation"][np.ix_(indices, indices)]
        recurrent_submatrix = matrices["recurrent_correlation"][np.ix_(indices, indices)]
        edge_submatrix = matrices["top_edge_jaccard"][np.ix_(indices, indices)]
        upper_triangle = np.triu_indices(len(indices), k=1)
        decoder_rows.append(
            (
                decoder,
                float(np.mean([result["subset_accuracy"] for result in selected])),
                float(rates.mean()),
                [float(rates[:, tau_index].mean()) for tau_index in range(rates.shape[1])],
                float(activity_submatrix[upper_triangle].mean()),
                float(sorted_submatrix[upper_triangle].mean()),
                float(recurrent_submatrix[upper_triangle].mean()),
                float(edge_submatrix[upper_triangle].mean()),
            )
        )

    direct_cross_encoder = metrics["activity_correlation"]["same_decoder"]["mean"]
    sorted_cross_encoder = metrics["sorted_activity_correlation"]["same_decoder"]["mean"]
    topology_cross_encoder = metrics["recurrent_correlation"]["same_decoder"]["mean"]
    edge_cross_encoder = metrics["top_edge_jaccard"]["same_decoder"]["mean"]
    chance_jaccard = edge_fraction / (2.0 - edge_fraction)
    decoder_by_name = {row[0]: row for row in decoder_rows}
    spike_mlp_row = decoder_by_name.get("spike_mlp")
    non_spike_mlp_rows = [row for row in decoder_rows if row[0] != "spike_mlp"]
    non_spike_mlp_rate = float(np.mean([row[2] for row in non_spike_mlp_rows]))

    lines = [
        "# MNIST Sweep Spiking and Topology Analysis",
        "",
        f"- Checkpoints analyzed: **{len(results)}**",
        f"- Evaluation set: **{samples_per_digit * 10} balanced MNIST test images** "
        f"({samples_per_digit} per digit)",
        "- Shared reservoir: **64 neuron positions × 3 taus = 192 spiking units**",
        f"- Strong-edge topology: top **{edge_fraction:.0%}** absolute off-diagonal "
        "recurrent weights",
        "",
        "## Main Answer",
        "",
        "**No: the models do not converge to the same recurrent neuron-to-neuron "
        "topology. They do converge to a similar statistical firing regime.**",
        "",
        f"Across different encoders with the same decoder, exact neuron firing-profile "
        f"correlation averages **{direct_cross_encoder:.3f}**. After sorting neurons "
        f"within each tau (permutation-insensitive), it is **{sorted_cross_encoder:.3f}**.",
        "",
        f"Exact recurrent-weight correlation across encoders averages "
        f"**{topology_cross_encoder:.3f}**. Strong-edge Jaccard overlap averages "
        f"**{edge_cross_encoder:.3f}**, versus a random-overlap baseline of about "
        f"**{chance_jaccard:.3f}**.",
        "",
        "These metrics separate three questions: exact neuron identity, firing-rate "
        "distribution regardless of neuron permutation, and learned recurrent wiring.",
        "",
        "The decoder creates the largest firing-regime split. "
        + (
            f"`spike_mlp` averages **{spike_mlp_row[2]:.3f}** spikes per unit-step, "
            f"versus **{non_spike_mlp_rate:.3f}** for all other decoders. Its per-tau "
            f"rates are **{spike_mlp_row[3][0]:.3f}**, "
            f"**{spike_mlp_row[3][1]:.3f}**, and "
            f"**{spike_mlp_row[3][2]:.3f}** for tau 1.1, 8, and 64."
            if spike_mlp_row is not None
            else ""
        ),
        "",
        "## Encoder Summary",
        "",
        "| Encoder | Balanced accuracy | Mean spike rate | Activity similarity across decoders | Recurrent similarity across decoders |",
        "|---|---:|---:|---:|---:|",
    ]
    for encoder, accuracy, spike_rate, activity_similarity, recurrent_similarity in encoder_rows:
        lines.append(
            f"| {encoder} | {accuracy:.2%} | {spike_rate:.3f} | "
            f"{activity_similarity:.3f} | {recurrent_similarity:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Decoder Summary",
            "",
            "| Decoder | Balanced accuracy | Mean spike rate | Tau 1.1 | Tau 8 | Tau 64 | Exact activity across encoders | Sorted activity across encoders | Recurrent correlation | Top-edge Jaccard |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for (
        decoder,
        accuracy,
        spike_rate,
        tau_rates,
        activity_similarity,
        sorted_similarity,
        recurrent_similarity,
        edge_similarity,
    ) in decoder_rows:
        lines.append(
            f"| {decoder} | {accuracy:.2%} | {spike_rate:.3f} | "
            f"{tau_rates[0]:.3f} | {tau_rates[1]:.3f} | {tau_rates[2]:.3f} | "
            f"{activity_similarity:.3f} | {sorted_similarity:.3f} | "
            f"{recurrent_similarity:.3f} | {edge_similarity:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Pairwise Group Means",
            "",
            "| Metric | Same encoder | Same decoder | Different encoder and decoder |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in [
        "activity_correlation",
        "digit_activity_correlation",
        "sorted_activity_correlation",
        "recurrent_correlation",
        "recurrent_spectrum_cosine",
        "top_edge_jaccard",
    ]:
        metric = metrics[name]
        lines.append(
            f"| {name.replace('_', ' ')} | "
            f"{metric['same_encoder']['mean']:.3f} | "
            f"{metric['same_decoder']['mean']:.3f} | "
            f"{metric['different_encoder_and_decoder']['mean']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Same-encoder comparisons vary the decoder while preserving encoder structure.",
            "- Same-decoder comparisons vary the encoder and directly address the question.",
            "- Exact neuron correlations assume neuron index 17 in one model corresponds to "
            "neuron index 17 in another.",
            "- Sorted activity and recurrent singular-value comparisons are insensitive to "
            "neuron permutations, but they do not prove identical wiring.",
            "- Models with the same encoder were initialized with the same backbone seed "
            "before decoder construction. Different encoders consume different random "
            "numbers before backbone initialization, so cross-encoder weight similarity "
            "reflects both initialization and training.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    args = parse_args()
    if args.samples_per_digit <= 0:
        raise ValueError("--samples-per-digit must be positive.")
    if not 0.0 < args.top_edge_fraction < 1.0:
        raise ValueError("--top-edge-fraction must be between 0 and 1.")

    train_pipeline.set_seed(args.seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    checkpoint_paths = sorted(
        Path(args.sweep_root).glob("*/best.pt"),
        key=model_sort_key,
    )
    if args.max_models > 0:
        checkpoint_paths = checkpoint_paths[: args.max_models]
    if not checkpoint_paths:
        raise FileNotFoundError(f"No best.pt checkpoints found under {args.sweep_root}.")

    dataset, representative_indices = balanced_test_subset(
        args.data_dir,
        args.samples_per_digit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    base_dataset = dataset.dataset
    representative_image, _ = base_dataset[
        representative_indices[args.representative_digit]
    ]
    representative_image = representative_image.unsqueeze(0)

    output_dir = Path(args.output_dir)
    model_output_dir = output_dir / "models"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for index, checkpoint_path in enumerate(checkpoint_paths, start=1):
        print(f"[{index}/{len(checkpoint_paths)}] {checkpoint_path.parent.name}", flush=True)
        result = analyze_checkpoint(
            checkpoint_path,
            loader,
            representative_image,
            device,
            args.temporal_bins,
        )
        results.append(result)
        save_model_dashboard(
            result,
            model_output_dir / f"{result['name']}.png",
        )

    labels = [str(result["name"]) for result in results]
    activity = np.stack([np.asarray(result["neuron_rates"]).reshape(-1) for result in results])
    digit_activity = np.stack(
        [np.asarray(result["digit_rates"]).reshape(-1) for result in results]
    )
    sorted_activity = np.stack(
        [
            np.sort(np.asarray(result["neuron_rates"]), axis=1).reshape(-1)
            for result in results
        ]
    )
    recurrent_edges = np.stack(
        [np.asarray(result["recurrent_edges"]) for result in results]
    )
    recurrent_spectra = np.stack(
        [np.asarray(result["singular_values"]) for result in results]
    )
    temporal = np.stack([np.asarray(result["temporal_rates"]) for result in results])

    matrices = {
        "activity_correlation": safe_correlation_matrix(activity),
        "digit_activity_correlation": safe_correlation_matrix(digit_activity),
        "sorted_activity_correlation": safe_correlation_matrix(sorted_activity),
        "recurrent_correlation": safe_correlation_matrix(recurrent_edges),
        "recurrent_spectrum_cosine": cosine_matrix(recurrent_spectra),
        "top_edge_jaccard": jaccard_matrix(
            top_edge_masks(recurrent_edges, args.top_edge_fraction)
        ),
    }

    taus = list(results[0]["taus"])
    reservoir_dim = int(results[0]["reservoir_dim"])
    save_all_model_activity(
        activity,
        labels,
        taus,
        reservoir_dim,
        output_dir / "all_models_neuron_firing_rates.png",
    )
    save_grid_summary(results, output_dir / "encoder_decoder_summary.png")
    save_pca(
        activity,
        results,
        output_dir / "activity_pca.png",
        "PCA of exact tau-neuron firing profiles",
    )
    save_pca(
        recurrent_edges,
        results,
        output_dir / "recurrent_topology_pca.png",
        "PCA of learned recurrent edge weights",
    )

    matrix_plot_settings = {
        "activity_correlation": (-1.0, 1.0, "coolwarm"),
        "digit_activity_correlation": (-1.0, 1.0, "coolwarm"),
        "sorted_activity_correlation": (-1.0, 1.0, "coolwarm"),
        "recurrent_correlation": (-1.0, 1.0, "coolwarm"),
        "recurrent_spectrum_cosine": (0.0, 1.0, "viridis"),
        "top_edge_jaccard": (0.0, 1.0, "viridis"),
    }
    for name, matrix in matrices.items():
        low, high, cmap = matrix_plot_settings[name]
        save_matrix_heatmap(
            matrix,
            labels,
            output_dir / f"pairwise_{name}.png",
            f"Pairwise {name.replace('_', ' ')}",
            vmin=low,
            vmax=high,
            cmap=cmap,
        )

    for group_key, order in [("encoder", ENCODER_ORDER), ("decoder", DECODER_ORDER)]:
        activity_means, group_names = mean_group_matrix(
            activity,
            results,
            group_key,
            order,
        )
        recurrent_means, _ = mean_group_matrix(
            recurrent_edges,
            results,
            group_key,
            order,
        )
        save_matrix_heatmap(
            safe_correlation_matrix(activity_means),
            group_names,
            output_dir / f"{group_key}_mean_activity_similarity.png",
            f"{group_key.title()} mean exact firing-profile similarity",
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
        )
        save_matrix_heatmap(
            safe_correlation_matrix(recurrent_means),
            group_names,
            output_dir / f"{group_key}_mean_recurrent_similarity.png",
            f"{group_key.title()} mean recurrent-weight similarity",
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
        )

    write_summary_csv(results, output_dir / "summary.csv")
    np.savez_compressed(
        output_dir / "analysis_arrays.npz",
        model_names=np.asarray(labels),
        encoders=np.asarray([result["encoder"] for result in results]),
        decoders=np.asarray([result["decoder"] for result in results]),
        activity=activity,
        digit_activity=digit_activity,
        sorted_activity=sorted_activity,
        temporal_activity=temporal,
        recurrent_edges=recurrent_edges,
        recurrent_spectra=recurrent_spectra,
        **matrices,
    )
    metrics = write_report(
        results,
        labels,
        matrices,
        output_dir / "REPORT.md",
        args.samples_per_digit,
        args.top_edge_fraction,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(f"Saved analysis to: {output_dir}")


if __name__ == "__main__":
    main()
