"""Checkpoint-only functional and structural analysis for the MNIST sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as torch_functional
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from scipy.stats import kurtosis, skew, spearmanr
from torch.utils.data import DataLoader

import train_pipeline
from analyze_mnist_sweep_spiking import (
    DECODER_ORDER,
    ENCODER_ORDER,
    balanced_test_subset,
    checkpoint_args,
    logits_from_states,
    model_sort_key,
    save_matrix_heatmap,
    top_edge_masks,
)


PAIR_GROUPS = [
    "same_encoder",
    "same_decoder",
    "different_encoder_and_decoder",
]
MODEL_METRIC_LABELS = {
    "accuracy": "Accuracy",
    "spike_rate": "Spike rate",
    "spike_participation": "Spike participation",
    "spike_selective_fraction": "Digit-selective fraction",
    "spike_centroid_accuracy": "Spike centroid accuracy",
    "membrane_centroid_accuracy": "Membrane centroid accuracy",
    "stable_rank": "Stable rank",
    "effective_rank": "Effective rank",
    "top1_energy": "Leading-mode energy",
    "spectral_radius": "Spectral radius",
    "non_normality": "Non-normality",
    "reciprocity": "Weight reciprocity",
    "row_norm_cv": "Row-strength variation",
    "top_edge_reciprocity": "Strong-edge reciprocity",
}
NULL_METRICS = [
    "stable_rank",
    "effective_rank",
    "top1_energy",
    "spectral_radius",
    "non_normality",
    "reciprocity",
    "row_norm_cv",
    "top_edge_reciprocity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", default="sweep_runs_20ep_mnist")
    parser.add_argument(
        "--output-dir",
        default="visualizations/mnist_sweep_checkpoint_analysis",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--samples-per-digit", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--temporal-bins", type=int, default=32)
    parser.add_argument("--shuffle-repeats", type=int, default=50)
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


def pearson_vector(first: np.ndarray, second: np.ndarray) -> float:
    first_centered = first.reshape(-1).astype(np.float64)
    second_centered = second.reshape(-1).astype(np.float64)
    first_centered -= first_centered.mean()
    second_centered -= second_centered.mean()
    denominator = np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(first_centered, second_centered) / denominator)


def cosine_vector(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(first, second) / denominator)


def jaccard(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / union) if union else 1.0


def infer_checkpoint(
    checkpoint_path: Path,
    loader: DataLoader,
    device: torch.device,
    temporal_bins: int,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = checkpoint_args(checkpoint)
    model = train_pipeline.build_model(architecture).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tau_count = len(train_pipeline.parse_taus(architecture.taus))
    reservoir_dim = int(architecture.reservoir_dim)
    targets_all: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    spike_mean_all: list[np.ndarray] = []
    membrane_mean_all: list[np.ndarray] = []
    membrane_last_all: list[np.ndarray] = []
    temporal_all: list[np.ndarray] = []
    token_steps = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            tokens = model.encoder(images)
            states = model.backbone.forward_states(tokens, reset_state=True)
            logits = logits_from_states(model, states)
            spikes = states["spike"]
            membranes = states["membrane"]
            token_steps = int(tokens.size(1))

            population_temporal = spikes.reshape(
                images.size(0),
                token_steps,
                tau_count,
                reservoir_dim,
            ).mean(dim=-1)
            population_temporal = population_temporal.permute(0, 2, 1)
            population_temporal = torch_functional.interpolate(
                population_temporal,
                size=temporal_bins,
                mode="linear",
                align_corners=True,
            )

            targets_all.append(targets.numpy())
            logits_all.append(logits.cpu().numpy())
            spike_mean_all.append(spikes.mean(dim=1).cpu().numpy())
            membrane_mean_all.append(membranes.mean(dim=1).cpu().numpy())
            membrane_last_all.append(membranes[:, -1].cpu().numpy())
            temporal_all.append(population_temporal.cpu().numpy())

    targets = np.concatenate(targets_all)
    logits = np.concatenate(logits_all)
    spike_mean = np.concatenate(spike_mean_all)
    membrane_mean = np.concatenate(membrane_mean_all)
    membrane_last = np.concatenate(membrane_last_all)
    temporal = np.concatenate(temporal_all)
    recurrent = model.backbone.shared_recurrent.weight.detach().cpu().numpy()

    return {
        "name": checkpoint_path.parent.name,
        "encoder": str(architecture.encoder),
        "decoder": str(architecture.decoder),
        "accuracy": float((logits.argmax(axis=1) == targets).mean()),
        "best_val_accuracy": float(checkpoint.get("best_val_acc", float("nan"))),
        "steps": token_steps,
        "targets": targets,
        "logits": logits,
        "predictions": logits.argmax(axis=1),
        "spike_mean": spike_mean,
        "membrane_mean": membrane_mean,
        "membrane_last": membrane_last,
        "temporal": temporal,
        "recurrent": recurrent,
        "taus": list(train_pipeline.parse_taus(architecture.taus)),
        "reservoir_dim": reservoir_dim,
    }


def top_edge_mask(weight: np.ndarray, fraction: float) -> np.ndarray:
    flat = np.abs(weight).reshape(-1)
    diagonal = np.eye(weight.shape[0], dtype=bool).reshape(-1)
    candidates = np.flatnonzero(~diagonal)
    keep = max(1, int(round(len(candidates) * fraction)))
    chosen = candidates[np.argpartition(flat[candidates], -keep)[-keep:]]
    result = np.zeros_like(flat, dtype=bool)
    result[chosen] = True
    return result.reshape(weight.shape)


def matrix_metrics(weight: np.ndarray, edge_fraction: float) -> dict[str, object]:
    singular = np.linalg.svd(weight, compute_uv=False)
    energy = singular**2
    energy_probability = energy / max(float(energy.sum()), 1e-12)
    entropy = -float(
        np.sum(energy_probability * np.log(np.maximum(energy_probability, 1e-12)))
    )
    eigenvalues = np.linalg.eigvals(weight)
    frobenius_sq = float(np.sum(weight**2))
    commutator = weight.T @ weight - weight @ weight.T
    off_diagonal = ~np.eye(weight.shape[0], dtype=bool)
    reciprocal = pearson_vector(weight[off_diagonal], weight.T[off_diagonal])
    row_norms = np.linalg.norm(weight, axis=1)
    column_norms = np.linalg.norm(weight, axis=0)
    strong = top_edge_mask(weight, edge_fraction)
    strong_count = max(int(strong.sum()), 1)

    centered = weight - weight.mean()
    centered_singular = np.linalg.svd(centered, compute_uv=False)
    return {
        "weight_mean": float(weight.mean()),
        "weight_std": float(weight.std()),
        "weight_skew": float(skew(weight.reshape(-1))),
        "weight_kurtosis": float(kurtosis(weight.reshape(-1), fisher=True)),
        "spectral_norm": float(singular[0]),
        "frobenius_norm": float(np.sqrt(frobenius_sq)),
        "stable_rank": float(energy.sum() / max(float(energy[0]), 1e-12)),
        "effective_rank": float(np.exp(entropy)),
        "spectral_entropy": float(entropy / np.log(len(singular))),
        "top1_energy": float(energy_probability[0]),
        "top5_energy": float(energy_probability[:5].sum()),
        "spectral_radius": float(np.max(np.abs(eigenvalues))),
        "non_normality": float(
            np.linalg.norm(commutator, ord="fro") / max(frobenius_sq, 1e-12)
        ),
        "reciprocity": reciprocal,
        "symmetric_energy": float(
            np.sum(((weight + weight.T) * 0.5) ** 2) / max(frobenius_sq, 1e-12)
        ),
        "row_norm_cv": float(row_norms.std() / max(float(row_norms.mean()), 1e-12)),
        "column_norm_cv": float(
            column_norms.std() / max(float(column_norms.mean()), 1e-12)
        ),
        "top_edge_reciprocity": float(
            np.logical_and(strong, strong.T).sum() / strong_count
        ),
        "singular_values": singular,
        "normalized_singular_values": singular
        / max(float(np.linalg.norm(singular)), 1e-12),
        "centered_normalized_singular_values": centered_singular
        / max(float(np.linalg.norm(centered_singular)), 1e-12),
    }


def shuffled_weight_controls(
    weight: np.ndarray,
    edge_fraction: float,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[dict[str, dict[str, float]], np.ndarray]:
    values = {name: [] for name in NULL_METRICS}
    spectra = []
    flattened = weight.reshape(-1).copy()
    for _ in range(repeats):
        shuffled = rng.permutation(flattened).reshape(weight.shape)
        metrics = matrix_metrics(shuffled, edge_fraction)
        for name in NULL_METRICS:
            values[name].append(float(metrics[name]))
        spectra.append(np.asarray(metrics["normalized_singular_values"]))

    controls: dict[str, dict[str, float]] = {}
    observed = matrix_metrics(weight, edge_fraction)
    for name, samples in values.items():
        sample_array = np.asarray(samples)
        mean = float(sample_array.mean())
        std = float(sample_array.std(ddof=1))
        controls[name] = {
            "mean": mean,
            "std": std,
            "z": float((float(observed[name]) - mean) / max(std, 1e-12)),
        }
    return controls, np.mean(spectra, axis=0)


def split_centroid_accuracy(features: np.ndarray, targets: np.ndarray) -> float:
    reference_indices: list[int] = []
    query_indices: list[int] = []
    for digit in range(10):
        indices = np.flatnonzero(targets == digit)
        midpoint = len(indices) // 2
        reference_indices.extend(indices[:midpoint])
        query_indices.extend(indices[midpoint:])

    reference = features[reference_indices].astype(np.float64)
    query = features[query_indices].astype(np.float64)
    reference_targets = targets[reference_indices]
    query_targets = targets[query_indices]
    reference /= np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-12)
    query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    centroids = np.stack(
        [reference[reference_targets == digit].mean(axis=0) for digit in range(10)]
    )
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    predictions = np.argmax(query @ centroids.T, axis=1)
    return float((predictions == query_targets).mean())


def class_geometry(features: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, float]:
    centered = features.astype(np.float64) - features.mean(axis=0, keepdims=True)
    centered /= max(float(np.linalg.norm(centered)), 1e-12)
    centroids = np.stack(
        [centered[targets == digit].mean(axis=0) for digit in range(10)]
    )
    geometry = pdist(centroids, metric="euclidean")
    within = sum(
        float(np.sum((centered[targets == digit] - centroids[digit]) ** 2))
        for digit in range(10)
    )
    counts = np.asarray([(targets == digit).sum() for digit in range(10)])
    between = float(np.sum(counts[:, None] * centroids**2))
    return geometry, between / max(within, 1e-12)


def activity_metrics(result: dict[str, object]) -> dict[str, float]:
    spikes = np.asarray(result["spike_mean"], dtype=np.float64)
    membranes = np.asarray(result["membrane_mean"], dtype=np.float64)
    targets = np.asarray(result["targets"])
    temporal = np.asarray(result["temporal"], dtype=np.float64)
    total = spikes.sum(axis=1)
    participation = total**2 / np.maximum(
        spikes.shape[1] * np.sum(spikes**2, axis=1),
        1e-12,
    )

    digit_rates = np.stack([spikes[targets == digit].mean(axis=0) for digit in range(10)])
    maximum = digit_rates.max(axis=0)
    mean_other = (digit_rates.sum(axis=0) - maximum) / 9.0
    selectivity = (maximum - mean_other) / np.maximum(maximum + mean_other, 1e-12)
    active = maximum > 1e-4

    time_axis = np.linspace(0.0, 1.0, temporal.shape[-1])
    temporal_total = temporal.sum(axis=1)
    temporal_mass = temporal_total.sum(axis=1)
    center_of_mass = (temporal_total * time_axis).sum(axis=1) / np.maximum(
        temporal_mass,
        1e-12,
    )
    burst_ratio = temporal_total.max(axis=1) / np.maximum(
        temporal_total.mean(axis=1),
        1e-12,
    )

    spike_geometry, spike_fisher = class_geometry(spikes, targets)
    membrane_geometry, membrane_fisher = class_geometry(membranes, targets)
    result["spike_geometry"] = spike_geometry
    result["membrane_geometry"] = membrane_geometry
    return {
        "spike_rate": float(spikes.mean()),
        "spike_participation": float(participation.mean()),
        "active_unit_fraction": float(active.mean()),
        "spike_selective_fraction": float(
            (selectivity[active] > 0.5).mean() if active.any() else 0.0
        ),
        "mean_spike_selectivity": float(
            selectivity[active].mean() if active.any() else 0.0
        ),
        "temporal_center_of_mass": float(center_of_mass.mean()),
        "temporal_burst_ratio": float(burst_ratio.mean()),
        "spike_fisher_ratio": spike_fisher,
        "membrane_fisher_ratio": membrane_fisher,
        "spike_centroid_accuracy": split_centroid_accuracy(spikes, targets),
        "membrane_centroid_accuracy": split_centroid_accuracy(membranes, targets),
    }


def gram_statistics(features: np.ndarray) -> dict[str, object]:
    centered = features.astype(np.float64) - features.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    np.fill_diagonal(gram, 0.0)
    row_sum = gram.sum(axis=1)
    total = float(row_sum.sum())
    n = gram.shape[0]
    self_hsic = (
        float(np.sum(gram * gram))
        + total * total / ((n - 1) * (n - 2))
        - 2.0 * float(np.dot(row_sum, row_sum)) / (n - 2)
    )
    return {
        "gram": gram,
        "row_sum": row_sum,
        "total": total,
        "self_hsic": max(self_hsic, 1e-12),
    }


def unbiased_cka(first: dict[str, object], second: dict[str, object]) -> float:
    gram_first = np.asarray(first["gram"])
    gram_second = np.asarray(second["gram"])
    n = gram_first.shape[0]
    cross_hsic = (
        float(np.sum(gram_first * gram_second))
        + float(first["total"]) * float(second["total"]) / ((n - 1) * (n - 2))
        - 2.0
        * float(np.dot(first["row_sum"], second["row_sum"]))
        / (n - 2)
    )
    denominator = np.sqrt(float(first["self_hsic"]) * float(second["self_hsic"]))
    return float(np.clip(cross_hsic / max(denominator, 1e-12), -1.0, 1.0))


def cka_matrix(
    results: list[dict[str, object]],
    key: str,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    statistics = [
        gram_statistics(np.asarray(result[key]).reshape(len(result["targets"]), -1))
        for result in results
    ]
    size = len(results)
    matrix = np.eye(size, dtype=np.float64)
    for row in range(size):
        for col in range(row + 1, size):
            value = unbiased_cka(statistics[row], statistics[col])
            matrix[row, col] = value
            matrix[col, row] = value
    return matrix, statistics


def sample_permutation_cka_baseline(
    statistics: list[dict[str, object]],
    results: list[dict[str, object]],
    rng: np.random.Generator,
) -> dict[str, float]:
    grouped = {group: [] for group in PAIR_GROUPS}
    sample_count = np.asarray(statistics[0]["gram"]).shape[0]
    for row in range(len(results)):
        for col in range(row + 1, len(results)):
            permutation = rng.permutation(sample_count)
            second = statistics[col]
            permuted = {
                "gram": np.asarray(second["gram"])[np.ix_(permutation, permutation)],
                "row_sum": np.asarray(second["row_sum"])[permutation],
                "total": second["total"],
                "self_hsic": second["self_hsic"],
            }
            grouped[pair_group(results[row], results[col])].append(
                unbiased_cka(statistics[row], permuted)
            )
    return {
        group: float(np.mean(values))
        for group, values in grouped.items()
    }


def prediction_agreement_matrix(results: list[dict[str, object]]) -> np.ndarray:
    size = len(results)
    matrix = np.eye(size, dtype=np.float64)
    for row in range(size):
        for col in range(row + 1, size):
            value = float(
                np.mean(results[row]["predictions"] == results[col]["predictions"])
            )
            matrix[row, col] = value
            matrix[col, row] = value
    return matrix


def vector_similarity_matrix(vectors: list[np.ndarray], *, cosine: bool = False) -> np.ndarray:
    size = len(vectors)
    matrix = np.eye(size, dtype=np.float64)
    similarity = cosine_vector if cosine else pearson_vector
    for row in range(size):
        for col in range(row + 1, size):
            value = similarity(vectors[row], vectors[col])
            matrix[row, col] = value
            matrix[col, row] = value
    return matrix


def pair_group(first: dict[str, object], second: dict[str, object]) -> str:
    if first["encoder"] == second["encoder"]:
        return "same_encoder"
    if first["decoder"] == second["decoder"]:
        return "same_decoder"
    return "different_encoder_and_decoder"


def build_pairwise_rows(
    results: list[dict[str, object]],
    matrices: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for first in range(len(results)):
        for second in range(first + 1, len(results)):
            row: dict[str, object] = {
                "model_a": results[first]["name"],
                "model_b": results[second]["name"],
                "group": pair_group(results[first], results[second]),
                "accuracy_difference": abs(
                    float(results[first]["accuracy"]) - float(results[second]["accuracy"])
                ),
                "spike_rate_difference": abs(
                    float(results[first]["spike_rate"])
                    - float(results[second]["spike_rate"])
                ),
            }
            for name, matrix in matrices.items():
                row[name] = float(matrix[first, second])
            rows.append(row)
    return rows


def grouped_pair_means(
    pairwise_rows: list[dict[str, object]],
    metric_names: list[str],
) -> dict[str, dict[str, float]]:
    return {
        metric: {
            group: float(
                np.mean(
                    [
                        float(row[metric])
                        for row in pairwise_rows
                        if row["group"] == group
                    ]
                )
            )
            for group in PAIR_GROUPS
        }
        for metric in metric_names
    }


def neuron_signatures(result: dict[str, object]) -> np.ndarray:
    reservoir_dim = int(result["reservoir_dim"])
    tau_count = len(result["taus"])
    parts = []
    for key in ["membrane_mean", "spike_mean"]:
        values = np.asarray(result[key]).reshape(-1, tau_count, reservoir_dim)
        values = values.transpose(2, 0, 1).reshape(reservoir_dim, -1)
        values -= values.mean(axis=1, keepdims=True)
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        parts.append(values)
    signatures = np.concatenate(parts, axis=1)
    signatures -= signatures.mean(axis=1, keepdims=True)
    signatures /= np.maximum(np.linalg.norm(signatures, axis=1, keepdims=True), 1e-12)
    return signatures


def subspace_overlap(first: np.ndarray, second: np.ndarray, rank: int = 5) -> float:
    first_u, _, first_vh = np.linalg.svd(first, full_matrices=False)
    second_u, _, second_vh = np.linalg.svd(second, full_matrices=False)
    left = np.linalg.norm(first_u[:, :rank].T @ second_u[:, :rank], ord="fro") ** 2
    right = np.linalg.norm(first_vh[:rank] @ second_vh[:rank].T, ord="fro") ** 2
    return float((left + right) / (2.0 * rank))


def alignment_rows(
    results: list[dict[str, object]],
    edge_fraction: float,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for decoder in DECODER_ORDER:
        indices = [
            index for index, result in enumerate(results) if result["decoder"] == decoder
        ]
        if len(indices) < 2:
            continue
        reference_index = next(
            (
                index
                for index in indices
                if results[index]["encoder"] == "cnn3"
            ),
            max(indices, key=lambda index: float(results[index]["accuracy"])),
        )
        reference = results[reference_index]
        reference_signature = neuron_signatures(reference)
        reference_weight = np.asarray(reference["recurrent"])
        reference_mask = top_edge_mask(reference_weight, edge_fraction)
        off_diagonal = ~np.eye(reference_weight.shape[0], dtype=bool)

        for index in indices:
            if index == reference_index:
                continue
            target = results[index]
            target_signature = neuron_signatures(target)
            similarity = reference_signature @ target_signature.T
            reference_neurons, target_neurons = linear_sum_assignment(
                similarity,
                maximize=True,
            )
            order = np.empty(reference_weight.shape[0], dtype=int)
            order[reference_neurons] = target_neurons
            target_weight = np.asarray(target["recurrent"])
            aligned_weight = target_weight[np.ix_(order, order)]

            before_correlation = pearson_vector(
                reference_weight[off_diagonal],
                target_weight[off_diagonal],
            )
            after_correlation = pearson_vector(
                reference_weight[off_diagonal],
                aligned_weight[off_diagonal],
            )
            random_correlations = []
            random_match_correlations = []
            random_edge_jaccards = []
            for _ in range(50):
                random_order = rng.permutation(reference_weight.shape[0])
                random_weight = target_weight[np.ix_(random_order, random_order)]
                random_match_correlations.append(
                    float(similarity[np.arange(len(random_order)), random_order].mean())
                )
                random_correlations.append(
                    pearson_vector(
                        reference_weight[off_diagonal],
                        random_weight[off_diagonal],
                    )
                )
                random_edge_jaccards.append(
                    jaccard(
                        reference_mask,
                        top_edge_mask(random_weight, edge_fraction),
                    )
                )
            random_correlations = np.asarray(random_correlations)
            random_match_correlations = np.asarray(random_match_correlations)
            random_edge_jaccards = np.asarray(random_edge_jaccards)
            target_mask = top_edge_mask(target_weight, edge_fraction)
            aligned_mask = top_edge_mask(aligned_weight, edge_fraction)
            matched_correlation = float(
                similarity[reference_neurons, target_neurons].mean()
            )
            rows.append(
                {
                    "decoder": decoder,
                    "reference": reference["name"],
                    "target": target["name"],
                    "target_encoder": target["encoder"],
                    "match_correlation": matched_correlation,
                    "match_random_mean": float(random_match_correlations.mean()),
                    "match_correlation_z": float(
                        (matched_correlation - random_match_correlations.mean())
                        / max(float(random_match_correlations.std(ddof=1)), 1e-12)
                    ),
                    "weight_correlation_before": before_correlation,
                    "weight_correlation_after": after_correlation,
                    "weight_correlation_random_mean": float(
                        random_correlations.mean()
                    ),
                    "weight_correlation_after_z": float(
                        (after_correlation - random_correlations.mean())
                        / max(float(random_correlations.std(ddof=1)), 1e-12)
                    ),
                    "edge_jaccard_before": jaccard(reference_mask, target_mask),
                    "edge_jaccard_after": jaccard(reference_mask, aligned_mask),
                    "edge_jaccard_random_mean": float(random_edge_jaccards.mean()),
                    "subspace_overlap_before": subspace_overlap(
                        reference_weight,
                        target_weight,
                    ),
                    "subspace_overlap_after": subspace_overlap(
                        reference_weight,
                        aligned_weight,
                    ),
                }
            )
    return rows


def additive_variance_decomposition(
    results: list[dict[str, object]],
    metric: str,
) -> dict[str, float]:
    values = np.asarray([float(result[metric]) for result in results])
    grand_mean = float(values.mean())
    total = float(np.sum((values - grand_mean) ** 2))
    if total <= 1e-12:
        return {"encoder": 0.0, "decoder": 0.0, "residual": 1.0}

    encoder_ss = 0.0
    for encoder in ENCODER_ORDER:
        selected = values[
            [result["encoder"] == encoder for result in results]
        ]
        if len(selected):
            encoder_ss += len(selected) * float((selected.mean() - grand_mean) ** 2)
    decoder_ss = 0.0
    for decoder in DECODER_ORDER:
        selected = values[
            [result["decoder"] == decoder for result in results]
        ]
        if len(selected):
            decoder_ss += len(selected) * float((selected.mean() - grand_mean) ** 2)
    residual = max(total - encoder_ss - decoder_ss, 0.0)
    return {
        "encoder": encoder_ss / total,
        "decoder": decoder_ss / total,
        "residual": residual / total,
    }


def metric_associations(
    pairwise_rows: list[dict[str, object]],
    metric_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(
        [[float(row[name]) for name in metric_names] for row in pairwise_rows]
    )
    correlation = np.eye(len(metric_names), dtype=np.float64)
    p_values = np.zeros_like(correlation)
    for row in range(len(metric_names)):
        for col in range(row + 1, len(metric_names)):
            statistic = spearmanr(values[:, row], values[:, col])
            correlation[row, col] = float(statistic.statistic)
            correlation[col, row] = correlation[row, col]
            p_values[row, col] = float(statistic.pvalue)
            p_values[col, row] = p_values[row, col]
    return correlation, p_values


def save_model_metric_grid(
    results: list[dict[str, object]],
    metrics: list[str],
    output_path: Path,
) -> None:
    encoders = [name for name in ENCODER_ORDER if any(r["encoder"] == name for r in results)]
    decoders = [name for name in DECODER_ORDER if any(r["decoder"] == name for r in results)]
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    for axis, metric in zip(axes.flat, metrics):
        values = np.full((len(encoders), len(decoders)), np.nan)
        for result in results:
            values[
                encoders.index(str(result["encoder"])),
                decoders.index(str(result["decoder"])),
            ] = float(result[metric])
        image = axis.imshow(values, cmap="viridis", aspect="auto")
        axis.set_xticks(range(len(decoders)))
        axis.set_xticklabels(decoders, rotation=45, ha="right")
        axis.set_yticks(range(len(encoders)))
        axis.set_yticklabels(encoders)
        axis.set_title(MODEL_METRIC_LABELS.get(metric, metric))
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                axis.text(
                    col,
                    row,
                    f"{values[row, col]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white"
                    if values[row, col] > np.nanmedian(values)
                    else "black",
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_null_zscores(
    results: list[dict[str, object]],
    labels: list[str],
    output_path: Path,
) -> None:
    values = np.asarray(
        [
            [float(result[f"{metric}_shuffle_z"]) for metric in NULL_METRICS]
            for result in results
        ]
    )
    limit = max(2.0, float(np.percentile(np.abs(values), 98)))
    fig, axis = plt.subplots(figsize=(14, 13))
    image = axis.imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(NULL_METRICS)))
    axis.set_xticklabels(
        [MODEL_METRIC_LABELS.get(name, name) for name in NULL_METRICS],
        rotation=35,
        ha="right",
    )
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels, fontsize=7)
    axis.set_title("Observed recurrent metrics relative to shuffled-weight null (z-score)")
    fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_spectral_profiles(
    results: list[dict[str, object]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = dict(zip(DECODER_ORDER, plt.get_cmap("tab10").colors))
    rank = np.arange(1, len(results[0]["normalized_singular_values"]) + 1)
    for decoder in DECODER_ORDER:
        selected = [result for result in results if result["decoder"] == decoder]
        if not selected:
            continue
        observed = np.stack(
            [np.asarray(result["normalized_singular_values"]) for result in selected]
        )
        shuffled = np.stack(
            [np.asarray(result["shuffle_mean_spectrum"]) for result in selected]
        )
        axes[0].plot(
            rank,
            observed.mean(axis=0),
            color=colors[decoder],
            label=decoder,
            linewidth=2,
        )
        axes[1].plot(
            rank,
            (observed - shuffled).mean(axis=0),
            color=colors[decoder],
            label=decoder,
            linewidth=2,
        )
    axes[0].set_title("Mean normalized recurrent singular spectrum")
    axes[0].set_xlabel("singular-value rank")
    axes[0].set_ylabel("normalized singular value")
    axes[0].set_yscale("log")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Observed spectrum minus shuffled-weight null")
    axes[1].set_xlabel("singular-value rank")
    axes[1].set_ylabel("difference")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_alignment_summary(
    alignment: list[dict[str, object]],
    output_path: Path,
) -> None:
    decoders = [
        decoder for decoder in DECODER_ORDER if any(row["decoder"] == decoder for row in alignment)
    ]
    metrics = [
        ("weight_correlation_before", "weight_correlation_after", "Weight correlation"),
        ("edge_jaccard_before", "edge_jaccard_after", "Top-edge Jaccard"),
        ("subspace_overlap_before", "subspace_overlap_after", "Top-5 subspace overlap"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    positions = np.arange(len(decoders))
    width = 0.36
    for axis, (before_name, after_name, title) in zip(axes, metrics):
        before = [
            np.mean(
                [float(row[before_name]) for row in alignment if row["decoder"] == decoder]
            )
            for decoder in decoders
        ]
        after = [
            np.mean(
                [float(row[after_name]) for row in alignment if row["decoder"] == decoder]
            )
            for decoder in decoders
        ]
        axis.bar(positions - width / 2, before, width, label="Before alignment")
        axis.bar(positions + width / 2, after, width, label="After alignment")
        axis.set_xticks(positions)
        axis.set_xticklabels(decoders, rotation=40, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend()
    fig.suptitle("Neuron permutation inferred from independent activation signatures")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_variance_decomposition(
    decompositions: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    metrics = list(decompositions)
    encoder = np.asarray([decompositions[name]["encoder"] for name in metrics])
    decoder = np.asarray([decompositions[name]["decoder"] for name in metrics])
    residual = np.asarray([decompositions[name]["residual"] for name in metrics])
    positions = np.arange(len(metrics))
    fig, axis = plt.subplots(figsize=(14, 7))
    axis.bar(positions, encoder, label="Encoder")
    axis.bar(positions, decoder, bottom=encoder, label="Decoder")
    axis.bar(positions, residual, bottom=encoder + decoder, label="Residual/interaction")
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [MODEL_METRIC_LABELS.get(name, name) for name in metrics],
        rotation=40,
        ha="right",
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("fraction of variance")
    axis.set_title("Additive encoder/decoder variance decomposition")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_group_similarity(
    grouped: dict[str, dict[str, float]],
    metrics: list[str],
    output_path: Path,
) -> None:
    positions = np.arange(len(metrics))
    width = 0.25
    fig, axis = plt.subplots(figsize=(16, 7))
    for offset, group in enumerate(PAIR_GROUPS):
        axis.bar(
            positions + (offset - 1) * width,
            [grouped[metric][group] for metric in metrics],
            width,
            label=group.replace("_", " "),
        )
    axis.set_xticks(positions)
    axis.set_xticklabels([metric.replace("_", " ") for metric in metrics], rotation=35, ha="right")
    axis.set_ylabel("mean pairwise similarity")
    axis.set_title("What remains similar when encoder or decoder changes?")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def serializable_model_rows(results: list[dict[str, object]]) -> list[dict[str, object]]:
    excluded = {
        "targets",
        "logits",
        "predictions",
        "spike_mean",
        "membrane_mean",
        "membrane_last",
        "temporal",
        "recurrent",
        "spike_geometry",
        "membrane_geometry",
        "singular_values",
        "normalized_singular_values",
        "centered_normalized_singular_values",
        "shuffle_mean_spectrum",
        "taus",
    }
    return [
        {
            key: value
            for key, value in result.items()
            if key not in excluded and np.isscalar(value)
        }
        for result in results
    ]


def write_report(
    results: list[dict[str, object]],
    grouped: dict[str, dict[str, float]],
    alignment: list[dict[str, object]],
    decompositions: dict[str, dict[str, float]],
    association_names: list[str],
    associations: np.ndarray,
    observed_spectrum_cosine: np.ndarray,
    shuffled_spectrum_cosine: np.ndarray,
    sample_permutation_baselines: dict[str, dict[str, float]],
    output_path: Path,
    samples_per_digit: int,
) -> dict[str, object]:
    off_diagonal = np.triu_indices(len(results), k=1)
    observed_spectrum_mean = float(observed_spectrum_cosine[off_diagonal].mean())
    shuffled_spectrum_mean = float(shuffled_spectrum_cosine[off_diagonal].mean())
    alignment_summary = {
        name: float(np.mean([float(row[name]) for row in alignment]))
        for name in [
            "match_correlation",
            "match_random_mean",
            "match_correlation_z",
            "weight_correlation_before",
            "weight_correlation_after",
            "weight_correlation_after_z",
            "edge_jaccard_before",
            "edge_jaccard_after",
            "edge_jaccard_random_mean",
            "subspace_overlap_before",
            "subspace_overlap_after",
        ]
    }
    topology_index = association_names.index("recurrent_correlation")
    functional_associations = {
        name: float(associations[topology_index, association_names.index(name)])
        for name in [
            "spike_cka",
            "membrane_cka",
            "temporal_cka",
            "prediction_agreement",
        ]
    }
    dominant_effects = {
        metric: max(values, key=values.get)
        for metric, values in decompositions.items()
    }
    strongest_null = sorted(
        [
            (
                abs(float(result[f"{metric}_shuffle_z"])),
                str(result["name"]),
                metric,
                float(result[f"{metric}_shuffle_z"]),
            )
            for result in results
            for metric in NULL_METRICS
        ],
        reverse=True,
    )[:10]
    null_summary = {
        metric: {
            "observed_mean": float(
                np.mean([float(result[metric]) for result in results])
            ),
            "shuffle_mean": float(
                np.mean([float(result[f"{metric}_shuffle_mean"]) for result in results])
            ),
            "mean_z": float(
                np.mean([float(result[f"{metric}_shuffle_z"]) for result in results])
            ),
            "significant_high_fraction": float(
                np.mean(
                    [float(result[f"{metric}_shuffle_z"]) > 1.96 for result in results]
                )
            ),
            "significant_low_fraction": float(
                np.mean(
                    [float(result[f"{metric}_shuffle_z"]) < -1.96 for result in results]
                )
            ),
        }
        for metric in NULL_METRICS
    }
    model_spearman = {}
    accuracy = np.asarray([float(result["accuracy"]) for result in results])
    for metric in [
        "spike_centroid_accuracy",
        "membrane_centroid_accuracy",
        "spike_selective_fraction",
        "effective_rank",
        "non_normality",
        "spike_rate",
    ]:
        statistic = spearmanr(
            accuracy,
            np.asarray([float(result[metric]) for result in results]),
        )
        model_spearman[metric] = {
            "rho": float(statistic.statistic),
            "p": float(statistic.pvalue),
        }
    decoder_means = {
        decoder: {
            metric: float(
                np.mean(
                    [
                        float(result[metric])
                        for result in results
                        if result["decoder"] == decoder
                    ]
                )
            )
            for metric in [
                "accuracy",
                "spike_rate",
                "effective_rank",
                "top1_energy",
                "spectral_radius",
            ]
        }
        for decoder in DECODER_ORDER
    }

    lines = [
        "# MNIST Sweep Checkpoint-Only Analysis",
        "",
        f"- Models: **{len(results)}**",
        f"- Shared evaluation set: **{samples_per_digit * 10} images**",
        "- No training or initialization checkpoints were used.",
        "- Weight null: each recurrent matrix was repeatedly shuffled while preserving its exact entries.",
        "",
        "## Main Findings",
        "",
        f"Across encoders with the same decoder, unbiased spike CKA is "
        f"**{grouped['spike_cka']['same_decoder']:.3f}**, membrane CKA is "
        f"**{grouped['membrane_cka']['same_decoder']:.3f}**, and normalized temporal "
        f"CKA is **{grouped['temporal_cka']['same_decoder']:.3f}**.",
        f"After randomly breaking image correspondence, the respective baselines are "
        f"**{sample_permutation_baselines['spike_cka']['same_decoder']:.3f}**, "
        f"**{sample_permutation_baselines['membrane_cka']['same_decoder']:.3f}**, and "
        f"**{sample_permutation_baselines['temporal_cka']['same_decoder']:.3f}**.",
        "",
        f"Prediction agreement across encoders with the same decoder is "
        f"**{grouped['prediction_agreement']['same_decoder']:.3f}**, while exact "
        f"recurrent-weight correlation is "
        f"**{grouped['recurrent_correlation']['same_decoder']:.3f}**.",
        "",
        "Changing the decoder while keeping the encoder produces more similar "
        "representations than changing the encoder while keeping the decoder: spike "
        f"CKA is **{grouped['spike_cka']['same_encoder']:.3f}** versus "
        f"**{grouped['spike_cka']['same_decoder']:.3f}**, and membrane CKA is "
        f"**{grouped['membrane_cka']['same_encoder']:.3f}** versus "
        f"**{grouped['membrane_cka']['same_decoder']:.3f}**.",
        "",
        f"Sorted firing-rate profiles remain highly similar across encoders "
        f"(**{grouped['sorted_activity_correlation']['same_decoder']:.3f}**) even "
        f"though sample-wise spike CKA is only "
        f"**{grouped['spike_cka']['same_decoder']:.3f}**. Similar activity histograms "
        "therefore do not imply that the same images recruit the population similarly.",
        "",
        f"Raw singular-spectrum cosine averages **{observed_spectrum_mean:.3f}** across "
        f"model pairs. The shuffled-weight control averages "
        f"**{shuffled_spectrum_mean:.3f}**. This control is required because sorted, "
        "nonnegative spectra have a high cosine even without shared topology.",
        "",
        "Activation-derived neuron matching tests whether permutation symmetry explains "
        "the weight mismatch. Averaged across cross-encoder comparisons:",
        "",
        f"- matched activation correlation: **{alignment_summary['match_correlation']:.3f}**, "
        f"versus **{alignment_summary['match_random_mean']:.3f}** for random assignments",
        f"- recurrent correlation: **{alignment_summary['weight_correlation_before']:.3f}** "
        f"before, **{alignment_summary['weight_correlation_after']:.3f}** after alignment",
        f"- top-edge Jaccard: **{alignment_summary['edge_jaccard_before']:.3f}** before, "
        f"**{alignment_summary['edge_jaccard_after']:.3f}** after alignment, and "
        f"**{alignment_summary['edge_jaccard_random_mean']:.3f}** for random assignments",
        f"- top-5 singular-subspace overlap: "
        f"**{alignment_summary['subspace_overlap_before']:.3f}** before, "
        f"**{alignment_summary['subspace_overlap_after']:.3f}** after alignment "
        f"(random-subspace expectation: **{5 / int(results[0]['reservoir_dim']):.3f}**)",
        "",
        "## Recurrent Structure Versus Shuffled Weights",
        "",
        "Shuffling preserves every recurrent weight value but destroys which neurons "
        "those values connect.",
        "",
        "| Metric | Observed mean | Shuffled mean | Models above +1.96 shuffle SD | Models below -1.96 shuffle SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in NULL_METRICS:
        values = null_summary[metric]
        lines.append(
            f"| {MODEL_METRIC_LABELS.get(metric, metric)} | "
            f"{values['observed_mean']:.3f} | {values['shuffle_mean']:.3f} | "
            f"{values['significant_high_fraction']:.1%} | "
            f"{values['significant_low_fraction']:.1%} |"
        )

    lines.extend(
        [
            "",
            "The consistent low effective/stable rank, high leading-mode energy, and "
            "high non-normality are genuine organization beyond the weight histogram. "
            "Strong-edge reciprocity is not: its observed mean is approximately the "
            "shuffle expectation.",
            "",
            "## Functional Readout",
            "",
            "A nearest-class-centroid classifier was evaluated on half of each digit "
            "using centroids formed from the other half. Across checkpoints:",
            "",
            f"- final accuracy versus spike-centroid accuracy: Spearman "
            f"**{model_spearman['spike_centroid_accuracy']['rho']:.3f}**",
            f"- final accuracy versus membrane-centroid accuracy: Spearman "
            f"**{model_spearman['membrane_centroid_accuracy']['rho']:.3f}**",
            f"- final accuracy versus digit-selective neuron fraction: Spearman "
            f"**{model_spearman['spike_selective_fraction']['rho']:.3f}**",
            f"- final accuracy versus effective recurrent rank: Spearman "
            f"**{model_spearman['effective_rank']['rho']:.3f}**",
            f"- final accuracy versus recurrent non-normality: Spearman "
            f"**{model_spearman['non_normality']['rho']:.3f}**",
            "",
            "`spike_mlp` is the clearest decoder outlier: "
            f"mean spike rate **{decoder_means['spike_mlp']['spike_rate']:.3f}**, "
            f"effective rank **{decoder_means['spike_mlp']['effective_rank']:.2f}**, "
            f"and accuracy **{decoder_means['spike_mlp']['accuracy']:.2%}**. "
            "The other decoders average "
            f"**{np.mean([decoder_means[name]['spike_rate'] for name in DECODER_ORDER if name != 'spike_mlp']):.3f}** "
            "spikes per unit-step.",
            "",
        ]
    )

    lines.extend(
        [
            "## Topology Versus Function",
            "",
            "Spearman correlations between exact recurrent-weight correlation and functional similarity:",
            "",
        ]
    )
    for name, value in functional_associations.items():
        lines.append(f"- {name.replace('_', ' ')}: **{value:.3f}**")

    lines.extend(
        [
            "",
            "## Encoder/Decoder Variance",
            "",
            "| Metric | Encoder | Decoder | Residual/interaction | Largest component |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for metric, values in decompositions.items():
        lines.append(
            f"| {MODEL_METRIC_LABELS.get(metric, metric)} | "
            f"{values['encoder']:.1%} | {values['decoder']:.1%} | "
            f"{values['residual']:.1%} | {dominant_effects[metric]} |"
        )

    lines.extend(
        [
            "",
            "## Largest Departures From Shuffled Topology",
            "",
            "Positive z means the observed matrix metric is larger than entry-shuffled "
            "versions of the same matrix; negative means smaller.",
            "",
            "| Model | Metric | z-score |",
            "|---|---|---:|",
        ]
    )
    for _, model, metric, z_score in strongest_null:
        lines.append(f"| {model} | {metric.replace('_', ' ')} | {z_score:.2f} |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "grouped_pair_means": grouped,
        "alignment_summary": alignment_summary,
        "variance_decomposition": decompositions,
        "recurrent_function_spearman": functional_associations,
        "observed_spectrum_cosine_mean": observed_spectrum_mean,
        "shuffled_spectrum_cosine_mean": shuffled_spectrum_mean,
        "sample_permutation_cka_baselines": sample_permutation_baselines,
        "shuffled_weight_summary": null_summary,
        "model_accuracy_spearman": model_spearman,
        "decoder_means": decoder_means,
    }


def main() -> None:
    args = parse_args()
    if args.samples_per_digit < 2:
        raise ValueError("--samples-per-digit must be at least 2.")
    if args.shuffle_repeats <= 0:
        raise ValueError("--shuffle-repeats must be positive.")
    if not 0.0 < args.top_edge_fraction < 1.0:
        raise ValueError("--top-edge-fraction must be between zero and one.")

    train_pipeline.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
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
        raise FileNotFoundError(f"No checkpoints found under {args.sweep_root}.")

    dataset, _ = balanced_test_subset(args.data_dir, args.samples_per_digit)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for index, checkpoint_path in enumerate(checkpoint_paths, start=1):
        print(f"[{index}/{len(checkpoint_paths)}] {checkpoint_path.parent.name}", flush=True)
        result = infer_checkpoint(
            checkpoint_path,
            loader,
            device,
            args.temporal_bins,
        )
        result.update(activity_metrics(result))
        weight_metrics = matrix_metrics(
            np.asarray(result["recurrent"]),
            args.top_edge_fraction,
        )
        result.update(weight_metrics)
        controls, shuffle_mean_spectrum = shuffled_weight_controls(
            np.asarray(result["recurrent"]),
            args.top_edge_fraction,
            args.shuffle_repeats,
            rng,
        )
        for metric, control in controls.items():
            result[f"{metric}_shuffle_mean"] = control["mean"]
            result[f"{metric}_shuffle_std"] = control["std"]
            result[f"{metric}_shuffle_z"] = control["z"]
        result["shuffle_mean_spectrum"] = shuffle_mean_spectrum
        results.append(result)

    labels = [str(result["name"]) for result in results]
    matrices: dict[str, np.ndarray] = {}
    sample_permutation_baselines: dict[str, dict[str, float]] = {}
    for matrix_name, feature_name in [
        ("spike_cka", "spike_mean"),
        ("membrane_cka", "membrane_mean"),
        ("membrane_last_cka", "membrane_last"),
        ("temporal_cka", "temporal"),
    ]:
        matrix, statistics = cka_matrix(results, feature_name)
        matrices[matrix_name] = matrix
        sample_permutation_baselines[matrix_name] = sample_permutation_cka_baseline(
            statistics,
            results,
            rng,
        )
    matrices.update({
        "prediction_agreement": prediction_agreement_matrix(results),
        "digit_geometry_correlation": vector_similarity_matrix(
            [np.asarray(result["membrane_geometry"]) for result in results]
        ),
        "recurrent_correlation": vector_similarity_matrix(
            [
                np.asarray(result["recurrent"])[
                    ~np.eye(int(result["reservoir_dim"]), dtype=bool)
                ]
                for result in results
            ]
        ),
        "spectrum_cosine": vector_similarity_matrix(
            [np.asarray(result["normalized_singular_values"]) for result in results],
            cosine=True,
        ),
        "centered_spectrum_cosine": vector_similarity_matrix(
            [
                np.asarray(result["centered_normalized_singular_values"])
                for result in results
            ],
            cosine=True,
        ),
        "shuffled_spectrum_cosine": vector_similarity_matrix(
            [np.asarray(result["shuffle_mean_spectrum"]) for result in results],
            cosine=True,
        ),
        "sorted_activity_correlation": vector_similarity_matrix(
            [
                np.sort(
                    np.asarray(result["spike_mean"])
                    .mean(axis=0)
                    .reshape(len(result["taus"]), int(result["reservoir_dim"])),
                    axis=1,
                ).reshape(-1)
                for result in results
            ]
        ),
    })

    pairwise_rows = build_pairwise_rows(results, matrices)
    metric_names = list(matrices)
    grouped = grouped_pair_means(pairwise_rows, metric_names)
    alignment = alignment_rows(results, args.top_edge_fraction, rng)

    decomposition_metrics = [
        "accuracy",
        "spike_rate",
        "spike_participation",
        "spike_selective_fraction",
        "spike_centroid_accuracy",
        "membrane_centroid_accuracy",
        "stable_rank",
        "effective_rank",
        "spectral_radius",
        "non_normality",
        "reciprocity",
        "top_edge_reciprocity",
    ]
    decompositions = {
        metric: additive_variance_decomposition(results, metric)
        for metric in decomposition_metrics
    }

    association_names = [
        "spike_cka",
        "membrane_cka",
        "temporal_cka",
        "prediction_agreement",
        "digit_geometry_correlation",
        "recurrent_correlation",
        "spectrum_cosine",
        "sorted_activity_correlation",
    ]
    associations, association_p_values = metric_associations(
        pairwise_rows,
        association_names,
    )

    heatmaps = {
        "spike_cka": (-0.1, 1.0, "viridis"),
        "membrane_cka": (-0.1, 1.0, "viridis"),
        "temporal_cka": (-0.1, 1.0, "viridis"),
        "prediction_agreement": (0.0, 1.0, "viridis"),
        "digit_geometry_correlation": (-1.0, 1.0, "coolwarm"),
        "recurrent_correlation": (-1.0, 1.0, "coolwarm"),
        "centered_spectrum_cosine": (0.0, 1.0, "viridis"),
    }
    for name, (low, high, cmap) in heatmaps.items():
        save_matrix_heatmap(
            matrices[name],
            labels,
            output_dir / f"pairwise_{name}.png",
            f"Pairwise {name.replace('_', ' ')}",
            vmin=low,
            vmax=high,
            cmap=cmap,
        )
    save_matrix_heatmap(
        associations,
        [name.replace("_", " ") for name in association_names],
        output_dir / "pairwise_metric_associations.png",
        "Spearman association between pairwise similarity metrics",
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
    )
    save_model_metric_grid(
        results,
        [
            "accuracy",
            "spike_centroid_accuracy",
            "membrane_centroid_accuracy",
            "stable_rank",
            "spectral_radius",
            "non_normality",
        ],
        output_dir / "encoder_decoder_checkpoint_metrics.png",
    )
    save_null_zscores(results, labels, output_dir / "shuffled_weight_null_zscores.png")
    save_spectral_profiles(results, output_dir / "recurrent_spectral_profiles.png")
    save_alignment_summary(alignment, output_dir / "neuron_alignment_effect.png")
    save_variance_decomposition(
        decompositions,
        output_dir / "encoder_decoder_variance_decomposition.png",
    )
    save_group_similarity(
        grouped,
        [
            "spike_cka",
            "membrane_cka",
            "temporal_cka",
            "prediction_agreement",
            "digit_geometry_correlation",
            "recurrent_correlation",
            "centered_spectrum_cosine",
            "sorted_activity_correlation",
        ],
        output_dir / "pair_group_similarity_summary.png",
    )

    write_csv(serializable_model_rows(results), output_dir / "model_metrics.csv")
    write_csv(pairwise_rows, output_dir / "pairwise_metrics.csv")
    write_csv(alignment, output_dir / "neuron_alignment_metrics.csv")
    np.savez_compressed(
        output_dir / "analysis_matrices.npz",
        model_names=np.asarray(labels),
        association_names=np.asarray(association_names),
        metric_associations=associations,
        metric_association_p_values=association_p_values,
        **matrices,
    )
    report_metrics = write_report(
        results,
        grouped,
        alignment,
        decompositions,
        association_names,
        associations,
        matrices["spectrum_cosine"],
        matrices["shuffled_spectrum_cosine"],
        sample_permutation_baselines,
        output_dir / "REPORT.md",
        args.samples_per_digit,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(report_metrics, indent=2),
        encoding="utf-8",
    )
    print(f"Saved checkpoint-only analysis to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
