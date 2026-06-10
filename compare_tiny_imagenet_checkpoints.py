"""Compare the sparse 702K and dense 2048-unit Tiny ImageNet checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import svdvals

from analyze_mnist_sweep_checkpoint_only import (
    matrix_metrics,
    shuffled_weight_controls,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--small-checkpoint",
        default="best_702k_cnn3_membrane_transformer.pt",
    )
    parser.add_argument("--large-checkpoint", default="best.pt")
    parser.add_argument(
        "--large-analysis-dir",
        default="visualizations/best_checkpoint_structure",
    )
    parser.add_argument(
        "--output-dir",
        default="visualizations/tiny_imagenet_702k_vs_2048",
    )
    parser.add_argument("--shuffle-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=False)


def parameter_counts(state: dict[str, torch.Tensor]) -> dict[str, int]:
    groups = {"encoder": 0, "backbone": 0, "decoder": 0, "other": 0}
    for name, value in state.items():
        group = next(
            (
                candidate
                for candidate in ["encoder", "backbone", "decoder"]
                if name.startswith(candidate + ".")
            ),
            "other",
        )
        groups[group] += value.numel()
    groups["total"] = sum(groups.values())
    return groups


def spectrum_metrics(singular: np.ndarray) -> dict[str, float]:
    singular = np.sort(np.asarray(singular, dtype=np.float64))[::-1]
    energy = singular**2
    probability = energy / max(float(energy.sum()), 1e-300)
    entropy = -float(
        np.sum(probability * np.log(np.maximum(probability, 1e-300)))
    )
    return {
        "stable_rank": float(energy.sum() / max(float(energy[0]), 1e-300)),
        "effective_rank": float(np.exp(entropy)),
        "top1_energy": float(probability[0]),
        "top5_energy": float(probability[:5].sum()),
        "top16_energy": float(probability[:16].sum()),
        "top64_energy": float(probability[:64].sum()),
    }


def repeated_operator_spectra(
    weight: np.ndarray,
    taus: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    singular = svdvals(weight, check_finite=False)
    raw = np.sort(np.tile(singular, len(taus)))[::-1]
    scaled = np.sort(
        np.concatenate([singular / tau for tau in taus])
    )[::-1]
    return raw, scaled


def checkpoint_summary(
    checkpoint: dict[str, object],
    checkpoint_path: Path,
    shuffle_repeats: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    state = checkpoint["model_state"]
    args = checkpoint["args"]
    weight = state["backbone.shared_recurrent.weight"].double().numpy()
    metrics = matrix_metrics(weight, 0.10)
    energy = np.asarray(metrics["singular_values"]) ** 2
    probability = energy / energy.sum()
    metrics["top16_energy"] = float(probability[:16].sum())
    metrics["top64_energy"] = float(probability[:64].sum())
    controls, null_spectrum = shuffled_weight_controls(
        weight,
        0.10,
        shuffle_repeats,
        rng,
    )
    null_spectrum *= np.linalg.norm(weight, ord="fro")
    taus = np.asarray([float(value) for value in str(args["taus"]).split(",")])
    raw_spectrum, scaled_spectrum = repeated_operator_spectra(weight, taus)
    mask = state.get("backbone.neuron_mask")
    active_neurons = (
        int(torch.count_nonzero(mask))
        if isinstance(mask, torch.Tensor)
        else int(args["reservoir_dim"])
    )
    nonzero_weights = int(np.count_nonzero(weight))
    total_weights = int(weight.size)
    return {
        "path": str(checkpoint_path.resolve()),
        "checkpoint": checkpoint,
        "args": args,
        "state": state,
        "weight": weight,
        "parameters": parameter_counts(state),
        "epoch": int(checkpoint.get("epoch", -1)),
        "accuracy": float(checkpoint.get("best_val_acc", float("nan"))),
        "reservoir_dim": int(args["reservoir_dim"]),
        "taus": taus,
        "state_units": int(len(taus) * int(args["reservoir_dim"])),
        "active_neurons": active_neurons,
        "nonzero_weights": nonzero_weights,
        "weight_sparsity": 1.0 - nonzero_weights / total_weights,
        "matrix_metrics": metrics,
        "shuffle_controls": controls,
        "shuffle_mean_spectrum": null_spectrum,
        "raw_operator_spectrum": raw_spectrum,
        "scaled_operator_spectrum": scaled_spectrum,
        "raw_operator_metrics": spectrum_metrics(raw_spectrum),
        "scaled_operator_metrics": spectrum_metrics(scaled_spectrum),
    }


def add_large_cross_metrics(
    summary: dict[str, object],
    analysis_dir: Path,
) -> None:
    metrics_path = analysis_dir / "metrics.json"
    spectra_path = analysis_dir / "spectra.npz"
    if not metrics_path.is_file() or not spectra_path.is_file():
        raise FileNotFoundError(
            "Run analyze_best_checkpoint_structure.py before this comparison."
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    spectra = np.load(spectra_path)
    summary["raw_operator_spectrum"] = spectra["full_raw"]
    summary["scaled_operator_spectrum"] = spectra["full_tau_scaled"]
    summary["raw_operator_metrics"] = metrics["full_operator"]["raw"]
    summary["scaled_operator_metrics"] = metrics["full_operator"]["tau_scaled"]
    summary["cross_metrics"] = metrics["cross_neuron"]
    summary["tau_mix_metrics"] = metrics["tau_mix"]


def normalized_spectrum(singular: np.ndarray) -> np.ndarray:
    singular = np.asarray(singular, dtype=np.float64)
    return singular / max(float(np.linalg.norm(singular)), 1e-300)


def cumulative_energy(singular: np.ndarray) -> np.ndarray:
    energy = np.asarray(singular, dtype=np.float64) ** 2
    return np.cumsum(energy) / max(float(energy.sum()), 1e-300)


def save_shared_comparison(
    small: dict[str, object],
    large: dict[str, object],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    names = ["702K sparse", "1.65M dense"]
    summaries = [small, large]

    for index, (name, summary) in enumerate(zip(names, summaries)):
        weight = np.asarray(summary["weight"])
        limit = float(np.percentile(np.abs(weight), 99))
        image = axes[0, index].imshow(
            weight,
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        axes[0, index].set_title(
            f"{name} shared recurrent matrix\n"
            f"{summary['weight_sparsity']:.1%} zero weights"
        )
        axes[0, index].set_xlabel("source neuron")
        axes[0, index].set_ylabel("target neuron")
        fig.colorbar(image, ax=axes[0, index], fraction=0.046)

    axes[0, 2].axis("off")
    table_rows = [
        ["Parameters", f"{small['parameters']['total']:,}", f"{large['parameters']['total']:,}"],
        ["Validation accuracy", f"{small['accuracy']:.2%}", f"{large['accuracy']:.2%}"],
        ["Tau-neuron units", f"{small['state_units']:,}", f"{large['state_units']:,}"],
        ["Shared matrix", f"{small['reservoir_dim']}x{small['reservoir_dim']}", f"{large['reservoir_dim']}x{large['reservoir_dim']}"],
        ["Weight sparsity", f"{small['weight_sparsity']:.1%}", f"{large['weight_sparsity']:.1%}"],
        ["Stable rank", f"{small['matrix_metrics']['stable_rank']:.2f}", f"{large['matrix_metrics']['stable_rank']:.2f}"],
        ["Effective rank", f"{small['matrix_metrics']['effective_rank']:.2f}", f"{large['matrix_metrics']['effective_rank']:.2f}"],
        ["First-mode energy", f"{small['matrix_metrics']['top1_energy']:.1%}", f"{large['matrix_metrics']['top1_energy']:.1%}"],
    ]
    table = axes[0, 2].table(
        cellText=table_rows,
        colLabels=["Metric", "702K", "1.65M"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.05, 1.65)
    axes[0, 2].set_title("Checkpoint comparison")

    for name, summary in zip(names, summaries):
        spectrum = normalized_spectrum(
            np.asarray(summary["matrix_metrics"]["singular_values"])
        )
        rank_fraction = np.arange(1, len(spectrum) + 1) / len(spectrum)
        axes[1, 0].plot(rank_fraction, spectrum, label=name, linewidth=2)
        axes[1, 1].plot(
            rank_fraction,
            cumulative_energy(
                np.asarray(summary["matrix_metrics"]["singular_values"])
            ),
            label=name,
            linewidth=2,
        )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Width-normalized shared spectrum")
    axes[1, 0].set_xlabel("fraction of singular directions")
    axes[1, 0].set_ylabel("L2-normalized singular value")
    axes[1, 1].set_title("Cumulative shared recurrent energy")
    axes[1, 1].set_xlabel("fraction of singular directions")
    axes[1, 1].set_ylabel("cumulative energy")

    metric_names = [
        "stable_rank",
        "effective_rank",
        "top1_energy",
        "non_normality",
        "row_norm_cv",
        "top_edge_reciprocity",
    ]
    metric_labels = [
        "Stable rank / width",
        "Effective rank / width",
        "First-mode energy",
        "Non-normality",
        "Row-strength CV",
        "Strong-edge reciprocity",
    ]
    small_values = [
        float(small["matrix_metrics"]["stable_rank"]) / int(small["reservoir_dim"]),
        float(small["matrix_metrics"]["effective_rank"]) / int(small["reservoir_dim"]),
        *[float(small["matrix_metrics"][name]) for name in metric_names[2:]],
    ]
    large_values = [
        float(large["matrix_metrics"]["stable_rank"]) / int(large["reservoir_dim"]),
        float(large["matrix_metrics"]["effective_rank"]) / int(large["reservoir_dim"]),
        *[float(large["matrix_metrics"][name]) for name in metric_names[2:]],
    ]
    positions = np.arange(len(metric_labels))
    width = 0.38
    axes[1, 2].bar(positions - width / 2, small_values, width, label=names[0])
    axes[1, 2].bar(positions + width / 2, large_values, width, label=names[1])
    axes[1, 2].set_xticks(positions)
    axes[1, 2].set_xticklabels(metric_labels, rotation=35, ha="right")
    axes[1, 2].set_title("Normalized structural metrics")
    axes[1, 2].legend()

    for axis in axes[1]:
        axis.grid(alpha=0.2)
        if axis is not axes[1, 2]:
            axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_operator_comparison(
    small: dict[str, object],
    large: dict[str, object],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    names = ["702K: 512 units", "1.65M: 2048 units"]
    colors = ["tab:blue", "tab:orange"]
    summaries = [small, large]

    for name, color, summary in zip(names, colors, summaries):
        for row, key in enumerate(
            ["raw_operator_spectrum", "scaled_operator_spectrum"]
        ):
            spectrum = np.asarray(summary[key])
            rank_fraction = np.arange(1, len(spectrum) + 1) / len(spectrum)
            axes[row, 0].plot(
                rank_fraction,
                normalized_spectrum(spectrum),
                label=name,
                color=color,
                linewidth=2,
            )
            axes[row, 1].plot(
                rank_fraction,
                cumulative_energy(spectrum),
                label=name,
                color=color,
                linewidth=2,
            )

    axes[0, 0].set_title("Complete recurrent-current spectrum")
    axes[0, 1].set_title("Complete operator cumulative energy")
    axes[1, 0].set_title("Spectrum after target 1/tau scaling")
    axes[1, 1].set_title("Tau-scaled cumulative energy")
    for row in range(2):
        axes[row, 0].set_yscale("log")
        axes[row, 0].set_ylabel("L2-normalized singular value")
        axes[row, 1].set_ylabel("cumulative energy")
        for col in range(2):
            axes[row, col].set_xlabel("fraction of singular directions")
            axes[row, col].grid(alpha=0.2)
            axes[row, col].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def serializable_summary(summary: dict[str, object]) -> dict[str, object]:
    matrix = summary["matrix_metrics"]
    return {
        "path": summary["path"],
        "epoch": summary["epoch"],
        "accuracy": summary["accuracy"],
        "parameters": summary["parameters"],
        "reservoir_dim": summary["reservoir_dim"],
        "tau_count": len(summary["taus"]),
        "state_units": summary["state_units"],
        "weight_sparsity": summary["weight_sparsity"],
        "nonzero_shared_weights": summary["nonzero_weights"],
        "shared": {
            name: float(matrix[name])
            for name in [
                "stable_rank",
                "effective_rank",
                "top1_energy",
                "top5_energy",
                "top16_energy",
                "top64_energy",
                "spectral_radius",
                "non_normality",
                "reciprocity",
                "row_norm_cv",
                "top_edge_reciprocity",
            ]
        },
        "shared_shuffle": summary["shuffle_controls"],
        "raw_operator": summary["raw_operator_metrics"],
        "tau_scaled_operator": summary["scaled_operator_metrics"],
    }


def write_report(
    small: dict[str, object],
    large: dict[str, object],
    output_path: Path,
) -> None:
    small_matrix = small["matrix_metrics"]
    large_matrix = large["matrix_metrics"]
    small_scaled = small["scaled_operator_metrics"]
    large_scaled = large["scaled_operator_metrics"]
    large_cross = large["cross_metrics"]

    small_recurrent_parameters = small["reservoir_dim"] ** 2
    large_recurrent_parameters = (
        large["reservoir_dim"] ** 2
        + int(large["args"]["cross_rank"]) * large["reservoir_dim"] * 2
        + len(large["taus"]) ** 2
    )
    small_dense_equivalent = small["state_units"] ** 2
    large_dense_equivalent = large["state_units"] ** 2

    lines = [
        "# Tiny ImageNet Checkpoint Comparison",
        "",
        "| Property | 702K sparse model | 1.65M cross-reservoir model |",
        "|---|---:|---:|",
        f"| Best validation accuracy | {small['accuracy']:.2%} | {large['accuracy']:.2%} |",
        f"| Checkpoint epoch | {small['epoch']} | {large['epoch']} |",
        f"| Parameters | {small['parameters']['total']:,} | {large['parameters']['total']:,} |",
        f"| Effective tau-neuron state | {small['state_units']:,} | {large['state_units']:,} |",
        f"| Shared recurrent matrix | {small['reservoir_dim']} x {small['reservoir_dim']} | {large['reservoir_dim']} x {large['reservoir_dim']} |",
        f"| Shared recurrent sparsity | {small['weight_sparsity']:.1%} | {large['weight_sparsity']:.1%} |",
        f"| Recurrent/cross parameters | {small_recurrent_parameters:,} | {large_recurrent_parameters:,} |",
        f"| Dense state-to-state equivalent | {small_dense_equivalent:,} | {large_dense_equivalent:,} |",
        "",
        "The smaller model is the completed/pruned checkpoint. The larger model was "
        "still training when its epoch-61 `best.pt` was analyzed, so accuracy is not a "
        "final architecture comparison.",
        "",
        "## Shared Matrix",
        "",
        "| Metric | 702K sparse | 1.65M dense |",
        "|---|---:|---:|",
        f"| Stable rank | {small_matrix['stable_rank']:.2f} | {large_matrix['stable_rank']:.2f} |",
        f"| Stable rank / width | {small_matrix['stable_rank'] / small['reservoir_dim']:.2%} | {large_matrix['stable_rank'] / large['reservoir_dim']:.2%} |",
        f"| Effective rank | {small_matrix['effective_rank']:.2f} | {large_matrix['effective_rank']:.2f} |",
        f"| Effective rank / width | {small_matrix['effective_rank'] / small['reservoir_dim']:.2%} | {large_matrix['effective_rank'] / large['reservoir_dim']:.2%} |",
        f"| First-mode energy | {small_matrix['top1_energy']:.1%} | {large_matrix['top1_energy']:.1%} |",
        f"| First-five energy | {small_matrix['top5_energy']:.1%} | {large_matrix['top5_energy']:.1%} |",
        f"| First-16 energy | {small_matrix['top16_energy']:.1%} | {large_matrix['top16_energy']:.1%} |",
        f"| Non-normality | {small_matrix['non_normality']:.3f} | {large_matrix['non_normality']:.3f} |",
        f"| Row-strength variation | {small_matrix['row_norm_cv']:.3f} | {large_matrix['row_norm_cv']:.3f} |",
        f"| Strong-edge reciprocity | {small_matrix['top_edge_reciprocity']:.3f} | {large_matrix['top_edge_reciprocity']:.3f} |",
        "",
        "Both models learn a strong dominant direction and similarly high non-normality. "
        "Their residual structure differs:",
        "",
        f"- The sparse model puts **{small_matrix['top5_energy']:.1%}** of energy in "
        "five modes. Its 95% pruning leaves a very uneven, hub-like row-strength "
        f"distribution (CV **{small_matrix['row_norm_cv']:.2f}**).",
        f"- The dense model puts more energy in its first mode "
        f"(**{large_matrix['top1_energy']:.1%}**) but retains a broader absolute tail: "
        f"effective rank **{large_matrix['effective_rank']:.1f}** versus "
        f"**{small_matrix['effective_rank']:.1f}**.",
        "- Relative to width, the dense model is more compressed: effective rank is "
        f"**{large_matrix['effective_rank'] / large['reservoir_dim']:.1%}** of width "
        f"versus **{small_matrix['effective_rank'] / small['reservoir_dim']:.1%}**.",
        "",
        "## Complete Tau-Neuron Operator",
        "",
        "The smaller model repeats the same matrix independently across eight taus. The "
        "larger model additionally couples the taus through its learned cross-reservoir "
        "path.",
        "",
        "| Tau-scaled metric | 702K: 512 units | 1.65M: 2048 units |",
        "|---|---:|---:|",
        f"| Stable rank | {small_scaled['stable_rank']:.2f} | {large_scaled['stable_rank']:.2f} |",
        f"| Effective rank | {small_scaled['effective_rank']:.2f} | {large_scaled['effective_rank']:.2f} |",
        f"| Effective rank / state width | {small_scaled['effective_rank'] / small['state_units']:.2%} | {large_scaled['effective_rank'] / large['state_units']:.2%} |",
        f"| First-mode energy | {small_scaled['top1_energy']:.1%} | {large_scaled['top1_energy']:.1%} |",
        f"| First-five energy | {small_scaled['top5_energy']:.1%} | {large_scaled['top5_energy']:.1%} |",
        "",
        "After tau scaling, both systems have almost the same stable rank and leading "
        "energy. The larger system nevertheless retains substantially more effective "
        f"dimensions (**{large_scaled['effective_rank']:.1f}** versus "
        f"**{small_scaled['effective_rank']:.1f}**), while using four times as many "
        "tau-neuron units.",
        "",
        "## Cross-Reservoir Difference",
        "",
        f"The large model's rank-16 cross-neuron path has effective rank "
        f"**{large_cross['effective_rank']:.2f}**, with "
        f"**{large_cross['top1_energy']:.1%}** of its energy in one mode. This creates "
        "a global communication channel absent from the smaller model.",
        "",
        "## Interpretation",
        "",
        "The 702K model obtains its compact dynamics through explicit magnitude pruning: "
        "a tiny number of recurrent edges and a handful of dominant modes. The 1.65M "
        "model remains fully dense but self-organizes into a dominant shared mode plus "
        "an almost rank-one cross-tau channel.",
        "",
        "The larger architecture is therefore not simply a wider version of the smaller "
        "one. It trades explicit sparsity for dense low-dimensional coordination and "
        "global communication across timescales. Whether that improves generalization "
        "must be judged after its training schedule and official validation evaluation "
        "are complete.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.shuffle_repeats <= 0:
        raise ValueError("--shuffle-repeats must be positive.")
    rng = np.random.default_rng(args.seed)
    small_path = Path(args.small_checkpoint)
    large_path = Path(args.large_checkpoint)
    small = checkpoint_summary(
        load_checkpoint(small_path),
        small_path,
        args.shuffle_repeats,
        rng,
    )
    large = checkpoint_summary(
        load_checkpoint(large_path),
        large_path,
        args.shuffle_repeats,
        rng,
    )
    add_large_cross_metrics(large, Path(args.large_analysis_dir))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_shared_comparison(
        small,
        large,
        output_dir / "shared_recurrent_comparison.png",
    )
    save_operator_comparison(
        small,
        large,
        output_dir / "full_operator_comparison.png",
    )
    write_report(small, large, output_dir / "REPORT.md")
    metrics = {
        "small_702k": serializable_summary(small),
        "large_2048": serializable_summary(large),
        "large_cross": large["cross_metrics"],
        "large_tau_mix": large["tau_mix_metrics"],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "spectra.npz",
        small_shared=np.asarray(small["matrix_metrics"]["singular_values"]),
        large_shared=np.asarray(large["matrix_metrics"]["singular_values"]),
        small_raw_operator=np.asarray(small["raw_operator_spectrum"]),
        large_raw_operator=np.asarray(large["raw_operator_spectrum"]),
        small_tau_scaled_operator=np.asarray(small["scaled_operator_spectrum"]),
        large_tau_scaled_operator=np.asarray(large["scaled_operator_spectrum"]),
    )
    print(f"Saved comparison to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
