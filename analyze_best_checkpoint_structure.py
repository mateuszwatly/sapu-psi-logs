"""Analyze recurrent structure in a cross-reservoir TPSAPU checkpoint."""

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
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument(
        "--output-dir",
        default="visualizations/best_checkpoint_structure",
    )
    parser.add_argument("--shuffle-repeats", type=int, default=100)
    parser.add_argument("--full-null-repeats", type=int, default=10)
    parser.add_argument("--top-edge-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def spectrum_metrics(values: np.ndarray) -> dict[str, object]:
    singular = svdvals(values, overwrite_a=False, check_finite=False)
    energy = singular**2
    probability = energy / max(float(energy.sum()), 1e-12)
    entropy = -float(
        np.sum(probability * np.log(np.maximum(probability, 1e-300)))
    )
    return {
        "dimension": int(min(values.shape)),
        "spectral_norm": float(singular[0]),
        "stable_rank": float(energy.sum() / max(float(energy[0]), 1e-12)),
        "effective_rank": float(np.exp(entropy)),
        "spectral_entropy": float(entropy / np.log(len(singular))),
        "top1_energy": float(probability[0]),
        "top5_energy": float(probability[:5].sum()),
        "top16_energy": float(probability[:16].sum()),
        "top64_energy": float(probability[:64].sum()),
        "singular_values": singular,
    }


def scalar_summary(samples: list[dict[str, object]], metric: str) -> dict[str, float]:
    values = np.asarray([float(sample[metric]) for sample in samples])
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def z_score(value: float, summary: dict[str, float]) -> float:
    return (value - summary["mean"]) / max(summary["std"], 1e-12)


def shuffle_factor_control(
    up: np.ndarray,
    down: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    results = []
    for _ in range(repeats):
        shuffled_up = rng.permutation(up.reshape(-1)).reshape(up.shape)
        shuffled_down = rng.permutation(down.reshape(-1)).reshape(down.shape)
        results.append(spectrum_metrics(shuffled_up @ shuffled_down))
    return results


def shuffle_tau_mix(
    tau_mix: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    off_diagonal = ~np.eye(tau_mix.shape[0], dtype=bool)
    entries = tau_mix[off_diagonal]
    results = []
    for _ in range(repeats):
        shuffled = np.zeros_like(tau_mix)
        shuffled[off_diagonal] = rng.permutation(entries)
        results.append(spectrum_metrics(shuffled))
    return results


def architecture_null(
    shared: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    tau_mix: np.ndarray,
    taus: np.ndarray,
    gain: float,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    tau_count = len(taus)
    reservoir_dim = shared.shape[0]
    identity_tau = np.eye(tau_count)
    tau_scale = np.kron(np.diag(1.0 / taus), np.eye(reservoir_dim))
    off_diagonal = ~np.eye(tau_count, dtype=bool)
    tau_entries = tau_mix[off_diagonal]
    raw_results = []
    scaled_results = []

    for repeat in range(repeats):
        print(f"  full structured null {repeat + 1}/{repeats}", flush=True)
        shuffled_shared = rng.permutation(shared.reshape(-1)).reshape(shared.shape)
        shuffled_up = rng.permutation(up.reshape(-1)).reshape(up.shape)
        shuffled_down = rng.permutation(down.reshape(-1)).reshape(down.shape)
        shuffled_mix = np.zeros_like(tau_mix)
        shuffled_mix[off_diagonal] = rng.permutation(tau_entries)
        shuffled_cross = shuffled_up @ shuffled_down
        operator = np.kron(identity_tau, shuffled_shared)
        operator += gain * np.kron(shuffled_mix, shuffled_cross)
        raw_results.append(spectrum_metrics(operator))
        scaled_results.append(spectrum_metrics(tau_scale @ operator))

    return raw_results, scaled_results


def save_dashboard(
    shared: np.ndarray,
    shared_spectrum: np.ndarray,
    shared_null_spectrum: np.ndarray,
    cross_spectrum: np.ndarray,
    cross_null_spectrum: np.ndarray,
    tau_mix: np.ndarray,
    taus: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    shared_limit = float(np.percentile(np.abs(shared), 99))
    image = axes[0, 0].imshow(
        shared,
        cmap="coolwarm",
        vmin=-shared_limit,
        vmax=shared_limit,
    )
    axes[0, 0].set_title("Shared recurrent matrix W")
    axes[0, 0].set_xlabel("source neuron")
    axes[0, 0].set_ylabel("target neuron")
    fig.colorbar(image, ax=axes[0, 0], fraction=0.046)

    rank = np.arange(1, len(shared_spectrum) + 1)
    axes[0, 1].plot(rank, shared_spectrum, label="trained", linewidth=2)
    axes[0, 1].plot(rank, shared_null_spectrum, label="entry-shuffled", linewidth=2)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Shared recurrent singular spectrum")
    axes[0, 1].set_xlabel("singular-value rank")
    axes[0, 1].set_ylabel("singular value")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.2)

    normalized_observed = shared_spectrum / np.linalg.norm(shared_spectrum)
    normalized_null = shared_null_spectrum / np.linalg.norm(shared_null_spectrum)
    axes[0, 2].plot(
        rank,
        normalized_observed - normalized_null,
        linewidth=2,
    )
    axes[0, 2].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 2].set_title("Shared spectrum minus shuffled control")
    axes[0, 2].set_xlabel("singular-value rank")
    axes[0, 2].set_ylabel("normalized difference")
    axes[0, 2].grid(alpha=0.2)

    cross_rank = np.arange(1, len(cross_spectrum) + 1)
    axes[1, 0].plot(cross_rank, cross_spectrum, label="trained factors", linewidth=2)
    axes[1, 0].plot(
        cross_rank,
        cross_null_spectrum,
        label="factor-shuffled",
        linewidth=2,
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlim(1, min(32, len(cross_spectrum)))
    axes[1, 0].set_title("Cross-neuron operator U @ D")
    axes[1, 0].set_xlabel("singular-value rank")
    axes[1, 0].set_ylabel("singular value")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.2)

    mix_limit = float(np.max(np.abs(tau_mix)))
    image = axes[1, 1].imshow(
        tau_mix,
        cmap="coolwarm",
        vmin=-mix_limit,
        vmax=mix_limit,
    )
    tau_labels = [f"{tau:g}" for tau in taus]
    axes[1, 1].set_xticks(range(len(taus)))
    axes[1, 1].set_yticks(range(len(taus)))
    axes[1, 1].set_xticklabels(tau_labels, rotation=45)
    axes[1, 1].set_yticklabels(tau_labels)
    axes[1, 1].set_title("Used tau mixing weights")
    axes[1, 1].set_xlabel("source tau")
    axes[1, 1].set_ylabel("target tau")
    fig.colorbar(image, ax=axes[1, 1], fraction=0.046)

    incoming = np.abs(tau_mix).sum(axis=1)
    outgoing = np.abs(tau_mix).sum(axis=0)
    positions = np.arange(len(taus))
    width = 0.38
    axes[1, 2].bar(positions - width / 2, incoming, width, label="incoming")
    axes[1, 2].bar(positions + width / 2, outgoing, width, label="outgoing")
    axes[1, 2].set_xticks(positions)
    axes[1, 2].set_xticklabels(tau_labels, rotation=45)
    axes[1, 2].set_title("Cross-tau mixing strength")
    axes[1, 2].set_xlabel("tau")
    axes[1, 2].set_ylabel("sum of absolute mixing weights")
    axes[1, 2].legend()
    axes[1, 2].grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_full_spectra(
    raw: dict[str, object],
    scaled: dict[str, object],
    raw_null: list[dict[str, object]],
    scaled_null: list[dict[str, object]],
    output_path: Path,
) -> None:
    raw_spectrum = np.asarray(raw["singular_values"])
    scaled_spectrum = np.asarray(scaled["singular_values"])
    raw_null_spectrum = np.stack(
        [np.asarray(result["singular_values"]) for result in raw_null]
    ).mean(axis=0)
    scaled_null_spectrum = np.stack(
        [np.asarray(result["singular_values"]) for result in scaled_null]
    ).mean(axis=0)
    rank = np.arange(1, len(raw_spectrum) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    axes[0].plot(rank, raw_spectrum, label="trained", linewidth=2)
    axes[0].plot(rank, raw_null_spectrum, label="structured null", linewidth=2)
    axes[0].set_title("Complete 8-tau recurrent-current operator")
    axes[0].set_ylabel("singular value")
    axes[0].set_yscale("log")
    axes[0].legend()

    axes[1].plot(rank, scaled_spectrum, label="trained", linewidth=2)
    axes[1].plot(rank, scaled_null_spectrum, label="structured null", linewidth=2)
    axes[1].set_title("Operator after target-tau 1/tau scaling")
    axes[1].set_yscale("log")
    axes[1].legend()
    for axis in axes:
        axis.set_xlabel("singular-value rank")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def compact_spectrum_metrics(metrics: dict[str, object]) -> dict[str, float]:
    return {
        key: float(metrics[key])
        for key in [
            "spectral_norm",
            "stable_rank",
            "effective_rank",
            "spectral_entropy",
            "top1_energy",
            "top5_energy",
            "top16_energy",
            "top64_energy",
        ]
    }


def write_report(
    checkpoint_path: Path,
    checkpoint: dict[str, object],
    shared_metrics: dict[str, object],
    shared_controls: dict[str, dict[str, float]],
    cross_metrics: dict[str, object],
    cross_null: list[dict[str, object]],
    tau_metrics: dict[str, object],
    tau_null: list[dict[str, object]],
    raw_metrics: dict[str, object],
    raw_null: list[dict[str, object]],
    scaled_metrics: dict[str, object],
    scaled_null: list[dict[str, object]],
    tau_mix: np.ndarray,
    taus: np.ndarray,
    cross_strength_ratio: float,
    output_path: Path,
) -> dict[str, object]:
    args = checkpoint["args"]
    cross_null_summary = {
        name: scalar_summary(cross_null, name)
        for name in ["stable_rank", "effective_rank", "top1_energy"]
    }
    tau_null_summary = {
        name: scalar_summary(tau_null, name)
        for name in ["stable_rank", "effective_rank", "top1_energy"]
    }
    raw_null_summary = {
        name: scalar_summary(raw_null, name)
        for name in ["stable_rank", "effective_rank", "top1_energy", "top5_energy"]
    }
    scaled_null_summary = {
        name: scalar_summary(scaled_null, name)
        for name in ["stable_rank", "effective_rank", "top1_energy", "top5_energy"]
    }
    incoming = np.abs(tau_mix).sum(axis=1)
    outgoing = np.abs(tau_mix).sum(axis=0)
    strongest_target = int(np.argmax(incoming))
    strongest_source = int(np.argmax(outgoing))

    lines = [
        "# best.pt Recurrent Structure",
        "",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Epoch: **{checkpoint.get('epoch')}**",
        f"- Best validation accuracy: **{float(checkpoint.get('best_val_acc')):.2%}**",
        f"- Architecture: `{args['encoder']} -> {args['backbone']} -> {args['decoder']}`",
        f"- Reservoirs: **{len(taus)} taus x {args['reservoir_dim']} neurons = "
        f"{len(taus) * int(args['reservoir_dim'])} tau-neuron units**",
        "",
        "## Shared Recurrent Matrix",
        "",
        f"The learned `256 x 256` shared matrix has stable rank "
        f"**{shared_metrics['stable_rank']:.2f}** and effective rank "
        f"**{shared_metrics['effective_rank']:.2f}**. Its leading singular mode holds "
        f"**{shared_metrics['top1_energy']:.1%}** of all shared recurrent energy.",
        "",
        f"Entry-shuffled controls average stable rank "
        f"**{shared_controls['stable_rank']['mean']:.2f}**, effective rank "
        f"**{shared_controls['effective_rank']['mean']:.2f}**, and leading-mode energy "
        f"**{shared_controls['top1_energy']['mean']:.1%}**.",
        "",
        f"The trained matrix is strongly non-normal: **{shared_metrics['non_normality']:.3f}** "
        f"versus **{shared_controls['non_normality']['mean']:.3f}** after shuffling. "
        f"Its spectral radius is **{shared_metrics['spectral_radius']:.2f}**, versus "
        f"**{shared_controls['spectral_radius']['mean']:.2f}**.",
        "",
        f"Strong-edge reciprocity is **{shared_metrics['top_edge_reciprocity']:.3f}**, "
        f"close to the shuffled expectation "
        f"**{shared_controls['top_edge_reciprocity']['mean']:.3f}**.",
        "",
        "## Cross-Reservoir Path",
        "",
        "The cross-neuron map is architecturally limited to rank 16, so rank <=16 alone "
        "is not a learned finding. However, training collapses nearly all permitted "
        "capacity into one mode:",
        "",
        f"- stable rank: **{cross_metrics['stable_rank']:.2f}**",
        f"- effective rank: **{cross_metrics['effective_rank']:.2f}**",
        f"- leading-mode energy: **{cross_metrics['top1_energy']:.1%}**",
        f"- factor-shuffled leading-mode energy: "
        f"**{cross_null_summary['top1_energy']['mean']:.1%}**",
        "",
        f"After the configured cross gain, this branch has Frobenius strength equal to "
        f"**{cross_strength_ratio:.1%}** of the repeated within-tau shared path.",
        "",
        f"The strongest receiving tau is **{taus[strongest_target]:g}** with absolute "
        f"incoming mixing strength **{incoming[strongest_target]:.3f}**. The strongest "
        f"source tau is **{taus[strongest_source]:g}** with outgoing strength "
        f"**{outgoing[strongest_source]:.3f}**.",
        "",
        f"The used tau-mixing matrix has effective rank "
        f"**{tau_metrics['effective_rank']:.2f}**, versus "
        f"**{tau_null_summary['effective_rank']['mean']:.2f}** after shuffling its "
        "off-diagonal entries.",
        "",
        "## Complete Recurrent-Current Operator",
        "",
        "This combines the shared within-tau matrix and every cross-tau block into one "
        "`2048 x 2048` linear map from previous spikes to recurrent current.",
        "",
        f"- raw stable rank: **{raw_metrics['stable_rank']:.2f}** "
        f"(structured null **{raw_null_summary['stable_rank']['mean']:.2f}**)",
        f"- raw effective rank: **{raw_metrics['effective_rank']:.2f}** "
        f"(structured null **{raw_null_summary['effective_rank']['mean']:.2f}**)",
        f"- raw leading-mode energy: **{raw_metrics['top1_energy']:.1%}** "
        f"(structured null **{raw_null_summary['top1_energy']['mean']:.1%}**)",
        "",
        "After applying each target reservoir's `1/tau` update scaling:",
        "",
        f"- stable rank: **{scaled_metrics['stable_rank']:.2f}** "
        f"(structured null **{scaled_null_summary['stable_rank']['mean']:.2f}**)",
        f"- effective rank: **{scaled_metrics['effective_rank']:.2f}** "
        f"(structured null **{scaled_null_summary['effective_rank']['mean']:.2f}**)",
        f"- leading-mode energy: **{scaled_metrics['top1_energy']:.1%}** "
        f"(structured null **{scaled_null_summary['top1_energy']['mean']:.1%}**)",
        "",
        "## Interpretation",
        "",
        "The checkpoint has not learned a broadly distributed recurrent transform. It "
        "has organized both the shared topology and the cross-tau path around a few "
        "collective neuron directions. The cross path is especially close to rank one.",
        "",
        "This does not mean only one neuron is active. A singular mode is a distributed "
        "combination of many neurons. It also does not establish dynamical instability: "
        "LIF leakage, thresholding, reset, detached spikes, and input drive all modify "
        "the nonlinear state evolution.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "checkpoint": str(checkpoint_path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "best_val_accuracy": float(checkpoint.get("best_val_acc", float("nan"))),
        "shared": {
            **compact_spectrum_metrics(shared_metrics),
            "non_normality": float(shared_metrics["non_normality"]),
            "spectral_radius": float(shared_metrics["spectral_radius"]),
            "reciprocity": float(shared_metrics["reciprocity"]),
            "top_edge_reciprocity": float(shared_metrics["top_edge_reciprocity"]),
            "shuffle": shared_controls,
        },
        "cross_neuron": {
            **compact_spectrum_metrics(cross_metrics),
            "factor_shuffle": cross_null_summary,
            "strength_ratio_to_shared": cross_strength_ratio,
        },
        "tau_mix": {
            **compact_spectrum_metrics(tau_metrics),
            "shuffle": tau_null_summary,
            "incoming_absolute_strength": incoming.tolist(),
            "outgoing_absolute_strength": outgoing.tolist(),
        },
        "full_operator": {
            "raw": compact_spectrum_metrics(raw_metrics),
            "raw_structured_null": raw_null_summary,
            "tau_scaled": compact_spectrum_metrics(scaled_metrics),
            "tau_scaled_structured_null": scaled_null_summary,
        },
    }


def main() -> None:
    args = parse_args()
    if args.shuffle_repeats <= 0 or args.full_null_repeats <= 0:
        raise ValueError("Null repeat counts must be positive.")
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = checkpoint["args"]
    state = checkpoint["model_state"]
    required = [
        "backbone.shared_recurrent.weight",
        "backbone.cross_reservoir.down",
        "backbone.cross_reservoir.tau_mix",
        "backbone.cross_reservoir.up",
    ]
    missing = [name for name in required if name not in state]
    if missing:
        raise KeyError(f"Checkpoint does not contain cross-reservoir keys: {missing}")

    rng = np.random.default_rng(args.seed)
    shared = state["backbone.shared_recurrent.weight"].double().numpy()
    down = state["backbone.cross_reservoir.down"].double().numpy()
    up = state["backbone.cross_reservoir.up"].double().numpy()
    tau_mix = state["backbone.cross_reservoir.tau_mix"].double().numpy()
    tau_mix *= 1.0 - np.eye(tau_mix.shape[0])
    cross = up @ down
    taus = np.asarray([float(value) for value in str(saved_args["taus"]).split(",")])
    gain = float(saved_args["cross_gain"])

    print("Analyzing shared recurrent matrix...", flush=True)
    shared_metrics = matrix_metrics(shared, args.top_edge_fraction)
    shared_energy = np.asarray(shared_metrics["singular_values"]) ** 2
    shared_probability = shared_energy / shared_energy.sum()
    shared_metrics["top16_energy"] = float(shared_probability[:16].sum())
    shared_metrics["top64_energy"] = float(shared_probability[:64].sum())
    shared_controls, shared_null_spectrum = shuffled_weight_controls(
        shared,
        args.top_edge_fraction,
        args.shuffle_repeats,
        rng,
    )
    shared_null_spectrum *= np.linalg.norm(shared, ord="fro")
    shared_spectrum = np.asarray(shared_metrics["singular_values"])

    print("Analyzing cross-neuron and tau-mixing factors...", flush=True)
    cross_metrics = spectrum_metrics(cross)
    cross_null = shuffle_factor_control(up, down, args.shuffle_repeats, rng)
    cross_null_spectrum = np.stack(
        [np.asarray(result["singular_values"]) for result in cross_null]
    ).mean(axis=0)
    tau_metrics = spectrum_metrics(tau_mix)
    tau_null = shuffle_tau_mix(tau_mix, args.shuffle_repeats, rng)

    identity_tau = np.eye(len(taus))
    operator = np.kron(identity_tau, shared) + gain * np.kron(tau_mix, cross)
    tau_scale = np.kron(np.diag(1.0 / taus), np.eye(shared.shape[0]))
    print("Analyzing complete 2048 x 2048 operators...", flush=True)
    raw_metrics = spectrum_metrics(operator)
    scaled_metrics = spectrum_metrics(tau_scale @ operator)
    raw_null, scaled_null = architecture_null(
        shared,
        up,
        down,
        tau_mix,
        taus,
        gain,
        args.full_null_repeats,
        rng,
    )

    shared_full_strength = np.sqrt(len(taus)) * np.linalg.norm(shared, ord="fro")
    cross_full_strength = gain * np.linalg.norm(tau_mix, ord="fro") * np.linalg.norm(
        cross,
        ord="fro",
    )
    cross_strength_ratio = float(cross_full_strength / shared_full_strength)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dashboard(
        shared,
        shared_spectrum,
        shared_null_spectrum,
        np.asarray(cross_metrics["singular_values"]),
        cross_null_spectrum,
        tau_mix,
        taus,
        output_dir / "recurrent_structure_dashboard.png",
    )
    save_full_spectra(
        raw_metrics,
        scaled_metrics,
        raw_null,
        scaled_null,
        output_dir / "full_operator_spectra.png",
    )
    metrics = write_report(
        checkpoint_path,
        checkpoint,
        shared_metrics,
        shared_controls,
        cross_metrics,
        cross_null,
        tau_metrics,
        tau_null,
        raw_metrics,
        raw_null,
        scaled_metrics,
        scaled_null,
        tau_mix,
        taus,
        cross_strength_ratio,
        output_dir / "REPORT.md",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "spectra.npz",
        shared=np.asarray(shared_metrics["singular_values"]),
        shared_shuffle_mean=shared_null_spectrum,
        cross=np.asarray(cross_metrics["singular_values"]),
        cross_factor_shuffle_mean=cross_null_spectrum,
        tau_mix=np.asarray(tau_metrics["singular_values"]),
        full_raw=np.asarray(raw_metrics["singular_values"]),
        full_tau_scaled=np.asarray(scaled_metrics["singular_values"]),
        full_raw_null=np.stack(
            [np.asarray(result["singular_values"]) for result in raw_null]
        ),
        full_tau_scaled_null=np.stack(
            [np.asarray(result["singular_values"]) for result in scaled_null]
        ),
    )
    print(f"Saved analysis to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
