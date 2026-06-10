#!/usr/bin/env python3
"""Add statistical and class-level analysis to saved Tiny ImageNet predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_rel


MODEL_ORDER = ("702K", "2048-unit", "ensemble")
MODEL_COLORS = {
    "702K": "#0072B2",
    "2048-unit": "#D55E00",
    "ensemble": "#009E73",
}
TOP_K = (1, 5, 10)
Z_95 = 1.959963984540054


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def wilson_interval(correct: int, total: int) -> tuple[float, float]:
    proportion = correct / total
    denominator = 1.0 + Z_95**2 / total
    center = (proportion + Z_95**2 / (2.0 * total)) / denominator
    radius = (
        Z_95
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + Z_95**2 / (4.0 * total**2)
        )
        / denominator
    )
    return center - radius, center + radius


def mean_interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    return mean, mean - Z_95 * se, mean + Z_95 * se


def correctness(probabilities: np.ndarray, targets: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return probabilities.argmax(axis=1) == targets
    top = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    return (top == targets[:, None]).any(axis=1)


def load_split(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    saved = np.load(path / "logits_and_targets.npz")
    targets = saved["targets"]
    probabilities = {
        "702K": softmax(saved["small_logits"]),
        "2048-unit": softmax(saved["large_logits"]),
        "ensemble": saved["ensemble_probabilities"].astype(np.float64),
    }
    return targets, probabilities


def format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def build_statistics(
    root: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    split_data = {}
    for split in ("validation", "test"):
        split_data[split] = load_split(root / split)

    combined_targets = np.concatenate([split_data[s][0] for s in ("validation", "test")])
    combined_probabilities = {
        model: np.concatenate(
            [split_data[s][1][model] for s in ("validation", "test")], axis=0
        )
        for model in MODEL_ORDER
    }
    split_data["combined"] = (combined_targets, combined_probabilities)

    accuracy_rows = []
    paired_rows = []
    loss_rows = []
    comparisons = (
        ("702K", "2048-unit"),
        ("ensemble", "702K"),
        ("ensemble", "2048-unit"),
    )

    for split, (targets, probabilities) in split_data.items():
        correct_by_model = {
            model: {
                k: correctness(probabilities[model], targets, k) for k in TOP_K
            }
            for model in MODEL_ORDER
        }
        row_indices = np.arange(len(targets))
        losses = {}

        for model in MODEL_ORDER:
            model_probabilities = probabilities[model]
            losses[model] = -np.log(
                np.clip(model_probabilities[row_indices, targets], 1e-15, 1.0)
            )
            loss, loss_low, loss_high = mean_interval(losses[model])
            loss_rows.append(
                {
                    "split": split,
                    "model": model,
                    "num_samples": len(targets),
                    "nll": loss,
                    "ci95_low": loss_low,
                    "ci95_high": loss_high,
                }
            )

            for k in TOP_K:
                correct = int(correct_by_model[model][k].sum())
                low, high = wilson_interval(correct, len(targets))
                accuracy_rows.append(
                    {
                        "split": split,
                        "model": model,
                        "k": k,
                        "correct": correct,
                        "num_samples": len(targets),
                        "accuracy": correct / len(targets),
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )

        for model_a, model_b in comparisons:
            for k in TOP_K:
                a = correct_by_model[model_a][k]
                b = correct_by_model[model_b][k]
                a_only = int(np.count_nonzero(a & ~b))
                b_only = int(np.count_nonzero(~a & b))
                discordant = a_only + b_only
                paired_delta = a.astype(np.int8) - b.astype(np.int8)
                difference, diff_low, diff_high = mean_interval(paired_delta)
                p_value = (
                    float(
                        binomtest(
                            min(a_only, b_only),
                            discordant,
                            p=0.5,
                            alternative="two-sided",
                        ).pvalue
                    )
                    if discordant
                    else 1.0
                )
                paired_rows.append(
                    {
                        "split": split,
                        "model_a": model_a,
                        "model_b": model_b,
                        "k": k,
                        "a_only_correct": a_only,
                        "b_only_correct": b_only,
                        "discordant": discordant,
                        "accuracy_difference": difference,
                        "ci95_low": diff_low,
                        "ci95_high": diff_high,
                        "mcnemar_exact_p": p_value,
                    }
                )

        for model_a, model_b in comparisons:
            difference = losses[model_a] - losses[model_b]
            delta, delta_low, delta_high = mean_interval(difference)
            loss_rows.append(
                {
                    "split": split,
                    "model": f"{model_a} minus {model_b}",
                    "num_samples": len(targets),
                    "nll": delta,
                    "ci95_low": delta_low,
                    "ci95_high": delta_high,
                    "paired_t_p": float(ttest_rel(losses[model_a], losses[model_b]).pvalue),
                }
            )

    class_frames = {}
    for model, slug in (
        ("702K", "702k"),
        ("2048-unit", "2048unit"),
        ("ensemble", "ensemble"),
    ):
        frame = pd.read_csv(root / "combined" / f"{slug}_per_class.csv")
        class_frames[model] = frame

    base_columns = ["label", "synset", "name", "support"]
    per_class = class_frames["702K"][base_columns].copy()
    metric_columns = (
        "top1_accuracy",
        "top5_accuracy",
        "top10_accuracy",
        "mean_loss",
        "mean_true_rank",
        "precision",
        "recall",
        "f1",
    )
    for model in MODEL_ORDER:
        slug = model.lower().replace("-", "").replace("unit", "unit")
        if model == "2048-unit":
            slug = "2048unit"
        for column in metric_columns:
            per_class[f"{slug}_{column}"] = class_frames[model][column]

    for metric in ("top1_accuracy", "top5_accuracy", "top10_accuracy", "mean_loss"):
        per_class[f"small_minus_large_{metric}"] = (
            per_class[f"702k_{metric}"] - per_class[f"2048unit_{metric}"]
        )
        per_class[f"ensemble_minus_small_{metric}"] = (
            per_class[f"ensemble_{metric}"] - per_class[f"702k_{metric}"]
        )

    selective_rows = []
    targets, probabilities = split_data["combined"]
    coverages = (1.0, 0.9, 0.75, 0.5, 0.25, 0.1)
    for model in MODEL_ORDER:
        confidence = probabilities[model].max(axis=1)
        correct = probabilities[model].argmax(axis=1) == targets
        order = np.argsort(-confidence)
        for coverage in coverages:
            kept = max(1, int(round(coverage * len(targets))))
            selected = order[:kept]
            selective_rows.append(
                {
                    "model": model,
                    "coverage": coverage,
                    "num_samples": kept,
                    "accuracy": float(correct[selected].mean()),
                    "risk": float(1.0 - correct[selected].mean()),
                    "minimum_confidence": float(confidence[selected].min()),
                    "mean_confidence": float(confidence[selected].mean()),
                }
            )

    return (
        pd.DataFrame(accuracy_rows),
        pd.DataFrame(paired_rows),
        pd.DataFrame(loss_rows),
        per_class,
        pd.DataFrame(selective_rows),
    )


def make_summary_figure(
    root: Path,
    accuracy: pd.DataFrame,
    per_class: pd.DataFrame,
    selective: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    ax = axes[0, 0]
    combined = accuracy[accuracy["split"] == "combined"]
    x = np.arange(len(TOP_K))
    width = 0.24
    for offset, model in enumerate(MODEL_ORDER):
        rows = combined[combined["model"] == model].set_index("k").loc[list(TOP_K)]
        values = rows["accuracy"].to_numpy()
        low = values - rows["ci95_low"].to_numpy()
        high = rows["ci95_high"].to_numpy() - values
        ax.bar(
            x + (offset - 1) * width,
            100 * values,
            width,
            color=MODEL_COLORS[model],
            label=model,
            yerr=100 * np.vstack([low, high]),
            capsize=3,
        )
    ax.set_xticks(x, [f"Top-{k}" for k in TOP_K])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Combined held-out accuracy with 95% Wilson intervals")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    paired = json.loads((root / "combined" / "metrics.json").read_text())["paired"]
    outcomes = {
        "Both correct": paired["both_correct"],
        "702K only": paired["small_only_correct"],
        "2048 only": paired["large_only_correct"],
        "Neither": paired["neither_correct"],
    }
    colors = ["#999999", MODEL_COLORS["702K"], MODEL_COLORS["2048-unit"], "#444444"]
    bars = ax.bar(outcomes.keys(), [100 * value for value in outcomes.values()], color=colors)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_ylabel("Share of images (%)")
    ax.set_title("Top-1 complementarity")
    ax.set_ylim(0, max(55, 100 * max(outcomes.values()) + 7))
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    x_values = 100 * per_class["702k_top1_accuracy"]
    y_values = 100 * per_class["2048unit_top1_accuracy"]
    ax.scatter(x_values, y_values, alpha=0.65, s=28, color="#6A3D9A")
    limits = [0, 100]
    ax.plot(limits, limits, linestyle="--", color="black", linewidth=1)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("702K per-class top-1 (%)")
    ax.set_ylabel("2048-unit per-class top-1 (%)")
    ax.set_title("Each point is one of 200 classes")
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    for model in MODEL_ORDER:
        rows = selective[selective["model"] == model].sort_values("coverage")
        ax.plot(
            100 * rows["coverage"],
            100 * rows["accuracy"],
            marker="o",
            color=MODEL_COLORS[model],
            label=model,
        )
    ax.set_xlabel("Coverage: most-confident images retained (%)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Selective accuracy")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.suptitle("Tiny ImageNet checkpoint comparison", fontsize=16)
    fig.tight_layout()
    fig.savefig(root / "statistical_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_detailed_report(
    root: Path,
    accuracy: pd.DataFrame,
    paired: pd.DataFrame,
    losses: pd.DataFrame,
    per_class: pd.DataFrame,
    selective: pd.DataFrame,
) -> None:
    metrics = json.loads((root / "combined" / "metrics.json").read_text())
    metadata = json.loads((root / "metadata.json").read_text())

    combined_accuracy = accuracy[accuracy["split"] == "combined"]
    combined_losses = losses[
        (losses["split"] == "combined") & losses["model"].isin(MODEL_ORDER)
    ]
    top1_paired = paired[
        (paired["split"] == "combined")
        & (paired["k"] == 1)
        & (
            ((paired["model_a"] == "702K") & (paired["model_b"] == "2048-unit"))
            | ((paired["model_a"] == "ensemble") & (paired["model_b"] == "702K"))
        )
    ]

    def accuracy_cell(model: str, k: int) -> str:
        row = combined_accuracy[
            (combined_accuracy["model"] == model) & (combined_accuracy["k"] == k)
        ].iloc[0]
        return (
            f"{format_percent(row['accuracy'])} "
            f"[{format_percent(row['ci95_low'])}, {format_percent(row['ci95_high'])}]"
        )

    def class_table(frame: pd.DataFrame, value_column: str, ascending: bool) -> str:
        rows = frame.sort_values(value_column, ascending=ascending).head(10)
        lines = [
            "| Class | 702K | 2048-unit | Ensemble | Difference |",
            "|---|---:|---:|---:|---:|",
        ]
        for _, row in rows.iterrows():
            lines.append(
                f"| {row['name']} | {100 * row['702k_top1_accuracy']:.1f}% | "
                f"{100 * row['2048unit_top1_accuracy']:.1f}% | "
                f"{100 * row['ensemble_top1_accuracy']:.1f}% | "
                f"{100 * row[value_column]:+.1f} pp |"
            )
        return "\n".join(lines)

    small_wins = class_table(
        per_class, "small_minus_large_top1_accuracy", ascending=False
    )
    large_wins = class_table(
        per_class, "small_minus_large_top1_accuracy", ascending=True
    )
    ensemble_wins = class_table(
        per_class, "ensemble_minus_small_top1_accuracy", ascending=False
    )

    selective_table = [
        "| Coverage | 702K | 2048-unit | Ensemble |",
        "|---:|---:|---:|---:|",
    ]
    for coverage in (1.0, 0.75, 0.5, 0.25, 0.1):
        values = {}
        for model in MODEL_ORDER:
            row = selective[
                (selective["model"] == model) & (selective["coverage"] == coverage)
            ].iloc[0]
            values[model] = row["accuracy"]
        selective_table.append(
            f"| {100 * coverage:.0f}% | {format_percent(values['702K'])} | "
            f"{format_percent(values['2048-unit'])} | "
            f"{format_percent(values['ensemble'])} |"
        )

    speed = {}
    for model in ("702K", "2048-unit"):
        total_images = sum(
            metadata["splits"][split]["num_samples"] for split in ("validation", "test")
        )
        total_seconds = sum(
            metadata["splits"][split]["runtimes"][model]["seconds"]
            for split in ("validation", "test")
        )
        speed[model] = total_images / total_seconds

    nll_rows = {}
    for model in MODEL_ORDER:
        nll_rows[model] = combined_losses[combined_losses["model"] == model].iloc[0]

    small_vs_large = top1_paired[
        (top1_paired["model_a"] == "702K")
        & (top1_paired["model_b"] == "2048-unit")
    ].iloc[0]
    ensemble_vs_small = top1_paired[
        (top1_paired["model_a"] == "ensemble")
        & (top1_paired["model_b"] == "702K")
    ].iloc[0]

    report = f"""# Detailed Tiny ImageNet Evaluation

This report uses all **9,832** labeled held-out images from the clean dataset's
validation and test splits. Both checkpoints were evaluated with deterministic
`Resize(73) -> CenterCrop(64) -> ImageNet normalization` preprocessing.

## Main Results

Accuracy cells show the estimate followed by a 95% Wilson confidence interval.

| Model | Parameters | Top-1 | Top-5 | Top-10 | NLL [95% CI] |
|---|---:|---:|---:|---:|---:|
| 702K | 701,640 | {accuracy_cell("702K", 1)} | {accuracy_cell("702K", 5)} | {accuracy_cell("702K", 10)} | {nll_rows["702K"]["nll"]:.4f} [{nll_rows["702K"]["ci95_low"]:.4f}, {nll_rows["702K"]["ci95_high"]:.4f}] |
| 2048-unit | 1,650,568 | {accuracy_cell("2048-unit", 1)} | {accuracy_cell("2048-unit", 5)} | {accuracy_cell("2048-unit", 10)} | {nll_rows["2048-unit"]["nll"]:.4f} [{nll_rows["2048-unit"]["ci95_low"]:.4f}, {nll_rows["2048-unit"]["ci95_high"]:.4f}] |
| Ensemble | 2,352,208 total | {accuracy_cell("ensemble", 1)} | {accuracy_cell("ensemble", 5)} | {accuracy_cell("ensemble", 10)} | {nll_rows["ensemble"]["nll"]:.4f} [{nll_rows["ensemble"]["ci95_low"]:.4f}, {nll_rows["ensemble"]["ci95_high"]:.4f}] |

The 702K model leads the 2048-unit model by
**{100 * small_vs_large["accuracy_difference"]:.2f} percentage points** in top-1
(95% paired CI {100 * small_vs_large["ci95_low"]:.2f} to
{100 * small_vs_large["ci95_high"]:.2f}, exact McNemar
`p={small_vs_large["mcnemar_exact_p"]:.3g}`). The simple untrained probability
ensemble adds **{100 * ensemble_vs_small["accuracy_difference"]:.2f} points**
over the 702K model (95% paired CI
{100 * ensemble_vs_small["ci95_low"]:.2f} to
{100 * ensemble_vs_small["ci95_high"]:.2f},
`p={ensemble_vs_small["mcnemar_exact_p"]:.3g}`).

## Complementarity

- Same top-1 prediction: **{format_percent(metrics["paired"]["prediction_agreement"])}**
- Correct only for 702K: **{format_percent(metrics["paired"]["small_only_correct"])}**
- Correct only for 2048-unit: **{format_percent(metrics["paired"]["large_only_correct"])}**
- Either model correct (oracle): **{format_percent(metrics["paired"]["oracle_top1"])}**
- Simple ensemble top-1: **{format_percent(metrics["ensemble"]["topk"]["1"])}**
- Remaining oracle-to-ensemble gap: **{100 * (metrics["paired"]["oracle_top1"] - metrics["ensemble"]["topk"]["1"]):.2f} points**

The larger model is weaker alone but still uniquely solves about 8.1% of all
images. That disagreement is why averaging the probability vectors improves
accuracy, NLL, Brier score, and calibration.

## Ranking And Calibration

| Model | Top-2 | Top-3 | Top-20 | Top-50 | Mean rank | Rank p90 | MRR | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 702K | {format_percent(metrics["702K"]["topk"]["2"])} | {format_percent(metrics["702K"]["topk"]["3"])} | {format_percent(metrics["702K"]["topk"]["20"])} | {format_percent(metrics["702K"]["topk"]["50"])} | {metrics["702K"]["mean_true_rank"]:.2f} | {metrics["702K"]["true_rank_p90"]:.0f} | {metrics["702K"]["mean_reciprocal_rank"]:.3f} | {metrics["702K"]["calibration"]["ece_15"]:.3f} | {metrics["702K"]["brier_score"]:.3f} |
| 2048-unit | {format_percent(metrics["2048-unit"]["topk"]["2"])} | {format_percent(metrics["2048-unit"]["topk"]["3"])} | {format_percent(metrics["2048-unit"]["topk"]["20"])} | {format_percent(metrics["2048-unit"]["topk"]["50"])} | {metrics["2048-unit"]["mean_true_rank"]:.2f} | {metrics["2048-unit"]["true_rank_p90"]:.0f} | {metrics["2048-unit"]["mean_reciprocal_rank"]:.3f} | {metrics["2048-unit"]["calibration"]["ece_15"]:.3f} | {metrics["2048-unit"]["brier_score"]:.3f} |
| Ensemble | {format_percent(metrics["ensemble"]["topk"]["2"])} | {format_percent(metrics["ensemble"]["topk"]["3"])} | {format_percent(metrics["ensemble"]["topk"]["20"])} | {format_percent(metrics["ensemble"]["topk"]["50"])} | {metrics["ensemble"]["mean_true_rank"]:.2f} | {metrics["ensemble"]["true_rank_p90"]:.0f} | {metrics["ensemble"]["mean_reciprocal_rank"]:.3f} | {metrics["ensemble"]["calibration"]["ece_15"]:.3f} | {metrics["ensemble"]["brier_score"]:.3f} |

## Confidence Filtering

Accuracy after retaining only the most confident predictions:

{chr(10).join(selective_table)}

## Classes Favoring The 702K Model

{small_wins}

## Classes Favoring The 2048-Unit Model

{large_wins}

## Largest Ensemble Gains Over 702K

{ensemble_wins}

## Runtime On This CPU

- 702K: **{speed["702K"]:.1f} images/s**
- 2048-unit: **{speed["2048-unit"]:.1f} images/s**
- The 702K checkpoint was **{speed["702K"] / speed["2048-unit"]:.1f}x faster** in this run.

GPU throughput was not measured because CUDA was unavailable.

## Files

- `accuracy_confidence_intervals.csv`: top-1/5/10 counts and Wilson intervals
- `paired_significance.csv`: paired accuracy differences and exact McNemar tests
- `loss_confidence_intervals.csv`: NLL intervals and paired loss tests
- `per_class_differences.csv`: all per-class model metrics and deltas
- `combined_selective_accuracy.csv`: confidence/coverage tradeoffs
- `statistical_summary.png`: compact visual comparison
- split directories: logits, every prediction, errors, confusion matrices,
  calibration bins, selective accuracy, and per-class metrics
"""
    (root / "DETAILED_REPORT.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("visualizations/tiny_imagenet_official_evaluation"),
    )
    args = parser.parse_args()
    root = args.root.resolve()

    accuracy, paired, losses, per_class, selective = build_statistics(root)
    accuracy.to_csv(root / "accuracy_confidence_intervals.csv", index=False)
    paired.to_csv(root / "paired_significance.csv", index=False)
    losses.to_csv(root / "loss_confidence_intervals.csv", index=False)
    per_class.to_csv(root / "per_class_differences.csv", index=False)
    selective.to_csv(root / "combined_selective_accuracy.csv", index=False)

    summary = {
        "accuracy_confidence_intervals": json.loads(
            accuracy.to_json(orient="records")
        ),
        "paired_significance": json.loads(paired.to_json(orient="records")),
        "loss_confidence_intervals": json.loads(losses.to_json(orient="records")),
        "selective_accuracy": json.loads(selective.to_json(orient="records")),
    }
    (root / "statistical_analysis.json").write_text(json.dumps(summary, indent=2))
    make_summary_figure(root, accuracy, per_class, selective)
    write_detailed_report(root, accuracy, paired, losses, per_class, selective)


if __name__ == "__main__":
    main()
