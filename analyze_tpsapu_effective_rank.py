"""Effective rank analysis of both TPSAPU checkpoints.

Computes effective rank on:
  1. The shared recurrent weight matrix W_rec (weight-space analysis)
  2. The reservoir membrane activations across 500 validation samples
     (activation-space analysis — analogous to what visualize_resnet18.py does
     for ResNet-18 layer activations)

For the ResCNN-2048 model the weight-level analysis already lives in
  visualizations/best_checkpoint_structure/
This script focuses on the activation-space analysis and produces unified
comparison figures for both models.

Outputs → visualizations/tpsapu_effective_rank/
  wrec_effective_rank.png        — W_rec spectra + eff-rank both models
  activation_effective_rank.png  — eff-rank of membrane states per tau
  singular_value_spectra.png     — normalised SV decay per tau, both models
  activation_statistics.png      — mean-abs, std, sparsity of membranes per tau
  comparison_with_resnet.png     — eff-rank side-by-side: ResNet layers vs TPSAPU taus
  summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import svdvals
from torch.utils.data import DataLoader
from torchvision import transforms

import evaluate_tiny_imagenet_validation as eval_script
import train_pipeline

# ---------------------------------------------------------------------------
MODELS = {
    "CNN3+TPSAPU 702k": {
        "ckpt": "checkpoints/best_702k_cnn3_membrane_transformer.pt",
        "color": "#4c9be8",
        "reservoir_dim": 64,
        "num_taus": 8,
        "taus": [1.1, 2, 4, 8, 16, 32, 64, 128],
    },
    "ResCNN+TPSAPU 2048": {
        "ckpt": "checkpoints/res_cnn_cross_both_transformer/best.pt",
        "color": "#6bbf59",
        "reservoir_dim": 256,
        "num_taus": 8,
        "taus": [1.1, 2, 4, 8, 16, 32, 64, 128],
    },
}
RESNET_SUMMARY = "visualizations/resnet18/summary.json"
OUT = Path("visualizations/tpsapu_effective_rank")
OUT.mkdir(parents=True, exist_ok=True)

TINY_IMAGENET_MEAN = (0.485, 0.456, 0.406)
TINY_IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Spectrum helpers (same definition as everywhere else in this project)
# ---------------------------------------------------------------------------


def spectrum_metrics(mat: np.ndarray) -> dict:
    sv = svdvals(mat, overwrite_a=False, check_finite=False)
    e = sv**2
    p = e / max(float(e.sum()), 1e-12)
    H = -float(np.sum(p * np.log(np.maximum(p, 1e-300))))
    return {
        "dimension": int(min(mat.shape)),
        "spectral_norm": float(sv[0]),
        "stable_rank": float(e.sum() / max(float(e[0]), 1e-12)),
        "effective_rank": float(np.exp(H)),
        "spectral_entropy": float(H / np.log(max(len(sv), 2))),
        "top1_energy": float(p[0]),
        "singular_values": sv.tolist(),
    }


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------


def load_dataset():
    hf = train_pipeline.load_hf_split(
        "slegroux/tiny-imagenet-200-clean",
        split="validation",
        cache_dir="data/tiny-imagenet-200-clean",
        no_download=True,
    )
    tfm = transforms.Compose(
        [
            transforms.Resize(int(round(64 * 256 / 224))),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
        ]
    )

    class _DS(torch.utils.data.Dataset):
        def __init__(self, d):
            self.d = d

        def __len__(self):
            return len(self.d)

        def __getitem__(self, i):
            item = self.d[i]
            img = item["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            return tfm(img), item["label"]

    return _DS(hf)


# ---------------------------------------------------------------------------
# Collect membrane activations
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_membranes(model, dataset, device, n_samples=500, batch_size=64):
    """Returns dict: tau_idx -> (N, reservoir_dim) membrane array."""
    subset = torch.utils.data.Subset(dataset, range(min(n_samples, len(dataset))))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()

    all_membranes = []  # list of (B, steps, num_taus*reservoir_dim)
    for images, _ in loader:
        images = images.to(device)
        tokens = model.encoder(images)
        states = model.backbone.forward_states(tokens, reset_state=True)
        # states["membrane"]: (B, steps, num_taus * reservoir_dim)
        # pool over steps → (B, num_taus * reservoir_dim)
        m = states["membrane"].mean(dim=1).detach().cpu().numpy()
        all_membranes.append(m)

    mat = np.concatenate(all_membranes, axis=0)[:n_samples]  # (N, T*d)
    return mat  # caller will split by tau


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_wrec_spectra(results: dict, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        "Shared Recurrent Weight Matrix W_rec — Spectral Analysis",
        fontsize=11,
        fontweight="bold",
    )

    for name, res in results.items():
        sv = np.array(res["wrec"]["singular_values"])
        sv_n = sv / (sv[0] + 1e-12)
        col = MODELS[name]["color"]
        d = len(sv)
        er = res["wrec"]["effective_rank"]
        sr = res["wrec"]["stable_rank"]

        axes[0].plot(
            range(1, d + 1), sv, color=col, linewidth=1.8, label=f"{name} (d={d})"
        )
        axes[1].plot(range(1, d + 1), sv_n, color=col, linewidth=1.8, label=f"{name}")
        axes[2].scatter(
            [MODELS[name]["reservoir_dim"]],
            [er],
            color=col,
            s=180,
            zorder=5,
            label=f"{name}\neff={er:.1f} / stable={sr:.2f}",
        )
        axes[2].annotate(
            name.split("+")[0],
            xy=(MODELS[name]["reservoir_dim"], er),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=7,
            color=col,
        )

    axes[0].set_title("Raw singular values")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("SV rank")
    axes[0].set_ylabel("σ")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Normalised σᵢ/σ₁")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("SV rank")
    axes[1].set_ylabel("σᵢ/σ₁")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Effective rank vs reservoir dim\n(vs null-shuffle ref)")
    axes[2].set_xlabel("reservoir_dim d")
    axes[2].set_ylabel("Effective rank")
    for name, res in results.items():
        col = MODELS[name]["color"]
        null_er = res["wrec_null"]["effective_rank"]
        axes[2].axhline(
            null_er,
            color=col,
            linestyle="--",
            alpha=0.5,
            linewidth=1,
            label=f"{name.split('+')[0]} shuffle null ({null_er:.0f})",
        )
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    p = out / "wrec_effective_rank.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")


def fig_activation_effective_rank(results: dict, out: Path):
    tau_labels = [str(t) for t in [1.1, 2, 4, 8, 16, 32, 64, 128]]
    n_taus = 8

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        "Reservoir Membrane Activations — Effective Rank per τ Reservoir\n"
        "(computed over 500 validation samples, mean-centred)",
        fontsize=10,
        fontweight="bold",
    )

    for name, res in results.items():
        col = MODELS[name]["color"]
        d = MODELS[name]["reservoir_dim"]
        er = [res["act_per_tau"][ti]["effective_rank"] for ti in range(n_taus)]
        sr = [res["act_per_tau"][ti]["stable_rank"] for ti in range(n_taus)]
        t1 = [res["act_per_tau"][ti]["top1_energy"] for ti in range(n_taus)]

        x = range(n_taus)
        axes[0].plot(
            x, er, color=col, linewidth=2, marker="o", label=f"{name} (dim={d})"
        )
        axes[1].plot(x, sr, color=col, linewidth=2, marker="o", label=name)
        axes[2].plot(x, t1, color=col, linewidth=2, marker="o", label=name)

    for ax in axes:
        ax.set_xticks(range(n_taus))
        ax.set_xticklabels(
            [f"τ={t}" for t in tau_labels], rotation=35, ha="right", fontsize=8
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Effective Rank (exp H)")
    axes[0].set_title("Effective Rank per τ")
    axes[1].set_ylabel("Stable Rank")
    axes[1].set_title("Stable Rank per τ")
    axes[2].set_ylabel("Frac. energy in top SV")
    axes[2].set_title("Leading SV Energy per τ")

    plt.tight_layout()
    p = out / "activation_effective_rank.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")


def fig_activation_statistics(results: dict, out: Path):
    tau_labels = [f"τ={t}" for t in [1.1, 2, 4, 8, 16, 32, 64, 128]]
    n_taus = 8
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.suptitle(
        "Membrane Activation Statistics per τ Reservoir", fontsize=11, fontweight="bold"
    )

    for row_i, (name, res) in enumerate(results.items()):
        col = MODELS[name]["color"]
        m = [res["act_stats_tau"][ti]["mean_abs"] for ti in range(n_taus)]
        s = [res["act_stats_tau"][ti]["std"] for ti in range(n_taus)]
        sp = [res["act_stats_tau"][ti]["sparsity"] * 100 for ti in range(n_taus)]
        mx = [res["act_stats_tau"][ti]["max_abs"] for ti in range(n_taus)]

        for col_i, (vals, ylabel, title) in enumerate(
            zip(
                [m, s, sp, mx],
                ["Mean |V|", "Std V", "Near-zero %", "Max |V|"],
                [
                    "Mean Abs Membrane",
                    "Membrane Std",
                    "Sparsity (|V|<1e-4)",
                    "Max Abs Membrane",
                ],
            )
        ):
            ax = axes[row_i, col_i]
            ax.bar(range(n_taus), vals, color=col, alpha=0.8, edgecolor="white")
            ax.set_xticks(range(n_taus))
            ax.set_xticklabels(tau_labels, rotation=40, ha="right", fontsize=7)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(f"{name.split('+')[0]}: {title}", fontsize=8)
            ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    p = out / "activation_statistics.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")


def fig_comparison_with_resnet(results: dict, out: Path):
    """Side-by-side effective rank: ResNet layers vs TPSAPU tau-reservoirs."""
    # Load ResNet summary
    resnet_summary = None
    try:
        with open(RESNET_SUMMARY) as f:
            resnet_summary = json.load(f)
    except FileNotFoundError:
        print(
            f"Warning: ResNet summary not found at {RESNET_SUMMARY}; skipping comparison."
        )
        return

    resnet_layers = ["stem", "layer1", "layer2", "layer3", "layer4", "avgpool"]
    rn_er = [resnet_summary["spectra"][l]["effective_rank"] for l in resnet_layers]
    rn_dim = [resnet_summary["spectra"][l]["dimension"] for l in resnet_layers]

    tau_labels = [f"τ={t}" for t in [1.1, 2, 4, 8, 16, 32, 64, 128]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Effective Rank Comparison: ResNet-18 Layers vs TPSAPU τ-Reservoirs\n"
        "(N=500 validation samples, mean-centred activations)",
        fontsize=10,
        fontweight="bold",
    )

    # Panel 1: ResNet effective rank per layer
    ax = axes[0]
    colors_rn = plt.cm.Oranges(np.linspace(0.4, 0.9, len(resnet_layers)))
    bars = ax.bar(resnet_layers, rn_er, color=colors_rn, edgecolor="white")
    for bar, v, d in zip(bars, rn_er, rn_dim):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 3,
            f"{v:.0f}\n/{d}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_ylabel("Effective Rank (exp H)")
    ax.set_title("ResNet-18: Eff. Rank per Layer\n(dim = min(N,features)=500)")
    ax.set_xticklabels(resnet_layers, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: TPSAPU effective rank per tau
    ax = axes[1]
    x = np.arange(8)
    width = 0.38
    for i, (name, res) in enumerate(results.items()):
        col = MODELS[name]["color"]
        d = MODELS[name]["reservoir_dim"]
        er = [res["act_per_tau"][ti]["effective_rank"] for ti in range(8)]
        off = (i - 0.5) * width
        bars2 = ax.bar(
            x + off,
            er,
            width,
            color=col,
            edgecolor="white",
            alpha=0.85,
            label=f"{name.split('+')[0]} (dim={d})",
        )
        for bar, v in zip(bars2, er):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.3,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(tau_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Effective Rank (exp H)")
    ax.set_title("TPSAPU: Eff. Rank per τ-Reservoir\n(dim = min(N, reservoir_dim))")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: Fractional effective rank (eff_rank / dim)
    ax = axes[2]
    rn_frac = [e / d for e, d in zip(rn_er, rn_dim)]
    ax.plot(
        range(len(resnet_layers)),
        rn_frac,
        color="#e07b39",
        linewidth=2,
        marker="s",
        label="ResNet-18",
        markersize=7,
    )
    ax.set_xticks(range(len(resnet_layers)))
    ax.set_xticklabels(resnet_layers, rotation=20, ha="right", fontsize=8)
    for name, res in results.items():
        col = MODELS[name]["color"]
        d = MODELS[name]["reservoir_dim"]
        er = [res["act_per_tau"][ti]["effective_rank"] for ti in range(8)]
        dims = [min(500, d)] * 8
        frac = [e / di for e, di in zip(er, dims)]
        ax2_x = np.linspace(0, len(resnet_layers) - 1, 8)
        ax.plot(
            ax2_x,
            frac,
            color=col,
            linewidth=2,
            marker="o",
            label=name.split("+")[0],
            markersize=6,
        )
    ax.set_ylabel("Effective Rank / Dimension")
    ax.set_title("Fractional Effective Rank\n(1.0 = fully uniform spectrum)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = out / "comparison_with_resnet.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")


def fig_singular_value_spectra(results: dict, out: Path):
    tau_labels = [f"τ={t}" for t in [1.1, 2, 4, 8, 16, 32, 64, 128]]
    n_taus = 8

    fig, axes = plt.subplots(2, n_taus, figsize=(2.8 * n_taus, 6), squeeze=False)
    fig.suptitle(
        "Normalised Singular Value Spectra of Membrane Activations per τ",
        fontsize=11,
        fontweight="bold",
    )

    for row_i, (name, res) in enumerate(results.items()):
        col = MODELS[name]["color"]
        for ti in range(n_taus):
            ax = axes[row_i, ti]
            sv = np.array(res["act_per_tau"][ti]["singular_values"])
            sv = sv / (sv[0] + 1e-12)
            ax.plot(range(1, len(sv) + 1), sv, color=col, linewidth=1.5)
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{tau_labels[ti]}", fontsize=8)
            if col_i := ti == 0:
                ax.set_ylabel(f"{name.split('+')[0]}\nσᵢ/σ₁", fontsize=7)
            ax.tick_params(labelsize=6)

    plt.tight_layout()
    p = out / "singular_value_spectra.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    train_pipeline.set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading dataset …")
    dataset = load_dataset()

    results = {}

    for name, cfg in MODELS.items():
        print(f"\n{'=' * 55}\n  {name}\n{'=' * 55}")
        ck = eval_script.load_checkpoint(cfg["ckpt"], device)
        arch = eval_script.architecture_from_checkpoint(ck)
        model = train_pipeline.build_model(arch).to(device)
        model.load_state_dict(ck["model_state"])
        model.eval()

        # --- W_rec analysis -----------------------------------------------
        wrec = model.backbone.shared_recurrent.weight.detach().cpu().numpy()
        # Apply neuron mask (pruning)
        mask = model.backbone.neuron_mask.detach().cpu().numpy()
        wrec_masked = wrec * mask[:, None] * mask[None, :]
        print(
            f"  W_rec shape: {wrec_masked.shape}, "
            f"sparsity: {(np.abs(wrec_masked) < 1e-10).mean():.1%}"
        )

        wrec_metrics = spectrum_metrics(wrec_masked)
        # Null: shuffle non-zero entries
        rng = np.random.default_rng(42)
        nz = wrec_masked.reshape(-1)
        null_vals = rng.permutation(nz).reshape(wrec_masked.shape)
        wrec_null = spectrum_metrics(null_vals)
        print(
            f"  W_rec eff rank: {wrec_metrics['effective_rank']:.2f} "
            f"(null: {wrec_null['effective_rank']:.2f})"
        )

        # --- Membrane activation analysis ----------------------------------
        print(f"  Collecting membrane activations …")
        mem_mat = collect_membranes(model, dataset, device, n_samples=500)
        # mem_mat: (N, num_taus * reservoir_dim)
        d = cfg["reservoir_dim"]
        T = cfg["num_taus"]
        assert mem_mat.shape[1] == T * d, f"shape mismatch {mem_mat.shape}"

        act_per_tau = {}
        act_stats_tau = {}
        for ti in range(T):
            chunk = mem_mat[:, ti * d : (ti + 1) * d]  # (N, d)
            chunk_c = chunk - chunk.mean(axis=0, keepdims=True)
            m = spectrum_metrics(chunk_c)
            act_per_tau[ti] = m
            act_stats_tau[ti] = {
                "mean_abs": float(np.abs(chunk).mean()),
                "std": float(chunk.std()),
                "sparsity": float((np.abs(chunk) < 1e-4).mean()),
                "max_abs": float(np.abs(chunk).max()),
            }
            er = m["effective_rank"]
            print(
                f"    τ={cfg['taus'][ti]:>5.1f}: eff_rank={er:.1f}/{m['dimension']}  "
                f"top1_e={m['top1_energy']:.1%}  mean_abs={act_stats_tau[ti]['mean_abs']:.3f}"
            )

        results[name] = {
            "wrec": wrec_metrics,
            "wrec_null": wrec_null,
            "act_per_tau": act_per_tau,
            "act_stats_tau": act_stats_tau,
        }

    # --- Figures -----------------------------------------------------------
    print("\nGenerating figures …")
    fig_wrec_spectra(results, OUT)
    fig_activation_effective_rank(results, OUT)
    fig_activation_statistics(results, OUT)
    fig_comparison_with_resnet(results, OUT)
    fig_singular_value_spectra(results, OUT)

    # --- Summary JSON ------------------------------------------------------
    summary = {}
    for name, res in results.items():
        summary[name] = {
            "wrec_effective_rank": res["wrec"]["effective_rank"],
            "wrec_stable_rank": res["wrec"]["stable_rank"],
            "wrec_null_eff_rank": res["wrec_null"]["effective_rank"],
            "wrec_sparsity": float(
                (np.abs(np.array(res["wrec"]["singular_values"])) < 1e-10).mean()
            ),
            "activation_per_tau": {
                str(MODELS[name]["taus"][ti]): {
                    "effective_rank": res["act_per_tau"][ti]["effective_rank"],
                    "stable_rank": res["act_per_tau"][ti]["stable_rank"],
                    "top1_energy": res["act_per_tau"][ti]["top1_energy"],
                    "mean_abs": res["act_stats_tau"][ti]["mean_abs"],
                    "sparsity": res["act_stats_tau"][ti]["sparsity"],
                }
                for ti in range(MODELS[name]["num_taus"])
            },
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary saved: {OUT / 'summary.json'}")
    print(f"All outputs: {OUT}")

    # Print table
    print("\n" + "=" * 70)
    for name, s in summary.items():
        print(f"\n{name}")
        print(
            f"  W_rec eff rank: {s['wrec_effective_rank']:.2f}  "
            f"(null: {s['wrec_null_eff_rank']:.2f})"
        )
        print(f"  Activation eff rank per tau:")
        for tau, v in s["activation_per_tau"].items():
            print(
                f"    τ={float(tau):>6.1f}:  eff={v['effective_rank']:>6.1f}  "
                f"stable={v['stable_rank']:>6.1f}  top1={v['top1_energy']:.1%}  "
                f"sparsity={v['sparsity']:.1%}"
            )
    print("=" * 70)


if __name__ == "__main__":
    main()
