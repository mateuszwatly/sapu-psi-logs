"""Thorough attention analysis for transformer decoders in TPSAPU models.

For each model the script:

  1. Extracts multi-head attention weights from every transformer layer by
     re-running self-attention with need_weights=True inside a hook.
  2. Maps CLS-token attention back to spatial image coordinates so we can see
     "which patch on the image did the decoder look at?".
  3. Produces per-sample figures (10 samples) and aggregate figures over a
     larger batch (~500 samples).

Spatial token layout
--------------------
CNN3-702k (membrane_transformer) on 64×64 images:
  Two stride-2 convs → 16×16 = 256 tokens (each token covers a 4×4 pixel block).
  Transformer sees: [CLS, t1, t2, …, t256] — each ti = one cell of the 16×16 grid.

ResCNN-2048 (both_transformer) on 64×64 images:
  Same strides → 256 layer2 tokens + 256 layer3 tokens = 512 tokens total.
  Transformer sees: [CLS, t1..t256 (layer2 16×16), t257..t512 (layer3 16×16)].

All figures go into:
  visualizations/attention_analysis/<model_name>/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import evaluate_tiny_imagenet_validation as eval_script
import train_pipeline

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Token count derivation for 64×64 Tiny ImageNet images:
#   CNN3:   two stride-2 convs → 64//2//2 = 16 → 16×16 = 256 tokens
#   ResCNN: same strides → 256 layer2 + 256 layer3 = 512 tokens total
MODELS = {
    "cnn3_702k": {
        "checkpoint": "checkpoints/best_702k_cnn3_membrane_transformer.pt",
        "decoder_type": "membrane_transformer",
        "spatial_tokens": 256,
        "spatial_grid": (16, 16),
        "token_groups": {"layer": (0, 256)},
        "group_labels": ["cnn3 16×16"],
        "num_taus": 8,
        "reservoir_dim": 64,
        "label": "CNN3+TPSAPU 702k",
    },
    "rescnn_2048": {
        "checkpoint": "checkpoints/res_cnn_cross_both_transformer/best.pt",
        "decoder_type": "both_transformer",
        "spatial_tokens": 512,
        "spatial_grid": (16, 16),
        "token_groups": {"layer2": (0, 256), "layer3": (256, 512)},
        "group_labels": ["res_cnn layer2 16×16", "res_cnn layer3 16×16"],
        "num_taus": 8,
        "reservoir_dim": 256,
        "label": "ResCNN+TPSAPU 2048 neurons",
    },
}

TINY_IMAGENET_MEAN = (0.485, 0.456, 0.406)
TINY_IMAGENET_STD = (0.229, 0.224, 0.225)

OUT_ROOT = Path("visualizations/attention_analysis")

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        choices=list(MODELS),
        default=None,
        help="Which model to analyse. Default: both.",
    )
    p.add_argument("--data-dir", default="data/tiny-imagenet-200-clean")
    p.add_argument("--split", default="validation")
    p.add_argument(
        "--vis-indices",
        default="0,1,2,3,4,5,6,7,8,9",
        help="Comma-separated sample indices for per-sample figures.",
    )
    p.add_argument(
        "--num-samples", type=int, default=500, help="Samples for aggregate statistics."
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-download", action="store_true", default=True)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Attention capture hook
# ---------------------------------------------------------------------------


class AttentionCapture:
    """Re-runs self-attention with need_weights=True inside a forward hook."""

    def __init__(self) -> None:
        self.weights: dict[int, torch.Tensor] = {}  # layer_i -> (B, H, S, S)
        self._hooks: list = []

    def register(self, transformer_encoder: nn.TransformerEncoder) -> None:
        for layer_i, layer in enumerate(transformer_encoder.layers):

            def _make_hook(idx):
                _inside = [False]  # reentrancy guard

                def hook(module, inputs, output):
                    if _inside[0]:
                        return
                    _inside[0] = True
                    try:
                        q, k, v = inputs[0], inputs[1], inputs[2]
                        with torch.no_grad():
                            _, w = module(
                                q,
                                k,
                                v,
                                need_weights=True,
                                average_attn_weights=False,
                            )
                        if w is not None:
                            self.weights[idx] = w.detach().cpu()
                    finally:
                        _inside[0] = False

                return hook

            h = layer.self_attn.register_forward_hook(_make_hook(layer_i))
            self._hooks.append(h)

    def clear(self) -> None:
        self.weights.clear()

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_dataset(args):
    hf = train_pipeline.load_hf_split(
        "slegroux/tiny-imagenet-200-clean",
        split=args.split,
        cache_dir=args.data_dir,
        no_download=args.no_download,
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
        def __init__(self, hf_data):
            self.hf = hf_data

        def __len__(self):
            return len(self.hf)

        def __getitem__(self, i):
            item = self.hf[i]
            img = item["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            return tfm(img), item["label"], i

    return _DS(hf), list(hf.features["label"].names)


def denormalize(t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(TINY_IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(TINY_IMAGENET_STD).view(3, 1, 1)
    return (t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def load_human_names() -> dict[str, str]:
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
# Core: run model + collect attention
# ---------------------------------------------------------------------------


def run_with_attention(
    model: nn.Module,
    capture: AttentionCapture,
    images: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    capture.clear()
    images = images.to(device)
    with torch.no_grad():
        logits = model(images)
    weights = {k: v.clone() for k, v in capture.weights.items()}
    return logits.cpu(), weights


# ---------------------------------------------------------------------------
# Utility: reshape attention to spatial
# ---------------------------------------------------------------------------


def cls_attention_to_spatial(
    attn: torch.Tensor,  # (H, S, S)  S = steps+1
    spatial_grid: tuple[int, int],
    token_groups: dict,  # group_name -> (start, end) 0-indexed in token space
) -> dict[str, np.ndarray]:
    """Extract CLS row [0, 1:] of attention and reshape to spatial grids."""
    # CLS is at position 0; token positions are 1..S-1
    cls_row = attn[:, 0, 1:]  # (H, steps)  attention FROM cls TO each token
    results = {}
    for name, (lo, hi) in token_groups.items():
        chunk = cls_row[:, lo:hi]  # (H, chunk_len)
        g_h, g_w = spatial_grid
        assert chunk.shape[1] == g_h * g_w, f"chunk {chunk.shape[1]} != {g_h}*{g_w}"
        results[name] = chunk.numpy().reshape(-1, g_h, g_w)  # (H, g_h, g_w)
    return results


# ---------------------------------------------------------------------------
# Plot: per-sample full picture
# ---------------------------------------------------------------------------

CMAP_ATTN = "hot"
CMAP_DIV = "RdBu_r"


def _add_image_axis(fig, ax, img_np, label_str, pred_str, correct):
    ax.imshow(img_np)
    c = "#6bbf59" if correct else "#e07b39"
    ax.set_title(f"GT: {label_str}\nPred: {pred_str}", fontsize=7, color=c)
    ax.axis("off")


def save_per_sample_attention(
    out_dir: Path,
    sample_idx: int,
    image_tensor: torch.Tensor,
    attn_all_layers: dict[int, torch.Tensor],  # layer -> (H, S, S)
    label: int,
    logits: torch.Tensor,
    class_names: list[str],
    display_names: list[str],
    cfg: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    img_np = denormalize(image_tensor)
    probs = torch.softmax(logits, dim=-1)
    pred = int(probs.argmax())
    correct = pred == label
    gt_name = display_names[label]
    pr_name = display_names[pred]

    n_layers = len(attn_all_layers)
    n_heads = next(iter(attn_all_layers.values())).shape[0]
    n_groups = len(cfg["token_groups"])
    g_h, g_w = cfg["spatial_grid"]

    # ── Figure 1: CLS attention heatmaps per layer/head ──────────────────
    n_cols = n_layers * n_groups
    fig, axes = plt.subplots(
        n_heads + 1,
        n_cols,
        figsize=(n_cols * 2.2, (n_heads + 1) * 2.0),
        squeeze=False,
    )
    fig.suptitle(
        f"[{sample_idx}] CLS Attention — {cfg['label']}\n"
        f"GT: {gt_name}  |  Pred: {pr_name}  {'✓' if correct else '✗'}",
        fontsize=9,
    )

    for layer_i, attn in attn_all_layers.items():
        spatial = cls_attention_to_spatial(
            attn, cfg["spatial_grid"], cfg["token_groups"]
        )
        for grp_j, (grp_name, maps) in enumerate(spatial.items()):
            col = layer_i * n_groups + grp_j

            ax = axes[0, col]
            ax.imshow(img_np)
            ax.set_title(f"L{layer_i + 1} {grp_name}", fontsize=7)
            ax.axis("off")

            mean_map = maps.mean(axis=0)
            mean_norm = mean_map / (mean_map.max() + 1e-9)
            resized_overlay = np.array(
                plt.cm.hot(
                    np.kron(mean_norm, np.ones((g_h, g_w)))
                    if g_h == 1
                    else np.repeat(
                        np.repeat(mean_norm, 64 // g_h, axis=0), 64 // g_w, axis=1
                    )
                )[:, :, :3]
            )
            ax.imshow(resized_overlay, alpha=0.55)

            vmax = maps.max()
            for h in range(n_heads):
                ax = axes[h + 1, col]
                im = ax.imshow(maps[h], cmap=CMAP_ATTN, vmin=0, vmax=vmax)
                ax.set_title(f"head {h}", fontsize=6)
                ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / f"{sample_idx:04d}_cls_attn_heatmap.png", dpi=130)
    plt.close(fig)

    # ── Figure 2: Full attention matrix (mean over heads) per layer ───────
    n_seq = next(iter(attn_all_layers.values())).shape[-1]
    fig, axes = plt.subplots(n_layers, 1, figsize=(5, 5 * n_layers), squeeze=False)
    fig.suptitle(
        f"[{sample_idx}] Full Attention Matrix (mean over {n_heads} heads) — {cfg['label']}",
        fontsize=9,
    )
    for layer_i, attn in attn_all_layers.items():
        full = attn.mean(dim=0).numpy()  # (S, S)
        ax = axes[layer_i, 0]
        im = ax.imshow(full, cmap="viridis", vmin=0, aspect="auto")
        ax.set_title(f"Layer {layer_i + 1}", fontsize=8)
        ax.set_xlabel("Key position (0=CLS, 1..N=tokens)", fontsize=6)
        ax.set_ylabel("Query position", fontsize=6)
        ax.tick_params(labelsize=5)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        ax.axvline(0.5, color="cyan", linewidth=0.5, alpha=0.6)
        ax.axhline(0.5, color="cyan", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / f"{sample_idx:04d}_full_attn_matrix.png", dpi=130)
    plt.close(fig)

    # ── Figure 3: Per-head full attention matrix (layer 0) ────────────────
    attn_l0 = attn_all_layers[0]  # (H, S, S)
    fig, axes = plt.subplots(n_heads, 1, figsize=(3.5, 3.5 * n_heads), squeeze=False)
    fig.suptitle(
        f"[{sample_idx}] Per-head Full Attention — Layer 1 — {cfg['label']}",
        fontsize=9,
    )
    for h in range(n_heads):
        ax = axes[h, 0]
        im = ax.imshow(attn_l0[h].numpy(), cmap="viridis", vmin=0, aspect="auto")
        ax.set_title(f"Head {h}", fontsize=7)
        ax.tick_params(labelsize=4)
        ax.set_xlabel("Key", fontsize=5)
        ax.set_ylabel("Query", fontsize=5)
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.01)
    fig.tight_layout()
    fig.savefig(out_dir / f"{sample_idx:04d}_per_head_matrix.png", dpi=130)
    plt.close(fig)

    # ── Figure 4: CLS attention rollout (layer L1 averaged then L2 applied) ──
    rollout = _attention_rollout(attn_all_layers, n_seq)
    cls_rollout = rollout[0, 1:]  # drop CLS self-attention

    fig, axes = plt.subplots(
        n_groups + 1, 1, figsize=(3, 3 * (n_groups + 1)), squeeze=False
    )
    fig.suptitle(
        f"[{sample_idx}] Attention Rollout — CLS token — {cfg['label']}",
        fontsize=9,
    )
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title(f"Image\n{gt_name}", fontsize=7)
    axes[0, 0].axis("off")

    for j, (grp_name, (lo, hi)) in enumerate(cfg["token_groups"].items()):
        chunk = cls_rollout[lo:hi].reshape(g_h, g_w)
        chunk = chunk / (chunk.max() + 1e-9)
        upscale = 64 // g_h
        big = np.repeat(np.repeat(chunk, upscale, axis=0), upscale, axis=1)
        axes[j + 1, 0].imshow(img_np)
        im = axes[j + 1, 0].imshow(big, cmap="hot", alpha=0.55, vmin=0, vmax=1)
        axes[j + 1, 0].set_title(f"Rollout: {grp_name}", fontsize=7)
        axes[j + 1, 0].axis("off")
        fig.colorbar(im, ax=axes[j + 1, 0], fraction=0.05, pad=0.02)

    fig.tight_layout()
    fig.savefig(out_dir / f"{sample_idx:04d}_attention_rollout.png", dpi=130)
    plt.close(fig)


def _attention_rollout(
    attn_all_layers: dict[int, torch.Tensor],  # layer -> (H, S, S)
    n_seq: int,
) -> np.ndarray:
    """
    Attention rollout (Abnar & Zuidema, 2020).
    Recursively multiplies attention maps across layers after adding residual identity.
    Returns (S, S) rollout matrix.
    """
    rollout = np.eye(n_seq, dtype=np.float32)
    for layer_i in sorted(attn_all_layers.keys()):
        attn = attn_all_layers[layer_i].mean(dim=0).numpy()  # (S, S)
        # add residual connection + re-normalize
        attn_aug = 0.5 * attn + 0.5 * np.eye(n_seq)
        attn_aug /= attn_aug.sum(axis=-1, keepdims=True) + 1e-9
        rollout = attn_aug @ rollout
    return rollout  # (S, S)


# ---------------------------------------------------------------------------
# Aggregate analysis helpers
# ---------------------------------------------------------------------------


def collect_attention_batch(
    model: nn.Module,
    capture: AttentionCapture,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> dict:
    """
    Returns a dict with:
      all_cls_attn[layer_i]   : np.ndarray (N, H, steps)
      all_preds               : np.ndarray (N,)
      all_labels              : np.ndarray (N,)
      all_probs               : np.ndarray (N, C)
      all_entropies[layer_i]  : np.ndarray (N,)  — entropy of CLS attention
    """
    out = {
        "cls": {},  # layer -> list of (H, steps)
        "full": {},  # layer -> list of (H, S, S)
        "preds": [],
        "labels": [],
        "probs": [],
    }
    n_seen = 0

    model.eval()
    with torch.no_grad():
        for images, labels, _ in loader:
            bs = images.size(0)
            logits, attn = run_with_attention(model, capture, images, device)
            probs = torch.softmax(logits, dim=-1).numpy()
            preds = probs.argmax(axis=-1)

            out["preds"].append(preds)
            out["labels"].append(labels.numpy())
            out["probs"].append(probs)

            for li, w in attn.items():
                # w: (B, H, S, S)
                cls_row = w[
                    :, :, 0, 1:
                ].numpy()  # (B, H, steps) — CLS attending to tokens
                if li not in out["cls"]:
                    out["cls"][li] = []
                    out["full"][li] = []
                out["cls"][li].append(cls_row)
                out["full"][li].append(w.numpy())

            n_seen += bs
            if max_samples > 0 and n_seen >= max_samples:
                break

    # Concatenate
    out["preds"] = np.concatenate(out["preds"], axis=0)[:max_samples]
    out["labels"] = np.concatenate(out["labels"], axis=0)[:max_samples]
    out["probs"] = np.concatenate(out["probs"], axis=0)[:max_samples]
    for li in out["cls"]:
        out["cls"][li] = np.concatenate(out["cls"][li], axis=0)[
            :max_samples
        ]  # (N,H,steps)
        out["full"][li] = np.concatenate(out["full"][li], axis=0)[
            :max_samples
        ]  # (N,H,S,S)
    return out


def attn_entropy(attn_row: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Shannon entropy of an attention distribution. Input (..., S)."""
    p = attn_row + eps
    return -np.sum(p * np.log(p), axis=-1)


# ---------------------------------------------------------------------------
# Aggregate figures
# ---------------------------------------------------------------------------


def _hide_column(axes: np.ndarray, col: int) -> None:
    for row in range(axes.shape[0]):
        axes[row, col].axis("off")


def _hide_row(axes: np.ndarray, row: int) -> None:
    for col in range(axes.shape[1]):
        axes[row, col].axis("off")


def _safe_corrcoef(x: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(x.T)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def fig_mean_cls_attention(collected: dict, cfg: dict, out_dir: Path) -> None:
    """Mean CLS attention over all samples, per layer, per head + combined."""
    n_layers = len(collected["cls"])
    n_heads = next(iter(collected["cls"].values())).shape[1]  # (N, H, steps)
    g_h, g_w = cfg["spatial_grid"]

    correct_mask = collected["preds"] == collected["labels"]
    splits = [
        (np.ones(len(collected["preds"]), dtype=bool), "All"),
        (correct_mask, "Correct"),
    ]
    groups = list(cfg["token_groups"].items())

    n_rows = n_heads + 2
    n_cols = n_layers * len(splits) * len(groups)

    fig, big_axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 2.0, n_rows * 2.2),
        squeeze=False,
    )
    fig.suptitle(
        f"Mean CLS Attention: {cfg['label']}\n"
        f"(columns=layer/split/token-group, rows=mean/std/heads)",
        fontsize=9,
    )

    for layer_i in sorted(collected["cls"].keys()):
        cls = collected["cls"][layer_i]  # (N, H, steps)

        for split_j, (mask, split_label) in enumerate(splits):
            sub = cls[mask]  # (M, H, steps)
            for grp_j, (grp_name, (lo, hi)) in enumerate(groups):
                col = ((layer_i * len(splits) + split_j) * len(groups)) + grp_j
                if sub.shape[0] == 0:
                    _hide_column(big_axes, col)
                    continue

                mean_all_heads = sub.mean(axis=(0, 1))  # (steps,)
                mean_map = mean_all_heads[lo:hi].reshape(g_h, g_w)
                ax = big_axes[0, col]
                im = ax.imshow(mean_map, cmap=CMAP_ATTN)
                ax.set_title(
                    f"L{layer_i + 1} {split_label}\n{grp_name}\nmean all heads",
                    fontsize=6,
                )
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.05, pad=0.01)

                std_map = sub.mean(axis=1).std(axis=0)[lo:hi].reshape(g_h, g_w)
                ax = big_axes[1, col]
                im = ax.imshow(std_map, cmap="Blues")
                ax.set_title(
                    f"L{layer_i + 1} {split_label}\n{grp_name}\nstd over samples",
                    fontsize=6,
                )
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.05, pad=0.01)

                for h in range(n_heads):
                    head_mean = sub[:, h, :].mean(axis=0)
                    hmap = head_mean[lo:hi].reshape(g_h, g_w)
                    ax = big_axes[h + 2, col]
                    im = ax.imshow(hmap, cmap=CMAP_ATTN)
                    ax.set_title(
                        f"L{layer_i + 1} H{h}\n{split_label}\n{grp_name}", fontsize=6
                    )
                    ax.axis("off")
                    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.01)

    fig.tight_layout()
    p = out_dir / "mean_cls_attention.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")


def fig_attention_entropy(collected: dict, cfg: dict, out_dir: Path) -> None:
    """Distribution of CLS attention entropy: correct vs wrong predictions."""
    n_layers = len(collected["cls"])
    correct = collected["preds"] == collected["labels"]

    fig, axes = plt.subplots(n_layers, 1, figsize=(5, 4 * n_layers), squeeze=False)
    fig.suptitle(
        f"CLS Attention Entropy — {cfg['label']}\n"
        f"(lower = more focused; correct {correct.sum()} / wrong {(~correct).sum()})",
        fontsize=9,
    )

    for layer_i in sorted(collected["cls"].keys()):
        cls = collected["cls"][layer_i]  # (N, H, steps)
        mean_cls = cls.mean(axis=1)  # (N, steps) mean over heads
        ent = attn_entropy(mean_cls)  # (N,)

        ax = axes[layer_i, 0]
        bins = np.linspace(ent.min(), ent.max(), 40)
        if correct.any():
            ax.hist(
                ent[correct],
                bins=bins,
                alpha=0.65,
                color="#6bbf59",
                label=f"correct (n={correct.sum()})",
                density=True,
            )
            ax.axvline(
                ent[correct].mean(), color="#6bbf59", linestyle="--", linewidth=1.5
            )
        if (~correct).any():
            ax.hist(
                ent[~correct],
                bins=bins,
                alpha=0.65,
                color="#e07b39",
                label=f"wrong   (n={(~correct).sum()})",
                density=True,
            )
            ax.axvline(
                ent[~correct].mean(), color="#e07b39", linestyle="--", linewidth=1.5
            )
        ax.set_title(f"Layer {layer_i + 1}", fontsize=8)
        ax.set_xlabel("Entropy H(CLS attn)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        correct_mean = ent[correct].mean() if correct.any() else float("nan")
        correct_std = ent[correct].std() if correct.any() else float("nan")
        wrong_mean = ent[~correct].mean() if (~correct).any() else float("nan")
        wrong_std = ent[~correct].std() if (~correct).any() else float("nan")
        print(
            f"  [{cfg['label']}] L{layer_i + 1} entropy: "
            f"correct={correct_mean:.3f}±{correct_std:.3f}  "
            f"wrong={wrong_mean:.3f}±{wrong_std:.3f}"
        )

    fig.tight_layout()
    p = out_dir / "attention_entropy.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"Saved: {p}")


def fig_head_specialization(collected: dict, cfg: dict, out_dir: Path) -> None:
    """
    How similar are different heads?
    • Cosine similarity matrix between mean CLS attention vectors per head.
    • Pairwise correlation heatmap between per-sample CLS-attention entropies.
    """
    n_layers = len(collected["cls"])
    n_heads = next(iter(collected["cls"].values())).shape[1]

    fig, axes = plt.subplots(n_layers, 2, figsize=(10, 4 * n_layers), squeeze=False)
    fig.suptitle(f"Head Specialization — {cfg['label']}", fontsize=10)

    for layer_i in sorted(collected["cls"].keys()):
        cls = collected["cls"][layer_i]  # (N, H, steps)
        mean_per_head = cls.mean(axis=0)  # (H, steps)

        norms = np.linalg.norm(mean_per_head, axis=-1, keepdims=True) + 1e-9
        normed = mean_per_head / norms  # (H, steps)
        cosim = normed @ normed.T  # (H, H)

        head_entropies = attn_entropy(cls)  # (N, H)
        corr = _safe_corrcoef(head_entropies)  # (H, H)

        ax_cos = axes[layer_i, 0]
        im = ax_cos.imshow(cosim, cmap="coolwarm", vmin=-1, vmax=1)
        ax_cos.set_title(
            f"L{layer_i + 1} Cosine similarity\nbetween head mean attn", fontsize=7
        )
        ax_cos.set_xticks(range(n_heads))
        ax_cos.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=6)
        ax_cos.set_yticks(range(n_heads))
        ax_cos.set_yticklabels([f"H{h}" for h in range(n_heads)], fontsize=6)
        for i in range(n_heads):
            for j in range(n_heads):
                ax_cos.text(
                    j, i, f"{cosim[i, j]:.2f}", ha="center", va="center", fontsize=5
                )
        fig.colorbar(im, ax=ax_cos, fraction=0.05, pad=0.02)

        ax_cor = axes[layer_i, 1]
        im2 = ax_cor.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax_cor.set_title(
            f"L{layer_i + 1} Pearson corr\nbetween per-sample entropies", fontsize=7
        )
        ax_cor.set_xticks(range(n_heads))
        ax_cor.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=6)
        ax_cor.set_yticks(range(n_heads))
        ax_cor.set_yticklabels([f"H{h}" for h in range(n_heads)], fontsize=6)
        for i in range(n_heads):
            for j in range(n_heads):
                ax_cor.text(
                    j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=5
                )
        fig.colorbar(im2, ax=ax_cor, fraction=0.05, pad=0.02)

    fig.tight_layout()
    p = out_dir / "head_specialization.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"Saved: {p}")


def fig_layer_comparison(collected: dict, cfg: dict, out_dir: Path) -> None:
    """How does attention change between layers? Diff + correlation."""
    if len(collected["cls"]) < 2:
        return
    g_h, g_w = cfg["spatial_grid"]
    groups = list(cfg["token_groups"].items())

    layer_keys = sorted(collected["cls"].keys())
    n_heads = collected["cls"][layer_keys[0]].shape[1]

    fig, axes = plt.subplots(
        n_heads + 1,
        3 * len(groups),
        figsize=(3 * len(groups) * 2.5, (n_heads + 1) * 2.4),
        squeeze=False,
    )
    fig.suptitle(f"Layer Comparison: L1 → L2 — {cfg['label']}", fontsize=9)

    cls_l0 = collected["cls"][layer_keys[0]].mean(axis=0)  # (H, steps)
    cls_l1 = collected["cls"][layer_keys[1]].mean(axis=0)  # (H, steps)
    diff = cls_l1 - cls_l0  # (H, steps)

    vmax_0 = cls_l0.max()
    vmax_1 = cls_l1.max()
    dlim = max(np.abs(diff).max(), 1e-9)

    for grp_j, (grp_name, (lo, hi)) in enumerate(groups):
        col_base = grp_j * 3

        map_mean_l0 = cls_l0.mean(axis=0)[lo:hi].reshape(g_h, g_w)
        im = axes[0, col_base].imshow(map_mean_l0, cmap=CMAP_ATTN)
        axes[0, col_base].set_title(f"{grp_name}\nL1 mean all heads", fontsize=7)
        axes[0, col_base].axis("off")
        fig.colorbar(im, ax=axes[0, col_base], fraction=0.05, pad=0.01)

        map_mean_l1 = cls_l1.mean(axis=0)[lo:hi].reshape(g_h, g_w)
        im = axes[0, col_base + 1].imshow(map_mean_l1, cmap=CMAP_ATTN)
        axes[0, col_base + 1].set_title(f"{grp_name}\nL2 mean all heads", fontsize=7)
        axes[0, col_base + 1].axis("off")
        fig.colorbar(im, ax=axes[0, col_base + 1], fraction=0.05, pad=0.01)

        diff_mean = diff.mean(axis=0)[lo:hi].reshape(g_h, g_w)
        im = axes[0, col_base + 2].imshow(
            diff_mean, cmap=CMAP_DIV, vmin=-dlim, vmax=dlim
        )
        axes[0, col_base + 2].set_title(f"{grp_name}\nDiff mean", fontsize=7)
        axes[0, col_base + 2].axis("off")
        fig.colorbar(im, ax=axes[0, col_base + 2], fraction=0.05, pad=0.01)

        for h in range(n_heads):
            m = cls_l0[h][lo:hi].reshape(g_h, g_w)
            im = axes[h + 1, col_base].imshow(m, cmap=CMAP_ATTN, vmax=vmax_0)
            axes[h + 1, col_base].set_title(f"L1 H{h}", fontsize=6)
            axes[h + 1, col_base].axis("off")

            m = cls_l1[h][lo:hi].reshape(g_h, g_w)
            im = axes[h + 1, col_base + 1].imshow(m, cmap=CMAP_ATTN, vmax=vmax_1)
            axes[h + 1, col_base + 1].set_title(f"L2 H{h}", fontsize=6)
            axes[h + 1, col_base + 1].axis("off")

            d = diff[h][lo:hi].reshape(g_h, g_w)
            im = axes[h + 1, col_base + 2].imshow(
                d, cmap=CMAP_DIV, vmin=-dlim, vmax=dlim
            )
            axes[h + 1, col_base + 2].set_title(f"Diff H{h}", fontsize=6)
            axes[h + 1, col_base + 2].axis("off")

    fig.tight_layout()
    p = out_dir / "layer_comparison.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"Saved: {p}")


def fig_class_conditional_attention(
    collected: dict,
    cfg: dict,
    out_dir: Path,
    class_names: list[str],
    display_names: list[str],
    top_n_classes: int = 8,
) -> None:
    """Mean CLS attention conditioned on ground-truth class (correct predictions only)."""
    correct_mask = collected["preds"] == collected["labels"]
    labels_correct = collected["labels"][correct_mask]
    cls_l0 = collected["cls"][0][correct_mask]  # (N_correct, H, steps)

    from collections import Counter

    counts = Counter(labels_correct.tolist())
    top_classes = [c for c, _ in counts.most_common(top_n_classes)]
    if not top_classes:
        return

    g_h, g_w = cfg["spatial_grid"]
    groups = list(cfg["token_groups"].items())
    panels = [(cls_label, group) for cls_label in top_classes for group in groups]
    n_panels = len(panels)

    rows = min(4, n_panels)
    cols = math.ceil(n_panels / rows)
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 2.8, rows * 2.8), squeeze=False
    )
    fig.suptitle(
        f"Class-Conditional Mean CLS Attention (L1, mean heads) — {cfg['label']}\n"
        f"Correct predictions only",
        fontsize=9,
    )

    for idx, (cls_label, (grp_name, (lo, hi))) in enumerate(panels):
        r = idx % rows
        c = idx // rows
        mask_cls = labels_correct == cls_label
        sub = cls_l0[mask_cls].mean(axis=(0, 1))  # (steps,)
        m = sub[lo:hi].reshape(g_h, g_w)
        m = m / (m.max() + 1e-9)
        ax = axes[r, c]
        im = ax.imshow(m, cmap=CMAP_ATTN, vmin=0, vmax=1)
        name = display_names[cls_label][:20]
        ax.set_title(f"{name}\n{grp_name} (n={mask_cls.sum()})", fontsize=7)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    for idx in range(n_panels, rows * cols):
        r = idx % rows
        c = idx // rows
        axes[r, c].axis("off")

    fig.tight_layout()
    p = out_dir / "class_conditional_attention.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")


def fig_attention_rollout_aggregate(
    collected: dict,
    cfg: dict,
    out_dir: Path,
) -> None:
    """Aggregate attention rollout map (mean over samples, correct vs wrong)."""
    correct_mask = collected["preds"] == collected["labels"]
    g_h, g_w = cfg["spatial_grid"]
    groups = list(cfg["token_groups"].items())
    n_seq = next(iter(collected["full"].values())).shape[-1]  # S = steps+1

    def compute_rollout_batch(full_attn_by_layer):
        N = next(iter(full_attn_by_layer.values())).shape[0]
        rollouts = []
        for n in range(N):
            layer_dict = {
                li: torch.from_numpy(v[n]) for li, v in full_attn_by_layer.items()
            }
            r = _attention_rollout(layer_dict, n_seq)  # (S, S)
            rollouts.append(r[0, 1:])  # CLS row, drop CLS self
        return np.stack(rollouts, axis=0)  # (N, steps)

    print("  Computing attention rollout (this may take a moment)…")
    all_rollout = compute_rollout_batch(collected["full"])  # (N, steps)

    splits = [
        ("All samples", np.ones(len(all_rollout), bool)),
        ("Correct only", correct_mask),
        ("Wrong only", ~correct_mask),
    ]
    fig, axes = plt.subplots(
        len(splits),
        2 * len(groups),
        figsize=(2 * len(groups) * 3, len(splits) * 3),
        squeeze=False,
    )
    fig.suptitle(f"Attention Rollout — {cfg['label']}", fontsize=10)

    for row, (title, mask) in enumerate(splits):
        sub = all_rollout[mask]
        if sub.shape[0] == 0:
            _hide_row(axes, row)
            continue

        mean_r = sub.mean(axis=0)
        std_r = sub.std(axis=0)

        for grp_j, (grp_name, (lo, hi)) in enumerate(groups):
            m_map = mean_r[lo:hi].reshape(g_h, g_w)
            s_map = std_r[lo:hi].reshape(g_h, g_w)
            m_map /= m_map.max() + 1e-9

            im = axes[row, 2 * grp_j].imshow(m_map, cmap=CMAP_ATTN)
            axes[row, 2 * grp_j].set_title(
                f"{title}\n{grp_name}\nmean rollout (n={mask.sum()})",
                fontsize=8,
            )
            axes[row, 2 * grp_j].axis("off")
            fig.colorbar(im, ax=axes[row, 2 * grp_j], fraction=0.04, pad=0.02)

            im2 = axes[row, 2 * grp_j + 1].imshow(s_map, cmap="Blues")
            axes[row, 2 * grp_j + 1].set_title(
                f"{title}\n{grp_name}\nstd rollout", fontsize=8
            )
            axes[row, 2 * grp_j + 1].axis("off")
            fig.colorbar(im2, ax=axes[row, 2 * grp_j + 1], fraction=0.04, pad=0.02)

    fig.tight_layout()
    p = out_dir / "attention_rollout_aggregate.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"Saved: {p}")


def fig_positional_attention_profile(
    collected: dict,
    cfg: dict,
    out_dir: Path,
) -> None:
    """
    1-D profile: mean CLS attention per token position, averaged over samples and heads.
    One line per layer. Helps answer "does the decoder prefer early or late tokens?".
    """
    n_layers = len(collected["cls"])
    steps = next(iter(collected["cls"].values())).shape[-1]
    correct = collected["preds"] == collected["labels"]

    fig, axes = plt.subplots(2, 1, figsize=(6, 8), squeeze=False)
    fig.suptitle(
        f"CLS Attention Profile per Token Position — {cfg['label']}", fontsize=10
    )

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_layers))

    for split_i, (mask, label) in enumerate(
        [(np.ones(len(collected["preds"]), bool), "All"), (correct, "Correct only")]
    ):
        ax = axes[split_i, 0]
        if not mask.any():
            ax.axis("off")
            continue

        for layer_i in sorted(collected["cls"].keys()):
            cls = collected["cls"][layer_i][mask]  # (M, H, steps)
            mean_profile = cls.mean(axis=(0, 1))  # (steps,)
            ax.plot(
                range(1, steps + 1),
                mean_profile,
                color=colors[layer_i],
                linewidth=1.8,
                label=f"Layer {layer_i + 1}",
            )

        for grp_name, (lo, hi) in cfg["token_groups"].items():
            ax.axvspan(
                lo + 1,
                hi + 1,
                alpha=0.05,
                color=f"C{list(cfg['token_groups'].keys()).index(grp_name)}",
            )
            ax.axvline(lo + 1, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
            ax.text(
                lo + 2,
                ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.01,
                grp_name,
                fontsize=6,
                color="gray",
            )

        ax.set_xlabel("Token position (1 = first token)")
        ax.set_ylabel("Mean CLS attention weight")
        ax.set_title(f"{label} (n={mask.sum()})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    p = out_dir / "positional_attention_profile.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"Saved: {p}")


def fig_token_to_token_patterns(
    collected: dict,
    cfg: dict,
    out_dir: Path,
) -> None:
    """
    Average full attention matrix (all tokens to all tokens), mean over samples.
    Reveals how tokens propagate information to each other.
    """
    n_layers = len(collected["full"])
    n_heads = next(iter(collected["full"].values())).shape[1]

    fig, axes = plt.subplots(
        n_heads + 1,
        n_layers,
        figsize=(n_layers * 3.5, (n_heads + 1) * 3.5),
        squeeze=False,
    )
    fig.suptitle(f"Mean Full Attention Matrix (all→all) — {cfg['label']}", fontsize=9)

    for layer_i in sorted(collected["full"].keys()):
        full = collected["full"][layer_i]  # (N, H, S, S)
        mean_full = full.mean(axis=0)  # (H, S, S)

        ax = axes[0, layer_i]
        im = ax.imshow(mean_full.mean(axis=0), cmap="viridis", aspect="auto", vmin=0)
        ax.set_title(f"L{layer_i + 1} mean all heads", fontsize=7)
        ax.set_xlabel("Key (0=CLS)", fontsize=6)
        ax.set_ylabel("Query (0=CLS)", fontsize=6)
        ax.tick_params(labelsize=4)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)

        for h in range(n_heads):
            ax = axes[h + 1, layer_i]
            im = ax.imshow(mean_full[h], cmap="viridis", aspect="auto", vmin=0)
            ax.set_title(f"L{layer_i + 1} H{h}", fontsize=7)
            ax.tick_params(labelsize=4)
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)

    fig.tight_layout()
    p = out_dir / "mean_full_attention_matrix.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"Saved: {p}")


def fig_top_k_attention_positions(
    collected: dict,
    cfg: dict,
    out_dir: Path,
) -> None:
    """
    For each sample, find the top-k attended positions.
    Show histogram of which positions are 'top-3' most attended across dataset.
    """
    g_h, g_w = cfg["spatial_grid"]
    groups = list(cfg["token_groups"].items())
    correct = collected["preds"] == collected["labels"]

    k = 3
    layer_keys = sorted(collected["cls"].keys())
    fig, axes = plt.subplots(
        len(layer_keys) * len(groups),
        2,
        figsize=(10, len(layer_keys) * len(groups) * 3.5),
        squeeze=False,
    )
    fig.suptitle(f"Top-{k} Attended Positions — {cfg['label']}", fontsize=10)

    for layer_pos, layer_i in enumerate(layer_keys):
        cls = collected["cls"][layer_i]  # (N, H, steps)
        cls_mn = cls.mean(axis=1)  # (N, steps)

        for grp_j, (grp_name, (lo, hi)) in enumerate(groups):
            row_idx = layer_pos * len(groups) + grp_j
            topk = np.argsort(-cls_mn[:, lo:hi], axis=-1)[:, :k]

            freq = np.zeros(g_h * g_w)
            for row in topk:
                freq[row] += 1
            freq_map = freq.reshape(g_h, g_w) / len(topk)

            ax = axes[row_idx, 0]
            im = ax.imshow(freq_map, cmap="Oranges", vmin=0)
            ax.set_title(
                f"L{layer_i + 1} {grp_name}: fraction of samples\nposition is in top-{k}",
                fontsize=7,
            )
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

            freq_c = np.zeros(g_h * g_w)
            freq_w = np.zeros(g_h * g_w)
            for n_i, row in enumerate(topk):
                if correct[n_i]:
                    freq_c[row] += 1
                else:
                    freq_w[row] += 1
            freq_c_map = (freq_c / (correct.sum() + 1e-9)).reshape(g_h, g_w)
            freq_w_map = (freq_w / ((~correct).sum() + 1e-9)).reshape(g_h, g_w)
            diff = freq_c_map - freq_w_map
            lim = max(abs(diff.max()), abs(diff.min()), 1e-5)

            ax2 = axes[row_idx, 1]
            im2 = ax2.imshow(diff, cmap=CMAP_DIV, vmin=-lim, vmax=lim)
            ax2.set_title(
                f"L{layer_i + 1} {grp_name}: top-{k} freq diff\ncorrect − wrong",
                fontsize=7,
            )
            ax2.axis("off")
            fig.colorbar(im2, ax=ax2, fraction=0.04, pad=0.02)

    fig.tight_layout()
    p = out_dir / "top_k_attended_positions.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"Saved: {p}")


def _global_peak_stats(vector: np.ndarray, cfg: dict) -> dict:
    g_h, g_w = cfg["spatial_grid"]
    best = None
    for grp_name, (lo, hi) in cfg["token_groups"].items():
        local_pos = int(np.argmax(vector[lo:hi]))
        value = float(vector[lo + local_pos])
        stats = {
            "token_group": grp_name,
            "peak_token_position": int(lo + local_pos),
            "peak_token_group_position": local_pos,
            "peak_token_grid_rc": [local_pos // g_w, local_pos % g_w],
            "peak_attention": value,
        }
        if best is None or value > best["peak_attention"]:
            best = stats
    return best


def save_summary_json(
    collected: dict,
    cfg: dict,
    out_dir: Path,
) -> dict:
    """Compute and save key aggregate statistics as JSON."""
    correct = collected["preds"] == collected["labels"]

    summary: dict = {
        "model": cfg["label"],
        "n_samples": int(len(collected["labels"])),
        "n_correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "layers": {},
    }

    for layer_i in sorted(collected["cls"].keys()):
        cls = collected["cls"][layer_i]  # (N, H, steps)
        cls_mn = cls.mean(axis=1)  # (N, steps) — mean over heads
        ent = attn_entropy(cls_mn)  # (N,)

        mean_map = cls_mn.mean(axis=0)  # (steps,)
        peak = _global_peak_stats(mean_map, cfg)

        layer_stats = {
            "mean_entropy": float(ent.mean()),
            "std_entropy": float(ent.std()),
            "mean_entropy_correct": float(ent[correct].mean())
            if correct.any()
            else None,
            "mean_entropy_wrong": float(ent[~correct].mean())
            if (~correct).any()
            else None,
            **peak,
            "mean_attention_vector": mean_map.tolist(),
        }

        head_stats = []
        n_heads = cls.shape[1]
        for h in range(n_heads):
            head_mean = cls[:, h, :].mean(axis=0)
            head_ent = attn_entropy(cls[:, h, :]).mean()
            hp = _global_peak_stats(head_mean, cfg)
            head_stats.append(
                {
                    "head": h,
                    "mean_entropy": float(head_ent),
                    **hp,
                }
            )
        layer_stats["heads"] = head_stats
        summary["layers"][str(layer_i)] = layer_stats

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved: {out_dir / 'summary.json'}")
    return summary


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 65)
    print(f"Attention Analysis Summary: {summary['model']}")
    print(f"  Samples: {summary['n_samples']}  |  Accuracy: {summary['accuracy']:.2%}")
    print("=" * 65)
    for li, ls in summary["layers"].items():
        print(f"\n  Layer {int(li) + 1}:")
        print(
            f"    Entropy:  mean={ls['mean_entropy']:.4f}  std={ls['std_entropy']:.4f}"
        )
        if ls["mean_entropy_correct"] is not None:
            print(
                f"    Correct:  entropy={ls['mean_entropy_correct']:.4f}  "
                f"Wrong: {ls['mean_entropy_wrong']:.4f}  "
                f"Δ={ls['mean_entropy_correct'] - ls['mean_entropy_wrong']:.4f}"
            )
        print(
            f"    Peak token: #{ls['peak_token_position']}  grid={ls['peak_token_grid_rc']}"
        )
        for hs in ls["heads"]:
            print(
                f"    Head {hs['head']}:  entropy={hs['mean_entropy']:.3f}  "
                f"peak=#{hs['peak_position']} @ {hs['peak_grid_rc']}"
            )
    print("=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    train_pipeline.set_seed(args.seed)

    model_keys = [args.model] if args.model else list(MODELS.keys())

    print(f"Loading dataset split={args.split} …")
    dataset, class_names = load_dataset(args)
    human = load_human_names()
    display_names = [human.get(s, s).split(",")[0].strip() for s in class_names]

    vis_indices = [int(s.strip()) for s in args.vis_indices.split(",") if s.strip()]

    for model_key in model_keys:
        cfg = MODELS[model_key]
        out_dir = OUT_ROOT / model_key
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 60}")
        print(f"Model: {cfg['label']}  —  {cfg['checkpoint']}")
        print(f"{'=' * 60}")

        # Load model
        ck = eval_script.load_checkpoint(cfg["checkpoint"], device)
        arch_args = eval_script.architecture_from_checkpoint(ck)
        model = train_pipeline.build_model(arch_args).to(device)
        model.load_state_dict(ck["model_state"])
        model.eval()

        # Register attention capture
        decoder = model.decoder
        capture = AttentionCapture()
        capture.register(decoder.transformer)

        # ── Per-sample figures ────────────────────────────────────────
        print(f"\nGenerating per-sample figures for indices {vis_indices} …")
        for idx in vis_indices:
            if idx >= len(dataset):
                continue
            img_t, label, _ = dataset[idx]
            logits, attn = run_with_attention(
                model, capture, img_t.unsqueeze(0), device
            )
            logits = logits.squeeze(0)
            # attn[li]: (1, H, S, S) → (H, S, S)
            attn_squeezed = {li: w.squeeze(0) for li, w in attn.items()}

            sample_dir = out_dir / f"validation_{idx}"
            save_per_sample_attention(
                sample_dir,
                idx,
                img_t,
                attn_squeezed,
                label,
                logits,
                class_names,
                display_names,
                cfg,
            )
            probs = torch.softmax(logits, dim=-1)
            pred = int(probs.argmax())
            print(
                f"  [{idx}] {display_names[label]} → {display_names[pred]} "
                f"({'✓' if pred == label else '✗'})  "
                f"conf={float(probs[pred]):.1%}"
            )

        # ── Aggregate analysis ────────────────────────────────────────
        print(f"\nCollecting attention over {args.num_samples} samples …")
        loader = DataLoader(
            torch.utils.data.Subset(
                dataset, range(min(args.num_samples, len(dataset)))
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        collected = collect_attention_batch(
            model, capture, loader, device, args.num_samples
        )
        acc = (collected["preds"] == collected["labels"]).mean()
        print(f"  Accuracy over {len(collected['preds'])} samples: {acc:.2%}")

        capture.remove()

        # ── Aggregate figures ─────────────────────────────────────────
        print("\nGenerating aggregate figures …")
        fig_mean_cls_attention(collected, cfg, out_dir)
        fig_attention_entropy(collected, cfg, out_dir)
        fig_head_specialization(collected, cfg, out_dir)
        fig_layer_comparison(collected, cfg, out_dir)
        fig_class_conditional_attention(
            collected, cfg, out_dir, class_names, display_names
        )
        fig_attention_rollout_aggregate(collected, cfg, out_dir)
        fig_positional_attention_profile(collected, cfg, out_dir)
        fig_token_to_token_patterns(collected, cfg, out_dir)
        fig_top_k_attention_positions(collected, cfg, out_dir)

        summary = save_summary_json(collected, cfg, out_dir)
        print_summary(summary)

    print(f"\nAll outputs saved under: {OUT_ROOT}")


if __name__ == "__main__":
    main()
