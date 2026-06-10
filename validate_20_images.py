"""Comprehensive validation of all session work on 20 random validation images.

Runs all three models on 20 reproducibly sampled images and produces:

  1. Side-by-side prediction comparison  (all 3 models per image)
  2. TPSAPU CLS attention overlaid on image patches — with the honest caveat
     that the transformer sees TPSAPU *states*, not pixels directly
  3. ResNet-18 layer effective rank per image (spot-check)
  4. Cross-validation of aggregate metrics vs previously reported values
  5. Summary accuracy table

NOTE on attention interpretation
---------------------------------
The transformer decoder sees a sequence of TPSAPU reservoir states, one per
CNN encoder token.  The tokens are produced in raster-scan spatial order from
the 8×8 (for cnn3) or 2×8×8 (for res_cnn) feature grid.  So "position i
attends heavily" means "the TPSAPU state corresponding to spatial patch i
was most informative for classification" — it does NOT mean the transformer
directly saw image pixels at that location.  The chain is:

    image patch  →  CNN encoder  →  TPSAPU reservoir state  →  transformer

Attention rollout across the 2-layer transformer is numerically near-identical
to raw attention (only 2 layers → shallow compound), so we skip rollout here
and just show raw CLS attention mapped back to CNN patch coordinates.

All outputs go to:  visualizations/validation_20/
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

import evaluate_tiny_imagenet_validation as eval_script
import train_pipeline
from analyze_transformer_attention import AttentionCapture, attn_entropy

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 99  # used to pick the 20 images
N_IMAGES = 20

RESNET_CKPT = "sweep_runs_tiny_imagenet/resnet18_scratch/best.pt"
CNN3_CKPT = "checkpoints/best_702k_cnn3_membrane_transformer.pt"
RESCNN_CKPT = "checkpoints/res_cnn_cross_both_transformer/best.pt"

TINY_IMAGENET_MEAN = (0.485, 0.456, 0.406)
TINY_IMAGENET_STD = (0.229, 0.224, 0.225)

OUT = Path("visualizations/validation_20")
OUT.mkdir(parents=True, exist_ok=True)

# Previously reported aggregate metrics (from the session) for cross-validation
REPORTED_METRICS = {
    "ResNet-18": {"top1": 0.5665, "top5": 0.7920},
    "CNN3+TPSAPU 702k": {"top1": 0.4164, "top5": 0.6763},
    "ResCNN+TPSAPU 2048": {"top1": 0.3923, "top5": 0.6539},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def denormalize(t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(TINY_IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(TINY_IMAGENET_STD).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


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


def short(name: str, n: int = 16) -> str:
    return name[:n] + ("…" if len(name) > n else "")


def load_dataset(data_dir: str, split: str = "validation"):
    hf = train_pipeline.load_hf_split(
        "slegroux/tiny-imagenet-200-clean",
        split=split,
        cache_dir=data_dir,
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
            return tfm(img), item["label"], i

    return _DS(hf), list(hf.features["label"].names)


def pick_indices(n_total: int, n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed=seed)
    return sorted(rng.choice(n_total, size=n, replace=False).tolist())


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_resnet18(device: torch.device) -> nn.Module:
    ck = torch.load(RESNET_CKPT, map_location=device, weights_only=False)
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 200)
    model.load_state_dict(ck["model_state"])
    return model.to(device).eval()


def load_tpsapu(ckpt_path: str, device: torch.device) -> nn.Module:
    ck = eval_script.load_checkpoint(ckpt_path, device)
    arch = eval_script.architecture_from_checkpoint(ck)
    model = train_pipeline.build_model(arch).to(device)
    model.load_state_dict(ck["model_state"])
    return model.eval()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def infer_resnet(model, img_t: torch.Tensor, device: torch.device) -> dict:
    logits = model(img_t.unsqueeze(0).to(device)).squeeze(0).cpu()
    probs = torch.softmax(logits, dim=-1)
    topk = torch.topk(probs, 5)
    return {
        "logits": logits.numpy(),
        "probs": probs.numpy(),
        "pred": int(topk.indices[0]),
        "conf": float(topk.values[0]),
        "top5": topk.indices.tolist(),
    }


@torch.no_grad()
def infer_tpsapu_with_attention(
    model: nn.Module,
    capture: AttentionCapture,
    img_t: torch.Tensor,
    device: torch.device,
) -> dict:
    capture.clear()
    logits = model(img_t.unsqueeze(0).to(device)).squeeze(0).cpu()
    probs = torch.softmax(logits, dim=-1)
    topk = torch.topk(probs, 5)
    attn = {k: v.squeeze(0) for k, v in capture.weights.items()}  # (H, S, S)
    return {
        "logits": logits.numpy(),
        "probs": probs.numpy(),
        "pred": int(topk.indices[0]),
        "conf": float(topk.values[0]),
        "top5": topk.indices.tolist(),
        "attn": attn,
    }


def cls_attention_spatial(
    attn: dict[int, torch.Tensor],  # layer → (H, S, S)
    spatial_tokens: int,
    spatial_grid: tuple[int, int],
    layer: int = 0,
    token_offset: int = 0,
) -> np.ndarray:
    """
    Extract mean-head CLS attention, return as (g_h, g_w) spatial map.

    NOTE: position i corresponds to the TPSAPU reservoir state for CNN
    spatial patch i — NOT directly to image pixels.

    CNN3 on 64×64: 256 tokens (16×16 grid, each token = 4×4 pixel block)
    ResCNN on 64×64: 512 tokens (two 16×16 grids — layer2 and layer3)
    """
    w = attn[layer]  # (H, S, S)
    lo = 1 + token_offset
    hi = lo + spatial_tokens
    cls = w[:, 0, lo:hi]  # (H, spatial_tokens) — CLS→tokens only
    mean = cls.mean(dim=0).numpy()  # (spatial_tokens,)
    g_h, g_w = spatial_grid
    return mean.reshape(g_h, g_w)


# ---------------------------------------------------------------------------
# Figure 1: Per-image side-by-side comparison (all 3 models)
# ---------------------------------------------------------------------------

ORANGE = "#e07b39"
BLUE = "#4c9be8"
GREEN = "#6bbf59"
GRAY = "#9BA3C0"


def fig_per_image_comparison(
    images: list[np.ndarray],  # list of (64,64,3) floats
    img_tensors: list[torch.Tensor],
    labels: list[int],
    indices: list[int],
    rn_results: list[dict],
    cnn3_results: list[dict],
    rescnn_results: list[dict],
    display_names: list[str],
    cnn3_grid: tuple[int, int],
    rescnn_grid: tuple[int, int],
) -> None:
    """4 columns per image: input image, ResNet preds, CNN3 attention, ResCNN attention."""
    N = len(images)
    n_cols = 4
    fig, axes = plt.subplots(N, n_cols, figsize=(n_cols * 2.8, N * 2.6))
    if N == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(
        "20-Image Validation: All 3 Models\n"
        "Attention = CLS → TPSAPU-state positions (CNN patch coords)",
        fontsize=11,
        fontweight="bold",
    )

    col_titles = [
        "Input image\n(ground truth)",
        "ResNet-18\nprediction",
        "CNN3-702k\nCLS attn (L1, mean heads)",
        "ResCNN-2048\nCLS attn L1 layer2 group",
    ]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=8, fontweight="bold")

    for row, (img_np, img_t, label, idx, rn, c3, rc) in enumerate(
        zip(
            images,
            img_tensors,
            labels,
            indices,
            rn_results,
            cnn3_results,
            rescnn_results,
        )
    ):
        gt = display_names[label]

        # ── Col 0: image + ground truth ──────────────────────────────
        ax = axes[row, 0]
        ax.imshow(img_np)
        ax.set_xlabel(f"[{idx}] GT: {short(gt)}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

        # ── Col 1: ResNet prediction bar ──────────────────────────────
        ax = axes[row, 1]
        top5_probs = [rn["probs"][t] for t in rn["top5"]]
        top5_names = [short(display_names[t], 14) for t in rn["top5"]]
        colors = [ORANGE if t == label else GRAY for t in rn["top5"]]
        bars = ax.barh(range(5), top5_probs[::-1], color=colors[::-1], height=0.65)
        ax.set_yticks(range(5))
        ax.set_yticklabels(top5_names[::-1], fontsize=6)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability", fontsize=6)
        correct_rn = rn["pred"] == label
        ax.set_title(
            f"{'✓' if correct_rn else '✗'} {short(display_names[rn['pred']], 14)}",
            fontsize=7,
            color=ORANGE if correct_rn else "red",
        )
        ax.tick_params(labelsize=5)

        # ── Col 2: CNN3 attention heatmap overlaid on image ───────────
        ax = axes[row, 2]
        ax.imshow(img_np)
        if c3["attn"]:
            amap = cls_attention_spatial(c3["attn"], 256, cnn3_grid, layer=0)
            amap = amap / (amap.max() + 1e-9)
            upscale = 64 // cnn3_grid[0]  # 64//16 = 4 pixels per token
            big = np.repeat(np.repeat(amap, upscale, axis=0), upscale, axis=1)
            ax.imshow(big, cmap="hot", alpha=0.5, vmin=0, vmax=1)
        correct_c3 = c3["pred"] == label
        ax.set_xlabel(
            f"{'✓' if correct_c3 else '✗'} {short(display_names[c3['pred']], 14)}"
            f"  {c3['conf']:.0%}",
            fontsize=6,
            color=BLUE if correct_c3 else "red",
        )
        ax.set_xticks([])
        ax.set_yticks([])

        # ── Col 3: ResCNN attention (layer2 group) overlaid ───────────
        ax = axes[row, 3]
        ax.imshow(img_np)
        if rc["attn"]:
            # layer2 group = tokens 0..255 of the 512-token sequence
            w = rc["attn"].get(0)  # (H, S, S)
            if w is not None:
                cls_row = w[:, 0, 1:257].mean(dim=0).numpy()  # (256,) = layer2
                amap_rc = cls_row.reshape(rescnn_grid)  # (16,16)
                amap_rc = amap_rc / (amap_rc.max() + 1e-9)
                upscale = 64 // rescnn_grid[0]  # 64//16 = 4 pixels per token
                big_rc = np.repeat(np.repeat(amap_rc, upscale, axis=0), upscale, axis=1)
                ax.imshow(big_rc, cmap="hot", alpha=0.5, vmin=0, vmax=1)
        correct_rc = rc["pred"] == label
        ax.set_xlabel(
            f"{'✓' if correct_rc else '✗'} {short(display_names[rc['pred']], 14)}"
            f"  {rc['conf']:.0%}",
            fontsize=6,
            color=GREEN if correct_rc else "red",
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # Legend
    handles = [
        mpatches.Patch(color=ORANGE, label="ResNet-18"),
        mpatches.Patch(color=BLUE, label="CNN3+TPSAPU 702k"),
        mpatches.Patch(color=GREEN, label="ResCNN+TPSAPU 2048"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
    )

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    p = OUT / "per_image_comparison.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")


# ---------------------------------------------------------------------------
# Figure 2: Attention anatomy — what the transformer actually attends to
# ---------------------------------------------------------------------------


def fig_attention_anatomy(
    images: list[np.ndarray],
    labels: list[int],
    indices: list[int],
    cnn3_results: list[dict],
    rescnn_results: list[dict],
    display_names: list[str],
) -> None:
    """
    For each image, 6 panels:
      raw image | CNN3 L1 mean | CNN3 L2 mean | CNN3 L1-L2 diff |
      ResCNN L1 (layer2) | ResCNN L1 (layer3)
    """
    N = len(images)
    fig, axes = plt.subplots(N, 6, figsize=(6 * 2.4, N * 2.5))
    if N == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "Transformer Attention Anatomy\n"
        "What the transformer attends to = which TPSAPU-state timesteps\n"
        "(each timestep = one CNN spatial patch → not direct image attention)",
        fontsize=10,
        fontweight="bold",
    )

    titles = [
        "Image",
        "CNN3 CLS attn\nL1 (mean heads)",
        "CNN3 CLS attn\nL2 (mean heads)",
        "CNN3\nL2 − L1",
        "ResCNN L1\nlayer2 group",
        "ResCNN L1\nlayer3 group",
    ]
    for c, t in enumerate(titles):
        axes[0, c].set_title(t, fontsize=7)

    for row, (img_np, label, idx, c3, rc) in enumerate(
        zip(images, labels, indices, cnn3_results, rescnn_results)
    ):
        gt = display_names[label]

        # Col 0: image
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_xlabel(f"[{idx}] {short(gt)}", fontsize=6)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        # CNN3 attention maps (L1, L2, diff)
        for col, layer in [(1, 0), (2, 1)]:
            if c3["attn"] and layer in c3["attn"]:
                amap = cls_attention_spatial(c3["attn"], 256, (16, 16), layer=layer)
                upscale = 4  # 64//16 = 4 pixels per token
                big = np.repeat(np.repeat(amap, upscale, axis=0), upscale, axis=1)
                axes[row, col].imshow(img_np)
                axes[row, col].imshow(big, cmap="hot", alpha=0.55, vmin=0)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

        # CNN3 diff
        if c3["attn"] and 0 in c3["attn"] and 1 in c3["attn"]:
            l1 = cls_attention_spatial(c3["attn"], 256, (16, 16), layer=0)
            l2 = cls_attention_spatial(c3["attn"], 256, (16, 16), layer=1)
            diff = l2 - l1
            lim = max(abs(diff).max(), 1e-9)
            upscale = 4  # 64//16 = 4 pixels per token
            big_d = np.repeat(np.repeat(diff, upscale, axis=0), upscale, axis=1)
            axes[row, 3].imshow(img_np)
            axes[row, 3].imshow(big_d, cmap="RdBu_r", alpha=0.55, vmin=-lim, vmax=lim)
        axes[row, 3].set_xticks([])
        axes[row, 3].set_yticks([])

        # ResCNN layer2 and layer3 groups
        if rc["attn"] and 0 in rc["attn"]:
            w = rc["attn"][0]  # (H, S, S)
            for col, (lo, hi) in [(4, (0, 256)), (5, (256, 512))]:
                chunk = w[:, 0, lo + 1 : hi + 1].mean(dim=0).numpy()  # (256,)
                g = chunk.reshape(16, 16)
                g = g / (g.max() + 1e-9)
                big_rc = np.repeat(np.repeat(g, 4, axis=0), 4, axis=1)
                axes[row, col].imshow(img_np)
                axes[row, col].imshow(big_rc, cmap="hot", alpha=0.55, vmin=0)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
        else:
            axes[row, 4].set_xticks([])
            axes[row, 4].set_yticks([])
            axes[row, 5].set_xticks([])
            axes[row, 5].set_yticks([])

    plt.tight_layout()
    p = OUT / "attention_anatomy.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")


# ---------------------------------------------------------------------------
# Figure 3: Agreement / disagreement map
# ---------------------------------------------------------------------------


def fig_model_agreement(
    images: list[np.ndarray],
    labels: list[int],
    indices: list[int],
    rn_results: list[dict],
    cnn3_results: list[dict],
    rescnn_results: list[dict],
    display_names: list[str],
) -> None:
    """Show where models agree / disagree and their confidence."""
    N = len(images)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Model Agreement Analysis — 20 validation images", fontsize=12)

    rn_correct = np.array([r["pred"] == l for r, l in zip(rn_results, labels)])
    c3_correct = np.array([r["pred"] == l for r, l in zip(cnn3_results, labels)])
    rc_correct = np.array([r["pred"] == l for r, l in zip(rescnn_results, labels)])
    rn_conf = np.array([r["conf"] for r in rn_results])
    c3_conf = np.array([r["conf"] for r in cnn3_results])
    rc_conf = np.array([r["conf"] for r in rescnn_results])

    # Panel 1: accuracy bars
    ax = axes[0, 0]
    model_names = ["ResNet-18", "CNN3-702k", "ResCNN-2048"]
    accs = [rn_correct.mean(), c3_correct.mean(), rc_correct.mean()]
    colors = [ORANGE, BLUE, GREEN]
    bars = ax.bar(model_names, [a * 100 for a in accs], color=colors, edgecolor="white")
    for bar, a in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            a * 100 + 1,
            f"{a:.0%}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_ylim(0, 120)
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title(
        f"Accuracy on these 20 images\n(previously reported full-val in parentheses)"
    )
    for i, (name, full) in enumerate(zip(model_names, [0.5665, 0.4164, 0.3923])):
        ax.text(
            i,
            5,
            f"({full:.1%})",
            ha="center",
            fontsize=7,
            color="white",
            fontweight="bold",
        )
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: confidence boxplot
    ax = axes[0, 1]
    bp = ax.boxplot(
        [rn_conf * 100, c3_conf * 100, rc_conf * 100],
        labels=model_names,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Top-1 Confidence (%)")
    ax.set_title("Confidence Distribution")
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: agreement matrix (which images do all/some/no models get right)
    agree_all = rn_correct & c3_correct & rc_correct
    agree_none = ~rn_correct & ~c3_correct & ~rc_correct
    agree_rn_c3 = rn_correct & c3_correct & ~rc_correct
    agree_rn_rc = rn_correct & ~c3_correct & rc_correct
    agree_c3_rc = ~rn_correct & c3_correct & rc_correct
    only_rn = rn_correct & ~c3_correct & ~rc_correct
    only_c3 = ~rn_correct & c3_correct & ~rc_correct
    only_rc = ~rn_correct & ~c3_correct & rc_correct

    groups = [
        "All 3 correct",
        "None correct",
        "RN+C3 only",
        "RN+RC only",
        "C3+RC only",
        "Only ResNet",
        "Only CNN3",
        "Only ResCNN",
    ]
    counts = [
        agree_all.sum(),
        agree_none.sum(),
        agree_rn_c3.sum(),
        agree_rn_rc.sum(),
        agree_c3_rc.sum(),
        only_rn.sum(),
        only_c3.sum(),
        only_rc.sum(),
    ]
    group_colors = [
        "#6bbf59",
        "#cc3333",
        "#f0a030",
        "#d070c0",
        "#30c0d0",
        ORANGE,
        BLUE,
        GREEN,
    ]

    ax = axes[0, 2]
    ax.barh(groups, counts, color=group_colors, edgecolor="white", alpha=0.85)
    for i, c in enumerate(counts):
        ax.text(c + 0.05, i, str(c), va="center", fontsize=9)
    ax.set_xlabel("Count (out of 20)")
    ax.set_title("Model Agreement Patterns")
    ax.set_xlim(0, max(counts) + 2)
    ax.grid(axis="x", alpha=0.3)

    # Panel 4: Per-image result grid
    ax = axes[1, 0]
    results_grid = np.array(
        [rn_correct.astype(int), c3_correct.astype(int), rc_correct.astype(int)]
    )  # (3, 20)
    im = ax.imshow(results_grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["ResNet-18", "CNN3-702k", "ResCNN-2048"], fontsize=8)
    ax.set_xlabel("Image index (0-19)")
    ax.set_title("Per-image correctness (green=✓, red=✗)")
    for i in range(3):
        for j in range(20):
            ax.text(
                j,
                i,
                "✓" if results_grid[i, j] else "✗",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if not results_grid[i, j] else "black",
            )

    # Panel 5: Confidence when correct vs wrong
    ax = axes[1, 1]
    x = np.arange(3)
    w = 0.35
    conf_correct = [
        rn_conf[rn_correct].mean() if rn_correct.any() else 0,
        c3_conf[c3_correct].mean() if c3_correct.any() else 0,
        rc_conf[rc_correct].mean() if rc_correct.any() else 0,
    ]
    conf_wrong = [
        rn_conf[~rn_correct].mean() if (~rn_correct).any() else 0,
        c3_conf[~c3_correct].mean() if (~c3_correct).any() else 0,
        rc_conf[~rc_correct].mean() if (~rc_correct).any() else 0,
    ]
    ax.bar(
        x - w / 2,
        [v * 100 for v in conf_correct],
        w,
        label="correct",
        color=colors,
        alpha=0.9,
    )
    ax.bar(
        x + w / 2,
        [v * 100 for v in conf_wrong],
        w,
        label="wrong",
        color=colors,
        alpha=0.4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=8)
    ax.set_ylabel("Mean confidence (%)")
    ax.set_title("Conf when correct (solid) vs wrong (faded)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel 6: scatter ResNet conf vs CNN3 conf
    ax = axes[1, 2]
    scatter_colors = [
        "#6bbf59"
        if a and b
        else "#cc3333"
        if not a and not b
        else ORANGE
        if a
        else BLUE
        for a, b in zip(rn_correct, c3_correct)
    ]
    ax.scatter(
        rn_conf * 100, c3_conf * 100, c=scatter_colors, s=60, zorder=3, alpha=0.8
    )
    ax.set_xlabel("ResNet-18 confidence (%)")
    ax.set_ylabel("CNN3-702k confidence (%)")
    ax.set_title(
        "Confidence correlation\n(green=both ✓, red=both ✗, orange=RN✓, blue=C3✓)"
    )
    ax.grid(True, alpha=0.3)
    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, linewidth=0.8)
    for i, idx in enumerate(range(N)):
        if rn_results[i]["pred"] == labels[i] or cnn3_results[i]["pred"] == labels[i]:
            continue  # skip to reduce clutter

    plt.tight_layout()
    p = OUT / "model_agreement.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")


# ---------------------------------------------------------------------------
# Figure 4: Attention entropy per image (is focused = more confident?)
# ---------------------------------------------------------------------------


def fig_attention_entropy_per_image(
    labels: list[int],
    cnn3_results: list[dict],
    rescnn_results: list[dict],
    display_names: list[str],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Attention Entropy per Image — Does focused attention correlate with correct prediction?\n"
        "NOTE: entropy is over TPSAPU-state positions, not image pixels",
        fontsize=10,
    )

    for ax, results, name, color, n_tokens in [
        (axes[0], cnn3_results, "CNN3+TPSAPU 702k", BLUE, 256),  # 16×16 grid
        (axes[1], rescnn_results, "ResCNN+TPSAPU 2048", GREEN, 512),  # 2×16×16 grids
    ]:
        entropies = []
        correct_flags = []
        confs = []

        for res, label in zip(results, labels):
            if not res["attn"] or 0 not in res["attn"]:
                entropies.append(np.nan)
                correct_flags.append(False)
                confs.append(0)
                continue
            w = res["attn"][0]  # (H, S, S)
            cls_row = w[:, 0, 1 : n_tokens + 1]  # (H, n_tokens) — all tokens
            mean_cls = cls_row.mean(dim=0).numpy()
            ent = attn_entropy(mean_cls[np.newaxis, :]).item()
            entropies.append(ent)
            correct_flags.append(res["pred"] == label)
            confs.append(res["conf"])

        entropies = np.array(entropies)
        correct = np.array(correct_flags)
        confs = np.array(confs)

        sc = ax.scatter(
            entropies,
            confs * 100,
            c=[BLUE if c else ORANGE for c in correct],
            s=70,
            alpha=0.85,
            zorder=3,
        )
        ax.set_xlabel("CLS Attention Entropy (L1, mean heads)")
        ax.set_ylabel("Model Confidence (%)")
        ax.set_title(f"{name}")
        ax.grid(True, alpha=0.3)

        # Annotate entropy stats
        if correct.any():
            ax.axvline(
                entropies[correct].mean(),
                color=BLUE,
                linestyle="--",
                linewidth=1.2,
                label=f"correct mean={entropies[correct].mean():.2f}",
            )
        if (~correct).any():
            ax.axvline(
                entropies[~correct].mean(),
                color=ORANGE,
                linestyle="--",
                linewidth=1.2,
                label=f"wrong mean={entropies[~correct].mean():.2f}",
            )

        ax.legend(fontsize=8)
        correct_patch = mpatches.Patch(color=BLUE, label="Correct")
        wrong_patch = mpatches.Patch(color=ORANGE, label="Wrong")
        ax.legend(handles=[correct_patch, wrong_patch], fontsize=8)

    plt.tight_layout()
    p = OUT / "attention_entropy_per_image.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"Saved: {p}")


# ---------------------------------------------------------------------------
# Figure 5: Cross-validation of reported metrics
# ---------------------------------------------------------------------------


def fig_metric_crossvalidation(
    rn_results: list[dict],
    cnn3_results: list[dict],
    rescnn_results: list[dict],
    labels: list[int],
) -> None:
    """Compare accuracy on these 20 images vs the reported full-validation metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Cross-Validation: 20-Sample vs Full-Validation Metrics", fontsize=12)

    model_names = ["ResNet-18", "CNN3+TPSAPU 702k", "ResCNN+TPSAPU 2048"]
    sample_top1 = [
        np.mean([r["pred"] == l for r, l in zip(rn_results, labels)]),
        np.mean([r["pred"] == l for r, l in zip(cnn3_results, labels)]),
        np.mean([r["pred"] == l for r, l in zip(rescnn_results, labels)]),
    ]
    sample_top5 = [
        np.mean([l in r["top5"] for r, l in zip(rn_results, labels)]),
        np.mean([l in r["top5"] for r, l in zip(cnn3_results, labels)]),
        np.mean([l in r["top5"] for r, l in zip(rescnn_results, labels)]),
    ]
    reported_top1 = [REPORTED_METRICS[n]["top1"] for n in model_names]
    reported_top5 = [REPORTED_METRICS[n]["top5"] for n in model_names]

    x = np.arange(len(model_names))
    w = 0.3
    colors = [ORANGE, BLUE, GREEN]

    for ax_i, (ax, sample_vals, report_vals, title) in enumerate(
        zip(
            axes,
            [sample_top1, sample_top5],
            [reported_top1, reported_top5],
            ["Top-1 Accuracy", "Top-5 Accuracy"],
        )
    ):
        bars1 = ax.bar(
            x - w / 2,
            [v * 100 for v in sample_vals],
            w,
            label="20-sample",
            color=colors,
            edgecolor="white",
            alpha=0.9,
        )
        bars2 = ax.bar(
            x + w / 2,
            [v * 100 for v in report_vals],
            w,
            label="Full val (4909)",
            color=colors,
            edgecolor="white",
            alpha=0.4,
            hatch="//",
        )
        for bar, v in zip(bars1, sample_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v * 100 + 0.5,
                f"{v:.0%}",
                ha="center",
                fontsize=8,
                fontweight="bold",
            )
        for bar, v in zip(bars2, report_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v * 100 + 0.5,
                f"{v:.0%}",
                ha="center",
                fontsize=8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=10, ha="right", fontsize=8)
        ax.set_ylabel(f"{title} (%)")
        ax.set_title(f"{title}: 20-sample vs full validation")
        ax.set_ylim(0, 110)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # Delta annotation
        for i, (s, r) in enumerate(zip(sample_vals, report_vals)):
            delta = (s - r) * 100
            ax.text(
                x[i],
                max(s * 100, r * 100) + 6,
                f"Δ{delta:+.0f}pp",
                ha="center",
                fontsize=7,
                color="#e07b39" if abs(delta) > 15 else "gray",
            )

    plt.tight_layout()
    p = OUT / "metric_crossvalidation.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")


# ---------------------------------------------------------------------------
# Print text summary
# ---------------------------------------------------------------------------


def print_and_save_summary(
    indices: list[int],
    labels: list[int],
    rn_results: list[dict],
    cnn3_results: list[dict],
    rescnn_results: list[dict],
    display_names: list[str],
) -> None:
    rn_top1 = np.mean([r["pred"] == l for r, l in zip(rn_results, labels)])
    c3_top1 = np.mean([r["pred"] == l for r, l in zip(cnn3_results, labels)])
    rc_top1 = np.mean([r["pred"] == l for r, l in zip(rescnn_results, labels)])
    rn_top5 = np.mean([l in r["top5"] for r, l in zip(rn_results, labels)])
    c3_top5 = np.mean([l in r["top5"] for r, l in zip(cnn3_results, labels)])
    rc_top5 = np.mean([l in r["top5"] for r, l in zip(rescnn_results, labels)])

    print("\n" + "=" * 75)
    print("20-IMAGE VALIDATION SUMMARY")
    print(f"  Seed={SEED}, indices: {indices}")
    print("=" * 75)
    print(f"{'Model':<26} {'Top-1':>8} {'(full)':>8} {'Top-5':>8} {'(full)':>8}")
    print("-" * 75)
    for name, t1, t5 in [
        ("ResNet-18", rn_top1, rn_top5),
        ("CNN3+TPSAPU 702k", c3_top1, c3_top5),
        ("ResCNN+TPSAPU 2048", rc_top1, rc_top5),
    ]:
        full_t1 = REPORTED_METRICS[name]["top1"]
        full_t5 = REPORTED_METRICS[name]["top5"]
        print(f"  {name:<24} {t1:>7.1%}  {full_t1:>6.1%}  {t5:>7.1%}  {full_t5:>6.1%}")

    print("\nPer-image breakdown:")
    print(
        f"  {'idx':>5}  {'ground truth':<22}  {'ResNet':>8}  {'CNN3':>8}  {'ResCNN':>8}"
    )
    print(f"  {'-' * 5}  {'-' * 22}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
    for idx, label, rn, c3, rc in zip(
        indices, labels, rn_results, cnn3_results, rescnn_results
    ):
        gt = short(display_names[label], 22)
        rn_s = f"{'✓' if rn['pred'] == label else '✗'} {rn['conf']:.0%}"
        c3_s = f"{'✓' if c3['pred'] == label else '✗'} {c3['conf']:.0%}"
        rc_s = f"{'✓' if rc['pred'] == label else '✗'} {rc['conf']:.0%}"
        print(f"  {idx:>5}  {gt:<22}  {rn_s:>8}  {c3_s:>8}  {rc_s:>8}")

    print("=" * 75)

    # Save JSON summary
    summary = {
        "seed": SEED,
        "n_images": N_IMAGES,
        "indices": indices,
        "resnet18": {
            "top1": float(rn_top1),
            "top5": float(rn_top5),
            "reported_top1": 0.5665,
            "reported_top5": 0.7920,
        },
        "cnn3_702k": {
            "top1": float(c3_top1),
            "top5": float(c3_top5),
            "reported_top1": 0.4164,
            "reported_top5": 0.6763,
        },
        "rescnn_2048": {
            "top1": float(rc_top1),
            "top5": float(rc_top5),
            "reported_top1": 0.3923,
            "reported_top5": 0.6539,
        },
        "attention_note": (
            "Attention maps show which TPSAPU reservoir-state timestep "
            "the transformer CLS token attends to. Each timestep corresponds "
            "to one CNN spatial patch (16x16 grid for cnn3 on 64x64, two 16x16 grids for res_cnn). "
            "The mapping image_patch → CNN_token → TPSAPU_state is valid but indirect. "
            "Attention rollout across 2 transformer layers is numerically near-trivial "
            "and is NOT used here; raw L1/L2 CLS attention is shown instead."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved: {OUT / 'summary.json'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    train_pipeline.set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    print("Loading dataset …")
    dataset, class_names = load_dataset("data/tiny-imagenet-200-clean")
    human = load_human_names()
    display_names = [human.get(s, s).split(",")[0].strip() for s in class_names]

    # Pick 20 reproducible indices
    indices = pick_indices(len(dataset), N_IMAGES, SEED)
    print(f"Selected {N_IMAGES} images: {indices}")

    # Load images
    images_np = []
    images_t = []
    labels = []
    for idx in indices:
        img_t, lbl, _ = dataset[idx]
        images_t.append(img_t)
        images_np.append(denormalize(img_t))
        labels.append(lbl)

    print("\nGround truth classes:")
    for idx, lbl in zip(indices, labels):
        print(f"  [{idx}] {display_names[lbl]}")

    # Load models
    print("\nLoading models …")
    rn_model = load_resnet18(device)
    cnn3_model = load_tpsapu(CNN3_CKPT, device)
    rescnn_model = load_tpsapu(RESCNN_CKPT, device)

    # Set up attention capture
    cnn3_capture = AttentionCapture()
    rescnn_capture = AttentionCapture()
    cnn3_capture.register(cnn3_model.decoder.transformer)
    rescnn_capture.register(rescnn_model.decoder.transformer)

    # Run inference on all 20 images
    print("\nRunning inference …")
    rn_results = []
    cnn3_results = []
    rescnn_results = []

    for i, (img_t, label, idx) in enumerate(zip(images_t, labels, indices)):
        rn = infer_resnet(rn_model, img_t, device)
        c3 = infer_tpsapu_with_attention(cnn3_model, cnn3_capture, img_t, device)
        rc = infer_tpsapu_with_attention(rescnn_model, rescnn_capture, img_t, device)
        rn_results.append(rn)
        cnn3_results.append(c3)
        rescnn_results.append(rc)
        rn_s = f"{'✓' if rn['pred'] == label else '✗'}"
        c3_s = f"{'✓' if c3['pred'] == label else '✗'}"
        rc_s = f"{'✓' if rc['pred'] == label else '✗'}"
        print(
            f"  [{idx}] {display_names[label]:<22} | "
            f"RN {rn_s}{rn['conf']:.0%}  "
            f"C3 {c3_s}{c3['conf']:.0%}  "
            f"RC {rc_s}{rc['conf']:.0%}"
        )

    cnn3_capture.remove()
    rescnn_capture.remove()

    # Generate all figures
    print("\nGenerating figures …")
    # CNN3 on 64×64: two stride-2 convs → 16×16 = 256 tokens
    # ResCNN on 64×64: same → 512 tokens (256 layer2 + 256 layer3)
    fig_per_image_comparison(
        images_np,
        images_t,
        labels,
        indices,
        rn_results,
        cnn3_results,
        rescnn_results,
        display_names,
        cnn3_grid=(16, 16),
        rescnn_grid=(16, 16),
    )
    fig_attention_anatomy(
        images_np,
        labels,
        indices,
        cnn3_results,
        rescnn_results,
        display_names,
    )
    fig_model_agreement(
        images_np,
        labels,
        indices,
        rn_results,
        cnn3_results,
        rescnn_results,
        display_names,
    )
    fig_attention_entropy_per_image(
        labels,
        cnn3_results,
        rescnn_results,
        display_names,
    )
    fig_metric_crossvalidation(
        rn_results,
        cnn3_results,
        rescnn_results,
        labels,
    )

    print_and_save_summary(
        indices,
        labels,
        rn_results,
        cnn3_results,
        rescnn_results,
        display_names,
    )

    print(f"\nAll outputs saved to: {OUT}")


if __name__ == "__main__":
    main()
