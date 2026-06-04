"""Visualize CNN3 tokens and TPSAPU voltages for Tiny ImageNet.

Example:
    python visualize_cnn3_voltages.py --checkpoint latest.pt --index 0

Outputs are written under ``visualizations/cnn3_voltages`` by default:
    - input_and_cnn3_tokens.png
    - cnn3_top_channels.png
    - tpsapu_spatial_voltage_maps.png
    - tpsapu_tau_traces.png
    - tpsapu_neuron_heatmaps.png
    - arrays.npz
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import train_pipeline


TINY_IMAGENET_MEAN = (0.485, 0.456, 0.406)
TINY_IMAGENET_STD = (0.229, 0.224, 0.225)


ARCHITECTURE_KEYS = {
    "dataset",
    "backbone",
    "encoder",
    "decoder",
    "pooling",
    "image_size",
    "in_channels",
    "num_classes",
    "embed_dim",
    "reservoir_dim",
    "taus",
    "input_hidden_dim",
    "cross_rank",
    "cross_gain",
    "patch_size",
    "encoder_hidden_dim",
    "cnn_channels",
    "lif_white_threshold",
    "encoder_dropout",
    "decoder_hidden_dim",
    "decoder_dropout",
    "decoder_transformer_layers",
    "decoder_transformer_heads",
    "decoder_transformer_ff_mult",
    "decoder_max_steps",
    "recurrent_drop",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/tpsapu_tiny_imagenet.pt",
        help="TPSAPU checkpoint. If missing, random weights are visualized.",
    )
    parser.add_argument("--data-dir", default="data/tiny-imagenet-200-clean")
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--image",
        default="",
        help="Optional local image path. Overrides --split/--index and has label -1.",
    )
    parser.add_argument("--output-dir", default="visualizations/cnn3_voltages")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-feature-channels", type=int, default=16)
    parser.add_argument(
        "--selected-taus",
        default="",
        help=(
            "Comma-separated tau indices for neuron heatmaps. "
            "Default shows first, middle, and last tau."
        ),
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        default=True,
        help="Use only cached Hugging Face data by default.",
    )
    parser.add_argument(
        "--download",
        action="store_false",
        dest="no_download",
        help="Allow Hugging Face dataset download.",
    )
    return parser.parse_args()


def default_tiny_imagenet_architecture() -> argparse.Namespace:
    return argparse.Namespace(
        dataset="tiny_imagenet",
        backbone="tpsapu",
        encoder="cnn3",
        decoder="membrane_transformer",
        pooling="last",
        image_size=64,
        in_channels=3,
        num_classes=200,
        embed_dim=128,
        reservoir_dim=64,
        taus="1.1,2.0,4.0,8.0,16.0,32.0,64.0,128.0",
        input_hidden_dim=0,
        cross_rank=16,
        cross_gain=0.1,
        patch_size=8,
        encoder_hidden_dim=0,
        cnn_channels=64,
        lif_white_threshold=0.6,
        encoder_dropout=0.05,
        decoder_hidden_dim=128,
        decoder_dropout=0.1,
        decoder_transformer_layers=2,
        decoder_transformer_heads=4,
        decoder_transformer_ff_mult=4.0,
        decoder_max_steps=256,
        recurrent_drop=0.1,
    )


def architecture_from_checkpoint(
    checkpoint: dict[str, object] | None,
) -> argparse.Namespace:
    architecture = vars(default_tiny_imagenet_architecture()).copy()
    if checkpoint is not None and isinstance(checkpoint.get("args"), dict):
        saved_args = checkpoint["args"]
        architecture.update(
            {key: saved_args[key] for key in ARCHITECTURE_KEYS if key in saved_args}
        )
    return argparse.Namespace(**architecture)


def load_checkpoint(path: str, device: torch.device) -> dict[str, object] | None:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return None
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def image_transform(image_size: int) -> transforms.Compose:
    resize = max(image_size, int(round(image_size * 256 / 224)))
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
        ]
    )


def load_image_from_path(path: str, image_size: int) -> tuple[torch.Tensor, int, str]:
    image = Image.open(path).convert("RGB")
    tensor = image_transform(image_size)(image).unsqueeze(0)
    return tensor, -1, Path(path).stem


def load_tiny_imagenet_sample(
    args: argparse.Namespace,
    architecture_args: argparse.Namespace,
) -> tuple[torch.Tensor, int, str]:
    dataset = train_pipeline.load_hf_split(
        "slegroux/tiny-imagenet-200-clean",
        split=args.split,
        cache_dir=args.data_dir,
        no_download=args.no_download,
    )
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"--index must be between 0 and {len(dataset) - 1}.")

    item = dataset[args.index]
    image = item["image"]
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = image_transform(architecture_args.image_size)(image).unsqueeze(0)
    return tensor, int(item["label"]), f"{args.split}_{args.index}"


def denormalize_tiny_imagenet(image: torch.Tensor) -> np.ndarray:
    image_np = image.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    mean = np.asarray(TINY_IMAGENET_MEAN).reshape(1, 1, 3)
    std = np.asarray(TINY_IMAGENET_STD).reshape(1, 1, 3)
    return np.clip(image_np * std + mean, 0.0, 1.0)


def infer_square_grid(steps: int) -> int | None:
    side = int(math.sqrt(steps))
    if side * side == steps:
        return side
    return None


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    low = float(np.percentile(values, 1))
    high = float(np.percentile(values, 99))
    if high <= low:
        low = float(values.min())
        high = float(values.max())
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


@torch.no_grad()
def run_visualization_forward(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    model.eval()
    image = image.to(device)
    tokens = model.encoder(image)
    states = model.backbone.forward_states(tokens, reset_state=True)
    logits = model(image)

    result = {
        "tokens": tokens.squeeze(0).detach().cpu(),
        "membranes": states["membrane"].squeeze(0).detach().cpu(),
        "spikes": states["spike"].squeeze(0).detach().cpu(),
        "dynamics": states["dynamics"].squeeze(0).detach().cpu(),
        "spike_history": states["spike_history"].squeeze(0).detach().cpu(),
        "logits": logits.squeeze(0).detach().cpu(),
    }

    encoder = model.encoder
    if hasattr(encoder, "net"):
        result["cnn_features"] = encoder.net(image).squeeze(0).detach().cpu()
    return result


def save_input_and_tokens(
    path: Path,
    image: np.ndarray,
    tokens: torch.Tensor,
    cnn_features: torch.Tensor | None,
    label: int,
) -> None:
    steps = tokens.size(0)
    grid = infer_square_grid(steps)
    token_norms = tokens.norm(dim=-1).numpy()

    if grid is None:
        token_view = token_norms.reshape(1, -1)
    else:
        token_view = token_norms.reshape(grid, grid)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title(f"input label={label}")
    axes[0].axis("off")

    im = axes[1].imshow(token_view, cmap="magma")
    axes[1].set_title("CNN3 token L2 norm")
    axes[1].set_xlabel("token x")
    axes[1].set_ylabel("token y")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    if cnn_features is None:
        axes[2].axis("off")
        axes[2].set_title("CNN feature map unavailable")
    else:
        mean_abs = cnn_features.abs().mean(dim=0).numpy()
        im = axes[2].imshow(mean_abs, cmap="viridis")
        axes[2].set_title("CNN3 mean abs channel activation")
        axes[2].set_xlabel("feature x")
        axes[2].set_ylabel("feature y")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_cnn_feature_channels(
    path: Path,
    cnn_features: torch.Tensor | None,
    channel_count: int,
) -> None:
    if cnn_features is None:
        return

    channel_count = min(max(1, channel_count), cnn_features.size(0))
    scores = cnn_features.abs().mean(dim=(1, 2))
    selected = torch.topk(scores, k=channel_count).indices.tolist()
    cols = min(4, channel_count)
    rows = int(math.ceil(channel_count / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 2.6, rows * 2.4),
        squeeze=False,
    )
    for panel_index, axis in enumerate(axes.flat):
        axis.axis("off")
        if panel_index >= channel_count:
            continue
        channel_index = selected[panel_index]
        activation = cnn_features[channel_index].numpy()
        axis.imshow(normalize_map(activation), cmap="viridis", vmin=0.0, vmax=1.0)
        axis.set_title(f"channel {channel_index}", fontsize=9)

    fig.suptitle("CNN3 strongest feature channels", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def tau_labels(taus: str) -> list[str]:
    return [part.strip() for part in taus.split(",") if part.strip()]


def reshape_state_by_tau(
    state: torch.Tensor,
    *,
    reservoir_dim: int,
    taus: str,
) -> torch.Tensor:
    labels = tau_labels(taus)
    expected = reservoir_dim * len(labels)
    if state.size(-1) != expected:
        raise ValueError(
            f"State has {state.size(-1)} features, expected {expected} from "
            f"reservoir_dim={reservoir_dim} and {len(labels)} taus."
        )
    return state.reshape(state.size(0), len(labels), reservoir_dim)


def save_spatial_voltage_maps(
    path: Path,
    membranes: torch.Tensor,
    spikes: torch.Tensor,
    *,
    reservoir_dim: int,
    taus: str,
) -> None:
    labels = tau_labels(taus)
    grid = infer_square_grid(membranes.size(0))
    membrane_by_tau = reshape_state_by_tau(
        membranes,
        reservoir_dim=reservoir_dim,
        taus=taus,
    )
    spike_by_tau = reshape_state_by_tau(spikes, reservoir_dim=reservoir_dim, taus=taus)

    fig, axes = plt.subplots(
        2,
        len(labels),
        figsize=(max(12, len(labels) * 2.2), 5.2),
        squeeze=False,
    )

    for tau_index, label in enumerate(labels):
        membrane_values = membrane_by_tau[:, tau_index, :].mean(dim=-1).numpy()
        spike_values = spike_by_tau[:, tau_index, :].mean(dim=-1).numpy()
        if grid is None:
            membrane_view = membrane_values.reshape(1, -1)
            spike_view = spike_values.reshape(1, -1)
        else:
            membrane_view = membrane_values.reshape(grid, grid)
            spike_view = spike_values.reshape(grid, grid)

        im0 = axes[0, tau_index].imshow(membrane_view, cmap="coolwarm")
        axes[0, tau_index].set_title(f"tau={label}")
        axes[0, tau_index].axis("off")
        fig.colorbar(im0, ax=axes[0, tau_index], fraction=0.046, pad=0.04)

        im1 = axes[1, tau_index].imshow(spike_view, cmap="magma", vmin=0.0, vmax=1.0)
        axes[1, tau_index].axis("off")
        fig.colorbar(im1, ax=axes[1, tau_index], fraction=0.046, pad=0.04)

    axes[0, 0].set_ylabel("mean membrane voltage")
    axes[1, 0].set_ylabel("mean spike rate")
    fig.suptitle("TPSAPU state projected back to CNN3 token grid", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_tau_traces(
    path: Path,
    membranes: torch.Tensor,
    spikes: torch.Tensor,
    *,
    reservoir_dim: int,
    taus: str,
) -> None:
    labels = tau_labels(taus)
    steps = np.arange(1, membranes.size(0) + 1)
    membrane_by_tau = reshape_state_by_tau(
        membranes,
        reservoir_dim=reservoir_dim,
        taus=taus,
    )
    spike_by_tau = reshape_state_by_tau(spikes, reservoir_dim=reservoir_dim, taus=taus)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for tau_index, label in enumerate(labels):
        membrane_trace = membrane_by_tau[:, tau_index, :].abs().mean(dim=-1).numpy()
        spike_trace = spike_by_tau[:, tau_index, :].mean(dim=-1).numpy()
        axes[0].plot(steps, membrane_trace, label=f"tau={label}", linewidth=1.5)
        axes[1].plot(steps, spike_trace, label=f"tau={label}", linewidth=1.5)

    axes[0].set_ylabel("mean abs voltage")
    axes[0].set_title("Membrane voltage by token step")
    axes[0].legend(ncol=min(4, len(labels)), fontsize=8)
    axes[1].set_xlabel("CNN3 token step")
    axes[1].set_ylabel("mean spike rate")
    axes[1].set_title("Spike activity by token step")
    axes[1].legend(ncol=min(4, len(labels)), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def selected_tau_indices(selected_taus: str, tau_count: int) -> list[int]:
    if selected_taus.strip():
        indices = [int(part.strip()) for part in selected_taus.split(",") if part.strip()]
        return [index for index in indices if 0 <= index < tau_count]
    default = sorted({0, tau_count // 2, tau_count - 1})
    return default


def save_neuron_heatmaps(
    path: Path,
    membranes: torch.Tensor,
    spikes: torch.Tensor,
    *,
    reservoir_dim: int,
    taus: str,
    selected_taus: str,
) -> None:
    labels = tau_labels(taus)
    membrane_by_tau = reshape_state_by_tau(
        membranes,
        reservoir_dim=reservoir_dim,
        taus=taus,
    )
    spike_by_tau = reshape_state_by_tau(spikes, reservoir_dim=reservoir_dim, taus=taus)
    selected = selected_tau_indices(selected_taus, len(labels))
    if not selected:
        selected = [0]

    fig, axes = plt.subplots(
        len(selected),
        2,
        figsize=(12, 3.2 * len(selected)),
        squeeze=False,
    )
    for row, tau_index in enumerate(selected):
        membrane_view = membrane_by_tau[:, tau_index, :].transpose(0, 1).numpy()
        spike_view = spike_by_tau[:, tau_index, :].transpose(0, 1).numpy()

        im0 = axes[row, 0].imshow(membrane_view, aspect="auto", cmap="coolwarm")
        axes[row, 0].set_title(f"membrane voltage, tau={labels[tau_index]}")
        axes[row, 0].set_ylabel("reservoir neuron")
        axes[row, 0].set_xlabel("token step")
        fig.colorbar(im0, ax=axes[row, 0], fraction=0.024, pad=0.02)

        im1 = axes[row, 1].imshow(
            spike_view,
            aspect="auto",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        axes[row, 1].set_title(f"spikes, tau={labels[tau_index]}")
        axes[row, 1].set_ylabel("reservoir neuron")
        axes[row, 1].set_xlabel("token step")
        fig.colorbar(im1, ax=axes[row, 1], fraction=0.024, pad=0.02)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_arrays(
    path: Path,
    result: dict[str, torch.Tensor],
    label: int,
    probabilities: torch.Tensor,
) -> None:
    arrays = {
        "label": np.asarray(label, dtype=np.int64),
        "probabilities": probabilities.numpy(),
    }
    for key, value in result.items():
        arrays[key] = value.numpy()
    np.savez_compressed(path, **arrays)


def print_prediction_summary(logits: torch.Tensor, label: int) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    topk = torch.topk(probabilities, k=min(5, probabilities.numel()))
    print(f"Label: {label}")
    print(f"Prediction: {int(topk.indices[0])} ({float(topk.values[0]):.2%})")
    print("Top probabilities:")
    for class_index, probability in zip(topk.indices.tolist(), topk.values.tolist()):
        print(f"  {class_index}: {probability:.2%}")
    return probabilities


def main() -> None:
    args = parse_args()
    train_pipeline.set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint = load_checkpoint(args.checkpoint, device)
    architecture_args = architecture_from_checkpoint(checkpoint)
    if architecture_args.encoder != "cnn3":
        print(
            f"Warning: checkpoint/config uses encoder={architecture_args.encoder}; "
            "CNN3 feature-map plots require an encoder with .net."
        )

    model = train_pipeline.build_model(architecture_args).to(device)
    if checkpoint is None:
        print(f"No checkpoint found at {args.checkpoint}; using random weights.")
    else:
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded checkpoint: {args.checkpoint}")

    if args.image:
        image, label, sample_name = load_image_from_path(
            args.image,
            architecture_args.image_size,
        )
    else:
        image, label, sample_name = load_tiny_imagenet_sample(args, architecture_args)

    result = run_visualization_forward(model, image, device)
    probabilities = print_prediction_summary(result["logits"], label)

    output_dir = Path(args.output_dir) / sample_name
    output_dir.mkdir(parents=True, exist_ok=True)
    input_image = denormalize_tiny_imagenet(image)
    cnn_features = result.get("cnn_features")

    save_input_and_tokens(
        output_dir / "input_and_cnn3_tokens.png",
        input_image,
        result["tokens"],
        cnn_features,
        label,
    )
    save_cnn_feature_channels(
        output_dir / "cnn3_top_channels.png",
        cnn_features,
        args.num_feature_channels,
    )
    save_spatial_voltage_maps(
        output_dir / "tpsapu_spatial_voltage_maps.png",
        result["membranes"],
        result["spikes"],
        reservoir_dim=architecture_args.reservoir_dim,
        taus=architecture_args.taus,
    )
    save_tau_traces(
        output_dir / "tpsapu_tau_traces.png",
        result["membranes"],
        result["spikes"],
        reservoir_dim=architecture_args.reservoir_dim,
        taus=architecture_args.taus,
    )
    save_neuron_heatmaps(
        output_dir / "tpsapu_neuron_heatmaps.png",
        result["membranes"],
        result["spikes"],
        reservoir_dim=architecture_args.reservoir_dim,
        taus=architecture_args.taus,
        selected_taus=args.selected_taus,
    )
    save_arrays(output_dir / "arrays.npz", result, label, probabilities)

    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
