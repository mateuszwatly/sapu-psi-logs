"""
Inference and timeline visualization for encoder -> TPSAPU -> decoder models.

The visualization reveals the pixels consumed by each encoder token and shows
how the class prediction changes as more of the image is read.

Example:
    python inference.py --checkpoint checkpoints/tpsapu_mnist.pt --index 0 --gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

from train_pipeline import build_model, set_seed


ARCHITECTURE_KEYS = {
    "dataset",
    "encoder",
    "decoder",
    "pooling",
    "embed_dim",
    "reservoir_dim",
    "taus",
    "input_hidden_dim",
    "patch_size",
    "encoder_hidden_dim",
    "cnn_channels",
    "lif_white_threshold",
    "encoder_dropout",
    "decoder_hidden_dim",
    "decoder_dropout",
    "recurrent_drop",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/tpsapu_mnist.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output-dir", default="inference_outputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--gif", action="store_true")

    parser.add_argument(
        "--encoder",
        choices=[
            "checkpoint",
            "linear_patch",
            "mlp_patch",
            "lif_2x2",
            "cnn2",
            "cnn3",
            "res_cnn",
            "patch",
            "rows",
        ],
        default="checkpoint",
    )
    parser.add_argument(
        "--decoder",
        choices=[
            "checkpoint",
            "linear",
            "membrane_mlp",
            "spike_mlp",
            "both_mlp",
            "all_state_mlp",
            "lif_count",
            "mlp",
        ],
        default="checkpoint",
    )
    parser.add_argument("--pooling", choices=["checkpoint", "last", "mean"], default="checkpoint")
    parser.add_argument("--embed-dim", type=int, default=0)
    parser.add_argument("--reservoir-dim", type=int, default=0)
    parser.add_argument("--taus", default="")
    parser.add_argument("--input-hidden-dim", type=int, default=-1)
    parser.add_argument("--patch-size", type=int, default=0)
    parser.add_argument("--encoder-hidden-dim", type=int, default=-1)
    parser.add_argument("--cnn-channels", type=int, default=0)
    parser.add_argument("--lif-white-threshold", type=float, default=-1.0)
    parser.add_argument("--encoder-dropout", type=float, default=-1.0)
    parser.add_argument("--decoder-hidden-dim", type=int, default=0)
    parser.add_argument("--decoder-dropout", type=float, default=-1.0)
    parser.add_argument("--recurrent-drop", type=float, default=-1.0)
    return parser.parse_args()


def default_architecture() -> dict[str, object]:
    return {
        "dataset": "mnist",
        "encoder": "linear_patch",
        "decoder": "membrane_mlp",
        "pooling": "last",
        "embed_dim": 128,
        "reservoir_dim": 64,
        "taus": "1.1,8.0,64.0",
        "input_hidden_dim": 0,
        "patch_size": 7,
        "encoder_hidden_dim": 0,
        "cnn_channels": 64,
        "lif_white_threshold": 0.6,
        "encoder_dropout": 0.0,
        "decoder_hidden_dim": 128,
        "decoder_dropout": 0.0,
        "recurrent_drop": 0.0,
    }


def load_checkpoint(path: str, device: torch.device) -> dict[str, object] | None:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return None
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def build_architecture_args(
    cli_args: argparse.Namespace,
    checkpoint: dict[str, object] | None,
) -> argparse.Namespace:
    architecture = default_architecture()
    if checkpoint is not None and isinstance(checkpoint.get("args"), dict):
        saved_args = checkpoint["args"]
        architecture.update(
            {key: saved_args[key] for key in ARCHITECTURE_KEYS if key in saved_args}
        )

    if cli_args.encoder != "checkpoint":
        architecture["encoder"] = cli_args.encoder
    if cli_args.decoder != "checkpoint":
        architecture["decoder"] = cli_args.decoder
    if cli_args.pooling != "checkpoint":
        architecture["pooling"] = cli_args.pooling
    if cli_args.embed_dim > 0:
        architecture["embed_dim"] = cli_args.embed_dim
    if cli_args.reservoir_dim > 0:
        architecture["reservoir_dim"] = cli_args.reservoir_dim
    if cli_args.taus:
        architecture["taus"] = cli_args.taus
    if cli_args.input_hidden_dim >= 0:
        architecture["input_hidden_dim"] = cli_args.input_hidden_dim
    if cli_args.patch_size > 0:
        architecture["patch_size"] = cli_args.patch_size
    if cli_args.encoder_hidden_dim >= 0:
        architecture["encoder_hidden_dim"] = cli_args.encoder_hidden_dim
    if cli_args.cnn_channels > 0:
        architecture["cnn_channels"] = cli_args.cnn_channels
    if cli_args.lif_white_threshold >= 0:
        architecture["lif_white_threshold"] = cli_args.lif_white_threshold
    if cli_args.encoder_dropout >= 0:
        architecture["encoder_dropout"] = cli_args.encoder_dropout
    if cli_args.decoder_hidden_dim > 0:
        architecture["decoder_hidden_dim"] = cli_args.decoder_hidden_dim
    if cli_args.decoder_dropout >= 0:
        architecture["decoder_dropout"] = cli_args.decoder_dropout
    if cli_args.recurrent_drop >= 0:
        architecture["recurrent_drop"] = cli_args.recurrent_drop

    return argparse.Namespace(**architecture)


def load_mnist_image(args: argparse.Namespace) -> tuple[torch.Tensor, int]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    dataset = datasets.MNIST(
        args.data_dir,
        train=args.split == "train",
        download=not args.no_download,
        transform=transform,
    )
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"--index must be between 0 and {len(dataset) - 1}.")
    image, target = dataset[args.index]
    return image.unsqueeze(0), int(target)


def denormalize_mnist(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().squeeze().numpy()
    return np.clip(image * 0.3081 + 0.1307, 0.0, 1.0)


def reveal_masks(encoder_name: str, steps: int, patch_size: int) -> list[np.ndarray]:
    masks = []
    current = np.zeros((28, 28), dtype=np.float32)

    if encoder_name == "patch":
        encoder_name = "linear_patch"

    if encoder_name == "rows":
        for step in range(steps):
            current = current.copy()
            current[step, :] = 1.0
            masks.append(current)
        return masks

    if encoder_name == "lif_2x2":
        return reveal_grid_masks(steps, grid_size=14, cell_size=2)

    if encoder_name in {"cnn2", "cnn3", "res_cnn"}:
        return reveal_grid_masks(steps, grid_size=7, cell_size=4)

    patches_per_side = 28 // patch_size
    for step in range(steps):
        row = step // patches_per_side
        col = step % patches_per_side
        current = current.copy()
        y0 = row * patch_size
        x0 = col * patch_size
        current[y0 : y0 + patch_size, x0 : x0 + patch_size] = 1.0
        masks.append(current)
    return masks


def reveal_grid_masks(steps: int, *, grid_size: int, cell_size: int) -> list[np.ndarray]:
    masks = []
    current = np.zeros((28, 28), dtype=np.float32)
    for step in range(steps):
        cell = step % (grid_size * grid_size)
        row = cell // grid_size
        col = cell % grid_size
        current = current.copy()
        y0 = row * cell_size
        x0 = col * cell_size
        current[y0 : y0 + cell_size, x0 : x0 + cell_size] = 1.0
        masks.append(current)
    return masks


@torch.no_grad()
def run_temporal_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    image = image.to(device)
    tokens = model.encoder(image)
    states = model.backbone.forward_states(tokens)
    membranes = states["membrane"]
    spikes = states["spike"]

    if model.decoder.input_state == "membrane":
        sequence_features = membranes
    elif model.decoder.input_state == "spike":
        sequence_features = spikes
    elif model.decoder.input_state == "both":
        sequence_features = torch.cat([membranes, spikes], dim=-1)
    elif model.decoder.input_state == "all":
        sequence_features = torch.cat(
            [
                states["membrane"],
                states["spike"],
                states["dynamics"],
                states["spike_history"],
            ],
            dim=-1,
        )
    else:
        raise ValueError(f"Unsupported decoder input_state: {model.decoder.input_state}")

    if model.decoder.needs_sequence:
        logits_by_step = torch.stack(
            [
                model.decoder(sequence_features[:, : step + 1, :])
                for step in range(sequence_features.size(1))
            ],
            dim=1,
        ).squeeze(0)
    elif model.pooling == "mean":
        counts = torch.arange(
            1,
            sequence_features.size(1) + 1,
            device=sequence_features.device,
            dtype=sequence_features.dtype,
        ).view(1, -1, 1)
        cumulative_features = sequence_features.cumsum(dim=1) / counts
        logits_by_step = model.decoder(cumulative_features).squeeze(0)
    else:
        logits_by_step = model.decoder(sequence_features).squeeze(0)

    probabilities = torch.softmax(logits_by_step, dim=-1)
    confidences, predictions = probabilities.max(dim=-1)
    return (
        probabilities.cpu(),
        predictions.cpu(),
        confidences.cpu(),
        membranes.squeeze(0).cpu(),
        spikes.squeeze(0).cpu(),
    )


def save_timeline_png(
    path: Path,
    image: np.ndarray,
    masks: list[np.ndarray],
    predictions: torch.Tensor,
    confidences: torch.Tensor,
    target: int,
) -> None:
    steps = len(masks)
    cols = min(8, steps)
    rows = int(np.ceil(steps / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.4), squeeze=False)

    for step, axis in enumerate(axes.flat):
        axis.axis("off")
        if step >= steps:
            continue
        visible = image * masks[step]
        axis.imshow(visible, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(
            f"t={step + 1}\npred={int(predictions[step])} "
            f"{float(confidences[step]):.1%}",
            fontsize=9,
        )

    fig.suptitle(f"MNIST temporal inference, target={target}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_confidence_png(
    path: Path,
    predictions: torch.Tensor,
    confidences: torch.Tensor,
    target: int,
) -> None:
    steps = np.arange(1, len(confidences) + 1)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot(steps, confidences.numpy(), marker="o")
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("tokens read")
    axis.set_ylabel("top-class confidence")
    axis.set_title(f"Prediction confidence over time, target={target}")
    for step, pred in zip(steps, predictions.numpy()):
        axis.annotate(str(int(pred)), (step, confidences[step - 1].item()), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_activity_png(
    path: Path,
    membranes: torch.Tensor,
    spikes: torch.Tensor,
    reservoir_dim: int,
    taus: str,
) -> None:
    tau_values = [part.strip() for part in taus.split(",") if part.strip()]
    steps = np.arange(1, membranes.size(0) + 1)
    n_reservoirs = len(tau_values)
    membrane_activity = membranes.reshape(membranes.size(0), n_reservoirs, reservoir_dim)
    spike_activity = spikes.reshape(spikes.size(0), n_reservoirs, reservoir_dim)
    membrane_activity = membrane_activity.abs().mean(dim=-1).numpy()
    spike_activity = spike_activity.mean(dim=-1).numpy()

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for reservoir_index, tau in enumerate(tau_values):
        axes[0].plot(
            steps,
            membrane_activity[:, reservoir_index],
            marker="o",
            label=f"tau={tau}",
        )
        axes[1].plot(
            steps,
            spike_activity[:, reservoir_index],
            marker="o",
            label=f"tau={tau}",
        )
    axes[0].set_ylabel("mean abs membrane")
    axes[0].set_title("TPSAPU reservoir activity over time")
    axes[0].legend()
    axes[1].set_xlabel("tokens read")
    axes[1].set_ylabel("mean spike rate")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_timeline_gif(
    path: Path,
    image: np.ndarray,
    masks: list[np.ndarray],
    predictions: torch.Tensor,
    confidences: torch.Tensor,
    target: int,
) -> None:
    fig, axis = plt.subplots(figsize=(4, 4))
    axis.axis("off")
    frame = axis.imshow(image * masks[0], cmap="gray", vmin=0.0, vmax=1.0)
    title = axis.set_title("")

    def update(step: int):
        frame.set_data(image * masks[step])
        title.set_text(
            f"target={target} | t={step + 1}/{len(masks)} | "
            f"pred={int(predictions[step])} | conf={float(confidences[step]):.1%}"
        )
        return frame, title

    anim = animation.FuncAnimation(fig, update, frames=len(masks), interval=450, blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=2))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint = load_checkpoint(args.checkpoint, device)
    architecture_args = build_architecture_args(args, checkpoint)
    model = build_model(architecture_args).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"No checkpoint found at {args.checkpoint}; using random weights.")

    image, target = load_mnist_image(args)
    probabilities, predictions, confidences, membranes, spikes = run_temporal_inference(
        model,
        image,
        device,
    )
    original = denormalize_mnist(image)
    masks = reveal_masks(
        architecture_args.encoder,
        steps=len(predictions),
        patch_size=architecture_args.patch_size,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.split}_{args.index}_{architecture_args.encoder}"
    timeline_path = output_dir / f"{stem}_timeline.png"
    confidence_path = output_dir / f"{stem}_confidence.png"
    activity_path = output_dir / f"{stem}_activity.png"
    save_timeline_png(timeline_path, original, masks, predictions, confidences, target)
    save_confidence_png(confidence_path, predictions, confidences, target)
    save_activity_png(
        activity_path,
        membranes,
        spikes,
        reservoir_dim=architecture_args.reservoir_dim,
        taus=architecture_args.taus,
    )

    if args.gif:
        gif_path = output_dir / f"{stem}_timeline.gif"
        save_timeline_gif(gif_path, original, masks, predictions, confidences, target)
        print(f"Saved GIF: {gif_path}")

    final_probs = probabilities[-1]
    topk = torch.topk(final_probs, k=5)
    print(f"Target: {target}")
    print(f"Final prediction: {int(predictions[-1])} ({float(confidences[-1]):.2%})")
    print("Top-5 final probabilities:")
    for label, prob in zip(topk.indices.tolist(), topk.values.tolist()):
        print(f"  {label}: {prob:.2%}")
    print(f"Saved timeline: {timeline_path}")
    print(f"Saved confidence plot: {confidence_path}")
    print(f"Saved activity plot: {activity_path}")


if __name__ == "__main__":
    main()
