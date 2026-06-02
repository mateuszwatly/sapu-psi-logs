"""
Run every encoder/decoder combination with three cosine cycles.

Each run is saved in its own folder:

    sweep_runs/<encoder>__<decoder>/latest.pt
    sweep_runs/<encoder>__<decoder>/best.pt

`latest.pt` is a full resume checkpoint: model, optimizer, progress counters,
pruning mask, metrics, args, and RNG state.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ENCODERS = [
    "linear_patch",
    "mlp_patch",
    "cnn2",
    "cnn3",
    "res_cnn",
    "rows",
]

DECODERS = [
    "linear",
    "membrane_mlp",
    "spike_mlp",
    "both_mlp",
    "all_state_mlp",
    "membrane_transformer",
    "spike_transformer",
    "both_transformer",
    "all_state_transformer",
    "lif_count",
]


def parse_csv(value: str, allowed: list[str], name: str) -> list[str]:
    if value == "all":
        return allowed
    selected = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown {name}: {', '.join(unknown)}")
    return selected


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="sweep_runs")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--encoders", default="all")
    parser.add_argument("--decoders", default="all")
    parser.add_argument("--cycles", type=float, default=3.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--train-script", default="train_pipeline.py")
    parser.add_argument("--resume-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed to train_pipeline.py after `--`.",
    )
    args = parser.parse_args()
    extra = args.train_args
    if extra and extra[0] == "--":
        extra = extra[1:]
    return args, extra


def command_for_run(
    *,
    args: argparse.Namespace,
    extra_train_args: list[str],
    encoder: str,
    decoder: str,
    run_dir: Path,
) -> list[str]:
    latest = run_dir / "latest.pt"
    command = [
        args.python,
        args.train_script,
        "--encoder",
        encoder,
        "--decoder",
        decoder,
        "--cosine-cycles",
        str(args.cycles),
        "--prune-cycles",
        str(args.cycles),
        "--checkpoint-out",
        str(latest),
        "--data-dir",
        args.data_dir,
    ]
    if args.no_download:
        command.append("--no-download")
    if args.resume_existing and latest.exists():
        command.extend(["--resume", str(latest)])
    command.extend(extra_train_args)
    return command


def write_command_file(run_dir: Path, command: list[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.txt").write_text(
        shlex.join(command) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args, extra_train_args = parse_args()
    encoders = parse_csv(args.encoders, ENCODERS, "encoders")
    decoders = parse_csv(args.decoders, DECODERS, "decoders")
    output_root = Path(args.output_root)

    for encoder in encoders:
        for decoder in decoders:
            run_name = f"{encoder}__{decoder}"
            run_dir = output_root / run_name
            latest = run_dir / "latest.pt"
            if args.skip_existing and latest.exists():
                print(f"Skipping existing run: {run_name}")
                continue

            command = command_for_run(
                args=args,
                extra_train_args=extra_train_args,
                encoder=encoder,
                decoder=decoder,
                run_dir=run_dir,
            )
            write_command_file(run_dir, command)
            print(f"\n=== {run_name} ===")
            print(shlex.join(command))
            if args.dry_run:
                continue
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
