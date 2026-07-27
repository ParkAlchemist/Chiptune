from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import sys
import tomllib
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def to_cli_flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    for section_name, section in config.items():
        if not isinstance(section, dict):
            flat[section_name] = section
            continue

        for key, value in section.items():
            flat[key] = value

    return flat


def value_to_cli_args(key: str, value: Any) -> list:
    if value is None:
        return []

    flag = to_cli_flag(key)

    if isinstance(value, bool):
        return [flag] if value else []

    return [flag, str(value)]


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("rb") as f:
        return tomllib.load(f)


def resolve_checkpoint(
    config: dict[str, Any],
    checkpoint_arg: str | None,
) -> Path:
    flat = flatten_config(config)

    output_root = Path(flat.get("output_root", "runs/cyclegan_cqt"))
    experiment_name = flat.get("experiment_name")

    if experiment_name is None:
        raise ValueError("Config must contain paths.experiment_name or experiment_name.")

    run_dir = output_root / experiment_name
    checkpoint_dir = run_dir / "checkpoints"

    if checkpoint_arg is None or checkpoint_arg == "latest":
        checkpoint_path = checkpoint_dir / "latest.pt"
    elif checkpoint_arg.startswith("epoch:"):
        epoch_number = int(checkpoint_arg.split(":", 1)[1])
        checkpoint_path = checkpoint_dir / f"epoch_{epoch_number:04d}.pt"
    else:
        checkpoint_path = Path(checkpoint_arg)

    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return checkpoint_path


def build_command(
    config_path: Path,
    checkpoint_path: Path,
    passthrough_args: list[str],
) -> list:
    config = load_config(config_path)
    flat = flatten_config(config)

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_cyclegan.py"),
    ]

    for key, value in flat.items():
        command.extend(value_to_cli_args(key, value))

    command.extend(["--resume", str(checkpoint_path)])

    # Passthrough arguments at the end override earlier CLI args in argparse
    # only if train_cyclegan.py handles repeated args by taking the last value.
    # Python argparse normally takes the last occurrence for simple scalar args.
    command.extend(passthrough_args)

    return command


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Resume CycleGAN training from a TOML config."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to TOML training config.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help=(
            "Checkpoint to resume from. Options: "
            "'latest', 'epoch:12', or explicit path."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated command without running it.",
    )

    args, passthrough = parser.parse_known_args()

    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    return args, passthrough


def main() -> None:
    args, passthrough = parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    config = load_config(config_path)
    checkpoint_path = resolve_checkpoint(config, args.checkpoint)

    command = build_command(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        passthrough_args=passthrough,
    )

    print("Resuming from checkpoint:")
    print(checkpoint_path)

    print("\nGenerated resume command:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))

    if args.dry_run:
        return

    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()

