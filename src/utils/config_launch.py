from __future__ import annotations

from pathlib import Path
from typing import Any
import subprocess
import sys
import tomllib


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


def load_toml_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def config_to_cli_args(config: dict[str, Any]) -> list:
    flat = flatten_config(config)
    args: list[str] = []

    for key, value in flat.items():
        args.extend(value_to_cli_args(key, value))

    return args


def quote_command_part(part: str) -> str:
    if " " in part:
        return f'"{part}"'
    return part


def format_command(command: list[str]) -> str:
    return " ".join(quote_command_part(part) for part in command)


def build_training_command(
    project_root: Path,
    train_script: Path,
    config_path: Path,
    passthrough_args: list[str],
) -> list:
    config = load_toml_config(config_path)

    command = [
        sys.executable,
        str(train_script),
    ]

    command.extend(config_to_cli_args(config))
    command.extend(passthrough_args)

    return command


def run_command(
    command: list[str],
    cwd: Path,
    dry_run: bool = False,
    label: str = "Generated command",
) -> None:
    print(f"{label}:")
    print(format_command(command))

    if dry_run:
        return

    subprocess.run(command, cwd=cwd, check=True)


def resolve_run_checkpoint(
    project_root: Path,
    config: dict[str, Any],
    default_output_root: str,
    checkpoint_arg: str | None,
) -> Path:
    flat = flatten_config(config)

    output_root = Path(flat.get("output_root", default_output_root))
    experiment_name = flat.get("experiment_name")

    if experiment_name is None:
        raise ValueError("Config must contain experiment_name.")

    run_dir = output_root / experiment_name
    checkpoint_dir = run_dir / "checkpoints"

    if checkpoint_arg is None or checkpoint_arg == "latest":
        checkpoint_path = checkpoint_dir / "latest.pt"

    elif checkpoint_arg.startswith("step:"):
        step = int(checkpoint_arg.split(":", 1)[1])
        checkpoint_path = checkpoint_dir / f"step_{step:09d}.pt"

    elif checkpoint_arg.startswith("epoch:"):
        epoch = int(checkpoint_arg.split(":", 1)[1])
        checkpoint_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"

    else:
        checkpoint_path = Path(checkpoint_arg)

    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return checkpoint_path

