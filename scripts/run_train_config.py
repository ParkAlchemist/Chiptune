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
    """
    Converts TOML key/value into train_cyclegan.py CLI args.

    bool true:
        amp = true -> --amp

    bool false:
        amp = false -> skipped

    normal value:
        batch_size = 4 -> --batch-size 4
    """
    if value is None:
        return []

    flag = to_cli_flag(key)

    if isinstance(value, bool):
        return [flag] if value else []

    return [flag, str(value)]


def build_command(config_path: Path, passthrough_args: list[str]) -> list:
    with config_path.open("rb") as f:
        config = tomllib.load(f)

    flat = flatten_config(config)

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_cyclegan.py"),
    ]

    for key, value in flat.items():
        command.extend(value_to_cli_args(key, value))

    command.extend(passthrough_args)

    return command


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Launch CycleGAN training from a TOML config."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to TOML training config.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated command without running it.",
    )

    args, passthrough = parser.parse_known_args()

    # Allow overrides after `--`.
    # Example:
    #   python scripts/run_train_config.py --config cfg.toml -- --batch-size 2
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

    command = build_command(config_path, passthrough)

    print("Generated training command:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))

    if args.dry_run:
        return

    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()

