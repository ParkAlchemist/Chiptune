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
    Converts TOML key/value into train_vocoder.py CLI args.

    Examples:
        amp = true      -> --amp
        amp = false     -> skipped
        batch_size = 1  -> --batch-size 1
    """
    if value is None:
        return []

    flag = to_cli_flag(key)

    if isinstance(value, bool):
        return [flag] if value else []

    return [flag, str(value)]


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("rb") as f:
        return tomllib.load(f)


def build_command(
    config_path: Path,
    passthrough_args: list[str],
) -> list:
    config = load_config(config_path)
    flat = flatten_config(config)

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_vocoder.py"),
    ]

    for key, value in flat.items():
        command.extend(value_to_cli_args(key, value))

    # Passthrough args go last so scalar args can override config values.
    command.extend(passthrough_args)

    return command


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Launch vocoder training from a TOML config."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to TOML vocoder training config.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated command without running it.",
    )

    args, passthrough = parser.parse_known_args()

    # Allows:
    #   python scripts/run_vocoder_config.py --config cfg.toml -- --batch-size 2
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    return args, passthrough


def quote_command_part(part: str) -> str:
    if " " in part:
        return f'"{part}"'
    return part


def main() -> None:
    args, passthrough = parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    command = build_command(
        config_path=config_path,
        passthrough_args=passthrough,
    )

    print("Generated vocoder training command:")
    print(" ".join(quote_command_part(part) for part in command))

    if args.dry_run:
        return

    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()

