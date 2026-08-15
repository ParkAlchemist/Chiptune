from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
import shutil
import sys
import tomllib
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_\-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip("_")
    if not value:
        raise ValueError("Branch name became empty after slugification.")
    return value


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def toml_quote(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )
    return f'"{escaped}"'


def format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        return repr(value)

    if isinstance(value, str):
        return toml_quote(value)

    if isinstance(value, list):
        return "[" + ", ".join(format_toml_value(v) for v in value) + "]"

    raise TypeError(f"Unsupported TOML value type: {type(value)} for value={value!r}")


def write_toml(path: Path, config: dict[str, Any]) -> None:
    """
    Minimal TOML writer for this project's simple training config structure.

    Supports:
      [section]
      key = string/int/float/bool/list

    Skips None values.
    """
    lines: list[str] = []

    # Write non-section scalar keys first, if any.
    scalar_items = {
        key: value
        for key, value in config.items()
        if not isinstance(value, dict)
    }

    for key, value in scalar_items.items():
        if value is None:
            continue
        lines.append(f"{key} = {format_toml_value(value)}")

    if scalar_items:
        lines.append("")

    for section_name, section in config.items():
        if not isinstance(section, dict):
            continue

        lines.append(f"[{section_name}]")

        for key, value in section.items():
            if value is None:
                continue
            lines.append(f"{key} = {format_toml_value(value)}")

        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def get_config_value(config: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    sec = config.get(section, {})
    if isinstance(sec, dict):
        return sec.get(key, default)
    return default


def set_config_value(config: dict[str, Any], section: str, key: str, value: Any) -> None:
    if section not in config or not isinstance(config[section], dict):
        config[section] = {}
    config[section][key] = value


def resolve_source_experiment_paths(config: dict[str, Any]) -> tuple[Path, str, Path]:
    output_root = Path(get_config_value(config, "paths", "output_root", "runs/cyclegan_cqt"))
    experiment_name = get_config_value(config, "paths", "experiment_name", None)

    if experiment_name is None:
        raise ValueError("Source config must contain [paths].experiment_name")

    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    run_dir = output_root / experiment_name
    checkpoint_dir = run_dir / "checkpoints"

    return output_root, experiment_name, checkpoint_dir


def resolve_checkpoint(
    checkpoint_dir: Path,
    checkpoint_arg: str,
) -> Path:
    """
    Supported checkpoint arguments:
      latest
      epoch
      step
      explicit/path/to/checkpoint.pt
    """
    if checkpoint_arg == "latest":
        checkpoint_path = checkpoint_dir / "latest.pt"

    elif checkpoint_arg.startswith("epoch:"):
        epoch = int(checkpoint_arg.split(":", 1)[1])
        checkpoint_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"

    elif checkpoint_arg.startswith("step:"):
        step = int(checkpoint_arg.split(":", 1)[1])
        checkpoint_path = checkpoint_dir / f"step_{step:09d}.pt"

    else:
        checkpoint_path = Path(checkpoint_arg)
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return checkpoint_path


def make_relative_to_project(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(PROJECT_ROOT.resolve())
        return rel.as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_start_script(
    path: Path,
    branch_config_path: Path,
    branch_base_checkpoint: Path,
) -> None:
    config_rel = make_relative_to_project(branch_config_path)
    checkpoint_rel = make_relative_to_project(branch_base_checkpoint)

    content = f"""$ErrorActionPreference = "Stop"

cd "{PROJECT_ROOT}"
$env:PYTHONPATH = "."

python scripts/train/run_cyclegan_train_config.py `
  --config "{config_rel}" `
  -- --resume "{checkpoint_rel}"
"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_resume_script(
    path: Path,
    branch_config_path: Path,
) -> None:
    config_rel = make_relative_to_project(branch_config_path)

    content = f"""$ErrorActionPreference = "Stop"

cd "{PROJECT_ROOT}"
$env:PYTHONPATH = "."

python scripts/train/resume_cyclegan_train_config.py `
  --config "{config_rel}"
"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_branch_readme(
    path: Path,
    branch_name: str,
    experiment_name: str,
    source_config: Path,
    source_checkpoint: Path,
    branch_config: Path,
    branch_base_checkpoint: Path,
    start_script: Path,
    resume_script: Path,
) -> None:
    content = f"""# Experiment Branch: {branch_name}

## New experiment

```text
--experiment name: {experiment_name}
    
--source config: {source_config}
    
--source checkpoint: {source_checkpoint}
    
--branch config: {branch_config}
    
--branch base checkpoint: {branch_base_checkpoint}

--start_script: {make_relative_to_project(start_script).replace("/", "\\\\")}
    
--branch base checkpoint name: {branch_base_checkpoint.name}
    
--resume_script: {make_relative_to_project(resume_script).replace("/", "\\\\")}
"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an experiment branch.")
    parser.add_argument(
        "--source-config",
        type=Path,
        required=True,
        help="Existing TOML config to copy.",
    )
    parser.add_argument(
        "--branch-name",
        type=str,
        required=True,
        help="Short branch name, used for filenames/scripts.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="New experiment_name written into the copied config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Base checkpoint: latest, epoch:12, step:450000, or explicit path.",
    )
    parser.add_argument(
        "--config-out-dir",
        type=Path,
        default=Path("configs/branches"),
        help="Where to write the branch TOML config.",
    )
    parser.add_argument(
        "--script-out-dir",
        type=Path,
        default=Path("scripts/branches"),
        help="Where to write branch start/resume PowerShell scripts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing branch config or copied base checkpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_config_path = args.source_config

    if not source_config_path.is_absolute(): source_config_path = PROJECT_ROOT / source_config_path
    if not source_config_path.exists(): raise FileNotFoundError(
        f"Source config not found: {source_config_path}")

    branch_slug = slugify(args.branch_name)
    config_out_dir = args.config_out_dir

    if not config_out_dir.is_absolute(): config_out_dir = PROJECT_ROOT / config_out_dir
    script_out_dir = args.script_out_dir

    if not script_out_dir.is_absolute(): script_out_dir = PROJECT_ROOT / script_out_dir
    source_config = load_toml(source_config_path)

    output_root, source_experiment_name, source_checkpoint_dir = resolve_source_experiment_paths(
        source_config)

    source_checkpoint = resolve_checkpoint(
        checkpoint_dir=source_checkpoint_dir,
        checkpoint_arg=args.checkpoint, )

    # Simple deep copy through JSON because the config contains only simple TOML-compatible values.
    branch_config = json.loads(json.dumps(source_config))

    set_config_value(
        branch_config,
        "paths",
        "experiment_name",
         args.experiment_name,
    )

    # Important:
    # Do not store resume in the config. The generated start script passes
    # --resume branch_base.pt once. Future resumes should use resume_cyclegan_train_config.py.

    if "training" in branch_config and isinstance(branch_config["training"], dict):
        branch_config["training"].pop("resume", None)

    branch_config_path = config_out_dir / f"{branch_slug}.toml"

    branch_run_dir = output_root / args.experiment_name
    branch_checkpoint_dir = branch_run_dir / "checkpoints"
    branch_base_checkpoint = branch_checkpoint_dir / "branch_base.pt"

    start_script = script_out_dir / f"start_{branch_slug}.ps1"
    resume_script = script_out_dir / f"resume_{branch_slug}.ps1"

    branch_readme = branch_run_dir / "BRANCH.md"
    branch_info = branch_run_dir / "branch_info.json"

    files_to_check = [
        branch_config_path,
        branch_base_checkpoint,
        start_script,
        resume_script,
        branch_readme,
        branch_info,
    ]

    if not args.overwrite:
        existing = [path for path in files_to_check if path.exists()]

        if existing:
            formatted_existing = "\n".join(str(path) for path in existing)
            raise FileExistsError(
                "Refusing to overwrite existing branch files. " 
                "Use --overwrite if this is intentional.\n\n" 
                f"Existing files:\n{formatted_existing}"
            )

    print("Branch creation plan:")
    print(f" Source config: {source_config_path}")
    print(f" Source experiment: {source_experiment_name}")
    print(f" Source checkpoint: {source_checkpoint}")
    print(f" Branch name: {branch_slug}")
    print(f" New experiment: {args.experiment_name}")
    print(f" Branch config: {branch_config_path}")
    print(f" Branch run dir: {branch_run_dir}")
    print(f" Branch base ckpt: {branch_base_checkpoint}")
    print(f" Start script: {start_script}")
    print(f" Resume script: {resume_script}")

    if args.dry_run:
        print("\nDry run complete. No files written.")
        return

    branch_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    write_toml(branch_config_path, branch_config)

    shutil.copy2(source_checkpoint, branch_base_checkpoint)

    write_start_script(
        path=start_script,
        branch_config_path=branch_config_path,
        branch_base_checkpoint=branch_base_checkpoint,
    )

    write_resume_script(
        path=resume_script,
        branch_config_path=branch_config_path,
    )

    write_branch_readme(
        path=branch_readme,
        branch_name=branch_slug,
        experiment_name=args.experiment_name,
        source_config=source_config_path,
        source_checkpoint=source_checkpoint,
        branch_config=branch_config_path,
        branch_base_checkpoint=branch_base_checkpoint,
        start_script=start_script,
        resume_script=resume_script,
    )

    branch_info_data = {
        "branch_name": branch_slug,
        "experiment_name": args.experiment_name,
        "source_config": str(source_config_path),
        "source_experiment_name": source_experiment_name,
        "source_checkpoint": str(source_checkpoint),
        "branch_config": str(branch_config_path),
        "branch_base_checkpoint": str(branch_base_checkpoint),
        "start_script": str(start_script),
        "resume_script": str(resume_script),
    }

    branch_info.parent.mkdir(parents=True, exist_ok=True)
    branch_info.write_text(
        json.dumps(branch_info_data,
                   indent=2,
                   ensure_ascii=False),
        encoding="utf-8",
    )

    start_script_rel = make_relative_to_project(start_script).replace("/", "\\")
    resume_script_rel = make_relative_to_project(resume_script).replace("/", "\\")

    print("\nBranch created successfully.")
    print("\nNext steps:")
    print(f" 1. Edit config: {branch_config_path}")
    print(" 2. Start branch:")
    print(f" .\\{start_script_rel}")
    print(" 3. Resume later:")
    print(f" .\\{resume_script_rel}")


if __name__ == "__main__":
    main()

