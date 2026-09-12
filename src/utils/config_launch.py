from __future__ import annotations

from pathlib import Path
import subprocess


def resolve_project_path(
    path: Path,
    project_root: Path,
) -> Path:
    path = path.expanduser()

    if not path.is_absolute():
        path = project_root / path

    return path.resolve()


def format_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def run_command(
    command: list[str],
    *,
    cwd: Path,
    dry_run: bool = False,
    label: str = "Generated command",
) -> None:
    print(f"{label}:")
    print(format_command(command))

    if dry_run:
        return

    subprocess.run(
        command,
        cwd=cwd,
        check=True,
    )

