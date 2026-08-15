from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """
    Walk upward until a project marker is found.

    This makes scripts robust even if moved under scripts/train,
    scripts/checks, scripts/eval, etc.
    """
    if start is None:
        start = Path(__file__).resolve()

    current = start.resolve()

    if current.is_file():
        current = current.parent

    markers = [
        "README.md",
        "pytest.ini",
        "requirements.txt",
    ]

    for parent in [current, *current.parents]:
        if all((parent / marker).exists() for marker in markers):
            return parent

    raise RuntimeError(f"Could not locate project root from: {start}")

