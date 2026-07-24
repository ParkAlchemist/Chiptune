from pathlib import Path
from collections import Counter

from src.config import config
from src.data.audio_utils import list_audio_files


def inspect_domain(name: str, path: Path) -> None:
    files = list_audio_files(path, config.data.audio_extensions)
    suffixes = Counter(p.suffix.lower() for p in files)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"path: {path}")
    print(f"files: {len(files)}")

    for suffix, count in sorted(suffixes.items()):
        print(f"  {suffix}: {count}")

    if files:
        print("examples:")
        for p in files[:5]:
            print(f"  {p}")


def main() -> None:
    data_root = config.data.dataset_root

    inspect_domain("poly", data_root / config.data.poly_subdir)
    inspect_domain("chip", data_root / config.data.chip_subdir)


if __name__ == "__main__":
    main()
