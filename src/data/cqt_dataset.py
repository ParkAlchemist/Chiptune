from pathlib import Path
import random
from typing import Literal

import torch
from torch.utils.data import Dataset

from src.config import config
from src.data.audio_utils import (
    list_audio_files,
    read_jsonl,
    load_audio,
    pad_or_crop,
    audio_to_cqt_tensor,
)


DomainMode = Literal["normal", "chiptune"]


class CachedCQTDataset(Dataset):
    """
    Loads precomputed CQT tensors from cache.

    Each cached file should contain:
        {
            "cqt": Tensor [1, n_bins, time_frames],
            "metadata": dict
        }
    """

    def __init__(
        self,
        cache_dir: Path,
        index_file: Path | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.index_file = index_file or self.cache_dir / "index.jsonl"
        self.dtype = dtype

        if not self.index_file.exists():
            raise FileNotFoundError(
                f"Missing CQT index file: {self.index_file}. "
                f"Run precompute_cqt.py first."
            )

        records = read_jsonl(self.index_file)

        self.records = [
            r for r in records
            if "cache_path" in r and "error" not in r and Path(r["cache_path"]).exists()
        ]

        if not self.records:
            raise RuntimeError(f"No usable cached CQT records found in {self.index_file}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        data = torch.load(record["cache_path"], map_location="cpu")

        cqt = data["cqt"].to(dtype=self.dtype)

        return {
            "cqt": cqt,
            "metadata": data.get("metadata", record),
        }


class OnTheFlyCQTDataset(Dataset):
    """
    Computes CQT during __getitem__.

    Useful for debugging, not recommended for long training runs.
    """

    def __init__(
        self,
        audio_dir: Path,
        random_crop: bool = True,
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.random_crop = random_crop
        self.files = list_audio_files(self.audio_dir, config.data.audio_extensions)

        if not self.files:
            raise RuntimeError(f"No audio files found in {self.audio_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        cqt_cfg = config.cqt

        audio_path = self.files[idx]
        target_samples = int(cqt_cfg.sample_rate * cqt_cfg.duration)

        y = load_audio(
            audio_path,
            sample_rate=cqt_cfg.sample_rate,
            mono=cqt_cfg.mono,
            normalize_peak=cqt_cfg.normalize_peak,
        )

        y = pad_or_crop(
            y,
            target_samples=target_samples,
            random_crop=self.random_crop,
        )

        cqt = audio_to_cqt_tensor(
            y,
            sample_rate=cqt_cfg.sample_rate,
            hop_length=cqt_cfg.hop_length,
            n_bins=cqt_cfg.n_bins,
            bins_per_octave=cqt_cfg.bins_per_octave,
            db_min=cqt_cfg.db_min,
            db_max=cqt_cfg.db_max,
            fmin=cqt_cfg.fmin,
        )

        return {
            "cqt": cqt,
            "metadata": {
                "source_path": str(audio_path),
            },
        }


class UnpairedCQTDataset(Dataset):
    """
    CycleGAN dataset.

    Returns:
        {
            "real_x": poly CQT tensor,
            "real_y": chip CQT tensor,
            "meta_x": metadata,
            "meta_y": metadata
        }

    X = polyphonic music
    Y = chiptune music
    """

    def __init__(
        self,
        poly_dataset: Dataset,
        chip_dataset: Dataset,
        length_mode: Literal["max", "min", "poly", "chip"] = "max",
    ) -> None:
        self.poly_dataset = poly_dataset
        self.chip_dataset = chip_dataset
        self.length_mode = length_mode

    def __len__(self) -> int:
        if self.length_mode == "max":
            return max(len(self.poly_dataset), len(self.chip_dataset))
        if self.length_mode == "min":
            return min(len(self.poly_dataset), len(self.chip_dataset))
        if self.length_mode == "poly":
            return len(self.poly_dataset)
        if self.length_mode == "chip":
            return len(self.chip_dataset)

        raise ValueError(f"Unknown length_mode: {self.length_mode}")

    def __getitem__(self, idx: int) -> dict:
        x_idx = idx % len(self.poly_dataset)
        y_idx = random.randint(0, len(self.chip_dataset) - 1)

        x_item = self.poly_dataset[x_idx]
        y_item = self.chip_dataset[y_idx]

        return {
            "real_x": x_item["cqt"],
            "real_y": y_item["cqt"],
            "meta_x": x_item["metadata"],
            "meta_y": y_item["metadata"],
        }


def build_cached_unpaired_dataset(cache_root: Path = config.data.cache_root,) -> UnpairedCQTDataset:
    poly_cache = Path(cache_root) / config.data.poly_subdir
    chip_cache = Path(cache_root) / config.data.chip_subdir

    poly_ds = CachedCQTDataset(poly_cache)
    chip_ds = CachedCQTDataset(chip_cache)

    return UnpairedCQTDataset(
        poly_dataset=poly_ds,
        chip_dataset=chip_ds,
        length_mode="max",
    )
