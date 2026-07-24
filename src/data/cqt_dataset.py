from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal
import csv
import json
import random

import torch
from torch.utils.data import Dataset, DataLoader

from torch.utils.data._utils.collate import default_collate


Domain = Literal["poly", "chip"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def read_csv_index(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def discover_cache_records(cache_root: Path) -> list[dict[str, Any]]:
    cache_root = Path(cache_root)

    index_jsonl = cache_root / "index.jsonl"
    index_csv = cache_root / "index.csv"

    if index_jsonl.exists():
        records = read_jsonl(index_jsonl)
    elif index_csv.exists():
        records = read_csv_index(index_csv)
    else:
        records = [
            {
                "cache_path": str(path),
                "status": "ok",
            }
            for path in sorted(cache_root.glob("*.pt"))
        ]

    usable: list[dict[str, Any]] = []

    for record in records:
        if record.get("status") not in (None, "ok", "skipped_existing"):
            continue

        cache_path = record.get("cache_path")
        if cache_path is None:
            continue

        cache_path = Path(cache_path)
        if not cache_path.exists():
            continue

        record = dict(record)
        record["cache_path"] = str(cache_path)
        usable.append(record)

    if not usable:
        raise RuntimeError(f"No usable cache records found under: {cache_root}")

    return usable


def seconds_to_frames(seconds: float, sample_rate: int, hop_length: int) -> int:
    return max(1, int(round(seconds * sample_rate / hop_length)))


def pad_feature_last_dim(
    tensor: torch.Tensor,
    target_frames: int,
    pad_value: float,
) -> torch.Tensor:
    current_frames = tensor.shape[-1]

    if current_frames >= target_frames:
        return tensor

    pad_frames = target_frames - current_frames
    pad_shape = list(tensor.shape)
    pad_shape[-1] = pad_frames

    padding = torch.full(
        pad_shape,
        fill_value=pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )

    return torch.cat([tensor, padding], dim=-1)


def cqt_window_energy(cqt_window: torch.Tensor) -> float:
    """
    CQT is normalized to [-1, 1], where silence / db_min is near -1.
    Energy is measured after mapping [-1, 1] -> [0, 1].
    """
    x = (cqt_window.float() + 1.0) * 0.5
    return float(x.mean().item())


class SmallTensorLRUCache:
    """
    Tiny in-memory LRU cache for loaded .pt track payloads.

    This is useful because random snippet sampling can repeatedly hit the same
    track within an epoch. Keeping a few recent tracks avoids excessive disk IO.
    """

    def __init__(self, max_items: int = 8) -> None:
        self.max_items = max(0, int(max_items))
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(self, path: Path) -> dict[str, Any]:
        key = str(path)

        if self.max_items <= 0:
            return torch.load(path, map_location="cpu")

        if key in self._items:
            value = self._items.pop(key)
            self._items[key] = value
            return value

        value = torch.load(path, map_location="cpu")

        self._items[key] = value

        while len(self._items) > self.max_items:
            self._items.popitem(last=False)

        return value


class CachedCQTTrackDataset(Dataset):
    """
    Dataset over full-track cached CQT files.

    Each __getitem__ returns a fixed-length snippet sampled from a cached full-track CQT.

    Expected cache payload keys:
        cqt_mag: Tensor [1, n_bins, T]
        chroma: optional Tensor [12, T]
        cqt_phase: optional Tensor [n_bins, T]
        metadata: dict

    Returned item:
        {
            "cqt": Tensor [1, n_bins, snippet_frames],
            "chroma": optional Tensor [12, snippet_frames],
            "phase": optional Tensor [n_bins, snippet_frames],
            "metadata": dict,
            "cache_path": str,
            "source_path": optional str,
            "frame_start": int,
            "frame_end": int,
            "track_index": int,
            "domain": str
        }
    """

    def __init__(
        self,
        cache_root: Path,
        domain: Domain,
        snippet_seconds: float = 4.0,
        snippet_frames: int | None = None,
        sample_rate: int = 22050,
        hop_length: int = 512,
        windows_per_track: int = 16,
        random_window: bool = True,
        random_track: bool = False,
        return_chroma: bool = True,
        return_phase: bool = False,
        return_metadata: bool = True,
        min_window_energy: float = 0.01,
        max_resample_attempts: int = 10,
        pad_short_tracks: bool = True,
        track_cache_size: int = 8,
        dtype: torch.dtype = torch.float32,
        expected_config_hash: str | None = None,
        require_config_hash_match: bool = False,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.domain = domain
        self.records = discover_cache_records(self.cache_root)

        self.sample_rate = sample_rate
        self.hop_length = hop_length

        self.snippet_frames = (
            snippet_frames
            if snippet_frames is not None
            else seconds_to_frames(snippet_seconds, sample_rate, hop_length)
        )

        self.windows_per_track = max(1, int(windows_per_track))
        self.random_window = random_window
        self.random_track = random_track

        self.return_chroma = return_chroma
        self.return_phase = return_phase
        self.return_metadata = return_metadata

        self.min_window_energy = float(min_window_energy)
        self.max_resample_attempts = max(1, int(max_resample_attempts))
        self.pad_short_tracks = pad_short_tracks

        self.dtype = dtype
        self.expected_config_hash = expected_config_hash
        self.require_config_hash_match = require_config_hash_match

        self.track_cache = SmallTensorLRUCache(max_items=track_cache_size)

        if self.require_config_hash_match and self.expected_config_hash is None:
            raise ValueError(
                "require_config_hash_match=True requires expected_config_hash."
            )

    def __len__(self) -> int:
        return len(self.records) * self.windows_per_track

    def _record_index_for_item(self, idx: int) -> int:
        if self.random_track:
            return random.randint(0, len(self.records) - 1)

        return idx % len(self.records)

    def _load_payload(self, record_index: int) -> tuple[dict[str, Any], Path]:
        record = self.records[record_index]
        cache_path = Path(record["cache_path"])
        payload = self.track_cache.get(cache_path)

        if "cqt_mag" not in payload:
            raise KeyError(f"Cache file missing 'cqt_mag': {cache_path}")

        if self.expected_config_hash is not None:
            found_hash = payload.get("config_hash") or payload.get("metadata", {}).get("config_hash")
            if found_hash != self.expected_config_hash and self.require_config_hash_match:
                raise RuntimeError(
                    f"Config hash mismatch in {cache_path}. "
                    f"Expected {self.expected_config_hash}, found {found_hash}."
                )

        return payload, cache_path

    def _choose_start_frame(self, total_frames: int, idx: int) -> int:
        if total_frames <= self.snippet_frames:
            return 0

        max_start = total_frames - self.snippet_frames

        if self.random_window:
            return random.randint(0, max_start)

        # Deterministic spread across the track.
        virtual_window_idx = idx // max(1, len(self.records))
        if self.windows_per_track <= 1:
            return 0

        fraction = virtual_window_idx / max(1, self.windows_per_track - 1)
        return int(round(max_start * fraction))

    def _slice_last_dim(
        self,
        tensor: torch.Tensor,
        start: int,
        end: int,
        pad_value: float,
    ) -> torch.Tensor:
        window = tensor[..., start:end]

        if window.shape[-1] < self.snippet_frames:
            if not self.pad_short_tracks:
                raise RuntimeError(
                    f"Track shorter than snippet length: "
                    f"{window.shape[-1]} < {self.snippet_frames}"
                )
            window = pad_feature_last_dim(window, self.snippet_frames, pad_value)

        return window

    def _make_item_from_payload(
        self,
        payload: dict[str, Any],
        cache_path: Path,
        record_index: int,
        idx: int,
    ) -> dict[str, Any]:
        cqt = payload["cqt_mag"].to(dtype=self.dtype)

        total_frames = int(cqt.shape[-1])
        start = self._choose_start_frame(total_frames, idx)
        end = start + self.snippet_frames

        cqt_window = self._slice_last_dim(
            cqt,
            start=start,
            end=end,
            pad_value=-1.0,
        )

        item: dict[str, Any] = {
            "cqt": cqt_window,
            "frame_start": int(start),
            "frame_end": int(end),
            "track_index": int(record_index),
            "domain": self.domain,
            "cache_path": str(cache_path),
            "energy": cqt_window_energy(cqt_window),
        }

        if self.return_chroma and "chroma" in payload:
            chroma = payload["chroma"].to(dtype=self.dtype)
            item["chroma"] = self._slice_last_dim(
                chroma,
                start=start,
                end=end,
                pad_value=0.0,
            )

        if self.return_phase and "cqt_phase" in payload:
            phase = payload["cqt_phase"].to(dtype=self.dtype)
            item["phase"] = self._slice_last_dim(
                phase,
                start=start,
                end=end,
                pad_value=0.0,
            )

        metadata = payload.get("metadata", {})

        if self.return_metadata:
            item["metadata"] = metadata

        source_path = metadata.get("source_path")
        if source_path is not None:
            item["source_path"] = source_path

        return item

    def __getitem__(self, idx: int) -> dict[str, Any]:
        last_item: dict[str, Any] | None = None

        for attempt in range(self.max_resample_attempts):
            record_index = self._record_index_for_item(idx)

            # If the selected snippet is too quiet, retry with a random track/window.
            if attempt > 0:
                record_index = random.randint(0, len(self.records) - 1)

            payload, cache_path = self._load_payload(record_index)
            item = self._make_item_from_payload(payload, cache_path, record_index, idx)

            last_item = item

            if item["energy"] >= self.min_window_energy:
                return item

        # Fallback: return the last sampled item rather than hard failing.
        assert last_item is not None
        return last_item


class UnpairedCQTDataset(Dataset):
    """
    CycleGAN-ready unpaired dataset.

    X = poly
    Y = chip

    Returned keys:
        real_x / real_poly
        real_y / real_chip

    Also returns optional chroma/phase with matching domain prefixes if available.
    """

    def __init__(
        self,
        poly_dataset: CachedCQTTrackDataset,
        chip_dataset: CachedCQTTrackDataset,
        length_mode: Literal["max", "min", "poly", "chip"] = "max",
        random_y: bool = True,
    ) -> None:
        self.poly_dataset = poly_dataset
        self.chip_dataset = chip_dataset
        self.length_mode = length_mode
        self.random_y = random_y

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

    def _chip_index(self, idx: int) -> int:
        if self.random_y:
            return random.randint(0, len(self.chip_dataset) - 1)
        return idx % len(self.chip_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        poly_item = self.poly_dataset[idx % len(self.poly_dataset)]
        chip_item = self.chip_dataset[self._chip_index(idx)]

        batch_item: dict[str, Any] = {
            "real_x": poly_item["cqt"],
            "real_y": chip_item["cqt"],
            "real_poly": poly_item["cqt"],
            "real_chip": chip_item["cqt"],

            "energy_poly": float(poly_item["energy"]),
            "energy_chip": float(chip_item["energy"]),

            "frame_start_poly": poly_item["frame_start"],
            "frame_end_poly": poly_item["frame_end"],
            "frame_start_chip": chip_item["frame_start"],
            "frame_end_chip": chip_item["frame_end"],

            "cache_path_poly": poly_item["cache_path"],
            "cache_path_chip": chip_item["cache_path"],
        }

        if "chroma" in poly_item:
            batch_item["chroma_x"] = poly_item["chroma"]
            batch_item["chroma_poly"] = poly_item["chroma"]

        if "chroma" in chip_item:
            batch_item["chroma_y"] = chip_item["chroma"]
            batch_item["chroma_chip"] = chip_item["chroma"]

        if "phase" in poly_item:
            batch_item["phase_x"] = poly_item["phase"]
            batch_item["phase_poly"] = poly_item["phase"]

        if "phase" in chip_item:
            batch_item["phase_y"] = chip_item["phase"]
            batch_item["phase_chip"] = chip_item["phase"]

        if "metadata" in poly_item:
            batch_item["metadata_poly"] = poly_item["metadata"]

        if "metadata" in chip_item:
            batch_item["metadata_chip"] = chip_item["metadata"]

        if "source_path" in poly_item:
            batch_item["source_path_poly"] = poly_item["source_path"]

        if "source_path" in chip_item:
            batch_item["source_path_chip"] = chip_item["source_path"]

        return batch_item


def build_unpaired_cqt_dataset(
    cache_root: Path,
    snippet_seconds: float = 4.0,
    snippet_frames: int | None = None,
    sample_rate: int = 22050,
    hop_length: int = 512,
    windows_per_track: int = 16,
    return_chroma: bool = True,
    return_phase: bool = False,
    return_metadata: bool = False,
    min_window_energy: float = 0.01,
    track_cache_size: int = 8,
) -> UnpairedCQTDataset:
    cache_root = Path(cache_root)

    poly_dataset = CachedCQTTrackDataset(
        cache_root=cache_root / "poly",
        domain="poly",
        snippet_seconds=snippet_seconds,
        snippet_frames=snippet_frames,
        sample_rate=sample_rate,
        hop_length=hop_length,
        windows_per_track=windows_per_track,
        random_window=True,
        random_track=False,
        return_chroma=return_chroma,
        return_phase=return_phase,
        return_metadata=return_metadata,
        min_window_energy=min_window_energy,
        track_cache_size=track_cache_size,
    )

    chip_dataset = CachedCQTTrackDataset(
        cache_root=cache_root / "chip",
        domain="chip",
        snippet_seconds=snippet_seconds,
        snippet_frames=snippet_frames,
        sample_rate=sample_rate,
        hop_length=hop_length,
        windows_per_track=windows_per_track,
        random_window=True,
        random_track=False,
        return_chroma=return_chroma,
        return_phase=return_phase,
        return_metadata=return_metadata,
        min_window_energy=min_window_energy,
        track_cache_size=track_cache_size,
    )

    return UnpairedCQTDataset(
        poly_dataset=poly_dataset,
        chip_dataset=chip_dataset,
        length_mode="max",
        random_y=True,
    )


def cqt_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Custom collate for CQT datasets.

    - Tensors are stacked normally.
    - Numbers are collated normally.
    - Strings/paths are kept as lists.
    - Metadata dicts are kept as lists to avoid PyTorch default_collate
      crashing on nested None values such as fmin=None.
    - None values are kept as lists or ignored depending on key usage.
    """
    if not batch:
        return {}

    output: dict[str, Any] = {}

    keys = set()
    for item in batch:
        keys.update(item.keys())

    for key in keys:
        values = [item.get(key) for item in batch]

        # Metadata can contain None, nested dicts, paths, etc.
        # Keep it uncollated.
        if key.startswith("metadata"):
            output[key] = values
            continue

        # Source/cache paths should remain readable strings.
        if key.startswith("source_path") or key.startswith("cache_path"):
            output[key] = values
            continue

        # If any value is None, default_collate will crash.
        # Keep as list.
        if any(v is None for v in values):
            output[key] = values
            continue

        try:
            output[key] = default_collate(values)
        except TypeError:
            output[key] = values

    return output


def build_unpaired_cqt_dataloader(
    cache_root: Path,
    batch_size: int = 4,
    num_workers: int = 0,
    shuffle: bool = True,
    pin_memory: bool = True,
    snippet_seconds: float = 4.0,
    snippet_frames: int | None = None,
    sample_rate: int = 22050,
    hop_length: int = 512,
    windows_per_track: int = 16,
    return_chroma: bool = True,
    return_phase: bool = False,
    return_metadata: bool = False,
    min_window_energy: float = 0.01,
    track_cache_size: int = 8,
) -> DataLoader:
    dataset = build_unpaired_cqt_dataset(
        cache_root=cache_root,
        snippet_seconds=snippet_seconds,
        snippet_frames=snippet_frames,
        sample_rate=sample_rate,
        hop_length=hop_length,
        windows_per_track=windows_per_track,
        return_chroma=return_chroma,
        return_phase=return_phase,
        return_metadata=return_metadata,
        min_window_energy=min_window_energy,
        track_cache_size=track_cache_size,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=cqt_collate_fn,
        drop_last=True,
    )
