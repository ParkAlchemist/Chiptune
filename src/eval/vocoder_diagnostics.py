from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.cqt_dataset import discover_cache_records
from src.data.vocoder_dataset import load_processed_waveform, pad_1d


def normalize_path_string(path: str | Path) -> str:
    return str(Path(path)).replace("\\", "/").lower()


def find_cache_payload_for_source_path(
    chip_cache_root: Path,
    source_path: Path,
) -> tuple[dict[str, Any], Path]:
    records = discover_cache_records(chip_cache_root)

    wanted = normalize_path_string(source_path)

    # First try matching index metadata.
    for record in records:
        record_source = record.get("source_path")

        if record_source is None:
            continue

        if normalize_path_string(record_source) == wanted:
            cache_path = Path(record["cache_path"])
            payload = torch.load(cache_path, map_location="cpu")
            return payload, cache_path

    # Fallback: load payloads and inspect metadata.
    for record in records:
        cache_path = Path(record["cache_path"])
        payload = torch.load(cache_path, map_location="cpu")
        metadata = payload.get("metadata", {})
        record_source = metadata.get("source_path")

        if record_source is None:
            continue

        if normalize_path_string(record_source) == wanted:
            return payload, cache_path

    raise FileNotFoundError(
        "Could not find matching CQT cache payload for source_path:\n"
        f"{source_path}"
    )


def build_fixed_vocoder_batch(
    chip_cache_root: Path,
    source_path: Path,
    frame_start: int,
    segment_frames: int,
    sample_rate: int,
    hop_length: int,
    trim_top_db: float = 60.0,
    target_peak: float = 0.95,
) -> dict[str, Any]:
    payload, cache_path = find_cache_payload_for_source_path(
        chip_cache_root=chip_cache_root,
        source_path=source_path,
    )

    metadata = payload.get("metadata", {})
    cqt = payload["cqt_mag"].float()

    # [1, F, T] -> [F, T]
    if cqt.ndim == 3 and cqt.shape[0] == 1:
        cqt = cqt[0]

    if cqt.ndim != 2:
        raise ValueError(f"Expected CQT [F,T], got {tuple(cqt.shape)}")

    frame_end = frame_start + segment_frames

    cqt_segment = cqt[:, frame_start:frame_end]

    if cqt_segment.shape[-1] < segment_frames:
        cqt_segment = torch.nn.functional.pad(
            cqt_segment,
            (0, segment_frames - cqt_segment.shape[-1]),
            value=-1.0,
        )

    waveform = load_processed_waveform(
        source_path=source_path,
        sample_rate=sample_rate,
        trim_top_db=trim_top_db,
        target_peak=target_peak,
    )

    segment_samples = segment_frames * hop_length
    sample_start = frame_start * hop_length
    sample_end = sample_start + segment_samples

    audio_segment = waveform[sample_start:sample_end]
    audio_segment = pad_1d(audio_segment, segment_samples).unsqueeze(0)

    return {
        "cqt": cqt_segment.unsqueeze(0),          # [1, F, T]
        "audio": audio_segment.unsqueeze(0),     # [1, 1, samples]
        "cache_path": [str(cache_path)],
        "source_path": [metadata.get("source_path", str(source_path))],
        "frame_start": torch.tensor([frame_start], dtype=torch.long),
        "frame_end": torch.tensor([frame_end], dtype=torch.long),
        "sample_start": torch.tensor([sample_start], dtype=torch.long),
        "sample_end": torch.tensor([sample_end], dtype=torch.long),
    }


def tensor_to_audio_np(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float()

    if x.ndim == 3:
        x = x[0, 0]
    elif x.ndim == 2:
        x = x[0]
    elif x.ndim == 1:
        pass
    else:
        raise ValueError(f"Unexpected audio tensor shape: {tuple(x.shape)}")

    y = x.numpy().astype(np.float32)
    return np.clip(y, -1.0, 1.0)


def tensor_to_cqt_np(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float()

    if x.ndim == 4:
        x = x[0, 0]
    elif x.ndim == 3:
        if x.shape[0] == 1:
            x = x[0]
        else:
            raise ValueError(f"Unexpected 3D CQT tensor shape: {tuple(x.shape)}")
    elif x.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected CQT tensor shape: {tuple(x.shape)}")

    return x.numpy().astype(np.float32)
