from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.vocoder_dataset import CQTVocoderDataset
from src.models.vocoder_hifigan import (
    CQTGeneratorConfig,
    CQTUHiFiGANGenerator,
)


def main() -> None:
    chip_cache_root = Path("E:/Projects/Datasets/cache/cqt/chip")

    dataset = CQTVocoderDataset(
        chip_cache_root=chip_cache_root,
        sample_rate=22050,
        hop_length=512,
        segment_frames=32,
        windows_per_track=2,
        random_window=True,
        cache_waveforms=4,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    batch = next(iter(loader))

    cqt = batch["cqt"]
    audio = batch["audio"]

    print("Batch:")
    print("  cqt:", tuple(cqt.shape), cqt.dtype)
    print("  audio:", tuple(audio.shape), audio.dtype)
    print("  source_path:", batch["source_path"][0])
    print("  frame_start:", batch["frame_start"][0].item())
    print("  sample_start:", batch["sample_start"][0].item())

    config = CQTGeneratorConfig(
        cqt_bins=96,
        upsample_initial_channel=128,
        upsample_rates=(8, 8, 4, 2),
        upsample_kernel_sizes=(16, 16, 8, 4),
        activation="leaky_relu",
    )

    model = CQTUHiFiGANGenerator(config)

    with torch.no_grad():
        pred = model(cqt)

    print("\nModel:")
    print("  total_upsample_factor:", model.total_upsample_factor)
    print("  pred:", tuple(pred.shape), pred.dtype)

    assert model.total_upsample_factor == 512
    assert cqt.shape == (2, 96, 32)
    assert audio.shape == (2, 1, 32 * 512)
    assert pred.shape == audio.shape

    print("\nVocoder data/model smoke test passed.")


if __name__ == "__main__":
    main()

