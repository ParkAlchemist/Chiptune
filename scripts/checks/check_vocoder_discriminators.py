from __future__ import annotations

from pathlib import Path
import sys

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.models.vocoder_hifigan import (
    CQTGeneratorConfig,
    CQTUHiFiGANGenerator,
)
from src.models.vocoder_discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    HiFiGANMultiDiscriminator,
    count_parameters,
)


def print_outputs(name: str, outputs: list[torch.Tensor]) -> None:
    print(name)
    for i, output in enumerate(outputs):
        print(f"  [{i}] {tuple(output.shape)}")


def print_feature_maps(name: str, feature_maps: list[list[torch.Tensor]]) -> None:
    print(name)
    for disc_idx, fmap_list in enumerate(feature_maps):
        shapes = [tuple(x.shape) for x in fmap_list]
        print(f"  discriminator {disc_idx}: {shapes}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 2
    cqt_bins = 96
    segment_frames = 32
    hop_length = 512
    segment_samples = segment_frames * hop_length

    cqt = torch.randn(batch_size, cqt_bins, segment_frames, device=device).clamp(-1, 1)
    real_audio = torch.randn(batch_size, 1, segment_samples, device=device).clamp(-1, 1)

    generator = CQTUHiFiGANGenerator(
        CQTGeneratorConfig(
            cqt_bins=96,
            upsample_initial_channel=128,
            activation="leaky_relu",
        )
    ).to(device)

    with torch.no_grad():
        fake_audio = generator(cqt)

    assert fake_audio.shape == real_audio.shape

    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    combined = HiFiGANMultiDiscriminator().to(device)

    mpd.eval()
    msd.eval()
    combined.eval()

    with torch.no_grad():
        mpd_real, mpd_fake, mpd_real_fmaps, mpd_fake_fmaps = mpd(real_audio, fake_audio)
        msd_real, msd_fake, msd_real_fmaps, msd_fake_fmaps = msd(real_audio, fake_audio)
        combined_outputs = combined(real_audio, fake_audio)

    print("Device:", device)
    print("Input shapes:")
    print("  cqt:", tuple(cqt.shape))
    print("  real_audio:", tuple(real_audio.shape))
    print("  fake_audio:", tuple(fake_audio.shape))

    print("\nParameter counts:")
    print("  generator:", count_parameters(generator))
    print("  MPD:", count_parameters(mpd))
    print("  MSD:", count_parameters(msd))
    print("  combined:", count_parameters(combined))

    print()
    print_outputs("MPD real outputs:", mpd_real)
    print_outputs("MPD fake outputs:", mpd_fake)
    print_feature_maps("MPD real feature maps:", mpd_real_fmaps)

    print()
    print_outputs("MSD real outputs:", msd_real)
    print_outputs("MSD fake outputs:", msd_fake)
    print_feature_maps("MSD real feature maps:", msd_real_fmaps)

    print()
    print_outputs("Combined real outputs:", combined_outputs["real_outputs"])
    print_outputs("Combined fake outputs:", combined_outputs["fake_outputs"])

    assert len(mpd_real) == 5
    assert len(mpd_fake) == 5
    assert len(msd_real) == 3
    assert len(msd_fake) == 3

    assert len(combined_outputs["real_outputs"]) == 8
    assert len(combined_outputs["fake_outputs"]) == 8
    assert len(combined_outputs["real_feature_maps"]) == 8
    assert len(combined_outputs["fake_feature_maps"]) == 8

    for real_pred, fake_pred in zip(
        combined_outputs["real_outputs"],
        combined_outputs["fake_outputs"],
    ):
        assert real_pred.ndim == 2
        assert fake_pred.ndim == 2
        assert real_pred.shape[0] == batch_size
        assert fake_pred.shape[0] == batch_size

    print("\nVocoder discriminator smoke test passed.")


if __name__ == "__main__":
    main()

