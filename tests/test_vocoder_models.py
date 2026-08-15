import torch

from src.models.vocoder_hifigan import (
    CQTGeneratorConfig,
    CQTUHiFiGANGenerator,
)
from src.models.vocoder_discriminators import (
    HiFiGANMultiDiscriminator,
    VocoderDiscriminatorConfig,
    MultiPeriodDiscriminatorConfig,
)


def test_vocoder_generator_shape_32_frames():
    model = CQTUHiFiGANGenerator(
        CQTGeneratorConfig(
            cqt_bins=96,
            upsample_initial_channel=128,
            upsample_rates=(8, 8, 4, 2),
            upsample_kernel_sizes=(16, 16, 8, 4),
            activation="leaky_relu",
        )
    )

    x = torch.randn(2, 96, 32)
    y = model(x)

    assert y.shape == (2, 1, 32 * 512)


def test_vocoder_generator_shape_4d_input():
    model = CQTUHiFiGANGenerator(CQTGeneratorConfig(cqt_bins=96))

    x = torch.randn(2, 1, 96, 16)
    y = model(x)

    assert y.shape == (2, 1, 16 * 512)


def test_vocoder_discriminator_outputs():
    discriminator = HiFiGANMultiDiscriminator(
        VocoderDiscriminatorConfig(
            mpd=MultiPeriodDiscriminatorConfig(
                channels=(8, 32, 128, 256, 256),
            )
        )
    )

    real = torch.randn(2, 1, 8192)
    fake = torch.randn(2, 1, 8192)

    out = discriminator(real, fake)

    assert len(out["real_outputs"]) == 8
    assert len(out["fake_outputs"]) == 8
    assert len(out["real_feature_maps"]) == 8
    assert len(out["fake_feature_maps"]) == 8

    for pred in out["real_outputs"] + out["fake_outputs"]:
        assert pred.ndim == 2
        assert pred.shape[0] == 2

