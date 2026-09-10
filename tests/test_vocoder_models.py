import torch

from src.models.vocoder_hifigan import (
    VocoderGeneratorModelConfig,
    CQTUHiFiGANGenerator,
)
from src.models.vocoder_discriminators import (
    HiFiGANMultiDiscriminator,
    VocoderDiscriminatorConfig,
    MultiPeriodDiscriminatorConfig,
)


def test_vocoder_generator_shape_32_frames_leaky_relu():
    model = CQTUHiFiGANGenerator(
        96,
        VocoderGeneratorModelConfig(
            upsample_initial_channel=128,
            upsample_rates=(8, 8, 4, 2),
            upsample_kernel_sizes=(16, 16, 8, 4),
            activation="leaky_relu",
        )
    )

    x = torch.randn(2, 96, 32)
    y = model(x)

    assert y.shape == (2, 1, 32 * 512)
    assert torch.isfinite(y).all()


def test_vocoder_generator_shape_32_frames_snake_beta():
    model = CQTUHiFiGANGenerator(
        96,
        VocoderGeneratorModelConfig(
            upsample_initial_channel=128,
            activation="snake_beta",
        )
    )

    x = torch.randn(2, 96, 32)
    y = model(x)

    assert y.shape == (2, 1, 32 * 512)
    assert torch.isfinite(y).all()


def test_vocoder_generator_accepts_4d_cqt():
    model = CQTUHiFiGANGenerator(
        96,
        VocoderGeneratorModelConfig(
            upsample_initial_channel=128,
        )
    )

    x = torch.randn(2, 1, 96, 16)
    y = model(x)

    assert y.shape == (2, 1, 16 * 512)
    assert torch.isfinite(y).all()


def test_vocoder_discriminator_outputs_synthetic():
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
        assert torch.isfinite(pred).all()
