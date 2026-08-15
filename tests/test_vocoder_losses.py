import torch

from src.losses.vocoder_losses import (
    VocoderLossConfig,
    VocoderLossBundle,
    MultiResolutionSTFTConfig,
    compute_vocoder_discriminator_loss,
)


def test_mrstft_loss_finite():
    bundle = VocoderLossBundle(
        VocoderLossConfig(
            mrstft=MultiResolutionSTFTConfig(
                fft_sizes=(256, 512),
                hop_sizes=(64, 128),
                win_lengths=(256, 512),
            )
        )
    )

    real = torch.randn(2, 1, 4096)
    fake = torch.randn(2, 1, 4096)

    loss = bundle.mrstft_loss(fake, real)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_vocoder_discriminator_loss_finite():
    outputs = {
        "real_outputs": [torch.ones(2, 8), torch.ones(2, 4)],
        "fake_outputs": [torch.zeros(2, 8), torch.zeros(2, 4)],
    }

    losses = compute_vocoder_discriminator_loss(outputs)

    assert losses.total.ndim == 0
    assert torch.isfinite(losses.total)
    assert torch.isfinite(losses.real)
    assert torch.isfinite(losses.fake)

