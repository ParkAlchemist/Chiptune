import torch

from src.losses.vocoder_losses import (
    MultiResolutionSTFTConfig,
    VocoderLossConfig,
    VocoderLossBundle,
    compute_vocoder_discriminator_loss,
    compute_vocoder_generator_loss,
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


def test_vocoder_generator_loss_finite_with_fake_discriminator_outputs():
    bundle = VocoderLossBundle(
        VocoderLossConfig(
            lambda_adv=1.0,
            lambda_feature_matching=2.0,
            lambda_mrstft=1.0,
            mrstft=MultiResolutionSTFTConfig(
                fft_sizes=(256,),
                hop_sizes=(64,),
                win_lengths=(256,),
            ),
        )
    )

    real_audio = torch.randn(2, 1, 4096)
    fake_audio = torch.randn(2, 1, 4096, requires_grad=True)

    discriminator_outputs = {
        "fake_outputs": [
            torch.randn(2, 8, requires_grad=True),
            torch.randn(2, 8, requires_grad=True),
        ],
        "real_feature_maps": [
            [torch.randn(2, 4, 16), torch.randn(2, 8, 8)],
        ],
        "fake_feature_maps": [
            [
                torch.randn(2, 4, 16, requires_grad=True),
                torch.randn(2, 8, 8, requires_grad=True),
            ],
        ],
    }

    losses = compute_vocoder_generator_loss(
        discriminator_outputs=discriminator_outputs,
        fake_audio=fake_audio,
        real_audio=real_audio,
        loss_bundle=bundle,
    )
    assert losses.total.ndim == 0
    assert torch.isfinite(losses.total)
    assert torch.isfinite(losses.adversarial)
    assert torch.isfinite(losses.feature_matching)
    assert torch.isfinite(losses.mrstft)

    losses.total.backward()

    assert fake_audio.grad is not None
    assert torch.isfinite(fake_audio.grad).all()
