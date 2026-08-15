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
from src.models.vocoder_discriminators import HiFiGANMultiDiscriminator
from src.losses.vocoder_losses import (
    VocoderLossConfig,
    VocoderLossBundle,
    compute_vocoder_generator_loss,
    compute_vocoder_discriminator_loss,
)


def assert_finite(name: str, value: torch.Tensor) -> None:
    assert torch.isfinite(value).all(), f"{name} is not finite: {value}"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 2
    cqt_bins = 96
    segment_frames = 32
    hop_length = 512
    segment_samples = segment_frames * hop_length

    cqt = torch.randn(
        batch_size,
        cqt_bins,
        segment_frames,
        device=device,
    ).clamp(-1.0, 1.0)

    real_audio = torch.randn(
        batch_size,
        1,
        segment_samples,
        device=device,
    ).clamp(-1.0, 1.0)

    generator = CQTUHiFiGANGenerator(
        CQTGeneratorConfig(
            cqt_bins=96,
            upsample_initial_channel=128,
            activation="leaky_relu",
        )
    ).to(device)

    discriminator = HiFiGANMultiDiscriminator().to(device)

    loss_bundle = VocoderLossBundle(
        VocoderLossConfig(
            lambda_adv=1.0,
            lambda_feature_matching=2.0,
            lambda_mrstft=45.0,
        )
    ).to(device)

    generator.train()
    discriminator.train()

    # ------------------------------------------------------------
    # Generator loss check
    # ------------------------------------------------------------
    fake_audio = generator(cqt)

    discriminator_outputs_for_g = discriminator(
        real=real_audio,
        fake=fake_audio,
    )

    g_losses = compute_vocoder_generator_loss(
        discriminator_outputs=discriminator_outputs_for_g,
        fake_audio=fake_audio,
        real_audio=real_audio,
        loss_bundle=loss_bundle,
    )

    print("Device:", device)
    print("Generator losses:")
    print(f"  total:            {float(g_losses.total.detach().cpu()):.6f}")
    print(f"  adversarial:      {float(g_losses.adversarial.detach().cpu()):.6f}")
    print(f"  feature_matching: {float(g_losses.feature_matching.detach().cpu()):.6f}")
    print(f"  mrstft:           {float(g_losses.mrstft.detach().cpu()):.6f}")

    assert_finite("g_total", g_losses.total)
    assert_finite("g_adversarial", g_losses.adversarial)
    assert_finite("g_feature_matching", g_losses.feature_matching)
    assert_finite("g_mrstft", g_losses.mrstft)

    generator.zero_grad(set_to_none=True)
    discriminator.zero_grad(set_to_none=True)

    g_losses.total.backward()

    generator_grads = [
        p.grad
        for p in generator.parameters()
        if p.requires_grad
    ]

    assert any(g is not None for g in generator_grads)
    assert all(torch.isfinite(g).all() for g in generator_grads if g is not None)

    print("  generator backward: ok")

    # ------------------------------------------------------------
    # Discriminator loss check
    # ------------------------------------------------------------
    generator.zero_grad(set_to_none=True)
    discriminator.zero_grad(set_to_none=True)

    with torch.no_grad():
        fake_audio_detached = generator(cqt)

    discriminator_outputs_for_d = discriminator(
        real=real_audio,
        fake=fake_audio_detached.detach(),
    )

    d_losses = compute_vocoder_discriminator_loss(
        discriminator_outputs=discriminator_outputs_for_d,
    )

    print("\nDiscriminator losses:")
    print(f"  total: {float(d_losses.total.detach().cpu()):.6f}")
    print(f"  real:  {float(d_losses.real.detach().cpu()):.6f}")
    print(f"  fake:  {float(d_losses.fake.detach().cpu()):.6f}")

    assert_finite("d_total", d_losses.total)
    assert_finite("d_real", d_losses.real)
    assert_finite("d_fake", d_losses.fake)

    d_losses.total.backward()

    discriminator_grads = [
        p.grad
        for p in discriminator.parameters()
        if p.requires_grad
    ]

    assert any(g is not None for g in discriminator_grads)
    assert all(torch.isfinite(g).all() for g in discriminator_grads if g is not None)

    print("  discriminator backward: ok")

    print("\nVocoder loss smoke test passed.")


if __name__ == "__main__":
    main()

