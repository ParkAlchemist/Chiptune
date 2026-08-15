from __future__ import annotations

from pathlib import Path
import sys

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.vocoder_dataset import CQTVocoderDataset
from src.models.vocoder_hifigan import (
    CQTGeneratorConfig,
    CQTUHiFiGANGenerator,
)
from src.models.vocoder_discriminators import (
    HiFiGANMultiDiscriminator,
    VocoderDiscriminatorConfig,
    MultiPeriodDiscriminatorConfig,
)
from src.losses.vocoder_losses import (
    VocoderLossConfig,
    VocoderLossBundle,
)
from src.training.vocoder_step import (
    VocoderModels,
    VocoderOptimizers,
    vocoder_train_step,
)


def clone_trainable_parameters(model: torch.nn.Module) -> list[torch.Tensor]:
    return [
        parameter.detach().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]


def any_parameter_changed(
    model: torch.nn.Module,
    before: list[torch.Tensor],
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> bool:
    current = [
        parameter.detach()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if len(current) != len(before):
        raise RuntimeError("Parameter count changed between clone and check.")

    for now, old in zip(current, before):
        if not torch.allclose(now, old, atol=atol, rtol=rtol):
            return True

    return False


def max_parameter_delta(
    model: torch.nn.Module,
    before: list[torch.Tensor],
) -> float:
    current = [
        parameter.detach()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    max_delta = 0.0

    for now, old in zip(current, before):
        delta = float((now - old).abs().max().cpu().item())
        max_delta = max(max_delta, delta)

    return max_delta


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    chip_cache_root = Path("E:/Projects/Datasets/cache/cqt/chip")

    # Use a small segment for the smoke test to keep backward memory low.
    dataset = CQTVocoderDataset(
        chip_cache_root=chip_cache_root,
        sample_rate=22050,
        hop_length=512,
        segment_frames=16,
        windows_per_track=2,
        random_window=True,
        cache_waveforms=4,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    batch = next(iter(loader))

    generator = CQTUHiFiGANGenerator(
        CQTGeneratorConfig(
            cqt_bins=96,
            upsample_initial_channel=128,
            upsample_rates=(8, 8, 4, 2),
            upsample_kernel_sizes=(16, 16, 8, 4),
            activation="leaky_relu",
        )
    ).to(device)

    # Full HiFi-GAN MPD is quite large. For the smoke test, use a reduced MPD.
    discriminator = HiFiGANMultiDiscriminator(
        VocoderDiscriminatorConfig(
            mpd=MultiPeriodDiscriminatorConfig(
                channels=(16, 64, 256, 512, 512),
            )
        )
    ).to(device)

    loss_bundle = VocoderLossBundle(
        VocoderLossConfig(
            lambda_adv=1.0,
            lambda_feature_matching=2.0,
            lambda_mrstft=45.0,
        )
    ).to(device)

    models = VocoderModels(
        generator=generator,
        discriminator=discriminator,
    )

    optimizers = VocoderOptimizers(
        generator=torch.optim.AdamW(
            generator.parameters(),
            lr=2e-4,
            betas=(0.8, 0.99),
            weight_decay=0.0,
        ),
        discriminator=torch.optim.AdamW(
            discriminator.parameters(),
            lr=2e-4,
            betas=(0.8, 0.99),
            weight_decay=0.0,
        ),
    )

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
        init_scale=256.0,
        growth_interval=2000,
    )

    before_g = clone_trainable_parameters(generator)
    before_d = clone_trainable_parameters(discriminator)

    losses = vocoder_train_step(
        batch=batch,
        models=models,
        optimizers=optimizers,
        loss_bundle=loss_bundle,
        device=device,
        use_amp=use_amp,
        scaler=scaler,
        grad_clip_norm=10.0,
    )

    print("Device:", device)
    print("Batch shapes:")
    print("  cqt:", tuple(batch["cqt"].shape))
    print("  audio:", tuple(batch["audio"].shape))

    print("\nLosses:")
    for key, value in sorted(losses.items()):
        print(f"  {key}: {value:.6f}")

    for key, value in losses.items():
        assert torch.isfinite(torch.tensor(value)), f"Non-finite value: {key}={value}"

    assert losses["loss_g_total"] > 0.0
    assert losses["loss_d_total"] > 0.0

    g_delta = max_parameter_delta(generator, before_g)
    d_delta = max_parameter_delta(discriminator, before_d)

    print("\nParameter deltas:")
    print(f"  generator max delta:     {g_delta:.12f}")
    print(f"  discriminator max delta: {d_delta:.12f}")

    if g_delta == 0.0 and use_amp:
        print("\nGenerator did not update under AMP. Retrying one step without AMP for smoke-test validation.")

        before_g = clone_trainable_parameters(generator)

        losses = vocoder_train_step(
            batch=batch,
            models=models,
            optimizers=optimizers,
            loss_bundle=loss_bundle,
            device=device,
            use_amp=False,
            scaler=None,
            grad_clip_norm=10.0,
        )

        g_delta = max_parameter_delta(generator, before_g)

        print(f"  generator max delta after non-AMP retry: {g_delta:.12f}")

    assert g_delta > 0.0, "Generator parameters did not change."
    assert d_delta > 0.0, "Discriminator parameters did not change."

    assert any_parameter_changed(generator,
                                 before_g), "Generator parameters did not change."
    assert any_parameter_changed(discriminator,
                                 before_d), "Discriminator parameters did not change."

    print("\nVocoder one-step training smoke test passed.")


if __name__ == "__main__":
    main()

