from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from src.losses.vocoder_losses import (
    VocoderLossBundle,
    compute_vocoder_discriminator_loss,
    compute_vocoder_generator_loss,
    detach_vocoder_loss_dict,
)


@dataclass
class VocoderModels:
    generator: nn.Module
    discriminator: nn.Module


@dataclass
class VocoderOptimizers:
    generator: torch.optim.Optimizer
    discriminator: torch.optim.Optimizer


def set_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = requires_grad


def vocoder_train_step(
    batch: dict[str, Any],
    models: VocoderModels,
    optimizers: VocoderOptimizers,
    loss_bundle: VocoderLossBundle,
    device: torch.device,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    dtype: str = "float16",
    grad_clip_generator: float | None = None,
    grad_clip_discriminator: float | None = None,
) -> dict[str, float]:
    """
    One HiFi-GAN-style vocoder training step.

    Batch:
        cqt:   [B, Bins, T]
        audio: [B, 1, T * hop_length]

    Update order:
        1. Discriminator update
        2. Generator update

    Losses:
        Discriminator:
            LSGAN real/fake loss

        Generator:
            adversarial loss
          + feature matching loss
          + multi-resolution STFT loss
    """

    cqt = batch["cqt"].to(device, non_blocking=True)
    real_audio = batch["audio"].to(device, non_blocking=True)

    generator = models.generator
    discriminator = models.discriminator

    amp_enabled = bool(use_amp and device.type == "cuda")

    if amp_enabled and scaler is None:
        raise ValueError("AMP is enabled, but scaler is None.")

    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "float32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"dtype {dtype} not supported.")

    autocast_context = (
        torch.amp.autocast(device_type="cuda", dtype=torch_dtype)
        if amp_enabled
        else nullcontext()
    )

    # ------------------------------------------------------------
    # 1. Train discriminator
    # ------------------------------------------------------------
    set_requires_grad(discriminator, True)

    optimizers.discriminator.zero_grad(set_to_none=True)

    # Generate fake audio without generator graph for discriminator update.
    with torch.no_grad():
        with autocast_context:
            fake_audio_for_d = generator(cqt)

    with autocast_context:
        discriminator_outputs_for_d = discriminator(
            real=real_audio,
            fake=fake_audio_for_d.detach(),
        )

        d_losses = compute_vocoder_discriminator_loss(
            discriminator_outputs=discriminator_outputs_for_d,
        )

    if amp_enabled:
        assert scaler is not None

        scaler.scale(d_losses.total).backward()

        if grad_clip_discriminator is not None:
            scaler.unscale_(optimizers.discriminator)
            torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(),
                max_norm=grad_clip_discriminator,
            )

        scaler.step(optimizers.discriminator)
    else:
        d_losses.total.backward()

        if grad_clip_discriminator is not None:
            torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(),
                max_norm=grad_clip_discriminator,
            )

        optimizers.discriminator.step()

    # ------------------------------------------------------------
    # 2. Train generator
    # ------------------------------------------------------------
    set_requires_grad(discriminator, False)

    optimizers.generator.zero_grad(set_to_none=True)

    with autocast_context:
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

    if amp_enabled:
        assert scaler is not None

        scaler.scale(g_losses.total).backward()

        if grad_clip_generator is not None:
            scaler.unscale_(optimizers.generator)
            torch.nn.utils.clip_grad_norm_(
                generator.parameters(),
                max_norm=grad_clip_generator,
            )

        scaler.step(optimizers.generator)
        scaler.update()
    else:
        g_losses.total.backward()

        if grad_clip_generator is not None:
            torch.nn.utils.clip_grad_norm_(
                generator.parameters(),
                max_norm=grad_clip_generator,
            )

        optimizers.generator.step()

    set_requires_grad(discriminator, True)

    losses = {
        "loss_g_total": g_losses.total,
        "loss_g_adv": g_losses.adversarial,
        "loss_g_fm": g_losses.feature_matching,
        "loss_g_mrstft": g_losses.mrstft,

        "loss_d_total": d_losses.total,
        "loss_d_real": d_losses.real,
        "loss_d_fake": d_losses.fake,

        "real_audio_min": real_audio.detach().min(),
        "real_audio_max": real_audio.detach().max(),
        "fake_audio_min": fake_audio.detach().min(),
        "fake_audio_max": fake_audio.detach().max(),
        "fake_audio_mean": fake_audio.detach().mean(),
        "fake_audio_std": fake_audio.detach().std(),
    }

    return detach_vocoder_loss_dict(losses)

