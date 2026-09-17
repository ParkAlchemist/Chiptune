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


@dataclass
class VocoderMicroStepResult:
    losses: dict[str, float]
    batch_size: int


def set_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = requires_grad


def vocoder_train_micro_step(
        *,
        batch: dict,
        models: VocoderModels,
        loss_bundle: VocoderLossBundle,
        device: torch.device,
        use_amp: bool = False,
        scaler: torch.amp.GradScaler | None = None,
        dtype: str,
        loss_divisor: int,
) -> VocoderMicroStepResult:
    if loss_divisor < 1:
        raise ValueError(f"loss_divisor must be >= 1, got {loss_divisor}")

    generator = models.generator
    discriminator = models.discriminator

    cqt = batch["cqt"].to(device, non_blocking=True)
    real_audio = batch["audio"].to(device, non_blocking=True)

    autocast_dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]
    autocast_enabled = (use_amp and device.type == "cuda")

    #--------------------------------------------------------
    # Discriminator Backwards Pass
    #--------------------------------------------------------
    with torch.amp.autocast(
        device_type=device.type,
        enabled=autocast_enabled,
        dtype=autocast_dtype,
    ):
        with torch.no_grad():
            fake_audio_for_d = generator(cqt)

        discriminator_outputs_d = discriminator(
            real_audio,
            fake=fake_audio_for_d.detach(),
        )

        discriminator_losses = compute_vocoder_discriminator_loss(discriminator_outputs_d)

        loss_d_total = discriminator_losses.total
        loss_d_backward = loss_d_total / loss_divisor

    if scaler is not None and use_amp:
        scaler.scale(loss_d_backward).backward()
    else:
        loss_d_backward.backward()

    #-------------------------------------------------------------
    # Generator Backwards Pass
    #-------------------------------------------------------------
    set_requires_grad(discriminator, False)

    try:
        with torch.amp.autocast(
            device_type=device.type,
            enabled=autocast_enabled,
            dtype=autocast_dtype,
        ):
            fake_audio_for_g = generator(cqt)

            discriminator_outputs_g = discriminator(
                real_audio,
                fake=fake_audio_for_g,
            )

            generator_losses = compute_vocoder_generator_loss(
                discriminator_outputs=discriminator_outputs_g,
                fake_audio=fake_audio_for_g,
                real_audio=real_audio,
                loss_bundle=loss_bundle,
            )

            loss_g_total = generator_losses.total
            loss_g_backward = loss_g_total / loss_divisor

        if scaler is not None and use_amp:
            scaler.scale(loss_g_backward).backward()
        else:
            loss_g_backward.backward()
    finally:
        set_requires_grad(discriminator, True)

    losses = {
        "loss_g_total": float(loss_g_total.detach().cpu()),
        "loss_d_total": float(loss_d_total.detach().cpu()),
        "loss_g_adversarial": float(generator_losses.adversarial.detach().cpu()),
        "loss_g_feature_matching": float(generator_losses.feature_matching.detach().cpu()),
        "loss_g_mrstft": float(generator_losses.mrstft.detach().cpu()),
        "loss_d_real": float(discriminator_losses.real.detach().cpu()),
        "loss_d_fake": float(discriminator_losses.fake.detach().cpu()),
    }

    return VocoderMicroStepResult(losses=losses, batch_size=int(cqt.shape[0]))


def finish_vocoder_optimizer_step(
        *,
        models: VocoderModels,
        optimizers: VocoderOptimizers,
        scaler: torch.amp.GradScaler | None = None,
        use_amp: bool = False,
        grad_clip_generator: float | None = None,
        grad_clip_discriminator: float | None = None,
) -> dict[str, float | None]:

    generator_grad_norm: float = 0.0
    discriminator_grad_norm: float = 0.0

    if scaler is not None and use_amp:
        scaler.unscale_(optimizers.generator)
        scaler.unscale_(optimizers.discriminator)

    if grad_clip_generator is not None:
        generator_grad_norm_tensor = (
            torch.nn.utils.clip_grad_norm_(
                models.generator.parameters(),
                max_norm=grad_clip_generator,
            )
        )

        generator_grad_norm = float(
            generator_grad_norm_tensor.detach().cpu()
        )

    if grad_clip_discriminator is not None:
        discriminator_grad_norm_tensor = (
            torch.nn.utils.clip_grad_norm_(
                models.discriminator.parameters(),
                max_norm=grad_clip_discriminator,
            )
        )

        discriminator_grad_norm = float(
            discriminator_grad_norm_tensor.detach().cpu()
        )

    if scaler is not None and use_amp:
        scaler.step(optimizers.discriminator)
        scaler.step(optimizers.generator)
        scaler.update()
    else:
        optimizers.discriminator.step()
        optimizers.generator.step()

    optimizers.discriminator.zero_grad(set_to_none=True)
    optimizers.generator.zero_grad(set_to_none=True)

    return {
        "grad_norm_generator": float(generator_grad_norm),
        "grad_norm_discriminator": float(discriminator_grad_norm),
    }


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

