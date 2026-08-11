from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MultiResolutionSTFTConfig:
    fft_sizes: tuple[int, ...] = (512, 1024, 2048)
    hop_sizes: tuple[int, ...] = (128, 256, 512)
    win_lengths: tuple[int, ...] = (512, 1024, 2048)

    eps: float = 1e-7
    spectral_convergence_weight: float = 1.0
    log_magnitude_weight: float = 1.0


@dataclass
class VocoderLossConfig:
    lambda_adv: float = 1.0
    lambda_feature_matching: float = 2.0
    lambda_mrstft: float = 45.0

    mrstft: MultiResolutionSTFTConfig = field(default_factory=MultiResolutionSTFTConfig)


@dataclass
class VocoderGeneratorLossOutput:
    total: torch.Tensor
    adversarial: torch.Tensor
    feature_matching: torch.Tensor
    mrstft: torch.Tensor


@dataclass
class VocoderDiscriminatorLossOutput:
    total: torch.Tensor
    real: torch.Tensor
    fake: torch.Tensor


class SingleResolutionSTFTLoss(nn.Module):
    """
    One STFT resolution loss.

    Computes:
        1. Spectral convergence loss
        2. Log-magnitude L1 loss

    Input:
        real_audio: [B, 1, T] or [B, T]
        fake_audio: [B, 1, T] or [B, T]
    """

    def __init__(
        self,
        fft_size: int,
        hop_size: int,
        win_length: int,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()

        self.fft_size = int(fft_size)
        self.hop_size = int(hop_size)
        self.win_length = int(win_length)
        self.eps = float(eps)

        self.register_buffer(
            "window",
            torch.hann_window(self.win_length),
            persistent=False,
        )

    def _to_2d_audio(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            if x.shape[1] != 1:
                raise ValueError(f"Expected channel dimension 1, got {tuple(x.shape)}")
            x = x[:, 0, :]
        elif x.ndim != 2:
            raise ValueError(f"Expected audio [B,1,T] or [B,T], got {tuple(x.shape)}")

        return x

    def _stft_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        # STFT with complex half precision is experimental and can produce unstable
        # gradients under AMP. Always compute STFT loss in FP32.
        autocast_context = (
            torch.amp.autocast(device_type="cuda", enabled=False)
            if x.is_cuda
            else nullcontext()
        )

        with autocast_context:
            x = self._to_2d_audio(x).float()

            window = self.window.to(device=x.device, dtype=torch.float32)

            stft = torch.stft(
                x,
                n_fft=self.fft_size,
                hop_length=self.hop_size,
                win_length=self.win_length,
                window=window,
                center=True,
                return_complex=True,
            )

            magnitude = torch.abs(stft).clamp_min(self.eps)

        return magnitude

    def forward(
        self,
        fake_audio: torch.Tensor,
        real_audio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fake_mag = self._stft_magnitude(fake_audio)
        real_mag = self._stft_magnitude(real_audio)

        diff = real_mag - fake_mag

        spectral_convergence = torch.linalg.vector_norm(diff) / (
            torch.linalg.vector_norm(real_mag) + self.eps
        )

        log_magnitude = F.l1_loss(
            torch.log(fake_mag),
            torch.log(real_mag),
        )

        return spectral_convergence, log_magnitude


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss.

    This is the main non-adversarial waveform reconstruction loss for the
    vocoder. It compares generated and real audio at multiple time-frequency
    resolutions.
    """

    def __init__(
        self,
        config: MultiResolutionSTFTConfig = MultiResolutionSTFTConfig(),
    ) -> None:
        super().__init__()

        if not (
            len(config.fft_sizes)
            == len(config.hop_sizes)
            == len(config.win_lengths)
        ):
            raise ValueError("fft_sizes, hop_sizes, and win_lengths must match length.")

        self.config = config

        self.losses = nn.ModuleList(
            [
                SingleResolutionSTFTLoss(
                    fft_size=fft_size,
                    hop_size=hop_size,
                    win_length=win_length,
                    eps=config.eps,
                )
                for fft_size, hop_size, win_length in zip(
                    config.fft_sizes,
                    config.hop_sizes,
                    config.win_lengths,
                )
            ]
        )

    def forward(
        self,
        fake_audio: torch.Tensor,
        real_audio: torch.Tensor,
    ) -> torch.Tensor:
        total_sc = 0.0
        total_log_mag = 0.0

        for loss_fn in self.losses:
            sc, log_mag = loss_fn(fake_audio, real_audio)
            total_sc = total_sc + sc
            total_log_mag = total_log_mag + log_mag

        n = len(self.losses)

        total = (
            self.config.spectral_convergence_weight * total_sc / n
            + self.config.log_magnitude_weight * total_log_mag / n
        )

        return total


class VocoderLossBundle(nn.Module):
    """
    Container for vocoder losses.

    Kept as nn.Module so the STFT loss modules are properly moved with .to(device)
    if needed, although windows are created device-locally at runtime.
    """

    def __init__(
        self,
        config: VocoderLossConfig = VocoderLossConfig(),
    ) -> None:
        super().__init__()

        self.config = config
        self.mrstft_loss = MultiResolutionSTFTLoss(config.mrstft)


def lsgan_discriminator_loss(
    real_outputs: Sequence[torch.Tensor],
    fake_outputs: Sequence[torch.Tensor],
) -> VocoderDiscriminatorLossOutput:
    """
    LSGAN discriminator loss.

    Real targets are 1.
    Fake targets are 0.
    """
    if len(real_outputs) != len(fake_outputs):
        raise ValueError("real_outputs and fake_outputs must have same length.")

    loss_real = 0.0
    loss_fake = 0.0

    for real_pred, fake_pred in zip(real_outputs, fake_outputs):
        loss_real = loss_real + F.mse_loss(real_pred, torch.ones_like(real_pred))
        loss_fake = loss_fake + F.mse_loss(fake_pred, torch.zeros_like(fake_pred))

    n = max(1, len(real_outputs))

    loss_real = loss_real / n
    loss_fake = loss_fake / n
    total = loss_real + loss_fake

    return VocoderDiscriminatorLossOutput(
        total=total,
        real=loss_real,
        fake=loss_fake,
    )


def lsgan_generator_adversarial_loss(
    fake_outputs: Sequence[torch.Tensor],
) -> torch.Tensor:
    """
    LSGAN generator adversarial loss.

    Generator wants fake predictions to be classified as real, target 1.
    """
    loss = 0.0

    for fake_pred in fake_outputs:
        loss = loss + F.mse_loss(fake_pred, torch.ones_like(fake_pred))

    loss = loss / max(1, len(fake_outputs))

    return loss


def feature_matching_loss(
    real_feature_maps: Sequence[Sequence[torch.Tensor]],
    fake_feature_maps: Sequence[Sequence[torch.Tensor]],
) -> torch.Tensor:
    """
    Feature matching loss between discriminator intermediate activations.

    Real feature maps are detached so this loss updates only the generator
    during generator optimization.
    """
    if len(real_feature_maps) != len(fake_feature_maps):
        raise ValueError("real_feature_maps and fake_feature_maps must have same length.")

    total = 0.0
    count = 0

    for real_maps, fake_maps in zip(real_feature_maps, fake_feature_maps):
        if len(real_maps) != len(fake_maps):
            raise ValueError("Feature map lists must have matching lengths.")

        for real_fmap, fake_fmap in zip(real_maps, fake_maps):
            total = total + F.l1_loss(fake_fmap, real_fmap.detach())
            count += 1

    if count == 0:
        raise ValueError("No feature maps provided for feature matching loss.")

    return total / count


def compute_vocoder_generator_loss(
    discriminator_outputs: dict[str, list],
    fake_audio: torch.Tensor,
    real_audio: torch.Tensor,
    loss_bundle: VocoderLossBundle,
) -> VocoderGeneratorLossOutput:
    """
    Computes full generator loss:

        total =
            lambda_adv * adversarial
          + lambda_feature_matching * feature_matching
          + lambda_mrstft * multi_resolution_stft
    """
    cfg = loss_bundle.config

    loss_adv = lsgan_generator_adversarial_loss(
        discriminator_outputs["fake_outputs"]
    )

    loss_fm = feature_matching_loss(
        discriminator_outputs["real_feature_maps"],
        discriminator_outputs["fake_feature_maps"],
    )

    loss_mrstft = loss_bundle.mrstft_loss(
        fake_audio=fake_audio,
        real_audio=real_audio,
    )

    total = (
        cfg.lambda_adv * loss_adv
        + cfg.lambda_feature_matching * loss_fm
        + cfg.lambda_mrstft * loss_mrstft
    )

    return VocoderGeneratorLossOutput(
        total=total,
        adversarial=loss_adv,
        feature_matching=loss_fm,
        mrstft=loss_mrstft,
    )


def compute_vocoder_discriminator_loss(
    discriminator_outputs: dict[str, list],
) -> VocoderDiscriminatorLossOutput:
    """
    Computes discriminator LSGAN loss from combined MPD + MSD outputs.
    """
    return lsgan_discriminator_loss(
        real_outputs=discriminator_outputs["real_outputs"],
        fake_outputs=discriminator_outputs["fake_outputs"],
    )


def detach_vocoder_loss_dict(losses: dict[str, torch.Tensor | float]) -> dict[str, float]:
    out: dict[str, float] = {}

    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            out[key] = float(value.detach().cpu().item())
        else:
            out[key] = float(value)

    return out

