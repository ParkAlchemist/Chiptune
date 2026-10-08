from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.vocoder_config import MRSTFTConfig, VocoderLossConfig



@dataclass(frozen=True)
class WeightedLossComponent:
    """
    One generator loss before and after application of its configured scalar weight.
    """

    raw: torch.Tensor
    weight: float
    weighted: torch.Tensor


@dataclass(frozen=True)
class STFTResolutionLossOutput:
    fft_size: int
    hop_size: int
    win_length: int

    spectral_convergence: torch.Tensor
    log_magnitude: torch.Tensor

    spectral_convergence_weight: float
    log_magnitude_weight: float

    @property
    def total(self) -> torch.Tensor:
        return (
            self.spectral_convergence_weight
            * self.spectral_convergence
            + self.log_magnitude_weight
            * self.log_magnitude
        )


@dataclass(frozen=True)
class MultiResolutionSTFTLossOutput:
    total: torch.Tensor

    spectral_convergence: torch.Tensor
    log_magnitude: torch.Tensor

    resolutions: tuple[
        STFTResolutionLossOutput,
        ...,
    ]


@dataclass
class VocoderGeneratorLossOutput:
    total: torch.Tensor

    adversarial: WeightedLossComponent
    feature_matching: WeightedLossComponent
    mrstft: WeightedLossComponent
    waveform: WeightedLossComponent
    temporal_gradient: WeightedLossComponent

    mrstft_output: MultiResolutionSTFTLossOutput

    @property
    def learned_total(self) -> torch.Tensor:
        return self.adversarial.weighted + self.feature_matching.weighted

    @property
    def time_micro_total(self) -> torch.Tensor:
        return self.waveform.weighted + self.temporal_gradient.weighted

    @property
    def time_frequency_total(self) -> torch.Tensor:
        return self.mrstft.weighted


@dataclass
class VocoderDiscriminatorLossOutput:
    total: torch.Tensor
    real: torch.Tensor
    fake: torch.Tensor


def make_weighted_loss_component(
        raw: torch.Tensor,
        weight: float,
) -> WeightedLossComponent:
    return WeightedLossComponent(
        raw=raw,
        weight=weight,
        weighted=raw * float(weight)
    )


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
        config: MRSTFTConfig = MRSTFTConfig(),
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
    ) -> MultiResolutionSTFTLossOutput:
        resolution_outputs: list[STFTResolutionLossOutput] = []

        for (
                fft_size,
                hop_size,
                win_length,
                loss_func
        ) in zip(
            self.config.fft_sizes,
            self.config.hop_sizes,
            self.config.win_lengths,
            self.losses
        ):
            spectral_convergence, log_magnitude = loss_func(fake_audio, real_audio)

            resolution_outputs.append(
                STFTResolutionLossOutput(
                    fft_size=fft_size,
                    hop_size=hop_size,
                    win_length=win_length,
                    spectral_convergence=spectral_convergence,
                    log_magnitude=log_magnitude,
                    spectral_convergence_weight=self.config.spectral_convergence_weight,
                    log_magnitude_weight=self.config.log_magnitude_weight,
                )
            )

        if not resolution_outputs:
            raise RuntimeError(
                "MultiResolutionSTFTLoss contains no configured resolutions."
            )

        mean_spectral_convergence = torch.stack(
            [
                output.spectral_convergence for output in resolution_outputs
            ]
        ).mean()

        mean_log_magnitude = torch.stack(
            [
                output.log_magnitude for output in resolution_outputs
            ]
        ).mean()

        total = (
            self.config.spectral_convergence_weight * mean_spectral_convergence
            + self.config.log_magnitude_weight * mean_log_magnitude
        )

        return MultiResolutionSTFTLossOutput(
            total=total,
            spectral_convergence=mean_spectral_convergence,
            log_magnitude=mean_log_magnitude,
            resolutions=tuple(resolution_outputs),
        )


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
        self.temporal_gradient_loss = MultiScaleTemporalGradientLoss(
            lags=config.temporal_gradient.lags,
            normalization=config.temporal_gradient.normalization,
            scale_floor=config.temporal_gradient.scale_floor,
            eps=config.temporal_gradient.eps,
            edge_weighting=config.temporal_gradient.edge_weighting,
            edge_weight=config.temporal_gradient.edge_weight,
            max_edge_weight=config.temporal_gradient.max_edge_weight,
        ) if config.temporal_gradient.enabled else None


def lsgan_discriminator_loss(
    real_outputs: list[torch.Tensor],
    fake_outputs: list[torch.Tensor],
) -> VocoderDiscriminatorLossOutput:
    if not real_outputs:
        raise ValueError("No discriminator outputs supplied.")

    real_losses = [
        torch.mean((1.0 - output) ** 2)
        for output in real_outputs
    ]

    fake_losses = [
        torch.mean(output ** 2)
        for output in fake_outputs
    ]

    loss_real = torch.stack(real_losses).mean()
    loss_fake = torch.stack(fake_losses).mean()

    return VocoderDiscriminatorLossOutput(
        loss_real + loss_fake,
        loss_real,
        loss_fake,
    )


def lsgan_generator_adversarial_loss(
    fake_outputs: Sequence[torch.Tensor],
) -> torch.Tensor:
    """
    LSGAN generator adversarial loss.

    Generator wants fake predictions to be classified as real, target 1.
    """
    losses = [
        torch.mean((1.0 - output) ** 2)
        for output in fake_outputs
    ]

    loss = torch.stack(losses).mean()

    return loss


def waveform_loss(
        fake_audio: torch.Tensor,
        real_audio: torch.Tensor,
) -> torch.Tensor:

    return F.l1_loss(input=fake_audio, target=real_audio)


def generator_family_feature_matching_losses(
    discriminator_outputs: dict[str, list],
    *,
    normalization: str,
    scale_floor: float,
    eps: float,
) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}

    for family in ("mpd", "msd", "mrd"):
        real_maps = discriminator_outputs[
            f"{family}_real_feature_maps"
        ]
        fake_maps = discriminator_outputs[
            f"{family}_fake_feature_maps"
        ]

        if real_maps:
            losses[family] = feature_matching_loss(
                real_maps,
                fake_maps,
                normalization=normalization,
                scale_floor=scale_floor,
                eps=eps,
            )

    return losses


def feature_matching_loss(
    real_feature_maps: list[list[torch.Tensor]],
    fake_feature_maps: list[list[torch.Tensor]],
    *,
    normalization: str = "layer_mean",
    scale_floor: float = 1e-2,
    eps: float = 1e-8,
) -> torch.Tensor:
    if len(real_feature_maps) != len(fake_feature_maps):
        raise ValueError(
            "Real and fake feature-map collections must match."
        )

    discriminator_losses: list[torch.Tensor] = []

    for real_layers, fake_layers in zip(
        real_feature_maps,
        fake_feature_maps,
    ):
        if len(real_layers) != len(fake_layers):
            raise ValueError(
                "Real and fake discriminator layer counts "
                "must match."
            )

        layer_losses: list[torch.Tensor] = []

        for real_layer, fake_layer in zip(real_layers, fake_layers):
            real_reference = real_layer.detach()

            difference = F.l1_loss(fake_layer, real_reference, reduction="mean")

            if normalization == "layer_mean":
                layer_loss = difference
            elif normalization == "real_magnitude":
                reference_scale = real_reference.abs().mean().clamp_min(max(scale_floor, eps))
                layer_loss = difference / reference_scale
            else:
                raise ValueError(
                    f"Unknown feature-matching normalization: {normalization}"
                )

            layer_losses.append(layer_loss)

        discriminator_loss = torch.stack(layer_losses).mean()

        discriminator_losses.append(
            discriminator_loss
        )

    return torch.stack(
        discriminator_losses
    ).mean()


def lagged_difference(
        waveform: torch.Tensor,
        lag: int,
) -> torch.Tensor:
    if waveform.ndim != 3:
        raise ValueError(
            f"Expected waveform [B, C, T], got {tuple(waveform.shape)}"
        )

    if lag <= 0:
        raise ValueError(
            f"lag must be positive, got {lag}"
        )

    if waveform.shape[-1] <= lag:
        raise ValueError(
            f"Waveform must be longer than lag, got Length={waveform.shape[-1]}, lag={lag}"
        )

    return waveform[..., lag:] - waveform[..., :-lag]


class MultiScaleTemporalGradientLoss(nn.Module):
    def __init__(
            self,
            *,
            lags: tuple[int, ...] = (1, 2, 4),
            normalization: str = "none",
            scale_floor: float = 1e-3,
            eps: float = 1e-8,
            edge_weighting: bool = False,
            edge_weight: float = 1.0,
            max_edge_weight: float = 5.0,
    ) -> None:
        super().__init__()

        if not lags:
            raise ValueError("TemporalGradientLoss requires at least one lag")

        if any(lag <= 0 for lag in lags):
            raise ValueError("TemporalGradientLoss lags must be positive, "
                             f"got lags={lags}")

        if len(set(lags)) != len(lags):
            raise ValueError(
                f"Temporal gradient lags must be unique, got lags={lags}"
            )

        if normalization not in {"none", "target_magnitude"}:
            raise ValueError(
                f"Unsupported temporal gradient normalization: {normalization!r}"
            )

        if scale_floor <= 0.0:
            raise ValueError(
                f"Scale floor must be positive, got {scale_floor}"
            )

        if eps <= 0.0:
            raise ValueError(
                f"Eps must be positive, got {eps}"
            )

        if edge_weight < 0.0:
            raise ValueError(
                f"Edge weight must be positive, got {edge_weight}"
            )

        if max_edge_weight < 1.0:
            raise ValueError(
                f"max_edge_weight must be at least 1, got {max_edge_weight}"
            )

        self.lags = tuple(sorted(lags))
        self.normalization = normalization
        self.scale_floor = scale_floor
        self.eps = eps
        self.edge_weighting = edge_weighting
        self.edge_weight = edge_weight
        self.max_edge_weight = max_edge_weight

    def forward(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes do not match. "
                f"Prediction: {tuple(prediction.shape)}, target: {tuple(target.shape)}"
            )

        per_scale_losses: list[torch.Tensor] = []

        for lag in self.lags:
            prediction_difference = lagged_difference(prediction, lag)
            target_difference = lagged_difference(target, lag)

            absolute_error = (prediction_difference - target_difference).abs()

            if self.edge_weighting:
                target_scale = target_difference.detach().abs().mean(dim=-1, keepdim=True).clamp_min(max(self.scale_floor, self.eps))
                relative_edge_strength = target_difference.detach().abs() / target_scale

                weights = (1.0 + self.edge_weight * relative_edge_strength).clamp(max=self.max_edge_weight)

                absolute_error = absolute_error * weights

            scale_loss = absolute_error.mean()

            if self.normalization == "target_magnitude":
                reference_scale = target_difference.detach().abs().mean().clamp_min(max(self.scale_floor, self.eps))
                scale_loss = scale_loss / reference_scale

            per_scale_losses.append(scale_loss)

        return torch.stack(per_scale_losses).mean()



def generator_family_adversarial_losses(
    discriminator_outputs: dict[str, list],
) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}

    for family in ("mpd", "msd", "mrd"):
        fake_outputs = discriminator_outputs[
            f"{family}_fake_outputs"
        ]

        if fake_outputs:
            losses[family] = (
                lsgan_generator_adversarial_loss(
                    fake_outputs
                )
            )

    return losses


def compute_vocoder_generator_loss(
    discriminator_outputs: dict[str, list],
    fake_audio: torch.Tensor,
    real_audio: torch.Tensor,
    loss_bundle: VocoderLossBundle,
) -> VocoderGeneratorLossOutput:
    """
    Computes full generator loss
    """
    cfg = loss_bundle.config

    #------------------Raw losses-------------------

    loss_adversarial_raw = lsgan_generator_adversarial_loss(discriminator_outputs["fake_outputs"])

    feature_cfg = cfg.feature_matching

    loss_feature_matching_raw = feature_matching_loss(
        discriminator_outputs["real_feature_maps"],
        discriminator_outputs["fake_feature_maps"],
        normalization=feature_cfg.normalization,
        scale_floor=feature_cfg.scale_floor,
        eps=feature_cfg.eps,
    )

    mrstft_output = loss_bundle.mrstft_loss(
        fake_audio,
        real_audio,
    )

    loss_mrstft_raw = mrstft_output.total

    loss_waveform_raw = waveform_loss(
        fake_audio,
        real_audio,
    )

    if loss_bundle.temporal_gradient_loss is None:
        loss_temporal_gradient_raw = fake_audio.new_zeros(())
    else:
        loss_temporal_gradient_raw = loss_bundle.temporal_gradient_loss(
            fake_audio,
            real_audio,
        )

    #------------------Weighted losses-------------------

    adversarial = make_weighted_loss_component(
        raw=loss_adversarial_raw,
        weight=cfg.lambda_adversarial,
    )

    feature_matching = make_weighted_loss_component(
        raw=loss_feature_matching_raw,
        weight=cfg.lambda_feature_matching,
    )

    mrstft = make_weighted_loss_component(
        raw=loss_mrstft_raw,
        weight=cfg.lambda_mrstft,
    )

    waveform = make_weighted_loss_component(
        raw=loss_waveform_raw,
        weight=cfg.lambda_waveform,
    )

    temporal_gradient = make_weighted_loss_component(
        raw=loss_temporal_gradient_raw,
        weight=cfg.lambda_temporal_gradient,
    )

    total = (
        adversarial.weighted
        + feature_matching.weighted
        + mrstft.weighted
        + waveform.weighted
        + temporal_gradient.weighted
    )

    return VocoderGeneratorLossOutput(
        total=total,
        adversarial=adversarial,
        feature_matching=feature_matching,
        mrstft=mrstft,
        waveform=waveform,
        temporal_gradient=temporal_gradient,
        mrstft_output=mrstft_output,
    )


def discriminator_family_losses(
    discriminator_outputs: dict[str, list],
) -> dict[str, VocoderDiscriminatorLossOutput]:
    family_losses: dict[
        str,
        VocoderDiscriminatorLossOutput,
    ] = {}

    for family in ("mpd", "msd", "mrd"):
        real_outputs = discriminator_outputs[
            f"{family}_real_outputs"
        ]
        fake_outputs = discriminator_outputs[
            f"{family}_fake_outputs"
        ]

        if not real_outputs:
            continue

        family_losses[family] = (
            lsgan_discriminator_loss(
                real_outputs=real_outputs,
                fake_outputs=fake_outputs,
            )
        )

    return family_losses


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


def generator_loss_metrics(
    output: VocoderGeneratorLossOutput,
) -> dict[str, float]:
    metrics = {
        "loss_g_total": float(
            output.total.detach().cpu()
        ),

        "loss_g_adversarial": float(
            output.adversarial.raw.detach().cpu()
        ),
        "loss_g_feature_matching": float(
            output.feature_matching.raw.detach().cpu()
        ),
        "loss_g_mrstft": float(
            output.mrstft.raw.detach().cpu()
        ),
        "loss_g_waveform": float(
            output.waveform.raw.detach().cpu()
        ),
        "loss_g_temporal_gradient": float(
            output.temporal_gradient.raw.detach().cpu()
        ),

        "loss_g_weighted/adversarial": float(
            output.adversarial.weighted.detach().cpu()
        ),
        "loss_g_weighted/feature_matching": float(
            output.feature_matching.weighted.detach().cpu()
        ),
        "loss_g_weighted/mrstft": float(
            output.mrstft.weighted.detach().cpu()
        ),
        "loss_g_weighted/waveform": float(
            output.waveform.weighted.detach().cpu()
        ),
        "loss_g_weighted/temporal_gradient": float(
            output.temporal_gradient.weighted.detach().cpu()
        ),

        "loss_g_group/learned": float(
            output.learned_total.detach().cpu()
        ),
        "loss_g_group/time_micro": float(
            output.time_micro_total.detach().cpu()
        ),
        "loss_g_group/time_frequency": float(
            output.time_frequency_total.detach().cpu()
        ),

        "loss_g_mrstft_spectral_convergence": float(
            output.mrstft_output
            .spectral_convergence
            .detach()
            .cpu()
        ),
        "loss_g_mrstft_log_magnitude": float(
            output.mrstft_output
            .log_magnitude
            .detach()
            .cpu()
        ),
    }

    metrics.update(
        generator_loss_contribution_metrics(
            output=output,
        )
    )

    for resolution in (
        output.mrstft_output.resolutions
    ):
        prefix = (
            "loss_g_mrstft_resolution/"
            f"fft_{resolution.fft_size}"
            f"_hop_{resolution.hop_size}"
            f"_win_{resolution.win_length}"
        )

        metrics[
            f"{prefix}/spectral_convergence"
        ] = float(
            resolution
            .spectral_convergence
            .detach()
            .cpu()
        )

        metrics[
            f"{prefix}/log_magnitude"
        ] = float(
            resolution
            .log_magnitude
            .detach()
            .cpu()
        )

        metrics[
            f"{prefix}/total"
        ] = float(
            resolution.total
            .detach()
            .cpu()
        )

    return metrics


def generator_loss_contribution_metrics(
    output: VocoderGeneratorLossOutput,
    *,
    eps: float = 1e-8,
) -> dict[str, float]:
    weighted_components = {
        "adversarial": (
            output.adversarial.weighted
        ),
        "feature_matching": (
            output.feature_matching.weighted
        ),
        "mrstft": output.mrstft.weighted,
        "waveform": output.waveform.weighted,
        "temporal_gradient": (
            output.temporal_gradient.weighted
        ),
    }

    denominator = torch.stack(
        [
            value.detach().abs()
            for value in weighted_components.values()
        ]
    ).sum().clamp_min(eps)

    metrics: dict[str, float] = {}

    for name, value in weighted_components.items():
        contribution = (
            value.detach().abs()
            / denominator
        )

        metrics[
            f"loss_g_contribution/{name}"
        ] = float(
            contribution.cpu()
        )

    return metrics


