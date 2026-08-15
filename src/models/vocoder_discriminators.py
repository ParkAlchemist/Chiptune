from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm


NormType = Literal["weight",  "spectral", "none"]


def apply_norm(module: nn.Module, norm: NormType) -> nn.Module:
    if norm == "weight":
        return weight_norm(module)
    if norm == "spectral":
        return spectral_norm(module)
    if norm == "none":
        return module

    raise ValueError(f"Unknown norm type: {norm}")


def try_remove_weight_norm(module: nn.Module) -> None:
    try:
        remove_weight_norm(module)
    except ValueError:
        pass


@dataclass
class PeriodDiscriminatorConfig:
    period: int
    channels: tuple[int, ...] = (32, 128, 512, 1024, 1024)
    kernel_size: int = 5
    stride: int = 3
    norm: NormType = "weight"
    negative_slope: float = 0.2


class DiscriminatorPeriod(nn.Module):
    """
    HiFi-GAN-style period discriminator.
    It reshapes a waveform from:
        [B, 1, T]
    into:
        [B, C, T // period, period]
    """

    def __init__(self, config: PeriodDiscriminatorConfig) -> None:
        super().__init__()
        self.config = config
        self.period = int(config.period)

        channels = config.channels

        convs: list[nn.Module] = []

        in_channels = 1

        for i, out_channels in enumerate(channels):
            stride = (config.stride, 1) if i < len(channels) - 1 else (1, 1)

            convs.append(
                apply_norm(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=(config.kernel_size, 1),
                        stride=stride,
                        padding=((config.kernel_size - 1) // 2, 0),
                    ),
                    config.norm,
                )
            )

            in_channels = out_channels

        self.convs = nn.ModuleList(convs)

        self.conv_post = apply_norm(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=(3, 1),
                stride=(1, 1),
                padding=(1, 0),
            ),
            config.norm,
        )

    def _reshape_by_period(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, time = x.shape

        if time % self.period != 0:
            pad_length = self.period - (time % self.period)
            x = F.pad(x, (0, pad_length), mode="reflect")
            time = time + pad_length

        x = x.view(batch, channels, time // self.period, self.period)
        return x


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if x.ndim != 3:
            raise ValueError(f"Expected waveform [B, 1, T], got {tuple(x.shape)}]")

        feature_maps: list[torch.Tensor] = []

        x = self._reshape_by_period(x)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, negative_slope=self.config.negative_slope)
            feature_maps.append(x)

        x = self.conv_post(x)
        feature_maps.append(x)

        # Flatten patch predictions to [B, N]
        prediction = torch.flatten(x, start_dim=1)

        return prediction, feature_maps

    def remove_weight_norm(self) -> None:
        for conv in self.convs:
            try_remove_weight_norm(conv)

        try_remove_weight_norm(self.conv_post)


@dataclass
class MultiPeriodDiscriminatorConfig:
    periods: tuple[int, ...] = (2, 3, 5, 7, 11)
    channels: tuple[int, ...] = (32, 128, 512, 1024, 1024)
    norm: NormType = "weight"


class MultiPeriodDiscriminator(nn.Module):
    """
    Ensemble of period discriminators.
    Default periods follow the common HiFi-GAN MPD setup:
        [2, 3, 5, 7, 11]
    """

    def __init__(self, config: MultiPeriodDiscriminatorConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = MultiPeriodDiscriminatorConfig()
        self.config = config

        self.discriminators = nn.ModuleList(
            [
                DiscriminatorPeriod(
                    PeriodDiscriminatorConfig(
                        period=period,
                        channels=config.channels,
                        norm=config.norm,
                    )
                )
                for period in self.config.periods
            ]
        )

    def forward(
            self,
            real: torch.Tensor,
            fake: torch.Tensor,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[list[torch.Tensor]],
        list[list[torch.Tensor]]
    ]:
        real_outputs: list[torch.Tensor] = []
        fake_outputs: list[torch.Tensor] = []
        real_feature_maps: list[list[torch.Tensor]] = []
        fake_feature_maps: list[list[torch.Tensor]] = []

        for discriminator in self.discriminators:
            real_pred, real_fmap = discriminator(real)
            fake_pred, fake_fmap = discriminator(fake)

            real_outputs.append(real_pred)
            fake_outputs.append(fake_pred)
            real_feature_maps.append(real_fmap)
            fake_feature_maps.append(fake_fmap)

        return real_outputs, fake_outputs, real_feature_maps, fake_feature_maps

    def remove_weight_norm(self) -> None:
        for discriminator in self.discriminators:
            discriminator.remove_weight_norm()


@dataclass
class ScaleDiscriminatorConfig:
    channels: tuple[int, ...] = (128, 128, 256, 512, 1024, 1024)
    kernel_sizes: tuple[int, ...] = (15, 41, 41, 41, 41, 5)
    strides: tuple[int, ...] = (1, 2, 2, 4, 4, 1)
    groups: tuple[int, ...] = (1, 4, 16, 16, 16, 16)
    norm: NormType = "weight"
    negative_slope: float = 0.2


class DiscriminatorScale(nn.Module):
    """
    HiFi-GAN-style scale discriminator.
    It operates directly on waveform tensors:
        [B, 1, T]
    """

    def __init__(self, config: ScaleDiscriminatorConfig) -> None:
        super().__init__()
        self.config = config

        if not (
            len(config.channels)
            == len(config.kernel_sizes)
            == len(config.strides)
            == len(config.groups)
        ):
            raise ValueError("channels/kernel_sizes/strides/groups must have equal length")

        convs: list[nn.Module] = []

        in_channels = 1

        for out_channels, kernel_size, stride, groups, in zip(
            config.channels,
            config.kernel_sizes,
            config.strides,
            config.groups,
        ):
            convs.append(
                apply_norm(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=(kernel_size - 1) // 2,
                        groups=groups,
                    ),
                    config.norm,
                )
            )

            in_channels = out_channels

        self.convs = nn.ModuleList(convs)

        self.conv_post = apply_norm(
            nn.Conv1d(
                in_channels,
                1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            config.norm,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if x.ndim != 3:
            raise ValueError(f"Expected waveform [B, 1, T], got {tuple(x.shape)}]")

        feature_maps: list[torch.Tensor] = []

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, negative_slope=self.config.negative_slope)
            feature_maps.append(x)

        x = self.conv_post(x)
        feature_maps.append(x)

        prediction = torch.flatten(x, start_dim=1)

        return prediction, feature_maps

    def remove_weight_norm(self) -> None:
        for conv in self.convs:
            try_remove_weight_norm(conv)

        try_remove_weight_norm(self.conv_post)


@dataclass
class MultiScaleDiscriminatorConfig:
    num_scales: int = 3
    first_discriminator_norm: NormType = "spectral"
    other_discriminator_norm: NormType = "weight"


class MultiScaleDiscriminator(nn.Module):
    """
    Ensemble of scale discriminators.

    Scale 0:
        raw waveform
    Scale 1:
        pooled waveform
    Scale 2:
        pooled again
    """

    def __init__(self, config: MultiScaleDiscriminatorConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = MultiScaleDiscriminatorConfig()
        self.config = config

        discriminators: list[nn.Module] = []

        for scale_idx in range(config.num_scales):
            norm = (
                config.first_discriminator_norm
                if scale_idx == 0
                else config.other_discriminator_norm
            )

            discriminators.append(
                DiscriminatorScale(
                    ScaleDiscriminatorConfig(norm=norm),
                )
            )

        self.discriminators = nn.ModuleList(discriminators)

        self.pooling = nn.AvgPool1d(
            kernel_size=4,
            stride=2,
            padding=2,
            count_include_pad=False,
        )

    def forward(
            self,
            real: torch.Tensor,
            fake: torch.Tensor,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[list[torch.Tensor]],
        list[list[torch.Tensor]],
    ]:
        real_outputs: list[torch.Tensor] = []
        fake_outputs: list[torch.Tensor] = []
        real_feature_maps: list[list[torch.Tensor]] = []
        fake_feature_maps: list[list[torch.Tensor]] = []

        real_current = real
        fake_current = fake

        for scale_idx, discriminator in enumerate(self.discriminators):
            if scale_idx > 0:
                real_current = self.pooling(real_current)
                fake_current = self.pooling(fake_current)

            real_pred, real_fmap = discriminator(real_current)
            fake_pred, fake_fmap = discriminator(fake_current)

            real_outputs.append(real_pred)
            fake_outputs.append(fake_pred)
            real_feature_maps.append(real_fmap)
            fake_feature_maps.append(fake_fmap)

        return real_outputs, fake_outputs, real_feature_maps, fake_feature_maps

    def remove_weight_norm(self) -> None:
        for discriminator in self.discriminators:
            discriminator.remove_weight_norm()


@dataclass
class VocoderDiscriminatorConfig:
    mpd: MultiPeriodDiscriminatorConfig = field(default_factory=MultiPeriodDiscriminatorConfig)
    msd: MultiScaleDiscriminatorConfig = field(default_factory=MultiScaleDiscriminatorConfig)


class HiFiGANMultiDiscriminator(nn.Module):
    """
    Combined MPD + MSD wrapper.

    This object returns all predictions and feature maps needed for:
        - discriminator adversarial loss
        - generator adverarial loss
        - feature matching loss
    """

    def __init__(self, config: VocoderDiscriminatorConfig | None = None) -> None:
        super().__init__()

        if config is None:
            config = VocoderDiscriminatorConfig()

        self.config = config
        self.mpd = MultiPeriodDiscriminator(config.mpd)
        self.msd = MultiScaleDiscriminator(config.msd)

    def forward(self, real: torch.Tensor, fake: torch.Tensor) -> dict[str, list]:
        mpd_real, mpd_fake, mpd_real_fmaps, mpd_fake_fmaps = self.mpd(real, fake)
        msd_real, msd_fake, msd_real_fmaps, msd_fake_fmaps = self.msd(real, fake)

        return {
            "real_outputs": mpd_real + msd_real,
            "fake_outputs": mpd_fake + msd_fake,
            "real_feature_maps": mpd_real_fmaps + msd_real_fmaps,
            "fake_feature_maps": mpd_fake_fmaps + msd_fake_fmaps,
            "mpd_real_outputs": mpd_real,
            "mpd_fake_outputs": mpd_fake,
            "msd_real_outputs": msd_real,
            "msd_fake_outputs": msd_fake,
        }

    def remove_weight_norm(self) -> None:
        self.mpd.remove_weight_norm()
        self.msd.remove_weight_norm()


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    return sum(p.numel() for p in module.parameters())

