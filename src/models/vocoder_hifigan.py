from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm


from src.config.vocoder_config import (
    VocoderGeneratorModelConfig,
    ActivationType, )


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


class SnakeBeta(nn.Module):
    """
    Lightweight SnakeBeta activation for [B, C, T] tensors.

    Formula:
        x + (1 / beta) * sin(alpha * x)^2
    """

    def __init__(
        self,
        channels: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        eps: float = 1e-9,
    ) -> None:
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(1, channels, 1) * alpha)
        self.beta = nn.Parameter(torch.ones(1, channels, 1) * beta)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sin_term = torch.sin(self.alpha * x)
        return x + sin_term.mul(sin_term) / (self.beta.abs() + self.eps)


def get_activation(
    activation: ActivationType,
    channels: int,
) -> nn.Module:
    if activation == "leaky_relu":
        return nn.LeakyReLU(0.1)

    if activation == "snake_beta":
        return SnakeBeta(channels)

    raise ValueError(f"Unknown activation: {activation}")


class ResBlock1D(nn.Module):
    """
    HiFi-GAN-style residual block.

    Each dilation stage:
        activation -> dilated Conv1d -> activation -> Conv1d -> residual add
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: tuple[int, ...],
        activation: ActivationType = "leaky_relu",
    ) -> None:
        super().__init__()

        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.acts1 = nn.ModuleList()
        self.acts2 = nn.ModuleList()

        for dilation in dilations:
            self.acts1.append(get_activation(activation, channels))
            self.convs1.append(
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=dilation,
                        padding=get_padding(kernel_size, dilation),
                    )
                )
            )

            self.acts2.append(get_activation(activation, channels))
            self.convs2.append(
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for act1, conv1, act2, conv2 in zip(
            self.acts1,
            self.convs1,
            self.acts2,
            self.convs2,
        ):
            residual = x
            y = conv1(act1(x))
            y = conv2(act2(y))
            x = y + residual

        return x

    def remove_weight_norm(self) -> None:
        for conv in self.convs1:
            remove_weight_norm(conv)

        for conv in self.convs2:
            remove_weight_norm(conv)


class CQTUHiFiGANGenerator(nn.Module):
    """
    Lightweight CQT-conditioned HiFi-GAN-style generator.

    Input:
        [B, Bins, T]
        or [B, 1, Bins, T]

    Output:
        [B, 1, T * Hop]

    This is designed for the current CQT cache:
        sample_rate = 22050
        hop_length = 512
        n_bins = 96
    """

    def __init__(
            self,
            cqt_bins: int,
            model_config: VocoderGeneratorModelConfig = VocoderGeneratorModelConfig(),
    ) -> None:
        super().__init__()
        self.config = model_config

        if len(model_config.upsample_rates) != len(model_config.upsample_kernel_sizes):
            raise ValueError("upsample_rates and upsample_kernel_sizes must match.")

        if len(model_config.resblock_kernel_sizes) != len(model_config.resblock_dilation_sizes):
            raise ValueError(
                "resblock_kernel_sizes and resblock_dilation_sizes must match."
            )

        self.conv_pre = weight_norm(
            nn.Conv1d(
                cqt_bins,
                model_config.upsample_initial_channel,
                kernel_size=7,
                stride=1,
                padding=3,
            )
        )

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        current_channels = model_config.upsample_initial_channel

        for upsample_rate, upsample_kernel_size in zip(
            model_config.upsample_rates,
            model_config.upsample_kernel_sizes,
        ):
            next_channels = current_channels // 2

            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        current_channels,
                        next_channels,
                        kernel_size=upsample_kernel_size,
                        stride=upsample_rate,
                        padding=(upsample_kernel_size - upsample_rate) // 2,
                    )
                )
            )

            for kernel_size, dilations in zip(
                model_config.resblock_kernel_sizes,
                model_config.resblock_dilation_sizes,
            ):
                self.resblocks.append(
                    ResBlock1D(
                        channels=next_channels,
                        kernel_size=kernel_size,
                        dilations=dilations,
                        activation=model_config.activation,
                    )
                )

            current_channels = next_channels

        self.activation_post = get_activation(model_config.activation, current_channels)

        self.conv_post = weight_norm(
            nn.Conv1d(
                current_channels,
                1,
                kernel_size=7,
                stride=1,
                padding=3,
            )
        )

        self.reset_parameters()

    @property
    def total_upsample_factor(self) -> int:
        factor = 1
        for rate in self.config.upsample_rates:
            factor *= rate
        return factor

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, cqt: torch.Tensor) -> torch.Tensor:
        if cqt.ndim == 4:
            # [B, 1, F, T] -> [B, F, T]
            cqt = cqt[:, 0, :, :]

        if cqt.ndim != 3:
            raise ValueError(
                f"Expected CQT shape [B,F,T] or [B,1,F,T], got {tuple(cqt.shape)}"
            )

        x = self.conv_pre(cqt)

        num_resblocks_per_stage = len(self.config.resblock_kernel_sizes)
        resblock_index = 0

        for up in self.ups:
            x = F.leaky_relu(x, negative_slope=0.1)
            x = up(x)

            fused = None

            for _ in range(num_resblocks_per_stage):
                rb_out = self.resblocks[resblock_index](x)
                fused = rb_out if fused is None else fused + rb_out
                resblock_index += 1

            x = fused / num_resblocks_per_stage

        x = self.activation_post(x)
        x = self.conv_post(x)

        if self.config.final_tanh:
            x = torch.tanh(x)

        return x

    def remove_weight_norm(self) -> None:
        remove_weight_norm(self.conv_pre)

        for up in self.ups:
            remove_weight_norm(up)

        for rb in self.resblocks:
            rb.remove_weight_norm()

        remove_weight_norm(self.conv_post)

