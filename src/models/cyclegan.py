from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks.conv_norm_act import ConvNormAct
from src.models.blocks.downsample_block import DownsampleBlock
from src.models.blocks.upsample_block import UpsampleBlock
from src.models.blocks.residual_block import ResidualBlock
from src.models.blocks.spatial_self_attention import SpatialSelfAttention
from src.models.blocks.patch_discriminator_block import PatchDiscriminatorBlock

from ..models.model_utils import (
    NormType,
    PaddingMode,
    InitType,
    init_weights,
    count_parameters,
)


@dataclass
class GeneratorConfig:
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 32
    max_channels: int = 256

    num_downsamples: int = 2
    num_res_blocks: int = 6

    norm: NormType = "instance"
    padding_mode: PaddingMode = "reflect"

    residual_dropout: float = 0.0
    encoder_dropout: float = 0.0
    decoder_dropout: float = 0.0

    use_attention: bool = True
    attention_position: Literal["middle", "after_resblocks", "none"] = "middle"
    attention_reduction: int = 8

    upsample_mode: Literal["nearest", "bilinear"] = "nearest"

    final_activation: Literal["tanh", "none"] = "tanh"

    init_type: InitType = "normal"
    init_gain: float = 0.02


@dataclass
class DiscriminatorConfig:
    in_channels: int = 1
    base_channels: int = 32
    max_channels: int = 512

    num_layers: int = 3
    norm: NormType = "instance"

    spectral_norm: bool = False
    use_sigmoid: bool = False

    init_type: InitType = "normal"
    init_gain: float = 0.02


class AudioResnetGenerator(nn.Module):
    """
    ResNet-style CycleGAN generator for CQT tensors.

    Input:
    [B, 1, F, T]

    Output:
    [B, 1, F, T]

    For your current preprocessing:
    [B, 1, 96, 172] -> [B, 1, 96, 172]

    Structure:
    Initial 7x7 conv
    Downsample x N
    Residual blocks + optional bottleneck attention
    Upsample x N
    Final 7x7 conv + Tanh
    """

    def __init__(self, config: GeneratorConfig = GeneratorConfig()) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []

        current_channels = config.base_channels

        # Initial Projection
        layers.append(
            ConvNormAct(
                in_channels=config.in_channels,
                out_channels=current_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode=config.padding_mode,
                norm=config.norm,
                activation="relu",
                dropout=0.0,
            )
        )

        # Encoder/Downsampling
        for _ in range(config.num_downsamples):
            next_channels = min(current_channels * 2, config.max_channels)

            layers.append(
                DownsampleBlock(
                    in_channels=current_channels,
                    out_channels=next_channels,
                    norm=config.norm,
                    activation="relu",
                    padding_mode=config.padding_mode,
                    dropout=config.encoder_dropout,
                    spectral_norm=False,
                )
            )

            current_channels = next_channels

        # Bottleneck residual transformation
        attention_inserted = False
        midpoint = config.num_res_blocks // 2

        for block_idx in range(config.num_res_blocks):
            layers.append(
                ResidualBlock(
                    channels=current_channels,
                    padding_mode=config.padding_mode,
                    norm=config.norm,
                    dropout=config.residual_dropout,
                )
            )

            if config.use_attention and config.attention_position == "middle" and block_idx == midpoint:
                layers.append(
                    SpatialSelfAttention(
                        channels=current_channels,
                        reduction=config.attention_reduction,
                    )
                )
                attention_inserted = True

        if config.use_attention and config.attention_position == "after_resblocks" and not attention_inserted:
            layers.append(
                SpatialSelfAttention(
                    channels=config.base_channels,
                    reduction=config.attention_reduction,
                )
            )

        # Decoder/Upsampling
        for _ in range(config.num_downsamples):
            next_channels = max(config.base_channels, current_channels // 2)

            layers.append(
                UpsampleBlock(
                    in_channels=current_channels,
                    out_channels=next_channels,
                    norm=config.norm,
                    activation="relu",
                    padding_mode=config.padding_mode,
                    dropout=config.decoder_dropout,
                    upsample_mode=config.upsample_mode,
                )
            )

            current_channels = next_channels

        self.body = nn.Sequential(*layers)

        # Final projection
        final_layers: list[nn.Module] = [
            ConvNormAct(
                in_channels=current_channels,
                out_channels=config.out_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode=config.padding_mode,
                norm="none",
                activation="none",
                dropout=0.0,
            )
        ]

        if config.final_activation == "tanh":
            final_layers.append(nn.Tanh())

        self.final = nn.Sequential(*final_layers)

        self.apply(lambda m: init_weights(m, config.init_type, config.init_gain))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_spatial_size = x.shape[-2:]

        h = self.body(x)

        # Safety for odd dimensions or future parameter changes.
        if h.shape[-2:] != input_spatial_size:
            h = F.interpolate(
                h,
                size=input_spatial_size,
                mode="bilinear",
                align_corners=False,
            )

        y = self.final(h)

        return y


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator for CQT tensors.

    Input:
    [B, 1, F, T]

    Output:
    [B, 1, H_patch, W_patch]

    The discriminator evaluates local CQT patches rather than producing a single
    real/fake scalar.
    """

    def __init__(self, config: DiscriminatorConfig = DiscriminatorConfig()) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []

        current_channels = config.base_channels

        # First layer usually has no normalization
        layers.append(
            PatchDiscriminatorBlock(
                in_channels=config.in_channels,
                out_channels=current_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                norm=config.norm,
                spectral_norm=config.spectral_norm,
                use_norm=False,
            )
        )

        # Deeper strided layers
        for layer_idx in range(1, config.num_layers):
            next_channels = min(config.base_channels * (2 ** layer_idx), config.max_channels)

            layers.append(
                PatchDiscriminatorBlock(
                    in_channels=current_channels,
                    out_channels=next_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    norm=config.norm,
                    spectral_norm=config.spectral_norm,
                    use_norm=True,
                )
            )

            current_channels = next_channels

        # One stride-1 refinement layer
        next_channels = min(current_channels * 2, config.max_channels)

        layers.append(
            PatchDiscriminatorBlock(
                in_channels=current_channels,
                out_channels=next_channels,
                kernel_size=4,
                stride=1,
                padding=1,
                norm=config.norm,
                spectral_norm=config.spectral_norm,
                use_norm=True,
            )
        )

        current_channels = next_channels

        # Final patch prediction layer
        layers.append(
            nn.Conv2d(
                in_channels=current_channels,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=True,
            )
        )

        if config.use_sigmoid:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

        self.apply(lambda m: init_weights(m, config.init_type, config.init_gain))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiScalePatchGANDiscriminator(nn.Module):
    """
    Optional multi-scale discriminator wrapper.

    It runs several PatchGANDdiscriminators on progressively downsampled inputs.
    Useful later if one discriminator only learns very local texture and misses
    broader rhythmic/spectral structure.

    * For first baseline training, a single PatchGANDiscriminator is enough.
    """
    def __init__(
            self,
            config: DiscriminatorConfig = DiscriminatorConfig(),
            num_scales: int = 2,
    ) -> None:
        super().__init__()

        self.num_scales = num_scales

        self.discriminators = nn.ModuleList(
            [PatchGANDiscriminator(config) for _ in range(num_scales)]
        )

        self.downsample = nn.AvgPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
            count_include_pad=False,
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []

        current = x

        for discriminator in self.discriminators:
            outputs.append(discriminator(current))
            current = self.downsample(current)

        return outputs


def build_generator(
        **kwargs,
) -> AudioResnetGenerator:
    config = GeneratorConfig(**kwargs)
    return AudioResnetGenerator(config)


def build_discriminator(
        **kwargs,
) -> PatchGANDiscriminator:
    config = DiscriminatorConfig(**kwargs)
    return PatchGANDiscriminator(config)


def describe_model(model: nn.Module) -> str:
    params = count_parameters(model)
    return f"{model.__class__.__name__}(trainable_params={params})"

