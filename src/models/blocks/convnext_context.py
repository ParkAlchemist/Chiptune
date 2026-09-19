from __future__ import annotations

import torch
import torch.nn as nn

from collections.abc import Sequence


class ConvNeXtContextBlock1d(nn.Module):
    def __init__(
            self,
            channels: int,
            kernel_size: int = 7,
            dilation: int = 1,
            expansion_ratio: int = 2,
            layer_scale_initial: float = 1e-6,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")

        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")

        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")

        if dilation <= 0:
            raise ValueError(f"dilation must be positive, got {dilation}")

        if expansion_ratio <= 0:
            raise ValueError(f"expansion_ratio must be positive, got {expansion_ratio}")

        if layer_scale_initial < 0.0:
            raise ValueError(
                "layer_scale_initial must be non-negative, "
                f"got {layer_scale_initial}"
            )

        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.expansion_ratio = expansion_ratio
        self.layer_scale_initial = layer_scale_initial

        padding = dilation * (kernel_size - 1) // 2

        self.depthwise = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            groups=channels,
        )

        self.norm = nn.LayerNorm(normalized_shape=channels)

        expanded_channels = channels * expansion_ratio

        self.expand = nn.Linear(
            in_features=channels,
            out_features=expanded_channels,
        )

        self.activation = nn.GELU()

        self.contract = nn.Linear(
            in_features=expanded_channels,
            out_features=channels
        )

        self.layer_scale = nn.Parameter(
            torch.full(
                (channels,),
                fill_value=layer_scale_initial,
                dtype=torch.float32,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise RuntimeError(f"ConvNeXtContextBlock1d only supports 3 dimensional tensors, "
                               f"got {tuple(x.shape)}")

        if x.shape[1] != self.channels:
            raise RuntimeError(f"ConvNeXtContextBlock1d channel mismatch: "
                               f"expected {self.channels}, got {x.shape[1]}")

        res = x

        x = self.depthwise(x)

        # [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)

        x = self.norm(x)

        # [B, T, C] -> [B, T, rC]
        x = self.expand(x)

        x = self.activation(x)

        # [B, T, rC] -> [B, T, C]
        x = self.contract(x)

        x = x * self.layer_scale

        # [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)

        if x.shape != res.shape:
            raise RuntimeError(f"ConvNeXtContextBlock1d forward dim mismatch: "
                               f"Input: {tuple(res.shape)}, Transformed {tuple(x.shape)}")

        return x + res


def _expand_per_block_setting(
    value: int | Sequence[int],
    *,
    number_of_blocks: int,
    name: str,
) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * number_of_blocks

    values = tuple(value)

    if len(values) != number_of_blocks:
        raise ValueError(
            f"{name} must contain exactly "
            f"{number_of_blocks} values, got {len(values)}"
        )

    return values


class ConvNeXtContextTrunk1d(nn.Module):
    def __init__(
            self,
            channels: int,
            number_of_blocks: int,
            kernel_sizes: int | Sequence[int] = 7,
            dilations: int | Sequence[int] = 1,
            expansion_ratio: int = 2,
            layer_scale_initial: float = 1e-6,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")

        if number_of_blocks <= 0:
            raise ValueError(f"number_of_blocks must be positive, got {number_of_blocks}")

        self.kernel_sizes = _expand_per_block_setting(
            value=kernel_sizes,
            number_of_blocks=number_of_blocks,
            name="kernel_sizes"
        )

        self.dilations = _expand_per_block_setting(
            value=dilations,
            number_of_blocks=number_of_blocks,
            name="dilations"
        )

        self.blocks = nn.ModuleList()

        for kernel_size, dilation in zip(self.kernel_sizes, self.dilations):
            self.blocks.append(
                ConvNeXtContextBlock1d(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    expansion_ratio=expansion_ratio,
                    layer_scale_initial=layer_scale_initial,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)

        return x

