from __future__ import annotations

import torch
import torch.nn as nn


class SqueezeExciteBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super(SqueezeExciteBlock, self).__init__()

        hidden_channels = max(1, channels // reduction)

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y

