from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Mapping

import torch
from torch.nn import functional as F

DISCRIMINATOR_FAMILIES: tuple[str, ...] = (
    "mpd",
    "msd",
    "mrd",
)


@dataclass(frozen=True)
class FamilyLossOutput:
    total: torch.Tensor
    families: dict[str, torch.Tensor]


def mean_tensor_values(
    values: Sequence[torch.Tensor],
    *,
    name: str,
) -> torch.Tensor:
    if not values:
        raise ValueError(
            f"{name} requires at least one tensor."
        )

    return torch.stack(
        tuple(values)
    ).mean()


def mean_enabled_families(
    family_losses: Mapping[str, torch.Tensor],
    *,
    name: str,
) -> torch.Tensor:
    if not family_losses:
        raise ValueError(
            f"{name} requires at least one enabled "
            "discriminator family."
        )

    return mean_tensor_values(
        tuple(family_losses.values()),
        name=name,
    )


def generator_family_feature_matching_losses(
    discriminator_outputs: dict[str, list],
    *,
    normalization: str,
    scale_floor: float,
    eps: float,
) -> dict[str, torch.Tensor]:
    family_losses: dict[
        str,
        torch.Tensor,
    ] = {}

    for family in DISCRIMINATOR_FAMILIES:
        real_maps = discriminator_outputs[
            f"{family}_real_feature_maps"
        ]

        fake_maps = discriminator_outputs[
            f"{family}_fake_feature_maps"
        ]

        if not real_maps:
            continue

        family_losses[family] = (
            feature_matching_loss(
                real_maps,
                fake_maps,
                normalization=normalization,
                scale_floor=scale_floor,
                eps=eps,
            )
        )

    return family_losses


def family_balanced_feature_matching_loss(
    discriminator_outputs: dict[str, list],
    *,
    normalization: str,
    scale_floor: float,
    eps: float,
) -> FamilyLossOutput:
    family_losses = (
        generator_family_feature_matching_losses(
            discriminator_outputs,
            normalization=normalization,
            scale_floor=scale_floor,
            eps=eps,
        )
    )

    total = mean_enabled_families(
        family_losses,
        name="feature-matching loss",
    )

    return FamilyLossOutput(
        total=total,
        families=family_losses,
    )


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
            elif normalization == "real_rms":
                reference_rms = torch.sqrt(real_reference.float().square().mean() + eps)
                reference_scale = reference_rms.clamp_min(scale_floor)
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
