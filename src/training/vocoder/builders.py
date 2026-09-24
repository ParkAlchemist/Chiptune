from __future__ import annotations

import torch

from src.config.vocoder_config import (
    OptimizerConfig,
    SchedulerConfig,
    VocoderExperimentConfig,
)
from src.config.vocoder_config_adapter import build_loss_config
from src.losses.vocoder_losses import VocoderLossBundle
from src.models.vocoder_discriminators import (
    HiFiGANMultiDiscriminator,
)
from src.models.vocoder_hifigan import (
    CQTUHiFiGANGenerator,
)
from src.training.vocoder.context import TrainingComponents
from src.training.vocoder_step import (
    VocoderModels,
    VocoderOptimizers,
)


def build_optimizer(
    parameters,
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
    if config.name == "adam":
        optimizer_class = torch.optim.Adam
    elif config.name == "adamw":
        optimizer_class = torch.optim.AdamW
    else:
        raise ValueError(
            f"Unsupported optimizer: {config.name!r}"
        )

    return optimizer_class(
        parameters,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        eps=config.eps,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
):
    if not config.enabled or config.name == "none":
        return None

    if config.name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.gamma,
        )

    raise ValueError(
        f"Unsupported scheduler: {config.name!r}"
    )


def build_training_components(
    config: VocoderExperimentConfig,
    device: torch.device,
) -> TrainingComponents:
    print("Building models...")

    generator = CQTUHiFiGANGenerator(
        config.data.cqt_bins,
        config.generator,
    ).to(device)

    generator.to(device)

    discriminator = HiFiGANMultiDiscriminator(
        config.discriminator,
    ).to(device)

    models = VocoderModels(
        generator=generator,
        discriminator=discriminator,
    )

    optimizers = VocoderOptimizers(
        generator=build_optimizer(
            generator.parameters(),
            config.optimizer_generator,
        ),
        discriminator=build_optimizer(
            discriminator.parameters(),
            config.optimizer_discriminator,
        ),
    )

    scheduler_generator = build_scheduler(
        optimizers.generator,
        config.scheduler_generator,
    )

    scheduler_discriminator = build_scheduler(
        optimizers.discriminator,
        config.scheduler_discriminator,
    )

    loss_bundle = VocoderLossBundle(
        build_loss_config(config)
    ).to(device)

    scaler_enabled = (
        config.amp.enabled
        and device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=scaler_enabled,
        init_scale=config.amp.initial_scale,
        growth_interval=config.amp.growth_interval,
    )

    return TrainingComponents(
        models=models,
        optimizers=optimizers,
        scheduler_generator=scheduler_generator,
        scheduler_discriminator=scheduler_discriminator,
        loss_bundle=loss_bundle,
        scaler=scaler,
    )

