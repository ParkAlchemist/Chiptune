from __future__ import annotations

from math import prod
from pathlib import Path
from typing import Iterable

import librosa

from src.config.vocoder_config import (
    MRSTFTConfig,
    OptimizerConfig,
    SchedulerConfig,
    VocoderExperimentConfig,
)


class ConfigValidationError(ValueError):
    """Raised when configuration values are mutually incompatible."""


def _format_errors(errors: Iterable[str]) -> str:
    errors = list(errors)

    return (
        "Invalid vocoder experiment configuration:\n"
        + "\n".join(f"  - {error}" for error in errors)
    )


def _validate_positive_int(
    value: int,
    *,
    path: str,
    errors: list[str],
    allow_zero: bool = False,
) -> None:
    minimum = 0 if allow_zero else 1

    if value < minimum:
        comparison = "non-negative" if allow_zero else "positive"
        errors.append(
            f"{path} must be {comparison}, got {value}."
        )


def _validate_probability(
    value: float,
    *,
    path: str,
    errors: list[str],
) -> None:
    if not 0.0 <= value <= 1.0:
        errors.append(
            f"{path} must be between 0.0 and 1.0, got {value}."
        )


def _validate_data_config(
    config: VocoderExperimentConfig,
    errors: list[str],
) -> None:
    data = config.data

    if not data.chip_cache_root.strip():
        errors.append(
            "data.chip_cache_root must not be empty."
        )

    _validate_positive_int(
        data.sample_rate,
        path="data.sample_rate",
        errors=errors,
    )
    _validate_positive_int(
        data.hop_length,
        path="data.hop_length",
        errors=errors,
    )
    _validate_positive_int(
        data.cqt_bins,
        path="data.cqt_bins",
        errors=errors,
    )
    _validate_positive_int(
        data.bins_per_octave,
        path="data.bins_per_octave",
        errors=errors,
    )
    _validate_positive_int(
        data.segment_frames,
        path="data.segment_frames",
        errors=errors,
    )
    _validate_positive_int(
        data.windows_per_track,
        path="data.windows_per_track",
        errors=errors,
    )
    _validate_positive_int(
        data.batch_size,
        path="data.batch_size",
        errors=errors,
    )
    _validate_positive_int(
        data.num_workers,
        path="data.num_workers",
        errors=errors,
        allow_zero=True,
    )
    _validate_positive_int(
        data.cache_waveforms,
        path="data.cache_waveforms",
        errors=errors,
        allow_zero=True,
    )

    if data.fmin <= 0.0:
        errors.append(
            f"data.fmin must be positive, got {data.fmin}."
        )

    if data.num_workers == 0 and data.persistent_workers:
        errors.append(
            "data.persistent_workers must be false when "
            "data.num_workers is 0."
        )

    if data.num_workers > 0:
        _validate_positive_int(
            data.prefetch_factor,
            path="data.prefetch_factor",
            errors=errors,
        )

    if data.sample_rate > 0 and data.fmin > 0:
        frequencies = librosa.cqt_frequencies(
            n_bins=data.cqt_bins,
            fmin=data.fmin,
            bins_per_octave=data.bins_per_octave,
        )

        highest_center = float(frequencies[-1])
        nyquist = data.sample_rate / 2.0

        if highest_center >= nyquist:
            errors.append(
                "The highest CQT center frequency must be below "
                "Nyquist. "
                f"Got highest_center={highest_center:.2f} Hz and "
                f"Nyquist={nyquist:.2f} Hz."
            )


def _validate_generator_config(
    config: VocoderExperimentConfig,
    errors: list[str],
) -> None:
    data = config.data
    generator = config.generator

    _validate_positive_int(
        generator.upsample_initial_channel,
        path="generator.upsample_initial_channel",
        errors=errors,
    )

    if not generator.upsample_rates:
        errors.append(
            "generator.upsample_rates must not be empty."
        )

    if not generator.upsample_kernel_sizes:
        errors.append(
            "generator.upsample_kernel_sizes must not be empty."
        )

    if len(generator.upsample_rates) != len(
        generator.upsample_kernel_sizes
    ):
        errors.append(
            "generator.upsample_rates and "
            "generator.upsample_kernel_sizes must contain the "
            "same number of entries. "
            f"Got {len(generator.upsample_rates)} and "
            f"{len(generator.upsample_kernel_sizes)}."
        )

    for index, rate in enumerate(generator.upsample_rates):
        _validate_positive_int(
            rate,
            path=f"generator.upsample_rates[{index}]",
            errors=errors,
        )

    for index, kernel_size in enumerate(
        generator.upsample_kernel_sizes
    ):
        _validate_positive_int(
            kernel_size,
            path=f"generator.upsample_kernel_sizes[{index}]",
            errors=errors,
        )

    if generator.upsample_rates:
        upsample_product = prod(generator.upsample_rates)

        if upsample_product != data.hop_length:
            errors.append(
                "The product of generator.upsample_rates must equal "
                "data.hop_length. "
                f"Got product={upsample_product} and "
                f"hop_length={data.hop_length}."
            )

    for index, (rate, kernel_size) in enumerate(
        zip(
            generator.upsample_rates,
            generator.upsample_kernel_sizes,
        )
    ):
        if kernel_size < rate:
            errors.append(
                f"generator.upsample_kernel_sizes[{index}] should "
                "normally be at least as large as the corresponding "
                f"upsampling rate. Got kernel_size={kernel_size}, "
                f"rate={rate}."
            )

        if (kernel_size - rate) % 2 != 0:
            errors.append(
                f"generator upsampling stage {index} cannot preserve "
                "exact rate-multiplied length using symmetric "
                "ConvTranspose1d padding because "
                f"kernel_size - rate = {kernel_size - rate}, "
                "which is odd."
            )

    if not generator.resblock_kernel_sizes:
        errors.append(
            "generator.resblock_kernel_sizes must not be empty."
        )

    if len(generator.resblock_kernel_sizes) != len(
        generator.resblock_dilation_sizes
    ):
        errors.append(
            "generator.resblock_kernel_sizes and "
            "generator.resblock_dilation_sizes must contain the "
            "same number of entries. "
            f"Got {len(generator.resblock_kernel_sizes)} and "
            f"{len(generator.resblock_dilation_sizes)}."
        )

    for block_index, kernel_size in enumerate(
        generator.resblock_kernel_sizes
    ):
        _validate_positive_int(
            kernel_size,
            path=(
                "generator.resblock_kernel_sizes"
                f"[{block_index}]"
            ),
            errors=errors,
        )

        if kernel_size % 2 == 0:
            errors.append(
                "Residual block kernel sizes should be odd to "
                "preserve sequence length symmetrically. "
                f"Got generator.resblock_kernel_sizes"
                f"[{block_index}]={kernel_size}."
            )

    for block_index, dilations in enumerate(
        generator.resblock_dilation_sizes
    ):
        if not dilations:
            errors.append(
                "Each residual block dilation sequence must contain "
                "at least one value. "
                f"Block {block_index} is empty."
            )

        for dilation_index, dilation in enumerate(dilations):
            _validate_positive_int(
                dilation,
                path=(
                    "generator.resblock_dilation_sizes"
                    f"[{block_index}][{dilation_index}]"
                ),
                errors=errors,
            )

    snake = generator.snake_beta

    if snake.alpha_initial <= 0.0:
        errors.append(
            "generator.snake_beta.alpha_initial must be positive."
        )

    if snake.beta_initial <= 0.0:
        errors.append(
            "generator.snake_beta.beta_initial must be positive."
        )

    alias_free = generator.alias_free

    if alias_free.enabled:
        if generator.activation != "snake_beta":
            errors.append(
                "generator.alias_free.enabled currently requires "
                'generator.activation = "snake_beta".'
            )

        _validate_positive_int(
            alias_free.upsample_ratio,
            path="generator.alias_free.upsample_ratio",
            errors=errors,
        )
        _validate_positive_int(
            alias_free.downsample_ratio,
            path="generator.alias_free.downsample_ratio",
            errors=errors,
        )
        _validate_positive_int(
            alias_free.upsample_kernel_size,
            path="generator.alias_free.upsample_kernel_size",
            errors=errors,
        )
        _validate_positive_int(
            alias_free.downsample_kernel_size,
            path="generator.alias_free.downsample_kernel_size",
            errors=errors,
        )


def _validate_discriminator_config(
    config: VocoderExperimentConfig,
    errors: list[str],
) -> None:
    discriminator = config.discriminator

    if not discriminator.use_mpd and not discriminator.use_msd:
        errors.append(
            "At least one discriminator family must be enabled. "
            "Both discriminator.use_mpd and "
            "discriminator.use_msd are false."
        )

    mpd = discriminator.mpd

    if discriminator.use_mpd:
        if not mpd.periods:
            errors.append(
                "discriminator.mpd.periods must not be empty when "
                "MPD is enabled."
            )

        if not mpd.channels:
            errors.append(
                "discriminator.mpd.channels must not be empty when "
                "MPD is enabled."
            )

        for index, period in enumerate(mpd.periods):
            if period < 2:
                errors.append(
                    f"discriminator.mpd.periods[{index}] must be "
                    f"at least 2, got {period}."
                )

        if len(set(mpd.periods)) != len(mpd.periods):
            errors.append(
                "discriminator.mpd.periods must not contain "
                "duplicate periods."
            )

        for index, channels in enumerate(mpd.channels):
            _validate_positive_int(
                channels,
                path=f"discriminator.mpd.channels[{index}]",
                errors=errors,
            )

        _validate_positive_int(
            mpd.kernel_size,
            path="discriminator.mpd.kernel_size",
            errors=errors,
        )
        _validate_positive_int(
            mpd.stride,
            path="discriminator.mpd.stride",
            errors=errors,
        )

        if mpd.negative_slope < 0.0:
            errors.append(
                "discriminator.mpd.negative_slope must be "
                "non-negative."
            )

    msd = discriminator.msd

    if discriminator.use_msd:
        _validate_positive_int(
            msd.num_scales,
            path="discriminator.msd.num_scales",
            errors=errors,
        )
        _validate_positive_int(
            msd.pool_kernel_size,
            path="discriminator.msd.pool_kernel_size",
            errors=errors,
        )
        _validate_positive_int(
            msd.pool_stride,
            path="discriminator.msd.pool_stride",
            errors=errors,
        )

        if msd.pool_padding < 0:
            errors.append(
                "discriminator.msd.pool_padding must be "
                "non-negative."
            )

        scale = msd.discriminator

        lengths = {
            "channels": len(scale.channels),
            "kernel_sizes": len(scale.kernel_sizes),
            "strides": len(scale.strides),
            "groups": len(scale.groups),
        }

        if len(set(lengths.values())) != 1:
            details = ", ".join(
                f"{name}={length}"
                for name, length in lengths.items()
            )

            errors.append(
                "discriminator.msd.discriminator channels, "
                "kernel_sizes, strides and groups must have equal "
                f"lengths. Got {details}."
            )

        for index, channels in enumerate(scale.channels):
            _validate_positive_int(
                channels,
                path=(
                    "discriminator.msd.discriminator.channels"
                    f"[{index}]"
                ),
                errors=errors,
            )

        for index, kernel_size in enumerate(scale.kernel_sizes):
            _validate_positive_int(
                kernel_size,
                path=(
                    "discriminator.msd.discriminator.kernel_sizes"
                    f"[{index}]"
                ),
                errors=errors,
            )

        for index, stride in enumerate(scale.strides):
            _validate_positive_int(
                stride,
                path=(
                    "discriminator.msd.discriminator.strides"
                    f"[{index}]"
                ),
                errors=errors,
            )

        for index, groups in enumerate(scale.groups):
            _validate_positive_int(
                groups,
                path=(
                    "discriminator.msd.discriminator.groups"
                    f"[{index}]"
                ),
                errors=errors,
            )

        in_channels = 1

        for index, (out_channels, groups) in enumerate(
                zip(scale.channels, scale.groups)
        ):
            if in_channels % groups != 0:
                errors.append(
                    "Scale discriminator input channels must be divisible "
                    f"by groups at layer {index}. "
                    f"Got in_channels={in_channels}, groups={groups}."
                )

            if out_channels % groups != 0:
                errors.append(
                    "Scale discriminator output channels must be divisible "
                    f"by groups at layer {index}. "
                    f"Got out_channels={out_channels}, groups={groups}."
                )

            in_channels = out_channels

        if scale.negative_slope < 0.0:
            errors.append(
                "discriminator.msd.discriminator.negative_slope "
                "must be non-negative."
            )


def _validate_mrstft_config(
    mrstft: MRSTFTConfig,
    errors: list[str],
) -> None:
    lengths = {
        "fft_sizes": len(mrstft.fft_sizes),
        "hop_sizes": len(mrstft.hop_sizes),
        "win_lengths": len(mrstft.win_lengths),
    }

    if len(set(lengths.values())) != 1:
        details = ", ".join(
            f"{name}={length}"
            for name, length in lengths.items()
        )

        errors.append(
            "loss.mrstft fft_sizes, hop_sizes and win_lengths "
            f"must have equal lengths. Got {details}."
        )

    if not mrstft.fft_sizes:
        errors.append(
            "loss.mrstft must define at least one STFT resolution."
        )

    for index, values in enumerate(
        zip(
            mrstft.fft_sizes,
            mrstft.hop_sizes,
            mrstft.win_lengths,
        )
    ):
        fft_size, hop_size, win_length = values

        _validate_positive_int(
            fft_size,
            path=f"loss.mrstft.fft_sizes[{index}]",
            errors=errors,
        )
        _validate_positive_int(
            hop_size,
            path=f"loss.mrstft.hop_sizes[{index}]",
            errors=errors,
        )
        _validate_positive_int(
            win_length,
            path=f"loss.mrstft.win_lengths[{index}]",
            errors=errors,
        )

        if win_length > fft_size:
            errors.append(
                f"loss.mrstft.win_lengths[{index}]={win_length} "
                "must not exceed the corresponding "
                f"fft_size={fft_size}."
            )

        if hop_size > win_length:
            errors.append(
                f"loss.mrstft.hop_sizes[{index}]={hop_size} "
                "must not exceed the corresponding "
                f"win_length={win_length}."
            )

    if mrstft.spectral_convergence_weight < 0.0:
        errors.append(
            "loss.mrstft.spectral_convergence_weight must be "
            "non-negative."
        )

    if mrstft.log_magnitude_weight < 0.0:
        errors.append(
            "loss.mrstft.log_magnitude_weight must be non-negative."
        )

    if mrstft.eps <= 0.0:
        errors.append(
            "loss.mrstft.eps must be positive."
        )


def _validate_loss_config(
    config: VocoderExperimentConfig,
    errors: list[str],
) -> None:
    loss = config.loss

    weights = {
        "loss.lambda_adversarial": loss.lambda_adversarial,
        "loss.lambda_feature_matching": (
            loss.lambda_feature_matching
        ),
        "loss.lambda_mrstft": loss.lambda_mrstft,
    }

    for path, value in weights.items():
        if value < 0.0:
            errors.append(
                f"{path} must be non-negative, got {value}."
            )

    if all(value == 0.0 for value in weights.values()):
        errors.append(
            "At least one generator loss weight must be non-zero."
        )

    _validate_mrstft_config(loss.mrstft, errors)


def _validate_optimizer_config(
    optimizer: OptimizerConfig,
    *,
    path: str,
    errors: list[str],
) -> None:
    if optimizer.lr <= 0.0:
        errors.append(
            f"{path}.lr must be positive, got {optimizer.lr}."
        )

    if optimizer.weight_decay < 0.0:
        errors.append(
            f"{path}.weight_decay must be non-negative."
        )

    if not 0.0 <= optimizer.beta1 < 1.0:
        errors.append(
            f"{path}.beta1 must be in [0, 1), "
            f"got {optimizer.beta1}."
        )

    if not 0.0 <= optimizer.beta2 < 1.0:
        errors.append(
            f"{path}.beta2 must be in [0, 1), "
            f"got {optimizer.beta2}."
        )

    if optimizer.eps <= 0.0:
        errors.append(
            f"{path}.*ps must be positive."
        )


def _validate_scheduler_config(
    scheduler: SchedulerConfig,
    *,
    path: str,
    errors: list[str],
) -> None:
    if not scheduler.enabled:
        return

    if scheduler.name == "none":
        errors.append(
            f"{path}.enabled is true, but its name is 'none'."
        )

    if scheduler.name == "exponential":
        if not 0.0 < scheduler.gamma <= 1.0:
            errors.append(
                f"{path}.gamma must be in (0, 1] for an "
                "exponential scheduler. "
                f"Got {scheduler.gamma}."
            )


def _validate_runtime_config(
    config: VocoderExperimentConfig,
    errors: list[str],
) -> None:
    run = config.run
    training = config.training
    amp = config.amp
    augmentation = config.augmentation
    logging = config.logging
    control = config.control

    if not run.experiment_name.strip():
        errors.append(
            "run.experiment_name must not be empty."
        )

    if not run.output_root.strip():
        errors.append(
            "run.output_root must not be empty."
        )

    _validate_positive_int(
        training.epochs,
        path="training.epochs",
        errors=errors,
    )

    if training.max_steps is not None:
        _validate_positive_int(
            training.max_steps,
            path="training.max_steps",
            errors=errors,
        )

    _validate_positive_int(
        training.gradient_accumulation_steps,
        path="training.gradient_accumulation_steps",
        errors=errors,
    )

    if (
        training.grad_clip_generator is not None
        and training.grad_clip_generator <= 0.0
    ):
        errors.append(
            "training.grad_clip_generator must be positive or "
            "omitted."
        )

    if (
        training.grad_clip_discriminator is not None
        and training.grad_clip_discriminator <= 0.0
    ):
        errors.append(
            "training.grad_clip_discriminator must be positive or "
            "omitted."
        )

    step_fields = {
        "training.generator_start_step": (
            training.generator_start_step
        ),
        "training.discriminator_start_step": (
            training.discriminator_start_step
        ),
        "training.adversarial_start_step": (
            training.adversarial_start_step
        ),
        "training.overfit_batches": training.overfit_batches,
        "amp.start_step": amp.start_step,
    }

    for path, value in step_fields.items():
        if value < 0:
            errors.append(
                f"{path} must be non-negative, got {value}."
            )

    if amp.initial_scale <= 0.0:
        errors.append(
            "amp.initial_scale must be positive."
        )

    _validate_positive_int(
        amp.growth_interval,
        path="amp.growth_interval",
        errors=errors,
    )

    if augmentation.gain_db_min > augmentation.gain_db_max:
        errors.append(
            "augmentation.gain_db_min must not exceed "
            "augmentation.gain_db_max."
        )

    if augmentation.conditioning_noise_std < 0.0:
        errors.append(
            "augmentation.conditioning_noise_std must be "
            "non-negative."
        )

    _validate_probability(
        augmentation.time_mask_probability,
        path="augmentation.time_mask_probability",
        errors=errors,
    )
    _validate_probability(
        augmentation.frequency_mask_probability,
        path="augmentation.frequency_mask_probability",
        errors=errors,
    )

    logging_intervals = {
        "logging.log_every_steps = 10000"
    }


def validate_vocoder_config(
    config: VocoderExperimentConfig,
    *,
    check_paths: bool = False,
) -> None:
    """
    Validate cross-field and semantic constraints.

    Parsing and type validation belong to the TOML loader.
    This validator checks whether otherwise valid values form
    a usable experiment configuration.
    """
    errors: list[str] = []

    if config.schema_version != 1:
        errors.append(
            "schema_version must currently be 1. "
            f"Got {config.schema_version}."
        )

    _validate_data_config(config, errors)
    _validate_generator_config(config, errors)
    _validate_discriminator_config(config, errors)
    _validate_loss_config(config, errors)

    _validate_optimizer_config(
        config.optimizer_generator,
        path="optimizer.generator",
        errors=errors,
    )
    _validate_optimizer_config(
        config.optimizer_discriminator,
        path="optimizer.discriminator",
        errors=errors,
    )

    _validate_scheduler_config(
        config.scheduler_generator,
        path="scheduler.generator",
        errors=errors,
    )
    _validate_scheduler_config(
        config.scheduler_discriminator,
        path="scheduler.discriminator",
        errors=errors,
    )

    _validate_runtime_config(config, errors)

    if check_paths:
        cache_root = Path(
            config.data.chip_cache_root
        ).expanduser()

        if not cache_root.exists():
            errors.append(
                "data.chip_cache_root does not exist: "
                f"{cache_root}"
            )
        elif not cache_root.is_dir():
            errors.append(
                "data.chip_cache_root is not a directory: "
                f"{cache_root}"
            )

    if errors:
        raise ConfigValidationError(
            _format_errors(errors)
        )


