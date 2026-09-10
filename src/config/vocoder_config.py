from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


NormType = Literal["weight",  "spectral", "none"]
ActivationType = Literal["leaky_relu", "snake_beta"]


@dataclass
class RunConfig:
    experiment_name: str = "vocoder"
    output_root: str = "runs/vocoder_cqt"
    seed: int = 1337
    device: str = "cuda"


@dataclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 2e-4
    weight_decay: float = 0.0
    beta1: float = 0.8
    beta2: float = 0.99
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    name: str = "exponential"
    enabled: bool = False
    gamma: float = 0.999


@dataclass
class TrainingConfig:
    epochs: int = 500
    max_steps: int | None = None
    grad_clip_generator: float = 10.0
    grad_clip_discriminator: float = 10.0

    gradient_accumulation_steps: int = 1
    generator_start_step: int = 0
    discriminator_start_step: int = 0
    adversarial_start_step: int = 0

    overfit_batches: int = 0
    detect_anomaly: bool = False
    fail_on_nonfinite: bool = True


@dataclass
class AMPConfig:
    enabled: bool = False
    start_step: int = 0
    dtype: str = "float16"
    initial_scale: float = 256.0
    growth_interval: int = 2000


@dataclass
class AugmentationConfig:
    enabled: bool = False
    gain_db_min: float = 0.0
    gain_db_max: float = 0.0
    conditioning_noise_std: float = 0.0
    time_mask_probability: float = 0.0
    frequency_mask_probability: float = 0.0


@dataclass
class LogConfig:
    log_every_steps: int = 50
    status_every_steps: int = 50
    save_every_steps: int = 10000
    preview_every_steps: int = 10000
    validation_every_steps: int = 10000
    preview_num_samples: int = 4
    validation_num_samples: int = 16
    keep_numbered_checkpoints: int = 10
    tensorboard: bool = True


@dataclass
class ControlConfig:
    enabled: bool = True
    poll_every_steps: int = 10
    save_on_interrupt: bool = True


@dataclass
class MRSTFTConfig:
    fft_sizes: tuple[int, ...] = (256, 512, 1024, 2048)
    hop_sizes: tuple[int, ...] = (64, 128, 256, 512)
    win_lengths: tuple[int, ...] = (256, 512, 1024, 2048)
    spectral_convergence_weight: float = 1.0
    log_magnitude_weight: float = 1.0
    eps: float = 1e-7


@dataclass
class VocoderLossConfig:
    lambda_adversarial: float = 1.0
    lambda_feature_matching: float = 2.0
    lambda_mrstft: float = 45.0

    mrstft: MRSTFTConfig = field(default_factory=MRSTFTConfig)


@dataclass
class VocoderDataConfig:
    chip_cache_root: str = ""
    sample_rate: int = 22050
    hop_length: int = 512
    cqt_bins: int = 96
    bins_per_octave: int = 12
    fmin: float = 32.70319566

    segment_frames: int = 32
    windows_per_track: int = 8
    batch_size: int = 2
    num_workers: int = 1
    cache_waveforms: int = 8

    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    random_window: bool = True
    drop_last: bool = True


@dataclass
class SnakeBetaConfig:
    alpha_logscale: bool = True
    alpha_initial: float = 1.0
    beta_initial: float = 1.0


@dataclass
class AliasFreeConfig:
    enabled: bool = False
    implementation: str = "pytorch"
    upsample_ratio: int = 2
    downsample_ratio: int = 2
    upsample_kernel_size: int = 12
    downsample_kernel_size: int = 12


@dataclass
class VocoderGeneratorModelConfig:
    activation: ActivationType = "snake_beta"
    upsample_initial_channel: int = 128

    upsample_rates: tuple[int, ...] = (8, 8, 4, 2)
    upsample_kernel_sizes: tuple[int, ...] = (16, 16, 8, 4)

    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )

    final_tanh: bool = True
    use_weight_norm: bool = True

    snake_beta: SnakeBetaConfig = field(default_factory=SnakeBetaConfig)
    alias_free: AliasFreeConfig = field(default_factory=AliasFreeConfig)


@dataclass
class PeriodDiscriminatorConfig:
    period: int
    channels: tuple[int, ...] = (32, 128, 512, 1024, 1024)
    kernel_size: int = 5
    stride: int = 3
    norm: NormType = "weight"
    negative_slope: float = 0.2


@dataclass
class MultiPeriodDiscriminatorConfig:
    periods: tuple[int, ...] = (2, 3, 5, 7, 11)
    channels: tuple[int, ...] = (32, 128, 512, 1024, 1024)
    kernel_size: int = 5
    stride: int = 3
    norm: NormType = "weight"
    negative_slope: float = 0.2


@dataclass
class ScaleDiscriminatorConfig:
    channels: tuple[int, ...] = (128, 128, 256, 512, 1024, 1024)
    kernel_sizes: tuple[int, ...] = (15, 41, 41, 41, 41, 5)
    strides: tuple[int, ...] = (1, 2, 2, 4, 4, 1)
    groups: tuple[int, ...] = (1, 4, 16, 16, 16, 16)
    norm: NormType = "weight"
    negative_slope: float = 0.2


@dataclass
@dataclass
class MultiScaleDiscriminatorConfig:
    discriminator: ScaleDiscriminatorConfig = field(
        default_factory=ScaleDiscriminatorConfig
    )

    num_scales: int = 3

    first_discriminator_norm: NormType = "spectral"
    other_discriminator_norm: NormType = "weight"

    pool_kernel_size: int = 4
    pool_stride: int = 2
    pool_padding: int = 2


@dataclass
class VocoderDiscriminatorConfig:
    mpd: MultiPeriodDiscriminatorConfig = field(default_factory=MultiPeriodDiscriminatorConfig)
    msd: MultiScaleDiscriminatorConfig = field(default_factory=MultiScaleDiscriminatorConfig)
    use_mpd: bool = True
    use_msd: bool = True


@dataclass
class VocoderExperimentConfig:
    schema_version: int = 1

    run: RunConfig = field(default_factory=RunConfig)
    data: VocoderDataConfig = field(default_factory=VocoderDataConfig)
    generator: VocoderGeneratorModelConfig = field(
        default_factory=VocoderGeneratorModelConfig
    )
    discriminator: VocoderDiscriminatorConfig = field(
        default_factory=VocoderDiscriminatorConfig
    )
    loss: VocoderLossConfig = field(
        default_factory=VocoderLossConfig
    )
    optimizer_generator: OptimizerConfig = field(
        default_factory=OptimizerConfig
    )
    optimizer_discriminator: OptimizerConfig = field(
        default_factory=OptimizerConfig
    )
    scheduler_generator: SchedulerConfig = field(
        default_factory=SchedulerConfig
    )
    scheduler_discriminator: SchedulerConfig = field(
        default_factory=SchedulerConfig
    )
    augmentation: AugmentationConfig = field(
        default_factory=AugmentationConfig
    )
    training: TrainingConfig = field(
        default_factory=TrainingConfig
    )
    control: ControlConfig = field(
        default_factory=ControlConfig
    )
    amp: AMPConfig = field(default_factory=AMPConfig)
    logging: LogConfig = field(default_factory=LogConfig)

