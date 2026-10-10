from __future__ import annotations

from src.config.vocoder_config import VocoderExperimentConfig, VocoderGeneratorModelConfig, MRSTFTConfig, \
    VocoderLossConfig, TemporalGradientLossConfig, FeatureMatchingLossConfig, RMSEnvelopeLossConfig, BandPowerLossConfig

from src.models.vocoder_discriminators import (
    MultiPeriodDiscriminatorConfig as ModelMPDConfig,
    MultiScaleDiscriminatorConfig as ModelMSDConfig,
    ScaleDiscriminatorConfig as ModelScaleConfig,
    VocoderDiscriminatorConfig as ModelDiscriminatorConfig,
)



def build_generator_config(
    experiment: VocoderExperimentConfig,
) -> VocoderGeneratorModelConfig:
    source = experiment.generator

    return VocoderGeneratorModelConfig(
        upsample_initial_channel=source.upsample_initial_channel,
        upsample_rates=source.upsample_rates,
        upsample_kernel_sizes=source.upsample_kernel_sizes,
        resblock_kernel_sizes=source.resblock_kernel_sizes,
        resblock_dilation_sizes=source.resblock_dilation_sizes,
        activation=source.activation,
        final_tanh=source.final_tanh,
        use_weight_norm=source.use_weight_norm,
    )


def build_mrstft_config(
    experiment: VocoderExperimentConfig,
) -> MRSTFTConfig:
    source = experiment.loss.mrstft

    return MRSTFTConfig(
        fft_sizes=source.fft_sizes,
        hop_sizes=source.hop_sizes,
        win_lengths=source.win_lengths,
        spectral_convergence_weight=(
            source.spectral_convergence_weight
        ),
        log_magnitude_weight=source.log_magnitude_weight,
        eps=source.eps,
    )


def build_temporal_gradient_config(
        experiment: VocoderExperimentConfig,
) -> TemporalGradientLossConfig:
    source = experiment.loss.temporal_gradient

    return TemporalGradientLossConfig(
        enabled=source.enabled,
        lags=source.lags,
        normalization=source.normalization,
        scale_floor=source.scale_floor,
        eps=source.eps,
        edge_weighting=source.edge_weighting,
        edge_weight=source.edge_weight,
        max_edge_weight=source.max_edge_weight,
    )


def build_feature_matching_config(
    experiment: VocoderExperimentConfig,
) -> FeatureMatchingLossConfig:
    source = experiment.loss.feature_matching

    return FeatureMatchingLossConfig(
        normalization=source.normalization,
        scale_floor=source.scale_floor,
        eps=source.eps,
    )


def build_rms_envelope_loss_config(
    experiment: VocoderExperimentConfig,
) -> RMSEnvelopeLossConfig:
    source = experiment.loss.rms_envelope

    return RMSEnvelopeLossConfig(
        enabled=source.enabled,
        window_sizes=source.window_sizes,
        hop_ratio=source.hop_ratio,
        eps=source.eps,
    )


def build_band_power_loss_config(
        experiment: VocoderExperimentConfig,
) -> BandPowerLossConfig:
    source = experiment.loss.band_power

    return BandPowerLossConfig(
        enabled=source.enabled,
        fft_sizes=source.fft_sizes,
        hop_sizes=source.hop_sizes,
        win_lengths=source.win_lengths,
        number_of_bands=source.number_of_bands,
        fmin=source.fmin,
        fmax=source.fmax,
        absolute_weight=source.absolute_weight,
        shape_weight=source.shape_weight,
        eps=source.eps,
    )


def build_loss_config(
    experiment: VocoderExperimentConfig,
) -> VocoderLossConfig:
    source = experiment.loss

    return VocoderLossConfig(
        lambda_adversarial=source.lambda_adversarial,
        lambda_feature_matching=source.lambda_feature_matching,
        lambda_mrstft=source.lambda_mrstft,
        lambda_waveform=source.lambda_waveform,
        lambda_temporal_gradient=source.lambda_temporal_gradient,
        lambda_rms_envelope=source.lambda_rms_envelope,
        lambda_band_power=source.lambda_band_power,
        mrstft=build_mrstft_config(experiment),
        temporal_gradient=build_temporal_gradient_config(experiment),
        feature_matching=build_feature_matching_config(experiment),
        rms_envelope=build_rms_envelope_loss_config(experiment),
        band_power=build_band_power_loss_config(experiment),
    )


def build_discriminator_config(
    experiment: VocoderExperimentConfig,
) -> ModelDiscriminatorConfig:
    source = experiment.discriminator

    mpd = ModelMPDConfig(
        periods=source.mpd.periods,
        channels=source.mpd.channels,
        kernel_size=source.mpd.kernel_size,
        stride=source.mpd.stride,
        norm=source.mpd.norm,
        negative_slope=source.mpd.negative_slope,
    )

    scale = ModelScaleConfig(
        channels=source.msd.discriminator.channels,
        kernel_sizes=source.msd.discriminator.kernel_sizes,
        strides=source.msd.discriminator.strides,
        groups=source.msd.discriminator.groups,
        norm=source.msd.discriminator.norm,
        negative_slope=source.msd.discriminator.negative_slope,
    )

    msd = ModelMSDConfig(
        num_scales=source.msd.num_scales,
        first_discriminator_norm=(
            source.msd.first_discriminator_norm
        ),
        other_discriminator_norm=(
            source.msd.other_discriminator_norm
        ),
        pool_kernel_size=source.msd.pool_kernel_size,
        pool_stride=source.msd.pool_stride,
        pool_padding=source.msd.pool_padding,
        discriminator=scale,
    )

    return ModelDiscriminatorConfig(
        mpd=mpd,
        msd=msd,
        use_mpd=source.use_mpd,
        use_msd=source.use_msd,
    )

