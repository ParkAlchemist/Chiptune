from __future__ import annotations

from collections.abc import Mapping

import torch

from src.config.vocoder_config import (
    FeatureMatchingLossConfig,
    MRSTFTConfig,
    TemporalGradientLossConfig,
    VocoderLossConfig,
)
from src.losses.vocoder_losses import (
    MultiResolutionSTFTLossOutput,
    STFTResolutionLossOutput,
    VocoderGeneratorLossOutput,
    WeightedLossComponent,
    make_weighted_loss_component,
)


def scalar_tensor(
    value: float,
    *,
    requires_grad: bool = False,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Construct a scalar float32 tensor for loss-output tests.
    """
    return torch.tensor(
        float(value),
        dtype=torch.float32,
        device=device,
        requires_grad=requires_grad,
    )


def make_test_stft_resolution_output(
    *,
    fft_size: int = 256,
    hop_size: int = 64,
    win_length: int = 256,
    spectral_convergence: float = 0.4,
    log_magnitude: float = 0.6,
    spectral_convergence_weight: float = 1.0,
    log_magnitude_weight: float = 1.0,
    device: torch.device | str = "cpu",
) -> STFTResolutionLossOutput:
    return STFTResolutionLossOutput(
        fft_size=fft_size,
        hop_size=hop_size,
        win_length=win_length,
        spectral_convergence=scalar_tensor(
            spectral_convergence,
            device=device,
        ),
        log_magnitude=scalar_tensor(
            log_magnitude,
            device=device,
        ),
        spectral_convergence_weight=(
            spectral_convergence_weight
        ),
        log_magnitude_weight=(
            log_magnitude_weight
        ),
    )


def make_test_mrstft_output(
    *,
    resolutions: tuple[
        STFTResolutionLossOutput,
        ...,
    ] | None = None,
    spectral_convergence_weight: float = 1.0,
    log_magnitude_weight: float = 1.0,
    device: torch.device | str = "cpu",
) -> MultiResolutionSTFTLossOutput:
    if resolutions is None:
        resolutions = (
            make_test_stft_resolution_output(
                fft_size=256,
                hop_size=64,
                win_length=256,
                spectral_convergence=0.4,
                log_magnitude=0.6,
                spectral_convergence_weight=(
                    spectral_convergence_weight
                ),
                log_magnitude_weight=(
                    log_magnitude_weight
                ),
                device=device,
            ),
            make_test_stft_resolution_output(
                fft_size=512,
                hop_size=128,
                win_length=512,
                spectral_convergence=0.6,
                log_magnitude=0.8,
                spectral_convergence_weight=(
                    spectral_convergence_weight
                ),
                log_magnitude_weight=(
                    log_magnitude_weight
                ),
                device=device,
            ),
        )

    if not resolutions:
        raise ValueError(
            "At least one MR-STFT resolution is required."
        )

    mean_spectral_convergence = torch.stack(
        [
            resolution.spectral_convergence
            for resolution in resolutions
        ]
    ).mean()

    mean_log_magnitude = torch.stack(
        [
            resolution.log_magnitude
            for resolution in resolutions
        ]
    ).mean()

    total = (
        spectral_convergence_weight
        * mean_spectral_convergence
        + log_magnitude_weight
        * mean_log_magnitude
    )

    return MultiResolutionSTFTLossOutput(
        total=total,
        spectral_convergence=(
            mean_spectral_convergence
        ),
        log_magnitude=mean_log_magnitude,
        resolutions=resolutions,
    )


def make_zero_mrstft_output(
    *,
    device: torch.device | str = "cpu",
) -> MultiResolutionSTFTLossOutput:
    resolution = make_test_stft_resolution_output(
        fft_size=256,
        hop_size=64,
        win_length=256,
        spectral_convergence=0.0,
        log_magnitude=0.0,
        spectral_convergence_weight=1.0,
        log_magnitude_weight=1.0,
        device=device,
    )

    return MultiResolutionSTFTLossOutput(
        total=scalar_tensor(
            0.0,
            device=device,
        ),
        spectral_convergence=scalar_tensor(
            0.0,
            device=device,
        ),
        log_magnitude=scalar_tensor(
            0.0,
            device=device,
        ),
        resolutions=(resolution,),
    )


def make_test_weighted_component(
    *,
    raw: float,
    weight: float,
    device: torch.device | str = "cpu",
) -> WeightedLossComponent:
    raw_tensor = scalar_tensor(
        raw,
        device=device,
    )

    return make_weighted_loss_component(
        raw=raw_tensor,
        weight=weight,
    )



def make_test_generator_loss_output(
    *,
    adversarial_raw: float = 2.0,
    adversarial_weight: float = 0.5,
    feature_matching_raw: float = 3.0,
    feature_matching_weight: float = 2.0,
    mrstft_raw: float = 4.0,
    mrstft_weight: float = 1.5,
    waveform_raw: float = 5.0,
    waveform_weight: float = 0.2,
    temporal_gradient_raw: float = 6.0,
    temporal_gradient_weight: float = 0.1,
    mrstft_output: (
        MultiResolutionSTFTLossOutput
        | None
    ) = None,
    device: torch.device | str = "cpu",
) -> VocoderGeneratorLossOutput:
    adversarial = make_test_weighted_component(
        raw=adversarial_raw,
        weight=adversarial_weight,
        device=device,
    )

    feature_matching = make_test_weighted_component(
        raw=feature_matching_raw,
        weight=feature_matching_weight,
        device=device,
    )

    mrstft = make_test_weighted_component(
        raw=mrstft_raw,
        weight=mrstft_weight,
        device=device,
    )

    waveform = make_test_weighted_component(
        raw=waveform_raw,
        weight=waveform_weight,
        device=device,
    )

    temporal_gradient = make_test_weighted_component(
        raw=temporal_gradient_raw,
        weight=temporal_gradient_weight,
        device=device,
    )

    total = (
        adversarial.weighted
        + feature_matching.weighted
        + mrstft.weighted
        + waveform.weighted
        + temporal_gradient.weighted
    )

    if mrstft_output is None:
        mrstft_output = make_test_mrstft_output(
            device=device,
        )

    return VocoderGeneratorLossOutput(
        total=total,
        adversarial=adversarial,
        feature_matching=(
            feature_matching
        ),
        mrstft=mrstft,
        waveform=waveform,
        temporal_gradient=(
            temporal_gradient
        ),
        mrstft_output=mrstft_output,
    )


def make_zero_generator_loss_output(
    *,
    device: torch.device | str = "cpu",
) -> VocoderGeneratorLossOutput:
    return make_test_generator_loss_output(
        adversarial_raw=0.0,
        adversarial_weight=1.0,
        feature_matching_raw=0.0,
        feature_matching_weight=1.0,
        mrstft_raw=0.0,
        mrstft_weight=1.0,
        waveform_raw=0.0,
        waveform_weight=1.0,
        temporal_gradient_raw=0.0,
        temporal_gradient_weight=1.0,
        mrstft_output=make_zero_mrstft_output(
            device=device,
        ),
        device=device,
    )


def make_test_loss_config(
    *,
    lambda_adversarial: float = 0.5,
    lambda_feature_matching: float = 2.0,
    lambda_mrstft: float = 1.5,
    lambda_waveform: float = 0.2,
    lambda_temporal_gradient: float = 0.1,
    feature_matching_normalization: str = "layer_mean",
    temporal_gradient_enabled: bool = True,
) -> VocoderLossConfig:
    return VocoderLossConfig(
        lambda_adversarial=(
            lambda_adversarial
        ),
        lambda_feature_matching=(
            lambda_feature_matching
        ),
        lambda_mrstft=lambda_mrstft,
        lambda_waveform=lambda_waveform,
        lambda_temporal_gradient=(
            lambda_temporal_gradient
        ),
        feature_matching=FeatureMatchingLossConfig(
            normalization=(
                feature_matching_normalization
            ),
            scale_floor=1e-2,
            eps=1e-8,
        ),
        mrstft=MRSTFTConfig(
            fft_sizes=(128, 256),
            hop_sizes=(32, 64),
            win_lengths=(128, 256),
            spectral_convergence_weight=1.0,
            log_magnitude_weight=1.0,
            eps=1e-7,
        ),
        temporal_gradient=TemporalGradientLossConfig(
            enabled=temporal_gradient_enabled,
            lags=(1, 2, 4),
            normalization="none",
            scale_floor=1e-3,
            eps=1e-8,
            edge_weighting=False,
            edge_weight=1.0,
            max_edge_weight=5.0,
        ),
    )


def make_test_discriminator_outputs(
    *,
    batch_size: int = 2,
    device: torch.device | str = "cpu",
    fake_requires_grad: bool = True,
) -> dict[str, list]:
    real_output = torch.full(
        (batch_size, 4),
        0.8,
        dtype=torch.float32,
        device=device,
    )

    fake_output = torch.full(
        (batch_size, 4),
        0.2,
        dtype=torch.float32,
        device=device,
        requires_grad=fake_requires_grad,
    )

    real_feature_map = torch.full(
        (batch_size, 4, 8),
        0.5,
        dtype=torch.float32,
        device=device,
    )

    fake_feature_map = torch.full(
        (batch_size, 4, 8),
        0.3,
        dtype=torch.float32,
        device=device,
        requires_grad=fake_requires_grad,
    )

    return {
        "real_outputs": [
            real_output,
        ],
        "fake_outputs": [
            fake_output,
        ],
        "real_feature_maps": [
            [
                real_feature_map,
            ]
        ],
        "fake_feature_maps": [
            [
                fake_feature_map,
            ]
        ],

        "mpd_real_outputs": [
            real_output,
        ],
        "mpd_fake_outputs": [
            fake_output,
        ],
        "mpd_real_feature_maps": [
            [
                real_feature_map,
            ]
        ],
        "mpd_fake_feature_maps": [
            [
                fake_feature_map,
            ]
        ],

        "msd_real_outputs": [],
        "msd_fake_outputs": [],
        "msd_real_feature_maps": [],
        "msd_fake_feature_maps": [],

        "mrd_real_outputs": [],
        "mrd_fake_outputs": [],
        "mrd_real_feature_maps": [],
        "mrd_fake_feature_maps": [],
    }


def make_family_outputs(
    values: tuple[float, ...],
    *,
    batch_size: int = 2,
    patch_count: int = 4,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> list[torch.Tensor]:
    return [
        torch.full(
            (batch_size, patch_count),
            value,
            dtype=torch.float32,
            device=device,
            requires_grad=requires_grad,
        )
        for value in values
    ]


def make_test_waveform_pair(
    *,
    batch_size: int = 2,
    channels: int = 1,
    length: int = 2048,
    device: torch.device | str = "cpu",
    fake_requires_grad: bool = True,
    seed: int = 1337,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(
        device=device,
    )
    generator.manual_seed(seed)

    real = torch.randn(
        batch_size,
        channels,
        length,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )

    fake = torch.randn(
        batch_size,
        channels,
        length,
        dtype=torch.float32,
        device=device,
        generator=generator,
        requires_grad=fake_requires_grad,
    )

    return real, fake


def assert_scalar_tensor(
    value: torch.Tensor,
    *,
    finite: bool = True,
) -> None:
    assert isinstance(value, torch.Tensor)
    assert value.ndim == 0

    if finite:
        assert torch.isfinite(value)


def assert_weighted_component_consistent(
    component: WeightedLossComponent,
) -> None:
    assert_scalar_tensor(component.raw)
    assert_scalar_tensor(component.weighted)

    expected = (
        component.raw
        * component.weight
    )

    assert torch.allclose(
        component.weighted,
        expected,
        atol=1e-7,
        rtol=1e-6,
    )


def assert_generator_loss_output_consistent(
    output: VocoderGeneratorLossOutput,
) -> None:
    assert_scalar_tensor(output.total)

    components = (
        output.adversarial,
        output.feature_matching,
        output.mrstft,
        output.waveform,
        output.temporal_gradient,
    )

    for component in components:
        assert_weighted_component_consistent(
            component
        )

    expected_total = torch.stack(
        [
            component.weighted
            for component in components
        ]
    ).sum()

    assert torch.allclose(
        output.total,
        expected_total,
        atol=1e-7,
        rtol=1e-6,
    )

    assert torch.allclose(
        output.learned_total,
        (
            output.adversarial.weighted
            + output.feature_matching.weighted
        ),
    )

    assert torch.allclose(
        output.time_micro_total,
        (
            output.waveform.weighted
            + output.temporal_gradient.weighted
        ),
    )

    assert torch.allclose(
        output.time_frequency_total,
        output.mrstft.weighted,
    )


def assert_finite_metrics(
    metrics: Mapping[str, float],
) -> None:
    assert metrics

    for name, value in metrics.items():
        assert isinstance(value, float), (
            f"Metric {name!r} is not a float: "
            f"{type(value).__name__}"
        )

        assert torch.isfinite(
            torch.tensor(value)
        ), (
            f"Metric {name!r} is non-finite: {value}"
        )




