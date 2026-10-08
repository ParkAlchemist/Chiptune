import math

import pytest
import torch

from src.config.vocoder_config import MRSTFTConfig, VocoderDiscriminatorConfig, VocoderGeneratorModelConfig, \
    TemporalGradientLossConfig
from src.losses.vocoder_losses import (
    VocoderLossConfig,
    VocoderLossBundle,
    compute_vocoder_discriminator_loss,
    compute_vocoder_generator_loss, feature_matching_loss, MultiScaleTemporalGradientLoss, MultiResolutionSTFTLoss,
    generator_loss_contribution_metrics, VocoderGeneratorLossOutput, make_weighted_loss_component,
    generator_loss_metrics,
)
from src.models.vocoder_discriminators import HiFiGANMultiDiscriminator
from src.models.vocoder_hifigan import CQTUHiFiGANGenerator
from tests.helpers.vocoder_loss_helpers import make_test_generator_loss_output, make_zero_mrstft_output, \
    make_test_loss_config


def test_mrstft_loss_finite():
    bundle = VocoderLossBundle(
        VocoderLossConfig(
            mrstft=MRSTFTConfig(
                fft_sizes=(256, 512),
                hop_sizes=(64, 128),
                win_lengths=(256, 512),
            )
        )
    )

    real = torch.randn(2, 1, 4096)
    fake = torch.randn(2, 1, 4096)

    loss = bundle.mrstft_loss(fake, real).total
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_vocoder_discriminator_loss_finite():
    outputs = {
        "real_outputs": [torch.ones(2, 8), torch.ones(2, 4)],
        "fake_outputs": [torch.zeros(2, 8), torch.zeros(2, 4)],
    }

    losses = compute_vocoder_discriminator_loss(outputs)

    assert losses.total.ndim == 0
    assert torch.isfinite(losses.total)
    assert torch.isfinite(losses.real)
    assert torch.isfinite(losses.fake)


def test_vocoder_generator_loss_finite_with_fake_discriminator_outputs():
    bundle = VocoderLossBundle(
        VocoderLossConfig(
            lambda_adversarial=1.0,
            lambda_feature_matching=2.0,
            lambda_mrstft=1.0,
            lambda_waveform=1.0,
            mrstft=MRSTFTConfig(
                fft_sizes=(256,),
                hop_sizes=(64,),
                win_lengths=(256,),
            ),
            temporal_gradient=TemporalGradientLossConfig()
        )
    )

    real_audio = torch.randn(2, 1, 4096)
    fake_audio = torch.randn(2, 1, 4096, requires_grad=True)

    discriminator_outputs = {
        "fake_outputs": [
            torch.randn(2, 8, requires_grad=True),
            torch.randn(2, 8, requires_grad=True),
        ],
        "real_feature_maps": [
            [torch.randn(2, 4, 16), torch.randn(2, 8, 8)],
        ],
        "fake_feature_maps": [
            [
                torch.randn(2, 4, 16, requires_grad=True),
                torch.randn(2, 8, 8, requires_grad=True),
            ],
        ],
    }

    losses = compute_vocoder_generator_loss(
        discriminator_outputs=discriminator_outputs,
        fake_audio=fake_audio,
        real_audio=real_audio,
        loss_bundle=bundle,
    )
    assert losses.total.ndim == 0
    assert torch.isfinite(losses.total)
    assert torch.isfinite(losses.adversarial.raw)
    assert torch.isfinite(losses.feature_matching.raw)
    assert torch.isfinite(losses.mrstft.raw)
    assert torch.isfinite(losses.waveform.raw)
    assert torch.isfinite(losses.temporal_gradient.raw)

    losses.total.backward()

    assert fake_audio.grad is not None
    assert torch.isfinite(fake_audio.grad).all()


def test_feature_matching_averages_layers_equally() -> None:
    real_maps = [
        [
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 100),
        ]
    ]

    fake_maps = [
        [
            torch.ones(1, 1, 2),
            torch.full((1, 1, 100), 3.0),
        ]
    ]

    loss = feature_matching_loss(real_maps, fake_maps, normalization="layer_mean")
    assert loss.item() == pytest.approx(2.0)


def test_feature_matching_normalizes_real_scale() -> None:
    real_maps = [
        [
            torch.full((1, 1, 8), 2.0)
        ]
    ]

    fake_maps = [
        [
            torch.full((1, 1, 8), 3.0)
        ]
    ]

    loss = feature_matching_loss(real_maps, fake_maps, normalization="real_magnitude", scale_floor=0.01)

    assert loss.item() == pytest.approx(0.5)


def test_temporal_gradient_zero_for_exact_match() -> None:
    loss_module = MultiScaleTemporalGradientLoss(
        lags=(1, 2, 4)
    )

    waveform = torch.randn(2, 1, 128)

    loss = loss_module(waveform, waveform)

    assert loss.item() == pytest.approx(0.0)


def test_temporal_gradient_is_polarity_sensitive() -> None:
    loss_module = MultiScaleTemporalGradientLoss(
        lags=(1, 2, 4)
    )

    waveform = torch.randn(2, 1, 128)

    loss = loss_module(-waveform, waveform)

    assert loss.item() > 0.0


def test_temporal_gradient_backpropagates() -> None:
    loss_module = MultiScaleTemporalGradientLoss(
        lags=(1, 2, 4)
    )

    prediction = torch.randn(2, 1, 128, requires_grad=True)

    target = torch.randn(2, 1, 128)

    loss = loss_module(prediction, target)

    loss.backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_temporal_gradient_prefers_correct_edges() -> None:
    target = torch.tensor(
        [
            [
                [
                    -1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    1.0,
                    1.0,
                    -1.0,
                    -1.0,
                ]
            ]
        ]
    )

    correct = target.clone()

    smoothed = torch.tensor(
        [
            [
                [
                    -1.0,
                    -0.8,
                    -0.2,
                    0.4,
                    1.0,
                    0.5,
                    -0.3,
                    -1.0,
                ]
            ]
        ]
    )

    loss_module = MultiScaleTemporalGradientLoss(
        lags=(1, 2)
    )

    correct_loss = loss_module(correct, target)
    smoothed_loss = loss_module(smoothed, target)

    assert correct_loss < smoothed_loss


def test_mrstft_components_are_zero_for_exact_match() -> None:
    config = MRSTFTConfig(
        fft_sizes=(128, 256),
        hop_sizes=(32, 64),
        win_lengths=(128, 256),
        spectral_convergence_weight=1.0,
        log_magnitude_weight=1.0,
        eps=1e-7,
    )

    loss_function = MultiResolutionSTFTLoss(
        config
    )

    waveform = torch.randn(
        2,
        1,
        2048,
    )

    output = loss_function(
        waveform,
        waveform,
    )

    assert output.total.item() == pytest.approx(
        0.0,
        abs=1e-6,
    )

    assert (
        output.spectral_convergence.item()
        == pytest.approx(
            0.0,
            abs=1e-6,
        )
    )

    assert (
        output.log_magnitude.item()
        == pytest.approx(
            0.0,
            abs=1e-6,
        )
    )

    assert len(output.resolutions) == 2


def test_mrstft_total_matches_components() -> None:
    config = MRSTFTConfig(
        fft_sizes=(128, 256),
        hop_sizes=(32, 64),
        win_lengths=(128, 256),
        spectral_convergence_weight=2.0,
        log_magnitude_weight=0.5,
        eps=1e-7,
    )

    loss_function = MultiResolutionSTFTLoss(
        config
    )

    fake = torch.randn(2, 1, 2048)
    real = torch.randn(2, 1, 2048)

    output = loss_function(fake, real)

    expected = (
        config.spectral_convergence_weight
        * output.spectral_convergence
        + config.log_magnitude_weight
        * output.log_magnitude
    )

    assert torch.allclose(
        output.total,
        expected,
        atol=1e-7,
        rtol=1e-6,
    )


def test_mrstft_reports_every_resolution() -> None:
    config = MRSTFTConfig(
        fft_sizes=(128, 256, 512),
        hop_sizes=(32, 64, 128),
        win_lengths=(128, 256, 512),
    )

    loss_function = MultiResolutionSTFTLoss(
        config
    )

    fake = torch.randn(1, 1, 2048)
    real = torch.randn(1, 1, 2048)

    output = loss_function(fake, real)

    metadata = [
        (
            resolution.fft_size,
            resolution.hop_size,
            resolution.win_length,
        )
        for resolution in output.resolutions
    ]

    assert metadata == [
        (128, 32, 128),
        (256, 64, 256),
        (512, 128, 512),
    ]


def test_mrstft_means_match_resolution_outputs() -> None:
    config = MRSTFTConfig(
        fft_sizes=(128, 256),
        hop_sizes=(32, 64),
        win_lengths=(128, 256),
    )

    loss_function = MultiResolutionSTFTLoss(
        config
    )

    fake = torch.randn(1, 1, 2048)
    real = torch.randn(1, 1, 2048)

    output = loss_function(fake, real)

    expected_convergence = torch.stack(
        [
            resolution.spectral_convergence
            for resolution in output.resolutions
        ]
    ).mean()

    expected_log_magnitude = torch.stack(
        [
            resolution.log_magnitude
            for resolution in output.resolutions
        ]
    ).mean()

    assert torch.allclose(
        output.spectral_convergence,
        expected_convergence,
    )

    assert torch.allclose(
        output.log_magnitude,
        expected_log_magnitude,
    )


def test_mrstft_structured_output_backpropagates() -> None:
    config = MRSTFTConfig(
        fft_sizes=(128, 256),
        hop_sizes=(32, 64),
        win_lengths=(128, 256),
    )

    loss_function = MultiResolutionSTFTLoss(
        config
    )

    fake = torch.randn(
        2,
        1,
        2048,
        requires_grad=True,
    )

    real = torch.randn(
        2,
        1,
        2048,
    )

    output = loss_function(fake, real)

    output.total.backward()

    assert fake.grad is not None
    assert torch.isfinite(fake.grad).all()
    assert fake.grad.abs().sum() > 0


def test_structured_mrstft_matches_legacy_scalar() -> None:
    config = MRSTFTConfig(
        fft_sizes=(128, 256),
        hop_sizes=(32, 64),
        win_lengths=(128, 256),
        spectral_convergence_weight=1.3,
        log_magnitude_weight=0.7,
    )

    loss_function = MultiResolutionSTFTLoss(
        config
    )

    fake = torch.randn(2, 1, 2048)
    real = torch.randn(2, 1, 2048)

    output = loss_function(fake, real)

    legacy_resolution_totals = []

    for single_loss in loss_function.losses:
        convergence, log_magnitude = (
            single_loss(fake, real)
        )

        legacy_resolution_totals.append(
            config.spectral_convergence_weight
            * convergence
            + config.log_magnitude_weight
            * log_magnitude
        )

    legacy_total = torch.stack(
        legacy_resolution_totals
    ).mean()

    assert torch.allclose(
        output.total,
        legacy_total,
        atol=1e-7,
        rtol=1e-6,
    )


def test_generator_total_matches_weighted_components() -> None:
    output = make_test_generator_loss_output()

    expected = (
        output.adversarial.weighted
        + output.feature_matching.weighted
        + output.mrstft.weighted
        + output.waveform.weighted
        + output.temporal_gradient.weighted
    )

    assert torch.allclose(
        output.total,
        expected,
        atol=1e-7,
        rtol=1e-6,
    )


def test_generator_weighted_components_use_configured_lambdas() -> None:

    output = make_test_generator_loss_output(
        adversarial_weight=0.5,
        feature_matching_weight=1.5,
        mrstft_weight=2.0,
        waveform_weight=0.25,
        temporal_gradient_weight=0.1,
    )

    assert torch.allclose(
        output.adversarial.weighted,
        0.5 * output.adversarial.raw,
    )

    assert torch.allclose(
        output.feature_matching.weighted,
        1.5 * output.feature_matching.raw,
    )

    assert torch.allclose(
        output.mrstft.weighted,
        2.0 * output.mrstft.raw,
    )

    assert torch.allclose(
        output.waveform.weighted,
        0.25 * output.waveform.raw,
    )

    assert torch.allclose(
        output.temporal_gradient.weighted,
        0.1 * output.temporal_gradient.raw,
    )


def test_generator_group_totals() -> None:
    output = make_test_generator_loss_output()

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


def test_loss_contribution_metrics_sum_to_one() -> None:
    output = make_test_generator_loss_output()

    metrics = (
        generator_loss_contribution_metrics(
            output
        )
    )

    total = sum(metrics.values())

    assert total == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_zero_loss_contributions_are_finite() -> None:
    zero = torch.zeros(())

    output = VocoderGeneratorLossOutput(
        total=zero,
        adversarial=(
            make_weighted_loss_component(
                zero,
                1.0,
            )
        ),
        feature_matching=(
            make_weighted_loss_component(
                zero,
                1.0,
            )
        ),
        mrstft=(
            make_weighted_loss_component(
                zero,
                1.0,
            )
        ),
        waveform=(
            make_weighted_loss_component(
                zero,
                1.0,
            )
        ),
        temporal_gradient=(
            make_weighted_loss_component(
                zero,
                1.0,
            )
        ),
        mrstft_output=make_zero_mrstft_output(),
    )

    metrics = (
        generator_loss_contribution_metrics(
            output
        )
    )

    assert all(
        math.isfinite(value)
        for value in metrics.values()
    )

    assert all(
        value == pytest.approx(0.0)
        for value in metrics.values()
    )


def test_generator_loss_metrics_use_structured_output() -> None:
    output = make_test_generator_loss_output()

    metrics = generator_loss_metrics(output)

    assert metrics["loss_g_total"] == pytest.approx(
        output.total.item()
    )

    assert metrics[
        "loss_g_adversarial"
    ] == pytest.approx(
        output.adversarial.raw.item()
    )

    assert metrics[
        "loss_g_weighted/adversarial"
    ] == pytest.approx(
        output.adversarial.weighted.item()
    )

    assert metrics[
        "loss_g_group/learned"
    ] == pytest.approx(
        output.learned_total.item()
    )


