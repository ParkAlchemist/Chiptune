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
    generator_loss_metrics, mean_enabled_families, family_balanced_generator_adversarial_loss,
    family_balanced_feature_matching_loss, family_balanced_discriminator_loss, FamilyLossOutput,
    FamilyDiscriminatorLossOutput, discriminator_loss_metrics,
)
from tests.helpers.vocoder_loss_helpers import make_test_generator_loss_output, make_zero_mrstft_output, \
    make_test_loss_config, make_multi_family_discriminator_outputs, assert_scalar_tensor, make_discriminator_family, \
    assert_family_loss_output_consistent, assert_family_discriminator_loss_consistent


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


def test_generator_loss_metrics_include_family_values() -> None:
    output = make_test_generator_loss_output(
        adversarial_raw=2.0,
        adversarial_families={
            "mpd": torch.Tensor((1.0, )),
            "mrd": torch.Tensor((3.0, )),
        },
        feature_matching_raw=3.0,
        feature_matching_families={
            "mpd": torch.Tensor((2.0, )),
            "mrd": torch.Tensor((4.0, )),
        },
    )

    metrics = generator_loss_metrics(output)

    assert metrics[
        "loss_g_adversarial_mpd"
    ] == pytest.approx(1.0)

    assert metrics[
        "loss_g_adversarial_mrd"
    ] == pytest.approx(3.0)

    assert metrics[
        "loss_g_feature_matching_mpd"
    ] == pytest.approx(2.0)

    assert metrics[
        "loss_g_feature_matching_mrd"
    ] == pytest.approx(4.0)


def test_discriminator_loss_metrics_include_family_values() -> None:
    mpd = make_discriminator_family(
        real_output_values=(1.0,),
        fake_output_values=(0.0,),
    )

    mrd = make_discriminator_family(
        real_output_values=(0.0,),
        fake_output_values=(1.0,),
    )

    output = family_balanced_discriminator_loss(
        make_multi_family_discriminator_outputs(
            mpd=mpd,
            mrd=mrd,
        )
    )

    metrics = discriminator_loss_metrics(
        output
    )

    assert metrics[
        "loss_d_mpd_total"
    ] == pytest.approx(0.0)

    assert metrics[
        "loss_d_mrd_total"
    ] == pytest.approx(2.0)

    assert metrics[
        "loss_d_total"
    ] == pytest.approx(1.0)


def test_mean_enabled_families() -> None:
    losses = {
        "mpd": torch.tensor(2.0),
        "mrd": torch.tensor(4.0),
    }

    result = mean_enabled_families(
        losses,
        name="test",
    )

    assert result.item() == pytest.approx(
        3.0
    )


def test_mean_enabled_families_rejects_empty_input() -> None:
    with pytest.raises(
        ValueError,
        match="at least one enabled",
    ):
        mean_enabled_families(
            {},
            name="test",
        )


def test_generator_adversarial_balances_families_equally() -> None:
    mpd = make_discriminator_family(
        real_output_values=(
            1.0,
            1.0,
            1.0,
        ),
        fake_output_values=(
            0.0,
            0.0,
            0.0,
        ),
    )

    mrd = make_discriminator_family(
        real_output_values=(1.0,),
        fake_output_values=(1.0,),
    )

    outputs = (
        make_multi_family_discriminator_outputs(
            mpd=mpd,
            mrd=mrd,
        )
    )

    result = (
        family_balanced_generator_adversarial_loss(
            outputs
        )
    )

    assert_family_loss_output_consistent(
        result
    )

    # MPD fake=0:
    # (1 - 0)^2 = 1
    assert result.families[
         "mpd"
    ].item() == pytest.approx(1.0)

    # MRD fake=1:
    # (1 - 1)^2 = 0
    assert result.families[
        "mrd"
    ].item() == pytest.approx(0.0)

    # Families receive equal weight despite MPD having
    # three branches and MRD only one.
    assert result.total.item() == pytest.approx(
        0.5
    )


def test_generator_adversarial_is_invariant_to_duplicate_equal_branches() -> None:
    one_mpd_branch = make_discriminator_family(
        real_output_values=(1.0,),
        fake_output_values=(0.0,),
    )

    five_mpd_branches = make_discriminator_family(
        real_output_values=(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ),
       fake_output_values=(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    )

    mrd = make_discriminator_family(
        real_output_values=(1.0,),
        fake_output_values=(1.0,),
    )

    one_result = (
        family_balanced_generator_adversarial_loss(
            make_multi_family_discriminator_outputs(
                mpd=one_mpd_branch,
                mrd=mrd,
            )
        )
    )

    five_result = (
        family_balanced_generator_adversarial_loss(
            make_multi_family_discriminator_outputs(
                mpd=five_mpd_branches,
                mrd=mrd,
            )
        )
    )

    assert torch.allclose(
        one_result.total,
        five_result.total,
    )


def test_feature_matching_balances_families_equally() -> None:
    mpd = make_discriminator_family(
        real_output_values=(1.0, 1.0),
        fake_output_values=(0.0, 0.0),
        real_feature_values=(
            (1.0,),
            (1.0,),
        ),
        fake_feature_values=(
            (0.0,),
            (0.0,),
        ),
    )

    mrd = make_discriminator_family(
        real_output_values=(1.0,),
        fake_output_values=(0.0,),
        real_feature_values=(
            (3.0,),
        ),
        fake_feature_values=(
            (0.0,),
        ),
    )

    outputs = (
        make_multi_family_discriminator_outputs(
            mpd=mpd,
            mrd=mrd,
        )
    )

    result = (
        family_balanced_feature_matching_loss(
            outputs,
            normalization="layer_mean",
            scale_floor=0.01,
            eps=1e-8,
        )
    )

    assert_family_loss_output_consistent(
        result
    )

    assert result.families[
        "mpd"
    ].item() == pytest.approx(1.0)

    assert result.families[
        "mrd"
    ].item() == pytest.approx(3.0)

    assert result.total.item() == pytest.approx(
        2.0
    )


def test_family_feature_matching_respects_real_magnitude_normalization() -> None:
    mpd = make_discriminator_family(
        real_output_values=(1.0,),
        fake_output_values=(0.0,),
        real_feature_values=(
            (2.0,),
        ),
        fake_feature_values=(
            (3.0,),
        ),
    )

    mrd = make_discriminator_family(
        real_output_values=(1.0,),
        fake_output_values=(0.0,),
        real_feature_values=(
            (4.0,),
        ),
        fake_feature_values=(
            (6.0,),
        ),
    )

    outputs = (
        make_multi_family_discriminator_outputs(
            mpd=mpd,
            mrd=mrd,
        )
    )

    result = (
        family_balanced_feature_matching_loss(
            outputs,
            normalization="real_magnitude",
            scale_floor=0.01,
            eps=1e-8,
        )
    )

    # MPD: |3 - 2| / 2 = 0.5
    # MRD: |6 - 4| / 4 = 0.5
    assert result.families[
        "mpd"
    ].item() == pytest.approx(0.5)

    assert result.families[
        "mrd"
    ].item() == pytest.approx(0.5)

    assert result.total.item() == pytest.approx(
        0.5
    )


def test_discriminator_loss_balances_families_equally() -> None:
    mpd = make_discriminator_family(
        real_output_values=(
            1.0,
            1.0,
            1.0,
        ),
        fake_output_values=(
            0.0,
            0.0,
            0.0,
        ),
    )

    mrd = make_discriminator_family(
        real_output_values=(0.0,),
        fake_output_values=(1.0,),
    )

    outputs = (
        make_multi_family_discriminator_outputs(
            mpd=mpd,
            mrd=mrd,
        )
    )

    result = (
        family_balanced_discriminator_loss(
            outputs
        )
    )

    assert_family_discriminator_loss_consistent(
        result
    )

    assert result.family_totals[
        "mpd"
    ].item() == pytest.approx(0.0)

    assert result.family_real[
        "mrd"
    ].item() == pytest.approx(1.0)

    assert result.family_fake[
        "mrd"
    ].item() == pytest.approx(1.0)

    assert result.family_totals[
        "mrd"
    ].item() == pytest.approx(2.0)

    assert result.total.item() == pytest.approx(
        1.0
    )


def test_single_enabled_family_is_not_rescaled() -> None:
    mpd = make_discriminator_family(
        real_output_values=(1.0, 0.8),
        fake_output_values=(0.0, 0.2),
    )

    outputs = (
        make_multi_family_discriminator_outputs(
            mpd=mpd,
        )
    )

    generator_result = (
        family_balanced_generator_adversarial_loss(
            outputs
        )
    )

    discriminator_result = (
        family_balanced_discriminator_loss(
            outputs
        )
    )

    assert set(
        generator_result.families
    ) == {"mpd"}

    assert set(
        discriminator_result.family_totals
    ) == {"mpd"}

    assert torch.allclose(
        generator_result.total,
        generator_result.families["mpd"],
    )

    assert torch.allclose(
        discriminator_result.total,
        discriminator_result.family_totals[
            "mpd"
        ],
    )


def test_family_balanced_losses_reject_no_enabled_families() -> None:
    outputs = (
        make_multi_family_discriminator_outputs()
    )

    with pytest.raises(
        ValueError,
        match="at least one enabled",
    ):
        family_balanced_generator_adversarial_loss(
            outputs
        )

    with pytest.raises(
        ValueError,
        match="at least one enabled",
    ):
        family_balanced_feature_matching_loss(
            outputs,
            normalization="layer_mean",
            scale_floor=0.01,
            eps=1e-8,
        )

    with pytest.raises(
        ValueError,
        match="at least one enabled",
    ):
        family_balanced_discriminator_loss(
            outputs
        )


def test_family_balanced_adversarial_loss_backpropagates_to_every_family() -> None:
    mpd = make_discriminator_family(
        real_output_values=(1.0, 1.0),
        fake_output_values=(0.2, 0.3),
        fake_requires_grad=True,
    )

    mrd = make_discriminator_family(
        real_output_values=(1.0,),
        fake_output_values=(0.4,),
        fake_requires_grad=True,
    )

    outputs = (
        make_multi_family_discriminator_outputs(
            mpd=mpd,
            mrd=mrd,
        )
    )

    result = (
        family_balanced_generator_adversarial_loss(
            outputs
        )
    )

    result.total.backward()

    for family in ("mpd", "mrd"):
        for prediction in outputs[
            f"{family}_fake_outputs"
        ]:
            assert prediction.grad is not None
            assert torch.isfinite(
                prediction.grad
            ).all()

