from pathlib import Path

import pytest

from src.config.vocoder_config_loader import (
    ConfigError,
    load_vocoder_config,
)


def write_config(
    tmp_path: Path,
    content: str,
) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_minimal_vocoder_config(
    tmp_path: Path,
) -> None:
    path = write_config(
        tmp_path,
        """
        schema_version = 1

        [run]
        experiment_name = "test_run"

        [data]
        chip_cache_root = "C:/cache"
        sample_rate = 44100
        hop_length = 256
        cqt_bins = 216

        [generator]
        activation = "snake_beta"
        upsample_rates = [8, 8, 2, 2]
        upsample_kernel_sizes = [16, 16, 4, 4]
        """,
    )

    config = load_vocoder_config(path)

    assert config.schema_version == 1
    assert config.run.experiment_name == "test_run"
    assert config.data.sample_rate == 44100
    assert config.data.hop_length == 256
    assert config.data.cqt_bins == 216

    assert config.generator.activation == "snake_beta"
    assert config.generator.upsample_rates == (8, 8, 2, 2)
    assert config.generator.upsample_kernel_sizes == (16, 16, 4, 4)


def test_load_nested_resblock_dilations(
    tmp_path: Path,
) -> None:
    path = write_config(
        tmp_path,
        """
        [generator]
        resblock_dilation_sizes = [
            [1, 3, 5],
            [1, 5, 7],
            [1, 7, 11],
        ]
        """,
    )

    config = load_vocoder_config(path)

    assert config.generator.resblock_dilation_sizes == (
        (1, 3, 5),
        (1, 5, 7),
        (1, 7, 11),
    )


def test_load_nested_snake_beta_config(
    tmp_path: Path,
) -> None:
    path = write_config(
        tmp_path,
        """
        [generator]
        activation = "snake_beta"

        [generator.snake_beta]
        alpha_logscale = false
        alpha_initial = 2.0
        beta_initial = 0.5
        """,
    )

    config = load_vocoder_config(path)
    assert config.generator.snake_beta.alpha_logscale is False
    assert config.generator.snake_beta.alpha_initial == 2.0
    assert config.generator.snake_beta.beta_initial == 0.5


def test_load_mrstft_config(
    tmp_path: Path,
) -> None:
    path = write_config(
        tmp_path,
        """
        [loss]
        lambda_mrstft = 30.0

        [loss.mrstft]
        fft_sizes = [512, 1024]
        hop_sizes = [128, 256]
        win_lengths = [512, 1024]
        """,
    )

    config = load_vocoder_config(path)

    assert config.loss.lambda_mrstft == 30.0
    assert config.loss.mrstft.fft_sizes == (512, 1024)
    assert config.loss.mrstft.hop_sizes == (128, 256)
    assert config.loss.mrstft.win_lengths == (
        512,
        1024,
    )


def test_load_optimizer_sections(
    tmp_path: Path,
) -> None:
    path = write_config(
        tmp_path,
        """
        [optimizer.generator]
        name = "adamw"
        lr = 0.0001

        [optimizer.discriminator]
        name = "adam"
        lr = 0.0002
        """,
    )

    config = load_vocoder_config(path)

    assert config.optimizer_generator.name == "adamw"
    assert config.optimizer_generator.lr == pytest.approx(1e-4)

    assert config.optimizer_discriminator.name == "adam"
    assert config.optimizer_discriminator.lr == pytest.approx(2e-4)


def test_unknown_nested_key_is_rejected(
    tmp_path: Path,
) -> None:
    path = write_config(
        tmp_path,
        """
        [generator]
        upsample_initial_channels = 256
        """,
    )

    with pytest.raises(
        ConfigError,
        match="generator.upsample_initial_channels",
    ):
        load_vocoder_config(path)


def test_unknown_root_section_is_rejected(
    tmp_path: Path,
) -> None:
    path = write_config(
        tmp_path,
        """
        [generatr]
        activation = "snake_beta"
        """,
    )

    with pytest.raises(
        ConfigError,
        match="generatr",
    ):
        load_vocoder_config(path)


def test_invalid_activation_is_rejected(
    tmp_path: Path,
) -> None:
    path = write_config(
        tmp_path,
        """
        [generator]
        activation = "relu"
        """,
    )

    with pytest.raises(
        ConfigError,
        match="generator.activation",
    ):
        load_vocoder_config(path)


def test_wrong_integer_type_is_rejected(
    tmp_path: Path,
) -> None:
    path = write_config(
        tmp_path,
        """
        [data]
        sample_rate = "44100"
        """,
    )

    with pytest.raises(
        ConfigError,
        match="data.sample_rate",
    ):
        load_vocoder_config(path)


def test_missing_fields_use_defaults(
    tmp_path: Path,
) -> None:
    path = write_config(
        tmp_path,
        """
        [run]
        experiment_name = "defaults_test"
        """,
    )

    config = load_vocoder_config(path)

    assert config.run.experiment_name == "defaults_test"
    assert config.data.sample_rate == 22050
    assert config.generator.upsample_initial_channel == 128
    assert config.training.gradient_accumulation_steps == 1

