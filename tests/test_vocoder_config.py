from src.config.vocoder_config import VocoderExperimentConfig


def test_default_vocoder_experiment_config():
    config = VocoderExperimentConfig()

    assert config.data.cqt_bins == 96
    assert config.data.hop_length == 512
    assert config.generator.activation == "snake_beta"
    assert config.generator.upsample_rates == (8, 8, 4, 2)
    assert config.loss.mrstft.fft_sizes == (
        256,
        512,
        1024,
        2048,
    )


def test_nested_config_instances_are_independent():
    first = VocoderExperimentConfig()
    second = VocoderExperimentConfig()

    first.generator.snake_beta.alpha_initial = 2.0

    assert second.generator.snake_beta.alpha_initial == 1.0