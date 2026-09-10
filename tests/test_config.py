from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

from configs.vocoder_config import VocoderExperimentConfig

# Update this if your cache location changes.
CACHE_ROOT = Path("C:/Datasets/cache/cqt/cqt96_sr22050_hop512")

SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_BINS = 96
SNIPPET_SECONDS = 4.0

BATCH_SIZE = 2

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

