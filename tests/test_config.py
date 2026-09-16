from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Update this if your cache location changes.
CACHE_ROOT = Path("C:/Datasets/cache/cqt/cqt116_sr44100_hop256_bpo12_fminA0")

SAMPLE_RATE = 44100
HOP_LENGTH = 256
N_BINS = 116
SNIPPET_SECONDS = 4.0

BATCH_SIZE = 2

