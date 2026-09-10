from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Update this if your cache location changes.
CACHE_ROOT = Path("C:/Datasets/cache/cqt/cqt96_sr22050_hop512")

SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_BINS = 96
SNIPPET_SECONDS = 4.0

BATCH_SIZE = 2

