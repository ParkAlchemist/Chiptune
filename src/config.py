from dataclasses import dataclass, field
from pathlib import Path
import os


DATA_ROOT = "E:/Projects/Datasets/raw"
CACHE_ROOT = "E:/Projects/Datasets/cache/cqt"

os.putenv("DATA_ROOT", DATA_ROOT)
os.putenv("CACHE_ROOT", CACHE_ROOT)

@dataclass
class CQTConfig:
    sample_rate: int = 22050
    duration: float = 4.0
    hop_length: int = 512
    n_bins: int = 84
    bins_per_octave: int = 12
    fmin: float | None = None

    db_min: float = -80.0
    db_max: float = 0.0

    mono: bool = True
    normalize_peak: bool = True


@dataclass
class DataConfig:
    dataset_root: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_ROOT", "data/raw")))
    cache_root: Path = field(
        default_factory=lambda: Path(os.getenv("CACHE_ROOT", "data/cache/cqt")))

    poly_subdir: str = "poly"
    chip_subdir: str = "chip"

    audio_extensions: tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg", ".aiff")

    random_crop: bool = True
    cache_dtype: str = "float16"


@dataclass
class TrainConfig:
    batch_size: int = 4
    num_workers: int = 1
    pin_memory: bool = True
    shuffle: bool = True


@dataclass
class Config:
    cqt: CQTConfig = CQTConfig()
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()


config = Config()
