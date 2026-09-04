from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.vocoder_dataset import CQTVocoderDataset
from src.models.vocoder_hifigan import (
    CQTGeneratorConfig,
    CQTUHiFiGANGenerator,
)
from src.losses.vocoder_losses import (
    MultiResolutionSTFTConfig,
    MultiResolutionSTFTLoss,
)

from src.eval.vocoder_diagnostics import build_fixed_vocoder_batch, tensor_to_cqt_np, tensor_to_audio_np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export vocoder copy-synthesis diagnostics."
    )

    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--chip-cache-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)

    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--cqt-bins", type=int, default=96)
    parser.add_argument("--bins-per-octave", type=int, default=12)
    parser.add_argument("--fmin-note", type=str, default="C1")

    parser.add_argument("--segment-frames", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--activation", type=str, default=None)

    parser.add_argument("--dpi", type=int, default=150)

    parser.add_argument("--sample-metadata", type=Path, default=None, help="Optional metadata.json from a previous diagnostic export. If provided, exports exactly this sample.")
    parser.add_argument("--fixed-source-path", type=Path, default=None, help="Optional source WAV path for exact fixed-sample export.")
    parser.add_argument("--fixed-frame-start", type=int, default=None, help="Optional CQT frame start for exact fixed-sample export.")

    parser.add_argument(
        "--overfit-batch-path",
        type=Path,
        default=None,
        help="Optional path to debug/overfit_batches.pt saved by train_vocoder.py.",
    )
    parser.add_argument("--overfit-batch-index", type=int, default=0)

    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return torch.load(path, map_location=device)


def load_sample_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Sample metadata not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_overfit_batch(path: Path, batch_index: int = 0) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Overfit batch file not found: {path}")

    batches = torch.load(path, map_location="cpu")

    if not isinstance(batches, list):
        raise ValueError(f"Expected list of batches in {path}")

    if batch_index < 0 or batch_index >= len(batches):
        raise IndexError(
            f"batch_index={batch_index} out of range for {len(batches)} saved batches"
        )

    return batches[batch_index]


def tupleize(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(tupleize(v) for v in value)
    return value


def build_generator_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
    activation_override: str | None = None,
) -> CQTUHiFiGANGenerator:
    cfg_dict = checkpoint.get("generator_config", {})

    # The config is saved through dataclasses.asdict(), so tuples may be lists.
    cfg_dict = {key: tupleize(value) for key, value in cfg_dict.items()}

    if activation_override is not None:
        saved_activation = cfg_dict.get("activation")
        if saved_activation is not None and saved_activation != activation_override:
            raise ValueError(
                "Activation override does not match checkpoint activation."
                f"Chekcpoint activation={saved_activation!r}, "
                f"override={activation_override!r}. "
                "Changing activation changes model parameters and breaks strict loading."
            )
        cfg_dict["activation"] = activation_override

    config = CQTGeneratorConfig(**cfg_dict)

    generator = CQTUHiFiGANGenerator(config).to(device)
    generator.load_state_dict(checkpoint["generator"], strict=True)
    generator.eval()

    return generator


def normalize_db_to_unit(
    db: np.ndarray,
    db_min: float = -80.0,
    db_max: float = 0.0,
) -> np.ndarray:
    db = np.clip(db, db_min, db_max)
    return ((db - db_min) / (db_max - db_min) * 2.0 - 1.0).astype(np.float32)


def audio_to_cqt_norm(
    y: np.ndarray,
    sample_rate: int,
    hop_length: int,
    n_bins: int,
    bins_per_octave: int,
    fmin_hz: float,
) -> np.ndarray:
    cqt = librosa.cqt(
        y,
        sr=sample_rate,
        hop_length=hop_length,
        fmin=fmin_hz,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
    )

    mag = np.abs(cqt)

    ref = float(np.max(mag)) if mag.size else 1.0
    if ref < 1e-8:
        ref = 1.0

    db = librosa.amplitude_to_db(mag, ref=ref)
    return normalize_db_to_unit(db)


def audio_to_stft_db(
    y: np.ndarray,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    stft = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window="hann",
        center=True,
    )
    mag = np.abs(stft)
    db = librosa.amplitude_to_db(mag, ref=1.0)
    return np.clip(db, -80.0, 0.0).astype(np.float32)


def match_time_dims(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = min(a.shape[-1], b.shape[-1])
    return a[..., :t], b[..., :t]


def l1_np(a: np.ndarray, b: np.ndarray) -> float:
    a, b = match_time_dims(a, b)
    return float(np.mean(np.abs(a - b)))


def corr_np(a: np.ndarray, b: np.ndarray) -> float:
    a, b = match_time_dims(a, b)
    av = a.reshape(-1)
    bv = b.reshape(-1)

    if np.std(av) < 1e-8 or np.std(bv) < 1e-8:
        return 0.0

    return float(np.corrcoef(av, bv)[0, 1])


def rms_np(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y)) + 1e-12))


def spectral_centroid_np(
    y: np.ndarray,
    sample_rate: int,
) -> float:
    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sample_rate,
    )
    return float(np.mean(centroid))


def plot_matrix(
    ax,
    matrix: np.ndarray,
    title: str,
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Time", fontsize=8)
    ax.set_ylabel("Freq", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)


def save_diagnostic_figure(
    path: Path,
    conditioning_cqt: np.ndarray,
    real_cqt: np.ndarray,
    fake_cqt: np.ndarray,
    real_stft: np.ndarray,
    fake_stft: np.ndarray,
    real_audio: np.ndarray,
    fake_audio: np.ndarray,
    sample_rate: int,
    title: str,
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    cond_aligned, fake_cqt_aligned = match_time_dims(conditioning_cqt, fake_cqt)
    real_cqt_aligned, fake_cqt_aligned_2 = match_time_dims(real_cqt, fake_cqt)

    cqt_diff_cond_fake = np.abs(cond_aligned - fake_cqt_aligned)
    cqt_diff_real_fake = np.abs(real_cqt_aligned - fake_cqt_aligned_2)

    real_stft_aligned, fake_stft_aligned = match_time_dims(real_stft, fake_stft)
    stft_diff = np.abs(real_stft_aligned - fake_stft_aligned)

    fig, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(16, 14),
        constrained_layout=True,
    )

    plot_matrix(
        axes[0, 0],
        conditioning_cqt,
        "Conditioning cached CQT",
        vmin=-1.0,
        vmax=1.0,
    )
    plot_matrix(
        axes[0, 1],
        real_cqt,
        "CQT extracted from real waveform",
        vmin=-1.0,
        vmax=1.0,
    )
    plot_matrix(
        axes[0, 2],
        fake_cqt,
        "CQT extracted from generated waveform",
        vmin=-1.0,
        vmax=1.0,
    )

    plot_matrix(
        axes[1, 0],
        cqt_diff_cond_fake,
        "|conditioning CQT - fake waveform CQT|",
        cmap="viridis",
        vmin=0.0,
        vmax=2.0,
    )
    plot_matrix(
        axes[1, 1],
        cqt_diff_real_fake,
        "|real waveform CQT - fake waveform CQT|",
        cmap="viridis",
        vmin=0.0,
        vmax=2.0,
    )
    axes[1, 2].axis("off")

    plot_matrix(
        axes[2, 0],
        real_stft,
        "Real waveform STFT dB",
        vmin=-80.0,
        vmax=0.0,
    )
    plot_matrix(
        axes[2, 1],
        fake_stft,
        "Generated waveform STFT dB",
        vmin=-80.0,
        vmax=0.0,
    )
    plot_matrix(
        axes[2, 2],
        stft_diff,
        "|real STFT - fake STFT|",
        cmap="viridis",
    )

    t_real = np.arange(len(real_audio)) / sample_rate
    t_fake = np.arange(len(fake_audio)) / sample_rate

    axes[3, 0].plot(t_real, real_audio, linewidth=0.7)
    axes[3, 0].set_title("Real waveform", fontsize=9)
    axes[3, 0].set_xlabel("Seconds")
    axes[3, 0].set_ylim(-1.05, 1.05)

    axes[3, 1].plot(t_fake, fake_audio, linewidth=0.7)
    axes[3, 1].set_title("Generated waveform", fontsize=9)
    axes[3, 1].set_xlabel("Seconds")
    axes[3, 1].set_ylim(-1.05, 1.05)

    min_len = min(len(real_audio), len(fake_audio))
    axes[3, 2].plot(t_real[:min_len], real_audio[:min_len], linewidth=0.7, label="real")
    axes[3, 2].plot(t_real[:min_len], fake_audio[:min_len], linewidth=0.7, alpha=0.7, label="fake")
    axes[3, 2].set_title("Overlay", fontsize=9)
    axes[3, 2].set_xlabel("Seconds")
    axes[3, 2].set_ylim(-1.05, 1.05)
    axes[3, 2].legend(fontsize=8)

    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_metric_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

@torch.no_grad()
def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint, device=device)
    generator = build_generator_from_checkpoint(
        checkpoint,
        device=device,
        activation_override=args.activation,
    )

    fixed_batch: dict[str, Any] | None = None

    if args.overfit_batch_path is not None:
        fixed_batch = load_overfit_batch(
            args.overfit_batch_path,
            batch_index=args.overfit_batch_index,
        )

    if args.sample_metadata is not None:
        sample_metadata = load_sample_metadata(args.sample_metadata)

        fixed_batch = build_fixed_vocoder_batch(
            chip_cache_root=args.chip_cache_root,
            source_path=Path(sample_metadata["source_path"]),
            frame_start=int(sample_metadata["frame_start"]),
            segment_frames=args.segment_frames,
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
        )

    elif args.fixed_source_path is not None or args.fixed_frame_start is not None:
        if args.fixed_source_path is None or args.fixed_frame_start is None:
            raise ValueError(
                "Both --fixed-source-path and --fixed-frame-start are required "
                "when using direct fixed-sample mode."
            )

        fixed_batch = build_fixed_vocoder_batch(
            chip_cache_root=args.chip_cache_root,
            source_path=args.fixed_source_path,
            frame_start=args.fixed_frame_start,
            segment_frames=args.segment_frames,
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
        )

    if fixed_batch is None:
        dataset = CQTVocoderDataset(
            chip_cache_root=args.chip_cache_root,
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
            segment_frames=args.segment_frames,
            windows_per_track=1,
            random_window=False,
            cache_waveforms=4,
        )

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        batches = loader
        total_samples = min(args.num_samples, len(dataset))

    else:
        batches = [fixed_batch]
        total_samples = 1

    mrstft = MultiResolutionSTFTLoss(
        MultiResolutionSTFTConfig()
    ).to(device)

    fmin_hz = librosa.note_to_hz(args.fmin_note)

    metrics: list[dict[str, Any]] = []

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {args.out_dir}")
    print(f"Samples: {args.num_samples}")

    for sample_idx, batch in tqdm(
            enumerate(batches),
            total=total_samples,
            desc="Exporting diagnostics",
    ):
        if sample_idx >= total_samples:
            break

        cqt = batch["cqt"].to(device)
        real_audio_t = batch["audio"].to(device)

        fake_audio_t = generator(cqt)

        conditioning_cqt = tensor_to_cqt_np(cqt[0])
        real_audio = tensor_to_audio_np(real_audio_t[0])
        fake_audio = tensor_to_audio_np(fake_audio_t[0])

        real_cqt = audio_to_cqt_norm(
            real_audio,
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
            n_bins=args.cqt_bins,
            bins_per_octave=args.bins_per_octave,
            fmin_hz=fmin_hz,
        )

        fake_cqt = audio_to_cqt_norm(
            fake_audio,
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
            n_bins=args.cqt_bins,
            bins_per_octave=args.bins_per_octave,
            fmin_hz=fmin_hz,
        )

        real_stft = audio_to_stft_db(
            real_audio,
            sample_rate=args.sample_rate,
            n_fft=1024,
            hop_length=256,
        )

        fake_stft = audio_to_stft_db(
            fake_audio,
            sample_rate=args.sample_rate,
            n_fft=1024,
            hop_length=256,
        )

        sample_dir = args.out_dir / f"sample_{sample_idx:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        sf.write(sample_dir / "real.wav", real_audio, args.sample_rate)
        sf.write(sample_dir / "fake.wav", fake_audio, args.sample_rate)

        fig_path = sample_dir / "diagnostics.png"

        save_diagnostic_figure(
            path=fig_path,
            conditioning_cqt=conditioning_cqt,
            real_cqt=real_cqt,
            fake_cqt=fake_cqt,
            real_stft=real_stft,
            fake_stft=fake_stft,
            real_audio=real_audio,
            fake_audio=fake_audio,
            sample_rate=args.sample_rate,
            title=f"Vocoder diagnostics sample {sample_idx}",
            dpi=args.dpi,
        )

        cqt_cond_real_l1 = l1_np(conditioning_cqt, real_cqt)
        cqt_cond_fake_l1 = l1_np(conditioning_cqt, fake_cqt)
        cqt_real_fake_l1 = l1_np(real_cqt, fake_cqt)

        cqt_cond_real_corr = corr_np(conditioning_cqt, real_cqt)
        cqt_cond_fake_corr = corr_np(conditioning_cqt, fake_cqt)
        cqt_real_fake_corr = corr_np(real_cqt, fake_cqt)

        stft_l1 = l1_np(real_stft, fake_stft)
        waveform_l1 = float(
            np.mean(
                np.abs(
                    real_audio[: min(len(real_audio), len(fake_audio))]
                    - fake_audio[: min(len(real_audio), len(fake_audio))]
                )
            )
        )

        mrstft_value = float(
            mrstft(
                fake_audio_t,
                real_audio_t,
            ).detach().cpu().item()
        )

        row: dict[str, Any] = {
            "sample_index": sample_idx,
            "source_path": batch.get("source_path", [""])[0],
            "frame_start": int(batch["frame_start"][0]),
            "frame_end": int(batch["frame_end"][0]),
            "sample_start": int(batch["sample_start"][0]),
            "sample_end": int(batch["sample_end"][0]),

            "cqt_cond_real_l1": cqt_cond_real_l1,
            "cqt_cond_fake_l1": cqt_cond_fake_l1,
            "cqt_real_fake_l1": cqt_real_fake_l1,

            "cqt_cond_real_corr": cqt_cond_real_corr,
            "cqt_cond_fake_corr": cqt_cond_fake_corr,
            "cqt_real_fake_corr": cqt_real_fake_corr,

            "stft_real_fake_l1_db": stft_l1,
            "waveform_l1": waveform_l1,
            "mrstft": mrstft_value,

            "real_rms": rms_np(real_audio),
            "fake_rms": rms_np(fake_audio),
            "rms_ratio_fake_real": rms_np(fake_audio) / max(rms_np(real_audio), 1e-8),

            "real_spectral_centroid": spectral_centroid_np(real_audio, args.sample_rate),
            "fake_spectral_centroid": spectral_centroid_np(fake_audio, args.sample_rate),

            "diagnostic_image": str(fig_path),
            "real_wav": str(sample_dir / "real.wav"),
            "fake_wav": str(sample_dir / "fake.wav"),
        }

        real_peak = float(np.max(np.abs(real_audio))) if real_audio.size else 0.0
        fake_peak = float(np.max(np.abs(fake_audio))) if fake_audio.size else 0.0

        row.update(
            {
                "real_mean": float(np.mean(real_audio)),
                "fake_mean": float(np.mean(fake_audio)),
                "real_peak": real_peak,
                "fake_peak": fake_peak,
                "peak_ratio_fake_real": fake_peak / max(real_peak, 1e-8),
            }
        )

        metrics.append(row)

        with (sample_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(row, f, indent=2, ensure_ascii=False)

    save_metric_csv(args.out_dir / "metrics.csv", metrics)

    summary = {
        "checkpoint": str(args.checkpoint),
        "num_samples": len(metrics),
        "mean_metrics": {},
    }

    if metrics:
        numeric_keys = [
            key
            for key, value in metrics[0].items()
            if isinstance(value, (int, float))
        ]

        for key in numeric_keys:
            summary["mean_metrics"][key] = float(np.mean([row[key] for row in metrics]))

    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Diagnostics export complete.")
    print(f"Wrote: {args.out_dir}")
    print(f"Metrics: {args.out_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()

