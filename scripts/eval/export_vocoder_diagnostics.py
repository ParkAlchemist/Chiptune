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
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.vocoder_dataset import CQTVocoderDataset
from src.config.vocoder_config import (
    VocoderExperimentConfig,
)
from src.models.vocoder_hifigan import (
    CQTUHiFiGANGenerator,
)
from src.losses.vocoder_losses import (
    MultiResolutionSTFTConfig,
    MultiResolutionSTFTLoss,
)

from src.eval.vocoder_diagnostics import build_fixed_vocoder_batch, tensor_to_cqt_np, tensor_to_audio_np, \
    match_time_dims, audio_to_cqt_norm, audio_to_stft_db, save_diagnostic_figure


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

    config = VocoderExperimentConfig(**cfg_dict)

    generator = CQTUHiFiGANGenerator(config.data.cqt_bins, config.generator).to(device)
    generator.load_state_dict(checkpoint["generator"], strict=True)
    generator.eval()

    return generator


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

