from __future__ import annotations

from pathlib import Path
import runpy
import sys
import argparse
import json
import random
import signal
import time
from dataclasses import asdict, dataclass
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from src.data.vocoder_dataset import CQTVocoderDataset
from src.models.vocoder_hifigan import (
    CQTGeneratorConfig,
    CQTUHiFiGANGenerator,
)
from src.models.vocoder_discriminators import (
    HiFiGANMultiDiscriminator,
    VocoderDiscriminatorConfig,
    MultiPeriodDiscriminatorConfig,
)
from src.losses.vocoder_losses import (
    VocoderLossConfig,
    VocoderLossBundle,
)
from src.training.vocoder_step import (
    VocoderModels,
    VocoderOptimizers,
    vocoder_train_step,
)


@dataclass
class VocoderTrainConfig:
    chip_cache_root: str = "E:/Projects/Datasets/cache/cqt/chip"
    output_root: str = "runs/vocoder_cqt"
    experiment_name: str = "cqt96_hifigan_v1"

    sample_rate: int = 22050
    hop_length: int = 512
    cqt_bins: int = 96

    segment_frames: int = 16
    windows_per_track: int = 8
    batch_size: int = 1
    num_workers: int = 0
    cache_waveforms: int = 8

    epochs: int = 50
    max_steps: int | None = None

    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.8
    beta2: float = 0.99

    lambda_adv: float = 1.0
    lambda_feature_matching: float = 2.0
    lambda_mrstft: float = 45.0

    activation: str = "leaky_relu"
    upsample_initial_channel: int = 128
    discriminator_size: str = "small"

    amp: bool = False
    amp_start_step: int = 1000
    amp_init_scale: float = 256.0
    grad_clip_norm: float | None = 10.0

    log_every_steps: int = 25
    save_every_steps: int = 1000
    preview_every_steps: int = 1000
    preview_num_samples: int = 4

    device: str = "cuda"
    seed: int = 1337
    resume: str | None = None


class StopController:
    def __init__(self) -> None:
        self.stop_requested = False

    def request_stop(self, signum=None, frame=None) -> None:
        print("\n[CONTROL] Stop requested. Saving checkpoint before exit...")
        self.stop_requested = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CQT-conditioned vocoder.")

    parser.add_argument("--chip-cache-root", type=str, default="E:/Projects/Datasets/cache/cqt/chip")
    parser.add_argument("--output-root", type=str, default="runs/vocoder_cqt")
    parser.add_argument("--experiment-name", type=str, default="cqt96_hifigan_v1")

    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--cqt-bins", type=int, default=96)

    parser.add_argument("--segment-frames", type=int, default=16)
    parser.add_argument("--windows-per-track", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-waveforms", type=int, default=8)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None)

    parser.add_argument("--lr-g", type=float, default=2e-4)
    parser.add_argument("--lr-d", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.8)
    parser.add_argument("--beta2", type=float, default=0.99)

    parser.add_argument("--lambda-adv", type=float, default=1.0)
    parser.add_argument("--lambda-feature-matching", type=float, default=2.0)
    parser.add_argument("--lambda-mrstft", type=float, default=45.0)

    parser.add_argument("--activation", type=str, choices=["leaky_relu", "snake_beta"], default="leaky_relu")
    parser.add_argument("--upsample-initial-channel", type=int, default=128)
    parser.add_argument("--discriminator-size", type=str, choices=["small", "full"], default="small")

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-start-step", type=int, default=1000)
    parser.add_argument("--amp-init-scale", type=float, default=256.0)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)

    parser.add_argument("--log-every-steps", type=int, default=25)
    parser.add_argument("--save-every-steps", type=int, default=1000)
    parser.add_argument("--preview-every-steps", type=int, default=1000)
    parser.add_argument("--preview-num-samples", type=int, default=4)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", type=str, default=None)

    return parser.parse_args()


def make_config(args: argparse.Namespace) -> VocoderTrainConfig:
    return VocoderTrainConfig(**vars(args))


def build_generator_config(cfg: VocoderTrainConfig) -> CQTGeneratorConfig:
    return CQTGeneratorConfig(
        cqt_bins=cfg.cqt_bins,
        upsample_initial_channel=cfg.upsample_initial_channel,
        upsample_rates=(8, 8, 4, 2),
        upsample_kernel_sizes=(16, 16, 8, 4),
        activation=cfg.activation,
    )


def build_discriminator_config(cfg: VocoderTrainConfig) -> VocoderDiscriminatorConfig:
    if cfg.discriminator_size == "full":
        mpd_channels = (32, 128, 512, 1024, 1024)
    else:
        mpd_channels = (16, 64, 256, 512, 512)

    return VocoderDiscriminatorConfig(
        mpd=MultiPeriodDiscriminatorConfig(
            channels=mpd_channels,
        )
    )


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    models: VocoderModels,
    optimizers: VocoderOptimizers,
    loss_bundle: VocoderLossBundle,
    train_config: VocoderTrainConfig,
    generator_config: CQTGeneratorConfig,
    discriminator_config: VocoderDiscriminatorConfig,
    scaler: torch.amp.GradScaler | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,

        "generator": models.generator.state_dict(),
        "discriminator": models.discriminator.state_dict(),

        "optimizer_generator": optimizers.generator.state_dict(),
        "optimizer_discriminator": optimizers.discriminator.state_dict(),

        "loss_config": asdict(loss_bundle.config),
        "train_config": asdict(train_config),
        "generator_config": asdict(generator_config),
        "discriminator_config": {
            "mpd_channels": discriminator_config.mpd.channels,
            "mpd_periods": discriminator_config.mpd.periods,
            "mpd_norm": discriminator_config.mpd.norm,
            "msd_num_scales": discriminator_config.msd.num_scales,
            "msd_first_norm": discriminator_config.msd.first_discriminator_norm,
            "msd_other_norm": discriminator_config.msd.other_discriminator_norm,
        },

        "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    models: VocoderModels,
    optimizers: VocoderOptimizers,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device)

    models.generator.load_state_dict(checkpoint["generator"])
    models.discriminator.load_state_dict(checkpoint["discriminator"])

    optimizers.generator.load_state_dict(checkpoint["optimizer_generator"])
    optimizers.discriminator.load_state_dict(checkpoint["optimizer_discriminator"])

    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = int(checkpoint["epoch"]) + 1
    global_step = int(checkpoint["global_step"])

    return start_epoch, global_step


def tensor_to_audio_np(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float()

    if x.ndim == 3:
        x = x[0, 0]
    elif x.ndim == 2:
        x = x[0]

    y = x.numpy()

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak * 0.95

    return y.astype(np.float32)


@torch.no_grad()
def export_preview_wavs(
    preview_dir: Path,
    generator: torch.nn.Module,
    preview_loader: DataLoader,
    device: torch.device,
    sample_rate: int,
    num_samples: int,
    use_amp: bool,
) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)

    generator.eval()

    amp_enabled = bool(use_amp and device.type == "cuda")

    written = 0

    for batch in preview_loader:
        cqt = batch["cqt"].to(device, non_blocking=True)
        real_audio = batch["audio"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            fake_audio = generator(cqt)

        for i in range(cqt.shape[0]):
            if written >= num_samples:
                generator.train()
                return

            real_np = tensor_to_audio_np(real_audio[i : i + 1])
            fake_np = tensor_to_audio_np(fake_audio[i : i + 1])

            sf.write(preview_dir / f"sample_{written:04d}_real.wav", real_np, sample_rate)
            sf.write(preview_dir / f"sample_{written:04d}_fake.wav", fake_np, sample_rate)

            manifest = {
                "sample_index": written,
                "source_path": batch.get("source_path", [""])[i],
                "frame_start": int(batch["frame_start"][i]),
                "frame_end": int(batch["frame_end"][i]),
                "sample_start": int(batch["sample_start"][i]),
                "sample_end": int(batch["sample_end"][i]),
            }

            save_json(preview_dir / f"sample_{written:04d}_metadata.json", manifest)

            written += 1

    generator.train()


def write_status(
    path: Path,
    epoch: int,
    global_step: int,
    losses: dict[str, float],
    latest_checkpoint: str | None = None,
    latest_preview: str | None = None,
) -> None:
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "losses": losses,
        "latest_checkpoint": latest_checkpoint,
        "latest_preview": latest_preview,
    }
    save_json(path, payload)


def main() -> None:
    args = parse_args()
    cfg = make_config(args)

    set_seed(cfg.seed)

    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)

    run_dir = Path(cfg.output_root) / cfg.experiment_name
    checkpoint_dir = run_dir / "checkpoints"
    preview_root = run_dir / "previews"
    log_dir = run_dir / "tensorboard"

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    preview_root.mkdir(parents=True, exist_ok=True)

    save_json(run_dir / "config.json", asdict(cfg))

    writer = SummaryWriter(log_dir=str(log_dir)) if SummaryWriter is not None else None

    stop_controller = StopController()
    signal.signal(signal.SIGINT, stop_controller.request_stop)

    print("Building datasets...")

    train_dataset = CQTVocoderDataset(
        chip_cache_root=Path(cfg.chip_cache_root),
        sample_rate=cfg.sample_rate,
        hop_length=cfg.hop_length,
        segment_frames=cfg.segment_frames,
        windows_per_track=cfg.windows_per_track,
        random_window=True,
        cache_waveforms=cfg.cache_waveforms,
    )

    preview_dataset = CQTVocoderDataset(
        chip_cache_root=Path(cfg.chip_cache_root),
        sample_rate=cfg.sample_rate,
        hop_length=cfg.hop_length,
        segment_frames=cfg.segment_frames,
        windows_per_track=1,
        random_window=False,
        cache_waveforms=cfg.cache_waveforms,
    )

    loader_kwargs = {
        "dataset": train_dataset,
        "batch_size": cfg.batch_size,
        "shuffle": True,
        "num_workers": cfg.num_workers,
        "drop_last": True,
        "pin_memory": device.type == "cuda",
    }

    if cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(**loader_kwargs)

    preview_loader = DataLoader(
        preview_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    print("Building models...")

    generator_config = build_generator_config(cfg)
    discriminator_config = build_discriminator_config(cfg)

    generator = CQTUHiFiGANGenerator(generator_config).to(device)
    discriminator = HiFiGANMultiDiscriminator(discriminator_config).to(device)

    models = VocoderModels(
        generator=generator,
        discriminator=discriminator,
    )

    optimizers = VocoderOptimizers(
        generator=torch.optim.AdamW(
            generator.parameters(),
            lr=cfg.lr_g,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=0.0,
        ),
        discriminator=torch.optim.AdamW(
            discriminator.parameters(),
            lr=cfg.lr_d,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=0.0,
        ),
    )

    loss_bundle = VocoderLossBundle(
        VocoderLossConfig(
            lambda_adv=cfg.lambda_adv,
            lambda_feature_matching=cfg.lambda_feature_matching,
            lambda_mrstft=cfg.lambda_mrstft,
        )
    ).to(device)

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(cfg.amp and device.type == "cuda"),
        init_scale=cfg.amp_init_scale,
        growth_interval=2000,
    )

    start_epoch = 0
    global_step = 0

    if cfg.resume is not None:
        resume_path = Path(cfg.resume)
        print(f"Resuming from: {resume_path}")
        start_epoch, global_step = load_checkpoint(
            resume_path,
            models=models,
            optimizers=optimizers,
            scaler=scaler,
            device=device,
        )
        print(f"Resumed at epoch={start_epoch}, global_step={global_step}")

    print("Starting vocoder training.")
    print(f"Run dir: {run_dir}")
    print(f"Device: {device}")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Segment frames: {cfg.segment_frames}")
    print(f"Segment samples: {cfg.segment_frames * cfg.hop_length}")
    print(f"AMP enabled after step: {cfg.amp_start_step if cfg.amp else 'disabled'}")

    latest_checkpoint_path: str | None = None
    latest_preview_path: str | None = None

    try:
        for epoch in range(start_epoch, cfg.epochs):
            generator.train()
            discriminator.train()

            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{cfg.epochs}",
                leave=True,
            )

            for batch in progress:
                global_step += 1

                use_amp_now = bool(
                    cfg.amp
                    and device.type == "cuda"
                    and global_step >= cfg.amp_start_step
                )

                losses = vocoder_train_step(
                    batch=batch,
                    models=models,
                    optimizers=optimizers,
                    loss_bundle=loss_bundle,
                    device=device,
                    use_amp=use_amp_now,
                    scaler=scaler if use_amp_now else None,
                    grad_clip_norm=cfg.grad_clip_norm,
                )

                if not all(np.isfinite(v) for v in losses.values()):
                    emergency_path = checkpoint_dir / f"nonfinite_step_{global_step:09d}.pt"
                    save_checkpoint(
                        emergency_path,
                        epoch,
                        global_step,
                        models,
                        optimizers,
                        loss_bundle,
                        cfg,
                        generator_config,
                        discriminator_config,
                        scaler,
                    )
                    raise RuntimeError(f"Non-finite loss detected. Saved {emergency_path}")

                if writer is not None and global_step % cfg.log_every_steps == 0:
                    for key, value in losses.items():
                        writer.add_scalar(key, value, global_step)

                    writer.add_scalar("train/use_amp", float(use_amp_now), global_step)

                progress.set_postfix(
                    {
                        "G": f"{losses['loss_g_total']:.2f}",
                        "D": f"{losses['loss_d_total']:.3f}",
                        "MR": f"{losses['loss_g_mrstft']:.3f}",
                        "AMP": int(use_amp_now),
                    }
                )

                if global_step % cfg.save_every_steps == 0:

                    #checkpoint_path = checkpoint_dir / f"step_{global_step:09d}.pt"
                    """
                    save_checkpoint(
                        checkpoint_path,
                        epoch,
                        global_step,
                        models,
                        optimizers,
                        loss_bundle,
                        cfg,
                        generator_config,
                        discriminator_config,
                        scaler,
                    )
                    """

                    latest_path = checkpoint_dir / "latest.pt"
                    save_checkpoint(
                        latest_path,
                        epoch,
                        global_step,
                        models,
                        optimizers,
                        loss_bundle,
                        cfg,
                        generator_config,
                        discriminator_config,
                        scaler,
                    )

                    #latest_checkpoint_path = str(checkpoint_path)
                    print(f"\nSaved checkpoint: {str(latest_path)}")

                if cfg.preview_every_steps > 0 and global_step % cfg.preview_every_steps == 0:
                    preview_dir = preview_root / f"step_{global_step:09d}"
                    export_preview_wavs(
                        preview_dir=preview_dir,
                        generator=generator,
                        preview_loader=preview_loader,
                        device=device,
                        sample_rate=cfg.sample_rate,
                        num_samples=cfg.preview_num_samples,
                        use_amp=use_amp_now,
                    )

                    latest_preview_path = str(preview_dir)
                    print(f"\nSaved preview WAVs: {preview_dir}")

                if global_step % cfg.log_every_steps == 0:
                    write_status(
                        run_dir / "status.json",
                        epoch=epoch,
                        global_step=global_step,
                        losses=losses,
                        latest_checkpoint=latest_checkpoint_path,
                        latest_preview=latest_preview_path,
                    )

                if cfg.max_steps is not None and global_step >= cfg.max_steps:
                    print("Reached max_steps.")
                    stop_controller.stop_requested = True

                if stop_controller.stop_requested:
                    raise KeyboardInterrupt

            epoch_path = checkpoint_dir / f"epoch_{epoch + 1:04d}.pt"
            save_checkpoint(
                epoch_path,
                epoch,
                global_step,
                models,
                optimizers,
                loss_bundle,
                cfg,
                generator_config,
                discriminator_config,
                scaler,
            )

            latest_path = checkpoint_dir / "latest.pt"
            save_checkpoint(
                latest_path,
                epoch,
                global_step,
                models,
                optimizers,
                loss_bundle,
                cfg,
                generator_config,
                discriminator_config,
                scaler,
            )

            print(f"Saved epoch checkpoint: {epoch_path}")

    except KeyboardInterrupt:
        stop_path = checkpoint_dir / f"stop_step_{global_step:09d}.pt"
        save_checkpoint(
            stop_path,
            epoch,
            global_step,
            models,
            optimizers,
            loss_bundle,
            cfg,
            generator_config,
            discriminator_config,
            scaler,
        )

        latest_path = checkpoint_dir / "latest.pt"
        save_checkpoint(
            latest_path,
            epoch,
            global_step,
            models,
            optimizers,
            loss_bundle,
            cfg,
            generator_config,
            discriminator_config,
            scaler,
        )

        print(f"\nTraining stopped cleanly. Saved: {stop_path}")

    finally:
        if writer is not None:
            writer.close()

    print("Vocoder training complete.")


if __name__ == "__main__":
    main()

