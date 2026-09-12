from __future__ import annotations

from pathlib import Path
import runpy
import sys
import argparse
import json
import random
import signal
import time
import shutil
from dataclasses import asdict, dataclass
from typing import Any
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from src.data.vocoder_dataset import CQTVocoderDataset
from src.models.vocoder_hifigan import (
    CQTUHiFiGANGenerator,
)
from src.config.vocoder_config import (
    VocoderGeneratorModelConfig,
    VocoderDiscriminatorConfig,
    VocoderDataConfig,
    VocoderLossConfig,
    ActivationType,
    VocoderExperimentConfig,
    RunConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    LogConfig,
    AMPConfig,
    AugmentationConfig,
    ControlConfig,
)
from src.config.vocoder_config_loader import load_vocoder_config
from src.config.vocoder_config_validator import validate_vocoder_config
from src.config.vocoder_config_adapter import build_loss_config
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
    parser = argparse.ArgumentParser(
        description="Train the CQT-conditioned vocoder."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Structured vocoder TOML configuration.",
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint from which to resume.",
    )

    return parser.parse_args()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


from dataclasses import asdict


def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    models: VocoderModels,
    optimizers: VocoderOptimizers,
    loss_bundle: VocoderLossBundle,
    experiment_config: VocoderExperimentConfig,
    scaler: torch.amp.GradScaler,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "generator": models.generator.state_dict(),
        "discriminator": models.discriminator.state_dict(),
        "optimizer_g": optimizers.generator.state_dict(),
        "optimizer_d": optimizers.discriminator.state_dict(),
        "scaler": scaler.state_dict(),
        "experiment_config": asdict(experiment_config),
    }

    torch.save(payload, path)


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

    optimizers.generator.load_state_dict(checkpoint["optimizer_g"])
    optimizers.discriminator.load_state_dict(checkpoint["optimizer_d"])

    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = int(checkpoint["epoch"])
    global_step = int(checkpoint["global_step"])

    return start_epoch, global_step


def prune_numbered_checkpoints(
    checkpoint_dir: Path,
    keep: int,
) -> None:
    if keep <= 0:
        return

    paths = sorted(
        checkpoint_dir.glob("step_*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for old_path in paths[keep:]:
        old_path.unlink()


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


def build_optimizer(
    parameters,
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
    if config.name == "adam":
        optimizer_class = torch.optim.Adam
    elif config.name == "adamw":
        optimizer_class = torch.optim.AdamW
    else:
        raise ValueError(
            f"Unsupported optimizer: {config.name!r}"
        )

    return optimizer_class(
        parameters,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        eps=config.eps,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
):
    if not config.enabled or config.name == "none":
        return None

    if config.name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.gamma,
        )

    raise ValueError(
        f"Unsupported scheduler: {config.name}"
    )


def main() -> None:
    args = parse_args()

    config_path = args.config.expanduser()

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config_path = config_path.resolve()

    config = load_vocoder_config(config_path)
    validate_vocoder_config(config, check_paths=True)

    from dataclasses import is_dataclass

    print(
        "Loaded experiment config:",
        type(config),
        f"is_dataclass={is_dataclass(config)}",
    )

    if not is_dataclass(config):
        raise TypeError(
            "load_vocoder_config() must return a dataclass instance. "
            f"Got {type(config).__name__}: "
            f"{config!r}"
        )

    run_cfg = config.run
    data_cfg = config.data
    generator_cfg = config.generator
    discriminator_cfg = config.discriminator
    optim_generator_cfg = config.optimizer_generator
    optim_discriminator_cfg = config.optimizer_discriminator
    scheduler_generator_cfg = config.scheduler_generator
    scheduler_discriminator_cfg = config.scheduler_discriminator
    loss_cfg = config.loss
    training_cfg = config.training
    amp_cfg = config.amp
    augmentation_cfg = config.augmentation
    log_cfg = config.logging
    control_cfg = config.control

    if training_cfg.gradient_accumulation_steps != 1:
        raise NotImplementedError(
            "Gradient accumulation is configured but has not yet "
            "been implemented in vocoder_train_step()."
        )

    if training_cfg.generator_start_step != 0:
        raise NotImplementedError(
            "training.generator_start_step is not implemented yet."
        )

    if training_cfg.discriminator_start_step != 0:
        raise NotImplementedError(
            "training.discriminator_start_step is not implemented yet."
        )

    if training_cfg.adversarial_start_step != 0:
        raise NotImplementedError(
            "training.adversarial_start_step is not implemented yet."
        )

    if augmentation_cfg.enabled:
        raise NotImplementedError(
            "Vocoder augmentation is configured but not yet implemented."
        )

    set_seed(run_cfg.seed)

    if run_cfg.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(run_cfg.device)

    run_dir = Path(run_cfg.output_root) / run_cfg.experiment_name
    checkpoint_dir = run_dir / "checkpoints"
    preview_root = run_dir / "previews"
    log_dir = run_dir / "tensorboard"

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    preview_root.mkdir(parents=True, exist_ok=True)

    resolved_config_path = run_dir / "config_resolved.json"
    source_config_path = run_dir / "config_source.toml"

    if not resolved_config_path.exists():
        save_json(resolved_config_path, asdict(config))

    if not source_config_path.exists():
        shutil.copy2(args.config, source_config_path)

    writer = (
        SummaryWriter(log_dir=str(log_dir))
        if log_cfg.tensorboard and SummaryWriter is not None
        else None
    )

    stop_controller = StopController()

    if control_cfg.enabled:
        signal.signal(
            signal.SIGINT,
            stop_controller.request_stop,
        )

    print("Building datasets...")

    train_dataset = CQTVocoderDataset(
        chip_cache_root=Path(data_cfg.chip_cache_root),
        sample_rate=data_cfg.sample_rate,
        hop_length=data_cfg.hop_length,
        segment_frames=data_cfg.segment_frames,
        windows_per_track=data_cfg.windows_per_track,
        random_window=data_cfg.random_window,
        cache_waveforms=data_cfg.cache_waveforms,
    )

    preview_dataset = CQTVocoderDataset(
        chip_cache_root=Path(data_cfg.chip_cache_root),
        sample_rate=data_cfg.sample_rate,
        hop_length=data_cfg.hop_length,
        segment_frames=data_cfg.segment_frames,
        windows_per_track=1,
        random_window=False,
        cache_waveforms=data_cfg.cache_waveforms,
    )

    loader_kwargs = {
        "dataset": train_dataset,
        "batch_size": data_cfg.batch_size,
        "shuffle": True,
        "num_workers": data_cfg.num_workers,
        "drop_last": data_cfg.drop_last,
        "pin_memory": data_cfg.pin_memory,
    }

    if data_cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = (
            data_cfg.persistent_workers
        )
        loader_kwargs["prefetch_factor"] = (
            data_cfg.prefetch_factor
        )

    train_loader = DataLoader(**loader_kwargs)

    if training_cfg.overfit_batches > 0:
        print(
            "Overfit mode enabled: using first "
            f"{training_cfg.overfit_batches} batches repeatedly."
        )

        overfit_batches: list[dict] = []
        iterator = iter(train_loader)

        for _ in range(training_cfg.overfit_batches):
            try:
                overfit_batches.append(next(iterator))
            except StopIteration as exc:
                raise RuntimeError(
                    "The training loader did not contain enough batches "
                    f"for overfit_batches={training_cfg.overfit_batches}."
                ) from exc
    else:
        overfit_batches = None

    if overfit_batches is not None:
        if len(overfit_batches) > 0:
            debug_dir = run_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)

            overfit_path = debug_dir / "overfit_batches.pt"
            torch.save(overfit_batches, overfit_path)

            print(f"Saved fixed overfit batches: {overfit_path}")

            metadata_rows = []

            for batch_idx, batch in enumerate(overfit_batches):
                batch_size = int(batch["cqt"].shape[0])

                for item_idx in range(batch_size):
                    metadata_rows.append(
                        {
                            "batch_index": batch_idx,
                            "item_index": item_idx,
                            "source_path": batch.get("source_path", [""])[
                                item_idx],
                            "frame_start": int(batch["frame_start"][item_idx]),
                            "frame_end": int(batch["frame_end"][item_idx]),
                            "sample_start": int(batch["sample_start"][item_idx]),
                            "sample_end": int(batch["sample_end"][item_idx]),
                        }
                    )

            with (debug_dir / "overfit_batches_metadata.json").open("w",
                                                                    encoding="utf-8") as f:
                json.dump(metadata_rows, f, indent=2, ensure_ascii=False)

            print(
                f"Saved fixed overfit metadata: {debug_dir / 'overfit_batches_metadata.json'}")


    preview_loader = DataLoader(
        preview_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=(device.type == "cuda" and data_cfg.pin_memory),
    )

    print("Building models...")

    generator = CQTUHiFiGANGenerator(data_cfg.cqt_bins, generator_cfg).to(device)
    discriminator = HiFiGANMultiDiscriminator(discriminator_cfg).to(device)

    models = VocoderModels(
        generator=generator,
        discriminator=discriminator,
    )

    optimizers = VocoderOptimizers(
        generator=build_optimizer(
            generator.parameters(),
            optim_generator_cfg,
        ),
        discriminator=build_optimizer(
            discriminator.parameters(),
            optim_discriminator_cfg,
        ),
    )

    scheduler_g = build_scheduler(
        optimizers.generator,
        config.scheduler_generator,
    )

    scheduler_d = build_scheduler(
        optimizers.discriminator,
        config.scheduler_discriminator,
    )

    model_loss_config = build_loss_config(config)

    loss_bundle = VocoderLossBundle(
        model_loss_config
    ).to(device)

    scaler_enabled = (
            amp_cfg.enabled
            and device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=scaler_enabled,
        init_scale=amp_cfg.initial_scale,
        growth_interval=amp_cfg.growth_interval,
    )

    start_epoch = 0
    global_step = 0

    resume_path: Path | None = None

    if args.resume is not None:
        resume_path = args.resume.expanduser()

        if not resume_path.is_absolute():
            resume_path = PROJECT_ROOT / resume_path

        resume_path = resume_path.resolve()

    if resume_path is not None:
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
    print(f"Segment frames: {data_cfg.segment_frames}")
    print(f"Segment samples: {data_cfg.segment_frames * data_cfg.hop_length}")
    print(f"AMP enabled after step: {amp_cfg.start_step if amp_cfg.enabled else 'disabled'}")

    latest_checkpoint_path: str | None = None
    latest_preview_path: str | None = None

    torch.autograd.set_detect_anomaly(
        training_cfg.detect_anomaly
    )

    try:
        for epoch in range(start_epoch, training_cfg.epochs):
            generator.train()
            discriminator.train()

            epoch_iterable = overfit_batches if overfit_batches is not None else train_loader

            progress = tqdm(
                epoch_iterable,
                desc=f"Epoch {epoch + 1}/{training_cfg.epochs}",
                leave=True,
            )

            for batch in progress:
                global_step += 1

                use_amp_now = bool(
                    amp_cfg.enabled
                    and device.type == "cuda"
                    and global_step >= amp_cfg.start_step
                )

                losses = vocoder_train_step(
                    batch=batch,
                    models=models,
                    optimizers=optimizers,
                    loss_bundle=loss_bundle,
                    device=device,
                    use_amp=use_amp_now,
                    scaler=scaler if use_amp_now else None,
                    dtype=amp_cfg.dtype,
                    grad_clip_generator=training_cfg.grad_clip_generator,
                    grad_clip_discriminator=training_cfg.grad_clip_discriminator,
                )

                if not all(np.isfinite(v) for v in losses.values()):
                    emergency_path = (
                            checkpoint_dir
                            / f"nonfinite_step_{global_step:09d}.pt"
                    )

                    save_checkpoint(
                        path=emergency_path,
                        epoch=epoch,
                        global_step=global_step,
                        models=models,
                        optimizers=optimizers,
                        loss_bundle=loss_bundle,
                        experiment_config=config,
                        scaler=scaler,
                    )

                    message = (
                        "Non-finite loss detected. "
                        f"Saved {emergency_path}"
                    )

                    if training_cfg.fail_on_nonfinite:
                        raise RuntimeError(message)

                    print(f"\nWARNING: {message}")

                if writer is not None and global_step % log_cfg.log_every_steps == 0:
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

                if global_step % log_cfg.save_every_steps == 0:

                    checkpoint_path = checkpoint_dir / f"step_{global_step:09d}.pt"
                    save_checkpoint(
                        path=checkpoint_path,
                        epoch=epoch,
                        global_step=global_step,
                        models=models,
                        optimizers=optimizers,
                        loss_bundle=loss_bundle,
                        experiment_config=config,
                        scaler=scaler,
                    )
                    latest_checkpoint_path = str(checkpoint_path)
                    print(f"\nSaved checkpoint: {str(latest_checkpoint_path)}")

                    prune_numbered_checkpoints(
                        checkpoint_dir,
                        log_cfg.keep_numbered_checkpoints,
                    )

                    latest_path = checkpoint_dir / "latest.pt"
                    save_checkpoint(
                        path=latest_path,
                        epoch=epoch,
                        global_step=global_step,
                        models=models,
                        optimizers=optimizers,
                        loss_bundle=loss_bundle,
                        experiment_config=config,
                        scaler=scaler,
                    )
                    latest_checkpoint_path = str(latest_path)

                if log_cfg.preview_every_steps > 0 and global_step % log_cfg.preview_every_steps == 0:
                    preview_dir = preview_root / f"step_{global_step:09d}"
                    export_preview_wavs(
                        preview_dir=preview_dir,
                        generator=generator,
                        preview_loader=epoch_iterable if overfit_batches is not None else preview_loader,
                        device=device,
                        sample_rate=data_cfg.sample_rate,
                        num_samples=log_cfg.preview_num_samples,
                        use_amp=use_amp_now,
                    )

                    latest_preview_path = str(preview_dir)
                    print(f"\nSaved preview WAVs: {preview_dir}")

                if log_cfg.status_every_steps > 0 and global_step % log_cfg.status_every_steps == 0:
                    write_status(
                        run_dir / "status.json",
                        epoch=epoch,
                        global_step=global_step,
                        losses=losses,
                        latest_checkpoint=latest_checkpoint_path,
                        latest_preview=latest_preview_path,
                    )

                if training_cfg.max_steps is not None and global_step >= training_cfg.max_steps:
                    print("Reached max_steps.")
                    stop_controller.stop_requested = True

                if stop_controller.stop_requested:
                    raise KeyboardInterrupt

            latest_path = checkpoint_dir / "latest.pt"
            save_checkpoint(
                path=latest_path,
                epoch=epoch,
                global_step=global_step,
                models=models,
                optimizers=optimizers,
                loss_bundle=loss_bundle,
                experiment_config=config,
                scaler=scaler,
            )

    except KeyboardInterrupt:
        stop_path = checkpoint_dir / f"stop_step_{global_step:09d}.pt"
        save_checkpoint(
            path=stop_path,
            epoch=epoch,
            global_step=global_step,
            models=models,
            optimizers=optimizers,
            loss_bundle=loss_bundle,
            experiment_config=config,
            scaler=scaler,
        )

        latest_path = checkpoint_dir / "latest.pt"
        save_checkpoint(
            path=latest_path,
            epoch=epoch,
            global_step=global_step,
            models=models,
            optimizers=optimizers,
            loss_bundle=loss_bundle,
            experiment_config=config,
            scaler=scaler,
        )

        print(f"\nTraining stopped cleanly. Saved: {stop_path}")

    finally:
        if writer is not None:
            writer.close()

    print("Vocoder training complete.")


if __name__ == "__main__":
    main()

