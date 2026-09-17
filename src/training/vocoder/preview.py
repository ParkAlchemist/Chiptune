from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import json

import numpy as np
import soundfile as sf
import torch


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


def save_metadata(
    path: Path,
    payload: dict,
) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


@torch.no_grad()
def export_preview_wavs(
    *,
    preview_dir: Path,
    generator: torch.nn.Module,
    preview_batches: Iterable[dict],
    device: torch.device,
    sample_rate: int,
    num_samples: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)

    was_training = generator.training
    generator.eval()

    try:
        written = 0
        for batch in preview_batches:
            cqt = batch["cqt"].to(
                device,
                non_blocking=True,
            )
            real_audio = batch["audio"].to(
                device,
                non_blocking=True,
            )

            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=(
                    use_amp
                    and device.type == "cuda"
                ),
            ):
                fake_audio = generator(cqt)

            batch_size = int(cqt.shape[0])
            source_paths = batch.get(
                "source_path",
                [""] * batch_size,
            )

            for item_index in range(batch_size):
                if written >= num_samples:
                    return

                real_np = tensor_to_audio_np(
                    real_audio[item_index:item_index + 1]
                )
                fake_np = tensor_to_audio_np(
                    fake_audio[item_index:item_index + 1]
                )

                prefix = f"sample_{written:04d}"

                sf.write(
                    preview_dir / f"{prefix}_real.wav",
                    real_np,
                    sample_rate,
                )
                sf.write(
                    preview_dir / f"{prefix}_fake.wav",
                    fake_np,
                    sample_rate,
                )

                save_metadata(
                    preview_dir / f"{prefix}_metadata.json",
                    {
                        "sample_index": written,
                        "source_path": source_paths[item_index],
                        "frame_start": int(
                            batch["frame_start"][item_index]
                        ),
                        "frame_end": int(
                            batch["frame_end"][item_index]
                        ),
                        "sample_start": int(
                            batch["sample_start"][item_index]
                        ),
                        "sample_end": int(
                            batch["sample_end"][item_index]
                        ),
                    },
                )

                written += 1

    finally:
        generator.train(was_training)

