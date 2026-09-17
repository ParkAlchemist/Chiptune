from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import json
import shutil
import time

from src.config.vocoder_config import VocoderExperimentConfig
from src.training.vocoder.context import (
    RunPaths,
    TrainingState,
)


def save_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(path)


def save_config_snapshot(
    *,
    config: VocoderExperimentConfig,
    source_path: Path,
    paths: RunPaths,
) -> None:
    save_json(
        paths.resolved_config_path,
        asdict(config),
    )

    shutil.copy2(
        source_path,
        paths.source_config_path,
    )


def write_status(
    *,
    paths: RunPaths,
    state: TrainingState,
    losses: dict[str, float],
) -> None:
    save_json(
        paths.status_path,
        {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "micro_step": state.micro_step,
            "segments_seen": state.segments_seen,
            "audio_samples_seen": (
                state.audio_samples_seen
            ),
            "time": time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "losses": losses,
            "latest_checkpoint": (
                state.latest_checkpoint
            ),
            "latest_preview": state.latest_preview,
        },
    )

