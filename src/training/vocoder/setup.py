from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from src.training.vocoder.builders import (
    build_training_components,
)
from src.training.vocoder.data import (
    build_dataloaders,
    collect_overfit_batches,
    save_overfit_debug_data,
)
from src.training.vocoder.context import (
    TrainingRuntime,
)
from src.training.vocoder.runtime import (
    build_run_paths,
    create_run_directories,
    configure_stop_controller,
)


def build_training_runtime(
    *,
    config,
    config_path: Path,
    device: torch.device,
):
    paths = build_run_paths(config)
    create_run_directories(paths)

    train_loader, preview_loader = build_dataloaders(
        config,
        device,
    )

    overfit_batches = collect_overfit_batches(
        train_loader,
        config.training.overfit_batches,
    )

    save_overfit_debug_data(
        overfit_batches,
        paths.debug_dir,
    )

    components = build_training_components(
        config,
        device,
    )

    writer = None

    if config.logging.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(
                log_dir=str(paths.tensorboard_dir)
            )
        except Exception as exc:
            print(
                "WARNING: TensorBoard writer unavailable: "
                f"{exc}"
            )

    stop_controller = configure_stop_controller(
        config.control
    )

    return TrainingRuntime(
        config=config,
        config_path=config_path,
        device=device,
        paths=paths,
        components=components,
        train_loader=train_loader,
        preview_loader=preview_loader,
        overfit_batches=overfit_batches,
        writer=writer,
        stop_controller=stop_controller,
    )

