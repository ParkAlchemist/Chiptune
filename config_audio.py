from pathlib import Path


def get_config():
    return {
        "batch_size": 4,
        "num_epochs": 20,
        "lr": 10**-4,
        "seg_len": 256,
        "d_model": 512,
        "channels": 2,
        "n_mels": 80,
        "sr": 16000,
        "model_folder": "weights",
        "model_basename": "amodel_",
        "preload": None,
        "augment": True,
        "experiment_name": "runs/amodel"
    }


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)
