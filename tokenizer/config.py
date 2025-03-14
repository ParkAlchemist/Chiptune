from pathlib import Path
from utils import Feature


def get_config():
    return {
        "wavenet_params": {
            'num_layers': 20,  # Number of residual blocks
            'in_channels': 2,  # Number of input channels
            'out_channels': 2,  # Number of output channels
            'kernel_size': 2,  # Kernel size for convolutions
            'latent_dim': 256,  # Latent dimension size for VQVAE
            'codebook_size': 512,  # Size of the codebook for the quantizer
            'commitment_beta': 0.25,
            'mu': 255,
            'sample_rate': 16000, # Sample rate of the input audio
            'tgt_token_rate': 50, # Target number of tokens per second of audio
            'seq_len': 15 # Length of input sequence in seconds
        },
        "model_params": {
            "latent_dim": 128,  # Latent dimension size
            "codebook_size": 512,  # Size of the codebook for the quantizer
            "num_codebooks": 4,
            "temperature": 1.0,
            "temperature_decay": 0.999,
            "codebook_dropout_prob": 0.01,
            "commitment_beta": 0.25,
            'seq_len': 15,  # Length of input sequence in seconds
            'sample_rate': 16000,
            'ema_decay': 0.99,
            'ema_eps': 1e-8
        },
        "training_params": {
            "batch_size": 8,
            "epochs_tok": 20,
            "epochs_enc": 10,
            "lr": 1e-3,
            "experiment_name_tok": "runs/tokenizer",
            "experiment_name_enc": "runs/encoder",
            "preload": None,
            "load_encoder": None,
            "model_basename_tok": "tokenizer_",
            "model_basename_enc": "encoder_",
            "model_basename_quant": "quantizer_",
            "shuffle_dataset": True,
            "validation_split": 0.2,
            "model_folder": "weights",
            "features_to_extract": [Feature.MFCC, Feature.CHROMA,
                                    Feature.SPECTRAL_CENTROID, Feature.SPECTRAL_CONTRAST,
                                    Feature.SPECTRAL_FLATNESS, Feature.SPECTRAL_ROLLOFF,
                                    Feature.ZCR, Feature.ONSET]
        },
        "resnet_params": {
            "in_dim": 128,
            "h_dim": 128,
            "res_h_dim": 64,
            "n_res_layers": 3
        }
    }

def get_enc_weights_file_path(config, epoch: str, appendix):
    model_folder = config["training_params"]["model_folder"]
    model_basename = config["training_params"]["model_basename_enc"] + appendix
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


def get_tok_weights_file_path(config, epoch: str, appendix):
    model_folder = config["training_params"]["model_folder"]
    model_basename = config["training_params"]["model_basename_tok"] + appendix
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


def get_quant_weights_file_path(config, epoch: str, appendix):
    model_folder = config["training_params"]["model_folder"]
    model_basename = config["training_params"]["model_basename_quant"] + appendix
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)
