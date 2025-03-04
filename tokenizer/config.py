from pathlib import Path


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
            "convbn_channels": [2, 16, 64, 128, 256],  # Channels for Conv layers in encoder
            "conv_kernel_size": [3, 3, 3, 3],  # Kernel sizes for Conv layers in encoder
            "conv_kernel_strides": [2, 2, 2, 2],  # Strides for Conv layers in encoder
            "conv_kernel_paddings": [1, 1, 1, 1], # Padding for Conv layers in encoder
            "conv_activation_fn": "relu",  # Activation function for Conv layers in encoder
            "convbn_blocks": 4,  # Number of Conv blocks in encoder

            "latent_dim": 256,  # Latent dimension size

            "transposebn_channels": [256, 128, 64, 16, 2],  # Channels for ConvTranspose layers in decoder
            "transpose_kernel_size": [3, 3, 3, 2],  # Kernel sizes for ConvTranspose layers in decoder
            "transpose_kernel_strides": [2, 2, 2, 2],  # Strides for ConvTranspose layers in decoder
            "transpose_kernel_paddings": [1, 1, 1, (5, 6)], # Padding for ConvTranspose layers in decoder
            "transpose_activation_fn": "relu",  # Activation function for ConvTranspose layers in decoder
            "transpose_bn_blocks": 4,  # Number of ConvTranspose blocks in decoder

            "codebook_size": 32768,  # Size of the codebook for the quantizer
            "commitment_beta": 0.25,
            'seq_len': 15,  # Length of input sequence in seconds
            'sample_rate': 16000,
            'ema_decay': 0.8,
            'ema_eps': 1e-5
        },
        "training_params": {
            "batch_size": 8,
            "epochs": 30,
            "lr": 1e-3,
            "experiment_name": "runs/tokenizer",
            "preload": None,
            "model_basename": "tokenizer_",
            "shuffle_dataset": True,
            "validation_split": 0.2,
            "model_folder": "weights",
        },
        "resnet_params": {
            "in_channels": 2,
            "num_hiddens": 64,
            "num_residual_hiddens": 32,
            "num_residual_layers": 2
        }
    }

def get_weights_file_path(config, epoch: str, appendix):
    model_folder = config["training_params"]["model_folder"]
    model_basename = config["training_params"]["model_basename"] + appendix
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)