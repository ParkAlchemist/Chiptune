from pathlib import Path


def get_config():
    return {
        "wavenet_params": {
            'num_layers': 5,  # Number of residual blocks
            'in_channels': 1,  # Number of input channels
            'out_channels': 1,  # Number of output channels
            'kernel_size': 2,  # Kernel size for convolutions
            'latent_dim': 128,  # Latent dimension size for VQVAE
            'codebook_size': 256,  # Size of the codebook for the quantizer
            'commitment_beta': 0.25,
            'mu': 255
        },
        "model_params": {
            "convbn_channels": [2, 64, 128, 256],  # Channels for Conv layers in encoder
            "conv_kernel_size": [4, 4, 4],  # Kernel sizes for Conv layers in encoder
            "conv_kernel_strides": [2, 2, 2],  # Strides for Conv layers in encoder
            "conv_activation_fn": "relu",  # Activation function for Conv layers in encoder
            "convbn_blocks": 3,  # Number of Conv blocks in encoder

            "latent_dim": 128,  # Latent dimension size

            "transposebn_channels": [256, 128, 64, 2],  # Channels for ConvTranspose layers in decoder
            "transpose_kernel_size": [4, 4, 4],  # Kernel sizes for ConvTranspose layers in decoder
            "transpose_kernel_strides": [2, 2, 2],  # Strides for ConvTranspose layers in decoder
            "transpose_activation_fn": "relu",  # Activation function for ConvTranspose layers in decoder
            "transpose_bn_blocks": 3,  # Number of ConvTranspose blocks in decoder

            "codebook_size": 512,  # Size of the codebook for the quantizer
            "commitment_beta": 0.25,
        },
        "training_params": {
            "batch_size": 2,
            "epochs": 20,
            "lr": 1e-3,
            "experiment_name": "runs/tokenizer",
            "preload": None,
            "model_basename": "tokenizer_",
            "shuffle_dataset": True,
            "validation_split": 0.2,
            "model_folder": "weights",
            "seq_len": 16000
        }
    }

def get_weights_file_path(config, epoch: str, appendix):
    model_folder = config["training_params"]["model_folder"]
    model_basename = config["training_params"]["model_basename"] + appendix
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)