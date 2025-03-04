import torch
import torch.nn as nn
from .resnet_encoder import get_encoder
from .resnet_decoder import get_decoder
from .quantizer import get_quantizer


class ResNetVQVAE(nn.Module):
    def __init__(self, config):
        super(ResNetVQVAE, self).__init__()
        self.encoder = get_encoder(config)
        self.pre_quant_conv = nn.Conv2d(
            config['resnet_params']['num_hiddens'],
            config['model_params']['latent_dim'],
            kernel_size=1)
        self.quantizer = get_quantizer(config)
        self.post_quant_conv = nn.Conv2d(config['model_params']['latent_dim'],
                                         config['resnet_params'][
                                             'num_hiddens'],
                                         kernel_size=1)
        self.decoder = get_decoder(config)

    def tokenize(self, x):
        enc = self.encoder(x)
        quant_input = self.pre_quant_conv(enc)
        quant_output, losses, quant_indices = self.quantizer(quant_input)
        return quant_indices

    def forward(self, x):
        enc = self.encoder(x)
        quant_input = self.pre_quant_conv(enc)
        quant_output, quant_loss, quant_idxs = self.quantizer(quant_input)
        dec_input = self.post_quant_conv(quant_output)
        out = self.decoder(dec_input)
        return {
            'generated_image': out,
            'quantized_output': quant_output,
            'quantized_losses': quant_loss,
            'quantized_indices': quant_idxs
        }

    def decode_from_codebook_indices(self, indices):
        quantized_output = self.quantizer.quantize_indices(indices)
        dec_input = self.post_quant_conv(quantized_output)
        return self.decoder(dec_input)


def get_model(config):
    model = ResNetVQVAE(
        config=config
    )
    return model
