import torch
import torch.nn as nn
from . import encoder
from . import decoder
from . import quantizer
from . import residual_quantizer

from memory_profiler import profile


class VQVAE(nn.Module):
    def __init__(self, config):
        super(VQVAE, self).__init__()
        self.encoder = encoder.get_encoder(config)
        self.pre_quant_conv = nn.Conv2d(
            config['resnet_params']['h_dim'],
            config['model_params']['latent_dim'],
            kernel_size=1)
        self.quantizer = quantizer.get_quantizer(config)
        self.post_quant_conv = nn.Conv2d(config['model_params']['latent_dim'],
                                         config['resnet_params']['h_dim'],
                                         kernel_size=1)
        self.decoder = decoder.get_decoder(config)

        # Initialize weights
        self._initialize_weights()

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
        return out, quant_output, quant_loss, quant_idxs

    def decode_from_codebook_indices(self, indices):
        quantized_output = self.quantizer.quantize_indices(indices)
        dec_input = self.post_quant_conv(quantized_output)
        return self.decoder(dec_input)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def get_model(config):
    model = VQVAE(
        config=config
    )
    return model
