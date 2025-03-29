import torch
import torch.nn as nn
import torch.nn.functional as F

from .mel_encoder import Encoder
from .mel_decoder import Decoder
from .mel_quantizer import VectorQuantizer
from ..residual_quantizer import ResidualQuantizer


double_z = False
z_channels = 64
resolution = 1288
in_channels =  2
out_ch = 2
ch = 32
ch_mult = [1, 2, 4, 8]  # num_down = len(ch_mult)-1
num_res_blocks = 2
attn_resolutions = [161]
dropout = 0.05


class VQVAE(nn.Module):
    def __init__(self, config):
        super(VQVAE, self).__init__()

        self.num_embeddings = config["model_params"]["codebook_size"]

        self.encoder = Encoder(ch = ch,
                               ch_mult=ch_mult,
                               num_res_blocks = num_res_blocks,
                              attn_resolutions = attn_resolutions,
                               dropout=dropout,
                               resamp_with_conv=True,
                               in_channels = in_channels,
                               resolution = resolution,
                               z_channels = z_channels,
                               double_z = double_z)

        self.quantizer = VectorQuantizer(config["model_params"])

        self.decoder = Decoder(ch=ch,
                               out_ch=out_ch,
                               ch_mult=ch_mult,
                               num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions,
                               dropout=dropout,
                               resamp_with_conv=True,
                               in_channels=in_channels,
                               resolution=resolution,
                               z_channels=z_channels)

        self.quant_conv = nn.Conv2d(z_channels, config["model_params"]["latent_dim"], 1)
        self.post_quant_conv = nn.Conv2d(config["model_params"]["latent_dim"], z_channels, 1)

        self.counts = [0 for _ in range(self.num_embeddings)]

    def encode(self, x):
        h = self.encoder(x)
        z = self.quant_conv(h)
        return z

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantizer(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, x):

        z = self.encode(x)

        loss, quantized, info = self.quantizer(z)

        x_recon = self.decode(quantized)

        if not self.training:
            self.counts = [info[2].squeeze().tolist().count(i) + self.counts[i]
                           for i in range(self.num_embeddings)]

        return loss, x_recon, info
