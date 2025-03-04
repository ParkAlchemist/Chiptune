import torch
import torch.nn as nn
import numpy as np
from .wavenet import get_encoder, get_decoder
from .quantizer_wave import get_quantizer


def mu_law_compress(audio, mu=255):
    """Apply μ-law compression to the audio signal."""
    audio = np.sign(audio) * np.log1p(mu * np.abs(audio)) / np.log1p(mu)
    return audio


def mu_law_expand(compressed_audio, mu=255):
    """Expand the μ-law compressed audio signal."""
    audio = np.sign(compressed_audio) * (1 / mu) * ((1 + mu) ** np.abs(compressed_audio) - 1)
    return audio


class VQVAEWave(nn.Module):
    def __init__(self, config):
        super(VQVAEWave, self).__init__()
        self.encoder = get_encoder(config)
        self.pre_quant_conv = nn.Conv1d(config['out_channels'], config['latent_dim'], kernel_size=1)
        self.quantizer = get_quantizer(config)
        self.post_quant_conv = nn.Conv1d(config['latent_dim'], config['out_channels'], kernel_size=1)
        self.decoder = get_decoder(config)
        self.mu = config["mu"]

    def forward(self, x):
        x = torch.tensor(mu_law_compress(x.numpy(), self.mu)).float().to(
            x.device)
        enc = self.encoder(x)
        quant_input = self.pre_quant_conv(enc)
        quant_output, quant_loss, quant_idxs = self.quantizer(quant_input)
        dec_input = self.post_quant_conv(quant_output)
        out = self.decoder(dec_input)
        out = torch.tensor(
            mu_law_expand(out.detach().cpu().numpy(), self.mu)).float().to(out.device)
        return {
            'generated_waveform': out,
            'quantized_output': quant_output,
            'quantized_losses': quant_loss,
            'quantized_indices': quant_idxs
        }


def get_model(config):
    model = VQVAEWave(config=config["wavenet_params"])
    return model
