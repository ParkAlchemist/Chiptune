import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Quantizer(nn.Module):
    def __init__(self, config):
        super(Quantizer, self).__init__()
        self.codebook_size = config["codebook_size"]
        self.latent_dim = config["latent_dim"]
        self.beta = config["commitment_beta"]
        self.embedding = nn.Embedding(num_embeddings=self.codebook_size,
                                      embedding_dim=self.latent_dim)
        self.embedding.weight.data.uniform_(-1 / config['codebook_size'],
                                             1 / config['codebook_size'])

    def forward(self, x):
        B, C, T = x.shape
        x_channel_last = x.permute(0, 2, 1)  # Shape: [B, T, C]
        x_flattened = x_channel_last.reshape(B*T, self.latent_dim)

        # Calculate distances
        distances = (
            torch.sum(x_flattened**2, dim=-1, keepdim=True)
            + torch.sum(self.embedding.weight.t()**2, dim=0, keepdim=True)
            - 2 * torch.matmul(x_flattened, self.embedding.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=-1)

        # Quantize
        quant_out = self.embedding(encoding_indices)
        quant_out = quant_out.reshape(B, T, self.latent_dim)
        quant_out = quant_out.permute(0, 2, 1)  # Shape: [B, C, T]

        # Losses
        commitment_loss = self.beta * F.mse_loss(quant_out.detach(), x)
        codebook_loss = F.mse_loss(quant_out, x.detach())
        quantize_losses = {
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss
        }

        # Straight-through estimator
        quant_out = x + (quant_out - x).detach()

        return quant_out, quantize_losses, encoding_indices.view(B, T)

    def quantize_indices(self, indices):
        return rearrange(indices, 'b t -> b t 1') @ self.embedding.weight


def get_quantizer(config):
    quantizer = Quantizer(
        config=config
    )
    return quantizer
