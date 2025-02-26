import faiss
import torch
import torch.nn as nn
from einops import rearrange


"""
class EfficientQuantizer(nn.Module):
    def __init__(self, config):
        super(EfficientQuantizer, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config['codebook_size'],
                                      config['latent_dim'])
        self.embedding.weight.data.uniform_(-1 / config['codebook_size'],
                                            1 / config['codebook_size'])
        self.index = faiss.IndexFlatL2(config['latent_dim'])

    def forward(self, x):
        B, C, T = x.shape
        x = x.permute(0, 2, 1).contiguous().view(B * T, C)
        x = x.to(torch.float32)

        # Add embeddings to FAISS index
        self.index.reset()  # Clear the index before adding new embeddings
        self.index.add(self.embedding.weight.detach().cpu().numpy())

        # Search for nearest neighbors
        _, min_encoding_indices = self.index.search(x.detach().cpu().numpy(),1)
        min_encoding_indices = torch.tensor(min_encoding_indices).to(
            x.device).view(B, T)

        # Quantize
        quant_out = self.embedding(min_encoding_indices).view(B, T, C).permute(
            0, 2, 1).contiguous()

        commitment_loss = self.config["commitment_beta"] * torch.mean((quant_out.detach() - x.view(B,T,C).permute(0,2,1).contiguous()) ** 2)
        codebook_loss = torch.mean((quant_out - x.view(B, T, C).permute(0, 2,1).contiguous().detach()) ** 2)
        quantize_losses = {
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss
        }

        # Straight-through estimator
        quant_out = x.view(B, T, C).permute(0, 2, 1).contiguous() + (quant_out - x.view(B, T, C).permute(0, 2,1).contiguous()).detach()

        return quant_out, quantize_losses, min_encoding_indices


def get_efficient_quantizer(config):
    quantizer = EfficientQuantizer(config=config)
    return quantizer
"""

class Quantizer(nn.Module):
    def __init__(self, config):
        super(Quantizer, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config['codebook_size'],
                                      config['latent_dim'])
        self.embedding.weight.data.uniform_(-1 / config['codebook_size'],
                                             1 / config['codebook_size'])

    def forward(self, x):
        B, C, T = x.shape
        x = x.permute(0, 2, 1).contiguous()  # Shape: [B, T, C]
        x = x.view(B * T, C)  # Shape: [B * T, C]

        # Calculate distances
        distances = torch.cdist(x.unsqueeze(1),
                                self.embedding.weight.unsqueeze(0), p=2)
        min_encoding_indices = torch.argmin(distances, dim=-1)

        # Quantize
        quant_out = self.embedding(min_encoding_indices).view(B, T, C)
        quant_out = quant_out.permute(0, 2, 1).contiguous()  # Shape: [B, C, T]

        # Losses
        commitment_loss = self.config["commitment_beta"] * torch.mean((quant_out.detach() - x.view(B,T,C).permute(0,2,1).contiguous()) ** 2)
        codebook_loss = torch.mean((quant_out - x.view(B, T, C).permute(0, 2,1).contiguous().detach()) ** 2)
        quantize_losses = {
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss
        }

        # Straight-through estimator
        quant_out = x.view(B, T, C).permute(0, 2, 1).contiguous() + (quant_out - x.view(B, T, C).permute(0, 2,1).contiguous()).detach()

        return quant_out, quantize_losses, min_encoding_indices.view(B, T)

    def quantize_indices(self, indices):
        return rearrange(indices, 'b t -> b t 1') @ self.embedding.weight


def get_quantizer(config):
    quantizer = Quantizer(
        config=config
    )
    return quantizer
