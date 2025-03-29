import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
from scipy.linalg import hadamard
import numpy as np


class ResidualQuantizer(nn.Module):
    def __init__(self, config):
        super(ResidualQuantizer, self).__init__()
        self.num_codebooks = config["num_codebooks"]
        self.codebook_size = config["codebook_size"]
        self.latent_dim = config["latent_dim"]
        self.beta = config["commitment_beta"]
        self.decay = config["ema_decay"]
        self.eps = config["ema_eps"]
        self.dead_code_threshold = 0.05
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training = True

        # Initialize multiple codebooks
        self.codebooks = nn.ModuleList([
            nn.Embedding(self.codebook_size, self.latent_dim).to(self.device) for _ in
            range(self.num_codebooks)
        ])
        for codebook in self.codebooks:
            codebook.weight.data.uniform_(-1 / self.codebook_size,
                                          1 / self.codebook_size)
        # EMA values for each codebook
        self.ema_cluster_size = [
            torch.ones(self.codebook_size).to(self.device) for _ in
            range(self.num_codebooks)]
        self.ema_w = [
            torch.Tensor(self.codebook_size, self.latent_dim).uniform_(
                -1 / self.codebook_size, 1 / self.codebook_size).to(
                self.device) for _ in range(self.num_codebooks)]

        # Hadamard rotation matrix
        self.rotation_matrix = nn.Parameter(
            torch.tensor(hadamard(self.latent_dim), dtype=torch.float32)).to(self.device)


    def forward(self, x):

        x = x.to(self.device)
        B, C, H, W = x.shape
        x_channel_last = x.permute(0, 2, 3, 1).contiguous()  # Shape: [B, H, W, C]
        x_flattened = x_channel_last.reshape(B*H*W, self.latent_dim)

        ## Apply rotation
        x_rotated = torch.matmul(x_flattened, self.rotation_matrix)

        residual = x_rotated
        quantized_outputs = []
        encoding_index_list = []
        encoding_list = []

        for i, codebook in enumerate(self.codebooks):
            distances = (
                    torch.sum(residual ** 2, dim=-1, keepdim=True)
                    + torch.sum(codebook.weight.T ** 2, dim=0, keepdim=True)
                    - 2 * torch.matmul(residual, codebook.weight.T)
            )
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1).to(self.device)
            encodings = torch.zeros(encoding_indices.shape[0],
                                    self.codebook_size, device=self.device)
            encodings.scatter_(1, encoding_indices, 1)

            quantized = torch.matmul(encodings, codebook.weight)
            quantized_outputs.append(quantized.detach())
            encoding_list.append(encodings.detach())
            encoding_index_list.append(encoding_indices.detach())
            residual = residual - quantized

            if self.training:
                self.ema_cluster_size[i] = (self.ema_cluster_size[i] * self.decay
                                            + (1 - self.decay) * torch.sum(encodings, 0))
                n = torch.sum(self.ema_cluster_size[i])
                self.ema_cluster_size[i] = ((self.ema_cluster_size[i] + self.eps)
                                            / (n + self.codebook_size * self.eps) * n)
                dw = torch.matmul(encodings.t(), residual.detach()).to(self.device)
                self.ema_w[i] = self.ema_w[i] * self.decay + (1 - self.decay) * dw
                codebook.weight.data = (self.ema_w[i]
                                        / self.ema_cluster_size[i].unsqueeze(1))

                # Adaptive replacement of dead codes
                """
                dead_codes = torch.where(self.ema_cluster_size[i] < self.dead_code_threshold)[0]
                if len(dead_codes) > 0:
                    active_codes = torch.where(self.ema_cluster_size[i] >= self.dead_code_threshold)[0]
                    if len(active_codes) > 0:
                        new_codes = codebook.weight.data[active_codes].mean(dim=0, keepdim=True)
                        codebook.weight.data[dead_codes] = new_codes
                """

        quant_out = torch.stack(quantized_outputs, dim=0).sum(dim=0).view(x.shape)

        commitment_loss = self.beta * F.mse_loss(quant_out.detach(), x)
        codebook_loss = F.mse_loss(quant_out, x.detach())
        avg_probs = torch.mean(torch.stack(encoding_list, dim=0), dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        loss = codebook_loss + commitment_loss

        quant_out = x + (quant_out - x).detach()

        return loss, quant_out, (perplexity, encoding_list, encoding_index_list)


    def quantize_indices(self, indices):
        return einsum(indices, self.embedding.weight, 'b n h w, n d -> b d h w')


def get_quantizer(config):
    quantizer = ResidualQuantizer(
        config=config["model_params"]
    )
    return quantizer


if __name__ == "__main__":
    config = {"codebook_size": 512,
              "latent_dim": 128,
              "commitment_beta": 0.25,
              "num_codebooks": 4,
              "ema_eps": 1e-8,
              "ema_decay": 0.99
              }
    quant = ResidualQuantizer(config)
    input = np.random.random_sample((1, 128, 16, 161))
    input = torch.tensor(input).float()

    loss, output, info = quant(input)
    print(info[2])
