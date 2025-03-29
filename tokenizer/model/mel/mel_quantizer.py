import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum


class VectorQuantizer(nn.Module):
    def __init__(self, config):
        super(VectorQuantizer, self).__init__()

        self.embedding_dim = config["latent_dim"]
        self.num_embeddings = config["codebook_size"]
        self.commitment_cost = config["commitment_beta"]

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings,
                                             1 / self.num_embeddings)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        input_shape = x.shape

        x_flatten = x.view(-1, self.embedding_dim)

        distances = (torch.sum(x_flatten**2, dim=-1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=0, keepdim=True)
                     - 2 * torch.matmul(x_flatten, self.embedding.weight.T))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=self.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized.permute(0, 3, 1, 2).contiguous(), (perplexity, encodings, encoding_indices)


    def decode_indices(self, indices):
        return einsum(indices, self.embedding.weight, 'b n h w, n d -> b d h w')
