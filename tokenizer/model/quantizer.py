import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
from scipy.linalg import hadamard
from sklearn.cluster import KMeans


class Quantizer(nn.Module):
    def __init__(self, config):
        super(Quantizer, self).__init__()
        self.codebook_size = config["codebook_size"]
        self.latent_dim = config["latent_dim"]
        self.beta = config["commitment_beta"]
        self.decay = config["ema_decay"]
        self.eps = config["ema_eps"]
        self.diversity_weight = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training = True
        self.embedding = nn.Embedding(num_embeddings=self.codebook_size,
                                      embedding_dim=self.latent_dim)
        self.embedding.weight.data.uniform_(-1/self.codebook_size,
                                            1/self.codebook_size)
        # EMA values
        self.ema_cluster_size = torch.ones(self.codebook_size).to(self.device)
        self.ema_w = torch.Tensor(self.codebook_size,
                                  self.latent_dim).uniform_(-1 / self.codebook_size, 1 / self.codebook_size).to(self.device)

        # Hadamard rotation matrix
        self.rotation_matrix = nn.Parameter(
            torch.tensor(hadamard(self.latent_dim), dtype=torch.float32))

        # Track code usage
        self.code_usage_threshold = 256
        self.num_codes_replaced = 0
        self.code_usage = torch.full(size=(self.codebook_size,),
                                     fill_value=self.code_usage_threshold).to(self.device)
        self.rho = 0.1

    def initialize_codebook(self, encoder, data_loader):
        # Collect all latent representations
        latents = []
        for x in data_loader:
            with torch.no_grad():
                z_e = encoder(x)
                latents.append(z_e.view(-1, self.latent_dim))
        latents = torch.cat(latents, dim=0).cpu().numpy()

        # Apply k-means clustering
        kmeans = KMeans(n_clusters=self.codebook_size, random_state=0).fit(
            latents)
        centroids = kmeans.cluster_centers_

        # Initialize codebook with centroids
        self.embedding.weight.data.copy_(torch.tensor(centroids))
        self.ema_w.data.copy_(torch.tensor(centroids))


    def forward(self, x):

        B, C, H, W = x.shape
        x_channel_last = x.permute(0, 2, 3, 1).contiguous()  # Shape: [B, H, W, C]
        x_flattened = x_channel_last.reshape(B*H*W, self.latent_dim)

        ## Apply rotation
        x_rotated = torch.matmul(x_flattened, self.rotation_matrix)

        ## Calculate distances
        distances = (
            torch.sum(x_rotated**2, dim=-1, keepdim=True)
            + torch.sum(self.embedding.weight.t()**2, dim=0, keepdim=True)
            - 2 * torch.matmul(x_rotated, self.embedding.weight.t())
        )

        ## Encoding
        encoding_indices = torch.argmin(distances, dim=-1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0],
                                self.codebook_size, device=self.device)
        encodings.scatter_(1, encoding_indices, 1)

        ## Quantize
        quant_out = torch.matmul(encodings.detach(), self.embedding.weight).view(x.shape)

        ## Losses
        # Commitment loss encourages the encoder ouputs to commit to the nearest codebook entry
        commitment_loss = self.beta * F.mse_loss(quant_out.detach(), x)
        # Codebook loss ensures that the codebook entries are updated to be close to the encoder outputs
        codebook_loss = F.mse_loss(quant_out, x.detach())
        avg_probs = torch.mean(encodings.detach(), dim=0)
        # Entropy loss encourages the usage of all codes more uniformly
        entropy_loss = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10)) * self.diversity_weight
        #perplexity = torch.exp(entropy_loss)
        loss = codebook_loss + commitment_loss + entropy_loss
        """
        quantize_losses = {
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss,
            'entropy_loss': entropy_loss,
            'perplexity': perplexity
        }
        """

        ## Codebook updates
        if self.training:
            # EMA
            self.ema_cluster_size = self.ema_cluster_size * self.decay + (
                        1 - self.decay) * torch.sum(encodings, 0)
            n = torch.sum(self.ema_cluster_size)
            self.ema_cluster_size = ((self.ema_cluster_size + self.eps) / (
                        n + self.codebook_size * self.eps) * n)
            dw = torch.matmul(encodings.t(), x_rotated.detach()).to(self.device)
            self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw
            self.embedding.weight.data = self.ema_w / self.ema_cluster_size.unsqueeze(1)

            # Update code usage
            encoding_mask = torch.zeros(self.codebook_size,
                                        device=self.device).to(self.device)
            encoding_mask.scatter_(0, encoding_indices.squeeze(), 1)
            self.code_usage = torch.where(encoding_mask == 1,
                                          self.code_usage_threshold,
                                          self.code_usage - 1)

            # Replace inactive codes
            self.replace(x_rotated)

        # Straight-through estimator
        quant_out = x + (quant_out - x).detach()

        return quant_out, loss, encoding_indices


    def replace(self, x):

        inactive_codes = torch.where(self.code_usage <= 0)[0]
        if inactive_codes.numel() > 0:
            mean = torch.mean(x, dim=0)
            noise = torch.rand_like(mean) * self.rho
            self.embedding.weight.data[inactive_codes] = (mean + noise).detach().to(self.device)
            self.ema_w.data[inactive_codes] = (mean + noise).detach().to(self.device)
            self.ema_cluster_size[inactive_codes] = 1.0
            self.num_codes_replaced += inactive_codes.numel()
            self.code_usage[inactive_codes] = self.code_usage_threshold


    def quantize_indices(self, indices):
        return einsum(indices, self.embedding.weight, 'b n h w, n d -> b d h w')


def get_quantizer(config):
    quantizer = Quantizer(
        config=config["model_params"]
    )
    return quantizer
