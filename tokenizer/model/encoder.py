import torch
import torch.nn as nn
import numpy as np
from .residual import ResidualStack


class Encoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder, self).__init__()

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        kernel = 4
        stride = 2
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.BatchNorm2d(h_dim // 2),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.BatchNorm2d(h_dim),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel - 1,
                      stride=stride - 1, padding=1),
            nn.BatchNorm2d(h_dim),
            nn.Dropout(p=0.2),
            ResidualStack(
                h_dim, h_dim, res_h_dim, n_res_layers)

        )

    def forward(self, x):
        return self.conv_stack(x)


def get_encoder(config):
    encoder = Encoder(
        in_dim=config["resnet_params"]["in_dim"],
        h_dim=config["resnet_params"]["h_dim"],
        n_res_layers=config["resnet_params"]["n_res_layers"],
        res_h_dim=config["resnet_params"]["res_h_dim"]
    )
    return encoder


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((16, 2, 39, 469))
    x = torch.tensor(x).float()

    # test encoder
    encoder = Encoder(2, 128, 3, 64)
    encoder_out = encoder(x)
    print('Encoder out shape:', encoder_out.shape)
