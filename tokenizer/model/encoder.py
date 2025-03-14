import torch
import torch.nn as nn
import numpy as np
from .residual import ResidualStack, Snake


class Encoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder, self).__init__()

        kernel = 3
        stride = 2
        self.conv1 = nn.Conv2d(2, h_dim // 4, kernel_size=kernel, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(h_dim // 4, h_dim // 2, kernel_size=kernel, stride=stride, padding=1)
        self.conv3 = nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1)
        self.layer_norm1 = nn.LayerNorm([h_dim // 4, 19, 235])
        self.layer_norm3 = nn.LayerNorm([h_dim, 5, 59])
        self.drop_out = nn.Dropout(p=0.05)
        self.snake = Snake()
        self.res_stack = ResidualStack(in_dim, h_dim, res_h_dim, n_res_layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.snake(x)
        x = self.layer_norm1(x)
        x = self.conv2(x)
        x = self.snake(x)
        x = self.conv3(x)
        x = self.snake(x)
        x = self.layer_norm3(x)
        x = self.drop_out(x)
        x = self.res_stack(x)
        return x


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
    x = np.random.random_sample((16, 2, 37, 469))
    x = torch.tensor(x).float()

    # test encoder
    encoder = Encoder(128, 128, 3, 64)
    encoder_out = encoder(x)
    print('Encoder out shape:', encoder_out.shape)
