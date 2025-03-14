import torch
import torch.nn as nn
import numpy as np
from .residual import ResidualStack, Snake

from memory_profiler import profile


class Decoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Decoder, self).__init__()
        kernel = 3
        stride = 2

        self.deconv1 = nn.ConvTranspose2d(in_dim, h_dim, kernel_size=kernel, stride=stride, padding=1)
        self.deconv2 = nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=kernel+1, stride=stride, padding=1)
        self.deconv3 = nn.ConvTranspose2d(h_dim // 2, 2, kernel_size=kernel, stride=stride, padding=(0, 0))
        self.layer_norm1 = nn.LayerNorm([h_dim, 9, 117])
        self.layer_norm3 = nn.LayerNorm([h_dim // 2, 18, 234])
        self.snake = Snake()
        self.drop_out = nn.Dropout(p=0.05)
        self.res_stack = ResidualStack(in_dim, h_dim, res_h_dim, n_res_layers)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.snake(x)
        x = self.layer_norm1(x)
        x = self.drop_out(x)
        x = self.res_stack(x)
        x = self.deconv2(x)
        x = self.snake(x)
        x = self.layer_norm3(x)
        x = self.drop_out(x)
        x = self.deconv3(x)
        return x


def get_decoder(config):
    decoder = Decoder(
        in_dim=config["resnet_params"]["h_dim"],
        h_dim=config["resnet_params"]["h_dim"],
        n_res_layers=config["resnet_params"]["n_res_layers"],
        res_h_dim=config["resnet_params"]["res_h_dim"]
    )
    return decoder


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((16, 128, 5, 59))
    x = torch.tensor(x).float()

    # test decoder
    decoder = Decoder(128, 128, 3, 64)
    decoder_out = decoder(x)
    print('Decoder out shape:', decoder_out.shape)
