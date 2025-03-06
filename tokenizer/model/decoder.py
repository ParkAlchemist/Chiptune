import torch
import torch.nn as nn
import numpy as np
from .residual import ResidualStack


class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
            nn.BatchNorm2d(h_dim),
            nn.Dropout(p=0.2),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2,
                               kernel_size=kernel, stride=stride, padding=1),
            nn.BatchNorm2d(h_dim // 2),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim//2, 2, kernel_size=kernel+3,
                               stride=stride, padding=(1, 2))
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)


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
    x = np.random.random_sample((16, 128, 9, 117))
    x = torch.tensor(x).float()

    # test decoder
    decoder = Decoder(128, 128, 3, 64)
    decoder_out = decoder(x)
    print('Decoder out shape:', decoder_out.shape)
