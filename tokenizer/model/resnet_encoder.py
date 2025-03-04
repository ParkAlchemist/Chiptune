import torch
import torch.nn as nn
import torch.nn.functional as F
from .residual import ResidualStack


class ResNetEncoder(nn.Module):
    def __init__(self, config):
        super(ResNetEncoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=config["in_channels"],
                                 out_channels=config["num_hiddens"] // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=config["num_hiddens"] // 2,
                                 out_channels=config["num_hiddens"],
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=config["num_hiddens"],
                                 out_channels=config["num_hiddens"],
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=config["num_hiddens"],
                                             num_hiddens=config["num_hiddens"],
                                             num_residual_layers=config["num_residual_layers"],
                                             num_residual_hiddens=config["num_residual_hiddens"])

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)


def get_encoder(config):
    model = ResNetEncoder(config=config["resnet_params"])
    return model
