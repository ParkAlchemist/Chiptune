import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.padding]  # Remove the extra padding


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.causal_conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.gate_conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.residual_conv = nn.Conv1d(out_channels, in_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(out_channels, in_channels, kernel_size=1)

    def forward(self, x):
        tanh_out = torch.tanh(self.causal_conv(x))
        sigm_out = torch.sigmoid(self.gate_conv(x))
        gated_out = tanh_out * sigm_out
        residual_out = self.residual_conv(gated_out)
        skip_out = self.skip_conv(gated_out)
        return residual_out + x, skip_out


class WaveNetEncoder(nn.Module):
    def __init__(self, config):
        super(WaveNetEncoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(config['num_layers']):
            dilation = 2 ** i
            self.layers.append(ResidualBlock(config['in_channels'], config['out_channels'], config['kernel_size'], dilation))

    def forward(self, x):
        skip_connections = []
        for layer in self.layers:
            x, skip = layer(x)
            skip_connections.append(skip)
        return sum(skip_connections)


class WaveNetDecoder(nn.Module):
    def __init__(self, config):
        super(WaveNetDecoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(config['num_layers']):
            dilation = 2 ** i
            self.layers.append(ResidualBlock(config['in_channels'], config['out_channels'], config['kernel_size'], dilation))

    def forward(self, x):
        skip_connections = []
        for layer in self.layers:
            x, skip = layer(x)
            skip_connections.append(skip)
        return sum(skip_connections)


class CheckpointedWaveNetEncoder(WaveNetEncoder):
    def forward(self, x):
        skip_connections = []
        for layer in self.layers:
            x, skip = checkpoint.checkpoint(layer, x)
            skip_connections.append(skip)
        return sum(skip_connections)


class CheckpointedWaveNetDecoder(WaveNetDecoder):
    def forward(self, x):
        skip_connections = []
        for layer in self.layers:
            x, skip = checkpoint.checkpoint(layer, x)
            skip_connections.append(skip)
        return sum(skip_connections)


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def get_encoder(config):
    encoder = CheckpointedWaveNetEncoder(config=config)
    encoder.apply(init_weights)
    return encoder

def get_decoder(config):
    decoder = CheckpointedWaveNetDecoder(config=config)
    decoder.apply(init_weights)
    return decoder
