import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .mel_residual import ResnetBlock, AttnBlock, Normalize, nonlinearity


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super(Downsample, self).__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels,
                                  kernel_size=3, stride=2, padding=0)
            self.pad = (0, 1, 0, 1)
        else:
            self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.with_conv:
            x = F.pad(x, self.pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = self.avg_pool(x)
        return x


class Encoder(nn.Module):
    def __init__(self, ch, num_res_blocks, attn_resolutions,
                 in_channels, resolution, z_channels, ch_mult=(1, 2, 4, 8),
                 dropout=0.0, resamp_with_conv=True, double_z=False):
        super(Encoder, self).__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch,
                                 kernel_size=3, stride=1, padding=1)
        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in,
                                  2*z_channels if double_z else z_channels,
                                  kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((1, 2, 128, 1288))
    x = torch.tensor(x).float()

    # test encoder
    encoder = Encoder(ch=64, num_res_blocks=2, attn_resolutions=[53], in_channels=2, resolution=1288, z_channels=128)
    encoder_out = encoder(x)
    print('Encoder out shape:', encoder_out.shape)
