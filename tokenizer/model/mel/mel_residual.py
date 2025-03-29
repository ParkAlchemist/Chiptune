import torch
import torch.nn as nn
import torch.nn.functional as F


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


def nonlinearity(x):
    return x * torch.sigmoid(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, temb_channels = 512, conv_shortcut=False):
        super(ResnetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(self.in_channels)
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels,
                               kernel_size=3, stride=1, padding=1)

        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)

        self.norm2 = Normalize(self.out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels,
                               kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttnBlock, self).__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels,
                           kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels,
                           kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels,
                           kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        ## Attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_

