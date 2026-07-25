import torch

from src.models.blocks.conv_norm_act import ConvNormAct
from src.models.blocks.downsample_block import DownsampleBlock
from src.models.blocks.upsample_block import UpsampleBlock
from src.models.blocks.residual_block import ResidualBlock
from src.models.blocks.spatial_self_attention import SpatialSelfAttention
from src.models.blocks.patch_discriminator_block import PatchDiscriminatorBlock


def test_base_blocks_shape_flow():
    x = torch.randn(2, 1, 96, 172)

    stem = ConvNormAct(
        in_channels=1,
        out_channels=32,
        kernel_size=7,
        padding=3,
        norm="instance",
        activation="relu",
        padding_mode="reflect",
    )

    down1 = DownsampleBlock(32, 64)
    down2 = DownsampleBlock(64, 128)

    res = ResidualBlock(
        channels=128,
        dropout=0.1,
        norm="instance",
        padding_mode="reflect",
    )

    attn = SpatialSelfAttention(128)

    up1 = UpsampleBlock(128, 64)
    up2 = UpsampleBlock(64, 32)

    with torch.no_grad():
        h0 = stem(x)
        h1 = down1(h0)
        h2 = down2(h1)
        h3 = res(h2)
        h4 = attn(h3)
        h5 = up1(h4)
        h6 = up2(h5)

    assert h0.shape == (2, 32, 96, 172)
    assert h1.shape == (2, 64, 48, 86)
    assert h2.shape == (2, 128, 24, 43)
    assert h3.shape == h2.shape
    assert h4.shape == h2.shape
    assert h5.shape == (2, 64, 48, 86)
    assert h6.shape == (2, 32, 96, 172)


def test_patch_discriminator_block_shape():
    x = torch.randn(2, 1, 96, 172)

    block = PatchDiscriminatorBlock(
        in_channels=1,
        out_channels=32,
        use_norm=False,
    )

    with torch.no_grad():
        y = block(x)

    assert y.ndim == 4
    assert y.shape[0] == 2
    assert y.shape[1] == 32

