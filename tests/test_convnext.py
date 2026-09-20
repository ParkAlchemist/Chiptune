import pytest
import torch


from src.config.vocoder_config import VocoderGeneratorModelConfig

from src.models.blocks.convnext_context import (
    ConvNeXtContextBlock1d,
    ConvNeXtContextTrunk1d
)

from src.models.vocoder_hifigan import CQTUHiFiGANGenerator

@pytest.mark.parametrize(
    ("kernel_size", "dilation"),
    [
        (3, 1),
        (7, 1),
        (7, 2),
        (15, 1),
    ],
)
def test_convnext_context_block_preserves_shape_and_gradient(
    kernel_size: int,
    dilation: int,
) -> None:
    channels = 16

    block = ConvNeXtContextBlock1d(
        channels=channels,
        kernel_size=kernel_size,
        dilation=dilation,
        expansion_ratio=2,
    )

    x = torch.randn(
        2,
        channels,
        128,
        requires_grad=True,
    )

    y = block(x)
    y.square().mean().backward()

    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    parameter_gradients = [
        parameter.grad
        for parameter in block.parameters()
        if parameter.requires_grad
    ]

    assert parameter_gradients
    assert all(
        gradient is not None for gradient in parameter_gradients
    )
    assert all(
        torch.isfinite(gradient).all() for gradient in parameter_gradients
    )


def test_convnext_context_trunk_preserves_shape_and_gradient() -> None:
    trunk = ConvNeXtContextTrunk1d(
        channels=16,
        number_of_blocks=4,
        kernel_sizes=(3, 7, 7, 15),
        dilations=(1, 1, 2, 1),
        expansion_ratio=2,
    )

    x = torch.randn(
        2,
        16,
        128,
        requires_grad=True,
    )

    y = trunk(x)
    y.square().mean().backward()

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    assert len(trunk.blocks) == 4
    assert trunk.kernel_sizes == (3, 7, 7, 15)
    assert trunk.dilations == (1, 1, 2, 1)

    for block in trunk.blocks:
        for parameter in block.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


def test_trunk_expands_scalar_settings() -> None:
    trunk = ConvNeXtContextTrunk1d(
        channels=16,
        number_of_blocks=4,
        kernel_sizes=7,
        dilations=2,
    )

    assert trunk.kernel_sizes == (7, 7, 7, 7)
    assert trunk.dilations == (2, 2, 2, 2)


def test_trunk_accepts_tuple_settings() -> None:
    trunk = ConvNeXtContextTrunk1d(
        channels=16,
        number_of_blocks=3,
        kernel_sizes=(3, 7, 15),
        dilations=(1, 2, 1),
    )

    assert len(trunk.blocks) == 3


def test_trunk_rejects_mismatched_kernel_count() -> None:
    with pytest.raises(
        ValueError,
        match="kernel_sizes",
    ):
        ConvNeXtContextTrunk1d(
            channels=16,
            number_of_blocks=4,
            kernel_sizes=(3, 7),
        )


def test_block_rejects_even_kernel() -> None:
    with pytest.raises(
        ValueError,
        match="kernel_size must be odd, got 4",
    ):
        ConvNeXtContextBlock1d(
            channels=16,
            kernel_size=4,
        )


def test_context_trunk_state_dict_roundtrip() -> None:
    first = ConvNeXtContextTrunk1d(
        channels=16,
        number_of_blocks=2,
        kernel_sizes=(3, 7),
        dilations=(1, 2),
    )

    second = ConvNeXtContextTrunk1d(
        channels=16,
        number_of_blocks=2,
        kernel_sizes=(3, 7),
        dilations=(1, 2),
    )

    second.load_state_dict(
        first.state_dict(),
        strict=True,
    )

    x = torch.randn(2, 16, 64)

    first.eval()
    second.eval()

    with torch.no_grad():
        first_output = first(x)
        second_output = second(x)

    assert torch.allclose(
        first_output,
        second_output,
        atol=1e-6,
        rtol=1e-5,
    )


def test_zero_layer_scale_makes_block_identity() -> None:
    block = ConvNeXtContextBlock1d(
        channels=16,
        layer_scale_initial=0.0,
    )

    x = torch.randn(2, 16, 64)
    y = block(x)

    assert torch.equal(y, x)


def test_vocoder_has_context_trunk() -> None:

    config = VocoderGeneratorModelConfig()

    config.context.enabled = True

    model = CQTUHiFiGANGenerator(
        cqt_bins=116,
        model_config=config
    )

    x = torch.randn(2, 116, 640, requires_grad=True)
    y = model(x)
    y.square().mean().backward()

    assert isinstance(model.context_trunk, ConvNeXtContextTrunk1d)



def test_vocoder_does_not_have_context_trunk() -> None:

    config = VocoderGeneratorModelConfig()

    config.context.enabled = False

    model = CQTUHiFiGANGenerator(
        cqt_bins=116,
        model_config=config
    )

    x = torch.randn(2, 116, 640, requires_grad=True)
    y = model(x)
    y.square().mean().backward()

    assert not isinstance(model.context_trunk, ConvNeXtContextTrunk1d)

