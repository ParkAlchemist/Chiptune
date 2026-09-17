import torch

from src.training.vocoder.accumulation import accumulation_windows


def test_accumulation_windows_full_groups():
    values = list(range(8))

    windows = list(
        accumulation_windows(values, 4)
    )

    assert windows == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]


def test_accumulation_windows_partial_final_group():
    values = list(range(6))

    windows = list(
        accumulation_windows(values, 4)
    )

    assert windows == [
        [0, 1, 2, 3],
        [4, 5],
    ]


def test_gradient_accumulation_matches_full_batch():
    torch.manual_seed(1337)

    full_model = torch.nn.Linear(4, 1, bias=False)
    accumulated_model = torch.nn.Linear(
        4,
        1,
        bias=False,
    )

    accumulated_model.load_state_dict(
        full_model.state_dict()
    )

    full_optimizer = torch.optim.SGD(
        full_model.parameters(),
        lr=0.1,
    )
    accumulated_optimizer = torch.optim.SGD(
        accumulated_model.parameters(),
        lr=0.1,
    )

    x = torch.randn(8, 4)
    target = torch.randn(8, 1)

    full_optimizer.zero_grad(set_to_none=True)

    full_prediction = full_model(x)
    full_loss = torch.nn.functional.mse_loss(
        full_prediction,
        target,
    )
    full_loss.backward()
    full_optimizer.step()

    accumulated_optimizer.zero_grad(set_to_none=True)

    microbatches = [
        (x[:4], target[:4]),
        (x[4:], target[4:]),
    ]

    for micro_x, micro_target in microbatches:
        prediction = accumulated_model(micro_x)
        loss = torch.nn.functional.mse_loss(
            prediction,
            micro_target,
        )
        (loss / len(microbatches)).backward()

    accumulated_optimizer.step()

    for full_parameter, accumulated_parameter in zip(
        full_model.parameters(),
        accumulated_model.parameters(),
    ):
        assert torch.allclose(
            full_parameter,
            accumulated_parameter,
            atol=1e-6,
            rtol=1e-5,
        )

