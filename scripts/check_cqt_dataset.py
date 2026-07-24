from torch.utils.data import DataLoader

from src.data.cqt_dataset import build_cached_unpaired_dataset
from src.config import config


def main() -> None:
    dataset = build_cached_unpaired_dataset()

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))

    print("real_x:", batch["real_x"].shape, batch["real_x"].dtype)
    print("real_y:", batch["real_y"].shape, batch["real_y"].dtype)
    print("x min/max:", batch["real_x"].min().item(), batch["real_x"].max().item())
    print("y min/max:", batch["real_y"].min().item(), batch["real_y"].max().item())

    expected_bins = config.cqt.n_bins
    assert batch["real_x"].shape[1] == 1
    assert batch["real_y"].shape[1] == 1
    assert batch["real_x"].shape[2] == expected_bins
    assert batch["real_y"].shape[2] == expected_bins

    print("CQT dataset smoke test passed.")


if __name__ == "__main__":
    main()
