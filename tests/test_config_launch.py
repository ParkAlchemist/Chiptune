from src.utils.config_launch import (
    flatten_config,
    value_to_cli_args,
    config_to_cli_args,
)


def test_flatten_config():
    config = {
        "paths": {
            "output_root": "runs",
            "experiment_name": "test",
        },
        "training": {
            "batch_size": 4,
            "amp": True,
        },
    }

    flat = flatten_config(config)

    assert flat["output_root"] == "runs"
    assert flat["experiment_name"] == "test"
    assert flat["batch_size"] == 4
    assert flat["amp"] is True


def test_value_to_cli_args_bool_true():
    assert value_to_cli_args("amp", True) == ["--amp"]


def test_value_to_cli_args_bool_false():
    assert value_to_cli_args("amp", False) == []


def test_value_to_cli_args_scalar():
    assert value_to_cli_args("batch_size", 2) == ["--batch-size", "2"]


def test_config_to_cli_args_contains_expected_values():
    config = {
        "data": {
            "batch_size": 2,
        },
        "amp": {
            "amp": False,
        },
    }

    args = config_to_cli_args(config)

    assert "--batch-size" in args
    assert "2" in args
    assert "--amp" not in args

