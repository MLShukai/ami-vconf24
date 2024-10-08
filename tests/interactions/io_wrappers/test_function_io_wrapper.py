import pytest
import torch
from pytest_mock import MockerFixture

from ami.interactions.io_wrappers.function_wrapper import (
    FunctionIOWrapper,
    normalize_tensor,
)


class TestFunctionIOWrapper:
    def test_wrap(self, mocker: MockerFixture):
        wrap_function = mocker.Mock(return_value="wrapped")
        wrapper = FunctionIOWrapper(wrap_function)
        assert wrapper.wrap("input") == "wrapped"


def test_normalize_tensor():
    for _ in range(10):
        out = normalize_tensor(torch.rand(100))
        assert out.mean() == pytest.approx(0.0, abs=1e-5)
        assert out.std() == pytest.approx(1.0, abs=1e-5)
