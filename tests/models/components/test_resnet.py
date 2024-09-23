import pytest
import torch

from ami.models.components.resnet import ResNetConv2d, ResNetFF


class TestResNetFF:
    @pytest.mark.parametrize(
        """
        batch,
        dim,
        dim_hidden,
        depth,
        """,
        [
            (3, 128, 256, 4),
            (6, 28, 56, 2),
        ],
    )
    def test_resnet_ff(self, batch, dim, dim_hidden, depth):
        mod = ResNetFF(dim, dim_hidden, depth)
        x = torch.randn(batch, dim)
        x = mod(x)
        assert x.shape == (batch, dim)


class TestResNetConv2d:
    @pytest.mark.parametrize(
        """
        batch,
        dim,
        dim_hidden,
        depth,
        size,
        """,
        [
            (3, 128, 256, 4, 128),
            (6, 28, 56, 2, 64),
        ],
    )
    def test_resnet_conv2d(self, batch, dim, dim_hidden, depth, size):
        mod = ResNetConv2d(dim, dim_hidden, depth)
        x = torch.randn(batch, dim, size, size)
        x = mod(x)
        assert x.shape == (batch, dim, size, size)
