import torch.nn as nn
from torch import Tensor


class ResNetFF(nn.Module):
    def __init__(self, dim: int, dim_hidden: int, depth: int, activation: nn.Module = nn.ReLU()):
        super().__init__()
        self.ff_list = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, dim_hidden), activation, nn.Linear(dim_hidden, dim)) for _ in range(depth)]
        )

    def forward(self, x: Tensor) -> Tensor:
        for ff in self.ff_list:
            x_ = x
            x = ff(x)
            x = x + x_
        return x


class ResNetConv2d(nn.Module):
    def __init__(self, dim: int, dim_hidden: int, depth: int, activation: nn.Module = nn.ReLU()) -> None:
        super().__init__()
        self.conv_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(dim, dim_hidden, 3, padding=1), activation, nn.Conv2d(dim_hidden, dim, 3, padding=1)
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.conv_list:
            x_ = x
            x = conv(x)
            x = x + x_
        return x
