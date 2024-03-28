import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

from .stacked_hidden_state import StackedHiddenState


class ForwardDynamics(nn.Module):
    def __init__(
        self,
        observation_flatten: nn.Module,
        action_flatten: nn.Module,
        obs_action_projection: nn.Module,
        core_model: StackedHiddenState,
        obs_hat_dist_head: nn.Module,
    ) -> None:
        super().__init__()
        self.observation_flatten = observation_flatten
        self.action_flatten = action_flatten
        self.obs_action_projection = obs_action_projection
        self.core_model = core_model
        self.obs_hat_dist_head = obs_hat_dist_head

    def forward(self, obs: Tensor, hidden: Tensor, action: Tensor) -> tuple[Normal, Tensor]:
        obs_flat = self.observation_flatten(obs)
        action_flat = self.action_flatten(action)
        x = self.obs_action_projection(torch.cat((obs_flat, action_flat), dim=-1))
        x, next_hidden = self.core_model(x, hidden)
        obs_hat_dist = self.obs_hat_dist_head(x)
        return obs_hat_dist, next_hidden
