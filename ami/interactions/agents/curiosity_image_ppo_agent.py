import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution

from ...data.buffers.buffer_names import BufferNames
from ...data.step_data import DataKeys, StepData
from ...models.forward_dynamics import ForwardDynamics
from ...models.model_names import ModelNames
from ...models.model_wrapper import ThreadSafeInferenceWrapper
from ...models.policy_value_common_net import PolicyValueCommonNet
from .base_agent import BaseAgent


class CuriosityImagePPOAgent(BaseAgent[Tensor, Tensor]):
    """Image input curiosity agent with ppo policy."""

    def __init__(self, initial_hidden: Tensor) -> None:
        """Constructs Agent.

        Args:
            initial_hidden: Initial hidden state for the forward dynamics model.
        """
        super().__init__()

        self.forward_dynamics_hidden_state = initial_hidden

    def on_inference_models_attached(self) -> None:
        super().on_inference_models_attached()
        self.image_encoder: ThreadSafeInferenceWrapper[nn.Module] = self.get_inference_model(ModelNames.IMAGE_ENCODER)
        self.forward_dynamics: ThreadSafeInferenceWrapper[ForwardDynamics] = self.get_inference_model(
            ModelNames.FORWARD_DYNAMICS
        )
        self.policy_value: ThreadSafeInferenceWrapper[PolicyValueCommonNet] = self.get_inference_model(
            ModelNames.POLICY_VALUE
        )

    # ------ Interaction Process ------
    predicted_next_embed_observation_dist: Distribution
    forward_dynamics_hidden_state: Tensor
    step_data: StepData

    def _common_step(self, observation: Tensor, initial_step: bool = False) -> Tensor:
        """Common step procedure for agent.

        If `initial_step` is False, some procedures are skipped.
        """

        # \phi(o_t) -> z_t
        embed_obs = self.image_encoder(observation)

        if not initial_step:
            # 報酬計算は初期ステップではできないためスキップ。
            reward = -self.predicted_next_embed_observation_dist.log_prob(embed_obs)
            self.step_data[DataKeys.REWARD] = reward  # r_{t+1}

            # ステップの冒頭でデータコレクトすることで前ステップのデータを収集する。
            self.data_collectors.collect(self.step_data)

        action_dist, value = self.policy_value(observation)

        self.step_data[DataKeys.OBSERVATION] = observation  # o_t
        self.step_data[DataKeys.EMBED_OBSERVATION] = embed_obs  # z_t

        action_dist, value = self.policy_value(observation)
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)

        self.step_data[DataKeys.ACTION] = action  # a_t
        self.step_data[DataKeys.ACTION_LOG_PROBABILITY] = action_log_prob  # log \pi(a_t | o_t)
        self.step_data[DataKeys.VALUE] = value  # v_t
        self.step_data[DataKeys.HIDDEN] = self.forward_dynamics_hidden_state  # h_t

        pred, hidden = self.forward_dynamics(embed_obs, self.forward_dynamics_hidden_state, action)
        self.predicted_next_embed_observation_dist = pred  # p(\hat{z}_{t+1} | z_t, h_t, a_t)
        self.forward_dynamics_hidden_state = hidden  # h_{t+1}

        return action

    def setup(self, observation: Tensor) -> Tensor:
        super().setup(observation)

        self.step_data = StepData()

        return self._common_step(observation, initial_step=True)

    def step(self, observation: Tensor) -> Tensor:
        return self._common_step(observation, initial_step=False)