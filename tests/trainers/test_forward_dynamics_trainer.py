from functools import partial

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from ami.data.buffers.buffer_names import BufferNames
from ami.data.buffers.causal_data_buffer import CausalDataBuffer
from ami.data.step_data import DataKeys, StepData
from ami.data.utils import DataCollectorsDict, DataUsersDict
from ami.models.components.fully_connected_fixed_std_normal import (
    FullyConnectedFixedStdNormal,
)
from ami.models.components.sconv import SConv
from ami.models.forward_dynamics import ForwardDynamics
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ModelWrapper
from ami.models.utils import ModelWrappersDict
from ami.models.vae import Conv2dEncoder, EncoderWrapper
from ami.tensorboard_loggers import StepIntervalLogger
from ami.trainers.forward_dynamics_trainer import ForwardDynamicsTrainer

BATCH = 4
DEPTH = 8
DIM = 16
DIM_FF_HIDDEN = 32
LEN = 64
DROPOUT = 0.1
DIM_OBS = 32
DIM_ACTION = 8
HEIGHT = 256
WIDTH = 256
CHANNELS = 3


class TestSconv:
    @pytest.fixture
    def core_model(self):
        sconv = SConv(DEPTH, DIM, DIM_FF_HIDDEN, DROPOUT)
        return sconv

    @pytest.fixture
    def observation_flatten(self):
        return nn.Identity()

    @pytest.fixture
    def action_flatten(self):
        return nn.Identity()

    @pytest.fixture
    def obs_action_projection(self):
        return nn.Linear(DIM_OBS + DIM_ACTION, DIM)

    @pytest.fixture
    def obs_hat_dist_head(self):
        return FullyConnectedFixedStdNormal(DIM, DIM_OBS)

    @pytest.fixture
    def forward_dynamics(
        self,
        observation_flatten,
        action_flatten,
        obs_action_projection,
        core_model,
        obs_hat_dist_head,
    ):
        return ForwardDynamics(
            observation_flatten, action_flatten, obs_action_projection, core_model, obs_hat_dist_head
        )

    @pytest.fixture
    def encoder(self):
        encoder = Conv2dEncoder(HEIGHT, WIDTH, CHANNELS, DIM_OBS)
        return encoder

    @pytest.fixture
    def encoder_wrapper(self, encoder, device):
        encoder_wrapper = EncoderWrapper(encoder, default_device=device)
        return encoder_wrapper

    @pytest.fixture
    def trajectory_step_data(self) -> StepData:
        d = StepData()
        d[DataKeys.OBSERVATION] = torch.randn(CHANNELS, HEIGHT, WIDTH)
        d[DataKeys.HIDDEN] = torch.randn(DEPTH, DIM)
        d[DataKeys.ACTION] = torch.randn(DIM_ACTION)
        return d

    @pytest.fixture
    def trajectory_buffer_dict(self, trajectory_step_data: StepData) -> DataCollectorsDict:
        d = DataCollectorsDict.from_data_buffers(
            **{
                BufferNames.FORWARD_DYNAMICS_TRAJECTORY: CausalDataBuffer.reconstructable_init(
                    32,
                    [
                        DataKeys.OBSERVATION,
                        DataKeys.HIDDEN,
                        DataKeys.ACTION,
                    ],
                )
            }
        )

        d.collect(trajectory_step_data)
        d.collect(trajectory_step_data)
        d.collect(trajectory_step_data)
        d.collect(trajectory_step_data)
        return d

    @pytest.fixture
    def partial_dataloader(self):
        partial_dataloader = partial(DataLoader, batch_size=2, shuffle=True)
        return partial_dataloader

    @pytest.fixture
    def partial_optimizer(self):
        partial_optimizer = partial(Adam, lr=0.001)
        return partial_optimizer

    @pytest.fixture
    def forward_dynamics_wrappers_dict(self, forward_dynamics, encoder_wrapper, device):
        d = ModelWrappersDict(
            {
                ModelNames.FORWARD_DYNAMICS: ModelWrapper(forward_dynamics, device, True),
                ModelNames.IMAGE_ENCODER: encoder_wrapper,
            }
        )
        d.send_to_default_device()
        return d

    @pytest.fixture
    def logger(self, tmp_path):
        return StepIntervalLogger(f"{tmp_path}/tensorboard", 1)

    @pytest.fixture
    def trainer(
        self,
        partial_dataloader,
        partial_optimizer,
        device,
        forward_dynamics_wrappers_dict,
        trajectory_buffer_dict,
        logger,
    ):
        trainer = ForwardDynamicsTrainer(
            partial_dataloader,
            partial_optimizer,
            device,
            logger,
            observation_encoder_name=ModelNames.IMAGE_ENCODER,
            minimum_new_data_count=2,
        )
        trainer.attach_model_wrappers_dict(forward_dynamics_wrappers_dict)
        trainer.attach_data_users_dict(trajectory_buffer_dict.get_data_users())
        return trainer

    def test_run(self, trainer) -> None:
        trainer.run()

    def test_is_trainable(self, trainer) -> None:
        assert trainer.is_trainable() is True
        trainer.trajectory_data_user.clear()
        assert trainer.is_trainable() is False

    def test_is_new_data_available(self, trainer: ForwardDynamicsTrainer):
        trainer.trajectory_data_user.update()
        assert trainer._is_new_data_available() is True
        trainer.run()
        assert trainer._is_new_data_available() is False

    def test_save_and_load_state(self, trainer: ForwardDynamicsTrainer, tmp_path, mocker) -> None:
        trainer_path = tmp_path / "forward_dynamics"
        trainer.save_state(trainer_path)
        assert trainer_path.exists()
        assert (trainer_path / "optimizer.pt").exists()
        assert (trainer_path / "logger.pt").exists()
        logger_state = trainer.logger.state_dict()

        mocked_logger_load_state_dict = mocker.spy(trainer.logger, "load_state_dict")
        trainer.optimizer_state.clear()
        assert trainer.optimizer_state == {}
        trainer.load_state(trainer_path)
        assert trainer.optimizer_state != {}
        mocked_logger_load_state_dict.assert_called_once_with(logger_state)
