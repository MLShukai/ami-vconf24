# Ref: https://github.com/facebookresearch/ijepa

from functools import partial
from pathlib import Path

import torch
import torchvision
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing_extensions import override

from ami.data.buffers.buffer_names import BufferNames
from ami.data.buffers.random_data_buffer import RandomDataBuffer
from ami.data.interfaces import ThreadSafeDataUser
from ami.models.i_jepa import IJEPAEncoder
from ami.models.i_jepa_latent_visualization_decoder import (
    IJEPALatentVisualizationDecoder,
)
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ModelWrapper
from ami.tensorboard_loggers import StepIntervalLogger

from .base_trainer import BaseTrainer


class IJEPALatentVisualizationDecoderTrainer(BaseTrainer):
    def __init__(
        self,
        partial_dataloader: partial[DataLoader[torch.Tensor]],
        partial_optimizer: partial[Optimizer],
        device: torch.device,
        logger: StepIntervalLogger,
        max_epochs: int = 1,
        minimum_dataset_size: int = 1,
        minimum_new_data_count: int = 0,
        moving_average_rate: float = 0.999,
    ) -> None:
        """Initializes an IJEPALatentVisualizationDecoderTrainer object.

        Args:
            partial_dataloader: A partially instantiated dataloader lacking a provided dataset.
            partial_optimizer: A partially instantiated optimizer lacking provided parameters.
            device: The accelerator device (e.g., CPU, GPU) utilized for training the model.
            minimum_new_data_count: Minimum number of new data count required to run the training.
        """
        super().__init__()
        self.partial_optimizer = partial_optimizer
        self.partial_dataloader = partial_dataloader
        self.device = device
        self.logger = logger
        self.max_epochs = max_epochs
        self.minimum_dataset_size = minimum_dataset_size
        self.minimum_new_data_count = minimum_new_data_count
        self.moving_average_rate = moving_average_rate

    def on_data_users_dict_attached(self) -> None:
        self.image_data_user: ThreadSafeDataUser[RandomDataBuffer] = self.get_data_user(BufferNames.IMAGE)

    def on_model_wrappers_dict_attached(self) -> None:
        self.target_encoder: ModelWrapper[IJEPAEncoder] = self.get_training_model(ModelNames.I_JEPA_TARGET_ENCODER)
        self.latent_visualization_decoder: ModelWrapper[IJEPALatentVisualizationDecoder] = self.get_training_model(
            ModelNames.I_JEPA_LATENT_VISUALIZATION_DECODER
        )
        self.latent_visualization_decoder_moving_average: ModelWrapper[
            IJEPALatentVisualizationDecoder
        ] = self.get_training_model(ModelNames.I_JEPA_LATENT_VISUALIZATION_DECODER_MOVING_AVERAGE)
        assert (
            self.latent_visualization_decoder.model is not self.latent_visualization_decoder_moving_average.model
        ), "latent_visualization_decoder and latent_visualization_decoder_moving_average must be allocated in memory as separate entities."

        # Since the model is swapped between the inference and training threads each time it is trained,
        # the model and optimizer are built within the `train()` method.
        # The following is the initial state generation of the optimizer.
        self.optimizer_state = self.partial_optimizer(self.latent_visualization_decoder.parameters()).state_dict()

        # copy weights from latent_visualization_decoder to latent_visualization_decoder_moving_average
        with torch.no_grad():
            for decoder_moving_average_param, decoder_param in zip(
                self.latent_visualization_decoder_moving_average.parameters(),
                self.latent_visualization_decoder.parameters(),
            ):
                decoder_moving_average_param.copy_(decoder_param)

    def is_trainable(self) -> bool:
        self.image_data_user.update()
        return len(self.image_data_user.buffer) >= self.minimum_dataset_size and self._is_new_data_available()

    def _is_new_data_available(self) -> bool:
        return self.image_data_user.buffer.new_data_count >= self.minimum_new_data_count

    def train(self) -> None:
        # move to device
        self.target_encoder = self.target_encoder.to(self.device)
        self.latent_visualization_decoder = self.latent_visualization_decoder.to(self.device)
        self.latent_visualization_decoder_moving_average = self.latent_visualization_decoder_moving_average.to(
            self.device
        )
        # define optimizer
        optimizer = self.partial_optimizer(self.latent_visualization_decoder.parameters())
        optimizer.load_state_dict(self.optimizer_state)
        # prepare about dataset
        dataset = self.image_data_user.get_dataset()
        dataloader = self.partial_dataloader(dataset=dataset)

        for _ in range(self.max_epochs):
            for batch in dataloader:
                (image_batch,) = batch
                image_batch = image_batch.to(self.device)
                # get latents from input images
                with torch.no_grad():
                    latents = self.target_encoder(image_batch)
                    latents = torch.nn.functional.layer_norm(
                        latents,
                        (latents.size(-1),),
                    )  # normalize over feature-dim
                    # latents: [batch_size, n_patches_height * n_patches_width, latents_dim]
                # decode from latents
                image_out = self.latent_visualization_decoder(input_latents=latents)
                # image_out: [batch_size, 3, n_patches_height * (2**(len(self.latent_visualization_decoder.decoder_blocks_in_and_out_channels)-1)), n_patches_width * (2**(len(self.latent_visualization_decoder.decoder_blocks_in_and_out_channels)-1))]
                # resize image_batch to calc loss with image_out
                with torch.no_grad():
                    _, _, height, width = image_out.size()
                    image_batch_resized = torchvision.transforms.functional.resize(image_out, size=(height, width))
                # calc loss
                loss = torch.nn.functional.smooth_l1_loss(
                    image_out,
                    image_batch_resized,
                    reduction="mean",
                )
                self.logger.log("i-jepa/losses/reconstruction", loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # self.latent_visualization_decoder_moving_average updates weights by moving average from self.latent_visualization_decoder
                with torch.no_grad():
                    m = self.moving_average_rate
                    for decoder_moving_average_param, decoder_param in zip(
                        self.latent_visualization_decoder_moving_average.parameters(),
                        self.latent_visualization_decoder.parameters(),
                    ):
                        decoder_moving_average_param.data.mul_(m).add_((1.0 - m) * decoder_param.detach().data)

                self.logger.update()

        self.optimizer_state = optimizer.state_dict()
        self.logger_state = self.logger.state_dict()

    @override
    def save_state(self, path: Path) -> None:
        path.mkdir()
        torch.save(self.optimizer_state, path / "optimizer.pt")
        torch.save(self.logger.state_dict(), path / "logger.pt")

    @override
    def load_state(self, path: Path) -> None:
        self.optimizer_state = torch.load(path / "optimizer.pt")
        self.logger.load_state_dict(torch.load(path / "logger.pt"))
