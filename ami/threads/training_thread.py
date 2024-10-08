import time
from pathlib import Path

from typing_extensions import override

from ..data.utils import DataUsersDict
from ..models.utils import ModelWrappersDict
from ..trainers.utils import TrainersList
from .background_thread import BackgroundThread
from .shared_object_names import SharedObjectNames
from .thread_types import ThreadTypes


class TrainingThread(BackgroundThread):

    THREAD_TYPE = ThreadTypes.TRAINING

    def __init__(self, trainers: TrainersList, models: ModelWrappersDict, training_interval: float = 1e-10) -> None:
        """Constructs the training thread class with the trainers and models.

        Args:
            training_interval: The interval to run a trainer.
        """
        super().__init__()

        self.trainers = trainers
        self.models = models
        self.training_interval = training_interval

        self.share_object(SharedObjectNames.INFERENCE_MODELS, models.inference_wrappers_dict)

        removed_names = self.models.remove_inference_thread_only_models()
        if len(removed_names) != 0:
            self.logger.debug(f"The inference thread only models are removed from training thread: {removed_names!r}")

    def on_shared_objects_pool_attached(self) -> None:
        super().on_shared_objects_pool_attached()

        self.data_users: DataUsersDict = self.get_shared_object(ThreadTypes.INFERENCE, SharedObjectNames.DATA_USERS)

        self.trainers.attach_data_users_dict(self.data_users)
        self.trainers.attach_model_wrappers_dict(self.models)

    def worker(self) -> None:
        self.logger.info("Starts the training thread.")

        while self.thread_command_handler.manage_loop():
            if len(self.trainers) == 0:
                time.sleep(self.training_interval)
                continue

            trainer = self.trainers.get_next_trainer()
            if trainer.is_trainable():
                self.logger.info(f"Running a trainer: {trainer.name!r}")
                start = time.perf_counter()
                trainer.run()
                self.logger.info(f"Training time: {time.perf_counter() - start:.2f} [s].")

            time.sleep(self.training_interval)  # See Issue: https://github.com/MLShukai/ami/issues/175

        self.logger.info("End the training thread.")

    @override
    def save_state(self, path: Path) -> None:
        path.mkdir()
        self.models.save_state(path / "models")
        self.trainers.save_state(path / "trainers")
        self.data_users.save_state(path / "data")

    @override
    def load_state(self, path: Path) -> None:
        self.models.load_state(path / "models")
        self.trainers.load_state(path / "trainers")
        self.data_users.load_state(path / "data")

    @override
    def on_paused(self) -> None:
        self.trainers.on_paused()

    @override
    def on_resumed(self) -> None:
        self.trainers.on_resumed()
