import pytest
import torch
from omegaconf import DictConfig, ListConfig

from ami.tensorboard_loggers import StepIntervalLogger, TimeIntervalLogger


class TestTimeIntervalLogger:
    @pytest.fixture
    def logger(self, tmp_path):
        return TimeIntervalLogger(tmp_path, 0.0)

    def test_log(self, logger: TimeIntervalLogger):
        for _ in range(64):
            logger.log("test1", torch.randn(1).item())
            logger.log("test2", torch.randn(1))
            logger.update()

    def test_log_hyperparameter(self, logger: TimeIntervalLogger):
        d = DictConfig({"a": 1, "b": [2, 3, {"c": 4, "d": [5, 6]}]})
        logger.log_hyperparameters(d)


class TestStepIntervalLogger:
    @pytest.fixture
    def logger(self, tmp_path):
        return StepIntervalLogger(tmp_path, 1)

    def test_log(self, logger: StepIntervalLogger):
        for _ in range(64):
            logger.log("test1", torch.randn(1).item())
            logger.log("test2", torch.randn(1))
            logger.update()

    def test_log_hyperparameter(self, logger: StepIntervalLogger):
        d = DictConfig({"a": 1, "b": [2, 3, {"c": 4, "d": [5, 6]}]})
        logger.log_hyperparameters(d)