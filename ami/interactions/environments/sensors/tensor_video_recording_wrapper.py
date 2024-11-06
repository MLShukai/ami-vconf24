from pathlib import Path
from typing import Any

from torch import Tensor
from typing_extensions import override

from ami.interactions.environments.sensors.base_sensor import BaseSensor

from ...io_wrappers.tensor_video_recorder import TensorVideoRecorder
from .base_sensor import BaseSensorWrapper


class TensorVideoRecordingWrapper(BaseSensorWrapper[Tensor, Tensor]):
    """Records the image tensor data using rguments for internal TensorVideoRecorder."""

    @override
    def __init__(self, sensor: BaseSensor[Tensor], *args: Any, **kwds: Any) -> None:
        """Initializes the TensorVideoRecordingWrapper.

        Args:
            sensor (BaseSensor[Tensor]): Sensor to be wrapped.
            *args (Any): Arguments for internal TensorVideoRecorder.
            **kwds (Any): Keyword arguments for internal TensorVideoRecorder.
        """
        super().__init__(sensor)

        self._recorder = TensorVideoRecorder(*args, **kwds)

    @override
    def setup(self) -> None:
        super().setup()
        self._recorder.setup()

    @override
    def wrap_observation(self, observation: Tensor) -> Tensor:
        self._recorder.wrap(observation)
        return observation

    @override
    def teardown(self) -> None:
        super().teardown()
        self._recorder.teardown()

    @override
    def on_paused(self) -> None:
        super().on_paused()
        self._recorder.on_paused()

    @override
    def on_resumed(self) -> None:
        super().on_resumed()
        self._recorder.on_resumed()
