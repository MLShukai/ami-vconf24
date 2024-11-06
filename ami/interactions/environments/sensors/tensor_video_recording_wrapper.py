from pathlib import Path
from typing import Any

from torch import Tensor
from typing_extensions import override

from ami.interactions.environments.sensors.base_sensor import BaseSensor

from ...io_wrappers.tensor_video_recorder import TensorVideoRecorder
from .base_sensor import BaseSensorWrapper


class TensorVideoRecordingWrapper(BaseSensorWrapper[Tensor, Tensor]):
    """Records the image tensor data to a video file."""

    @override
    def __init__(
        self,
        sensor: BaseSensor[Tensor],
        output_dir: str | Path,
        width: int,
        height: int,
        frame_rate: float,
        file_name_format: str = "%Y-%m-%d_%H-%M-%S.%f.mp4",
        fourcc: str = "mp4v",
        do_rgb_to_bgr: bool = True,
        do_scale_255: bool = True,
    ) -> None:
        super().__init__(sensor)

        self._recorder = TensorVideoRecorder(
            str(output_dir),
            width,
            height,
            frame_rate,
            file_name_format,
            fourcc,
            do_rgb_to_bgr,
            do_scale_255,
        )

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
