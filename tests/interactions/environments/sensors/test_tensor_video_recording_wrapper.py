from pathlib import Path

import torch
from pytest_mock import MockerFixture

from ami.interactions.environments.sensors.tensor_video_recording_wrapper import (
    BaseSensor,
    TensorVideoRecordingWrapper,
)


class TestTensorVideoRecordingWrapper:
    IMAGE = torch.zeros(3, 480, 640)

    def test_recording(self, mocker: MockerFixture, tmp_path: Path):
        output_dir = tmp_path / "video_recordings"
        wrapper = TensorVideoRecordingWrapper(
            sensor=mocker.Mock(BaseSensor), output_dir=output_dir, width=640, height=480, frame_rate=30
        )

        wrapper.setup()
        assert len(list(output_dir.glob("*.mp4"))) == 1
        assert wrapper.wrap_observation(self.IMAGE) is self.IMAGE
        wrapper.on_paused()
        wrapper.on_resumed()
        assert len(list(output_dir.glob("*.mp4"))) == 2
        wrapper.teardown()
