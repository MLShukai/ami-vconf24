import socket
import threading
import time
from typing import Any
from uuid import UUID

import torch
from mlagents_envs.environment import UnityEnvironment as RawUnityEnv
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from mlagents_envs.side_channel.side_channel import IncomingMessage, SideChannel
from numpy.typing import NDArray
from torch import Tensor
from typing_extensions import override

from .base_environment import BaseEnvironment


def find_free_port(start_port: int = 5005, max_port: int = 5999) -> int:
    for port in range(start_port, max_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise OSError("No free ports")


class TransformLogChannel(SideChannel):
    def __init__(self, id: UUID, log_file_path: str):
        super().__init__(id)
        self.log_file_path = log_file_path
        with open(self.log_file_path, mode="w") as f:
            f.write("frame_count, time, position_x, position_y, position_z, euler_x, euler_y, euler_z\n")

    def on_message_received(self, msg: IncomingMessage) -> None:
        with open(self.log_file_path, mode="a") as f:
            value_list = msg.read_float32_list()
            frame_count = value_list[0]
            time = value_list[1]
            position_x = value_list[2]
            position_y = value_list[3]
            position_z = value_list[4]
            euler_x = value_list[5]
            euler_y = value_list[6]
            euler_z = value_list[7]
            format_string = (
                f"{frame_count}, {time}, {position_x}, {position_y}, {position_z}, {euler_x}, {euler_y}, {euler_z}\n"
            )
            f.write(format_string)


class UnityEnvironment(BaseEnvironment[Tensor, Tensor]):
    """A class for connecting to and interacting with a Unity ML Agents
    environment.

    This class provides an interface to interact with Unity environments
    using the ML-Agents toolkit. It handles the setup, teardown, and
    communication with the Unity environment, allowing for observations
    to be received and actions to be sent.
    """

    def __init__(
        self,
        file_path: str,
        log_file_path: str | None = None,
        worker_id: int = 0,
        base_port: int | None = None,
        seed: int = 0,
        time_scale: float = 1.0,
        target_frame_rate: int = 60,
    ) -> None:
        """Initialize the UnityEnvironment.

        Args:
            file_path (str): Path to the Unity environment executable.
            worker_id (int, optional): Identifier for the worker. Defaults to 0.
            base_port (int | None, optional): Port to use for communication with the environment.
                If None, the default port will be used. Defaults to None.
            seed (int, optional): Random seed for the environment. Defaults to 0.
            time_scale (float, optional): Time scale for the Unity environment. Defaults to 1.0.
            target_frame_rate (int, optional): Target frame rate for the Unity environment. Defaults to 60.

        The UnityEnvironment is set up with the specified parameters, establishing a connection
        to the Unity executable and configuring the environment settings such as time scale and
        frame rate.
        """
        super().__init__()

        self.engine_configuration_channel = EngineConfigurationChannel()
        side_channels = [self.engine_configuration_channel]
        if log_file_path is not None:
            transform_log_channel = TransformLogChannel(UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"), log_file_path)
            side_channels.append(transform_log_channel)

        if base_port is None:
            base_port = find_free_port()

        self._env = UnityToGymWrapper(
            RawUnityEnv(file_path, worker_id, base_port, seed=seed, side_channels=side_channels),
            allow_multiple_obs=False,
        )
        self.engine_configuration_channel.set_configuration_parameters(
            time_scale=time_scale, target_frame_rate=target_frame_rate
        )

        self._action: Tensor | None = None
        self._observation: NDArray[Any]
        self._lock = threading.RLock()
        self._teardown_flag = threading.Event()
        self._interaction_thread = threading.Thread(target=self._interaction)

    @property
    def gym_unity_environment(self) -> UnityToGymWrapper:
        return self._env

    @property
    def raw_unity_environment(self) -> RawUnityEnv:
        return self._env._env

    @property
    def action(self) -> Tensor | None:
        with self._lock:
            if self._action is None:
                return None
            return self._action.clone()

    @action.setter
    def action(self, v: Tensor) -> None:
        with self._lock:
            self._action = v

    @property
    def observation(self) -> NDArray[Any]:
        with self._lock:
            return self._observation.copy()

    @observation.setter
    def observation(self, v: NDArray[Any]) -> None:
        with self._lock:
            self._observation = v

    def _interaction(self) -> None:
        """Affects the action and gets observation in background thread."""
        while not self._teardown_flag.is_set():
            if (action := self.action) is not None:
                self.observation, _, _, _ = self._env.step(action.detach().cpu().numpy())
            else:
                time.sleep(0.001)

    @override
    def setup(self) -> None:
        super().setup()
        self.observation = self._env.reset()
        self._interaction_thread.start()

    @override
    def affect(self, action: Tensor) -> None:
        self.action = action

    @override
    def observe(self) -> Tensor:
        return torch.from_numpy(self.observation)

    @override
    def teardown(self) -> None:
        super().teardown()
        self._teardown_flag.set()
        self._interaction_thread.join()
        self._env.close()
