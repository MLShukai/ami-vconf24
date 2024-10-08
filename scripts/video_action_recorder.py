"""Video Action Recorder.

This script records video and keyboard actions, sending them as OSC messages.

Installation:
    pip install opencv-python keyboard numpy python-osc

Usage:
    python video_action_recorder.py [--fps FPS] [--width WIDTH] [--height HEIGHT] [--osc-ip OSC_IP] [--osc-port OSC_PORT] [--device DEVICE]

    Optional arguments:
    --fps FPS           Frames per second for video recording (default: 10)
    --width WIDTH       Width of the video frame (default: 1280)
    --height HEIGHT     Height of the video frame (default: 720)
    --osc-ip OSC_IP     IP address for OSC server (default: 127.0.0.1)
    --osc-port OSC_PORT Port number for OSC server (default: 9000)
    --device DEVICE     Video capture device index (default: 0)

    Press 'q' to stop recording.

NOTE: Linux User
    You need to run this script as root user.
    Because keyboard module requires root user to access keyboard device.
    So run the following command in the project root directory.
    ```sh
    PYTHON_PATH=$(which python3) && sudo $PYTHON_PATH ./scripts/video_action_recorder.py
    ```

Output:
    - Video file: output_YYYY-MM-DD_HH-MM-SS.mp4
    - Action log: action_log_YYYY-MM-DD_HH-MM-SS.csv
"""

import argparse
import csv
import time
from datetime import datetime
from typing import Any, List, Optional, Tuple

import cv2
import keyboard
from numpy.typing import NDArray
from pythonosc import udp_client


class VideoRecorder:
    def __init__(self, output_file: str, fps: int, resolution: Tuple[int, int], device: int):
        self.output_file = output_file
        self.fps = fps
        self.resolution = resolution
        self.device = device
        self.capture: Optional[cv2.VideoCapture] = None
        self.out: Optional[cv2.VideoWriter] = None

    def start_recording(self) -> None:
        self.capture = cv2.VideoCapture(self.device)
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        self.out = cv2.VideoWriter(self.output_file, fourcc, self.fps, self.resolution)

    def record_frame(self) -> Optional[NDArray[Any]]:
        if self.capture is None or self.out is None:
            return None
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.resize(frame, self.resolution)
            self.out.write(frame)
            return frame
        return None

    def stop_recording(self) -> None:
        if self.capture:
            self.capture.release()
        if self.out:
            self.out.release()


class KeyboardActionHandler:
    def __init__(self, osc_ip: str = "127.0.0.1", osc_port: int = 12345):
        self.initial_actions = {"MoveVertical": 0, "MoveHorizontal": 0, "LookHorizontal": 0, "Jump": 0, "Run": 0}
        self.actions = self.initial_actions.copy()

        self.key_map = {
            "w": ("MoveVertical", 1),
            "s": ("MoveVertical", 2),
            "d": ("MoveHorizontal", 1),
            "a": ("MoveHorizontal", 2),
            ".": ("LookHorizontal", 1),
            ",": ("LookHorizontal", 2),
            "space": ("Jump", 1),
            "shift": ("Run", 1),
        }

        self.osc_client = udp_client.SimpleUDPClient(osc_ip, osc_port)
        self.osc_addresses = {
            "MoveForward": "/input/MoveForward",
            "MoveBackward": "/input/MoveBackward",
            "MoveLeft": "/input/MoveLeft",
            "MoveRight": "/input/MoveRight",
            "LookLeft": "/input/LookLeft",
            "LookRight": "/input/LookRight",
            "Jump": "/input/Jump",
            "Run": "/input/Run",
        }

    def update(self) -> bool:
        self.actions = self.initial_actions.copy()

        for key, (action, value) in self.key_map.items():
            if keyboard.is_pressed(key):
                self.actions[action] = value

        self.send_osc_messages()
        return not keyboard.is_pressed("q")  # Return False if 'q' is pressed to quit

    def send_osc_messages(self) -> None:
        self.osc_client.send_message(self.osc_addresses["MoveForward"], int(self.actions["MoveVertical"] == 1))
        self.osc_client.send_message(self.osc_addresses["MoveBackward"], int(self.actions["MoveVertical"] == 2))
        self.osc_client.send_message(self.osc_addresses["MoveRight"], int(self.actions["MoveHorizontal"] == 1))
        self.osc_client.send_message(self.osc_addresses["MoveLeft"], int(self.actions["MoveHorizontal"] == 2))
        self.osc_client.send_message(self.osc_addresses["LookRight"], int(self.actions["LookHorizontal"] == 1))
        self.osc_client.send_message(self.osc_addresses["LookLeft"], int(self.actions["LookHorizontal"] == 2))
        self.osc_client.send_message(self.osc_addresses["Jump"], int(self.actions["Jump"] == 1))
        self.osc_client.send_message(self.osc_addresses["Run"], int(self.actions["Run"] == 1))

    def get_action_array(self) -> List[int]:
        return [
            self.actions["MoveVertical"],
            self.actions["MoveHorizontal"],
            self.actions["LookHorizontal"],
            self.actions["Jump"],
            self.actions["Run"],
        ]


class ActionRecorder:
    def __init__(self, filename: str):
        self.filename = filename
        self.headers = ["Timestamp", "MoveVertical", "MoveHorizontal", "LookHorizontal", "Jump", "Run"]
        self._initialize_csv()

    def _initialize_csv(self) -> None:
        with open(self.filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.headers)

    def record_action(self, action_array: List[int]) -> None:
        timestamp = f"{time.time():.5f}"
        row = [timestamp] + action_array
        with open(self.filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)


def get_timestamp_string() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video and Action Recording Script")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for video recording")
    parser.add_argument("--width", type=int, default=1280, help="Width of the video frame")
    parser.add_argument("--height", type=int, default=720, help="Height of the video frame")
    parser.add_argument("--osc-ip", type=str, default="127.0.0.1", help="IP address for OSC server")
    parser.add_argument("--osc-port", type=int, default=9000, help="Port number for OSC server")
    parser.add_argument("--device", type=int, default=0, help="Video capture device index")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    # Configuration
    timestamp = get_timestamp_string()
    video_output = f"output_{timestamp}.mp4"
    action_log = f"action_log_{timestamp}.csv"
    resolution = (args.width, args.height)

    # Initialize components
    video_recorder = VideoRecorder(video_output, args.fps, resolution, args.device)
    action_handler = KeyboardActionHandler(osc_ip=args.osc_ip, osc_port=args.osc_port)
    action_recorder = ActionRecorder(action_log)

    # Start video recording
    video_recorder.start_recording()

    # Main loop
    frame_time = 1 / args.fps
    next_frame_time = time.time()

    print("Recording started. Press 'q' to stop.")

    while True:
        current_time = time.time()

        if current_time >= next_frame_time:
            # Record video frame
            frame = video_recorder.record_frame()
            if frame is not None:
                cv2.imshow("Recording", frame)

            # Update actions and send OSC
            if not action_handler.update():
                break

            # Record actions
            action_array = action_handler.get_action_array()
            action_recorder.record_action(action_array)
            print(f"\r{action_array}", end="")

            # Calculate next frame time
            next_frame_time += frame_time

        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    video_recorder.stop_recording()
    cv2.destroyAllWindows()

    print(f"Recording finished. Video saved as {video_output}, actions logged in {action_log}")


if __name__ == "__main__":
    main()
