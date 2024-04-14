import threading

from .base_thread import BaseThread
from .shared_object_names import SharedObjectNames
from .thread_control import ThreadCommandHandler
from .thread_types import ThreadTypes


class BackgroundThread(BaseThread):
    """Base class for all background thread objects.

    The `THREAD_TYPE` attribute must not be set to `ThreadType.MAIN`.
    """

    thread_command_handler: ThreadCommandHandler

    def __init__(self) -> None:
        super().__init__()
        self._thread = threading.Thread(target=self.run)

        if self.THREAD_TYPE is ThreadTypes.MAIN:
            raise ValueError("Background `THREAD_TYPE` must not be MAIN!")

    def on_shared_objects_pool_attached(self) -> None:
        super().on_shared_objects_pool_attached()

        self.thread_command_handler: ThreadCommandHandler = self.get_shared_object(
            ThreadTypes.MAIN,
            SharedObjectNames.THREAD_COMMAND_HANDLERS,
        )[self.THREAD_TYPE]
        self.thread_command_handler.register_on_paused_callback(self.on_paused)
        self.thread_command_handler.register_on_resumed_callback(self.on_resumed)

    def start(self) -> None:
        self.logger.info("Starts background thread.")
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def join(self) -> None:
        self._thread.join()
        self.logger.info("Joined background thread.")

    def on_paused(self) -> None:
        """Callback function to be called when the system is paused.

        This callback is called before `on_resumed` callback.
        """
        pass

    def on_resumed(self) -> None:
        """Callback function to be called when the system is resumed.

        This callback is called after `on_paused` callback.
        """
        pass
