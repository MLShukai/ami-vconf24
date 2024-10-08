import queue
import time
from collections.abc import Generator

import bottle
import pytest
import requests
from webtest import TestApp

from ami.threads.thread_control import ThreadController, ThreadControllerStatus
from ami.threads.web_api_handler import ControlCommands, WebApiHandler


class TestWebApiHandler:
    @pytest.fixture
    def controller(self) -> ThreadController:
        return ThreadController()

    @pytest.fixture
    def handler(self, controller) -> WebApiHandler:
        status = ThreadControllerStatus(controller)
        handler = WebApiHandler(status, host="localhost", port=8080)
        return handler

    @pytest.fixture
    def app(self, handler) -> Generator[TestApp, None, None]:
        app = bottle.default_app()

        # Test server for testing WSGI server
        yield TestApp(app)

        app.reset()

    def test_get_status(self, app: TestApp, controller: ThreadController) -> None:
        response = app.get("http://localhost:8080/api/status")
        assert response.status_code == 200
        assert response.json == {"status": "active"}

        controller.pause()
        response = app.get("http://localhost:8080/api/status")
        assert response.status_code == 200
        assert response.json == {"status": "paused"}

        controller.shutdown()
        response = app.get("http://localhost:8080/api/status")
        assert response.status_code == 200
        assert response.json == {"status": "stopped"}

    def test_has_and_receive_commands(self, app: TestApp, handler: WebApiHandler) -> bool:
        assert handler.has_commands() is False

        response = app.post("http://localhost:8080/api/pause")
        assert response.status_code == 200
        assert response.json == {"result": "ok"}
        assert handler.has_commands() is True

        assert handler.receive_command() is ControlCommands.PAUSE
        assert handler.has_commands() is False

        response = app.post("http://localhost:8080/api/resume")
        assert response.status_code == 200
        assert response.json == {"result": "ok"}
        assert handler.has_commands() is True

        response = app.post("http://localhost:8080/api/shutdown")
        assert response.status_code == 200
        assert response.json == {"result": "ok"}

        assert handler.receive_command() is ControlCommands.RESUME
        assert handler.receive_command() is ControlCommands.SHUTDOWN

        assert handler.has_commands() is False
        with pytest.raises(queue.Empty):
            handler.receive_command()

        response = app.post("http://localhost:8080/api/save-checkpoint")
        assert response.status_code == 200
        assert response.json == {"result": "ok"}
        assert handler.receive_command() is ControlCommands.SAVE_CHECKPOINT

    def test_error_404(self, app) -> None:
        response = app.get("http://localhost:8080/api/invalid", status=404)
        assert "error" in response.json

    def test_error_405(self, app) -> None:
        response = app.post("http://localhost:8080/api/status", status=405)
        assert "error" in response.json

    def test_increment_port_when_already_used(self, controller: ThreadController) -> None:
        PORT = 8092

        controller = ThreadController()
        status = ThreadControllerStatus(controller)
        handler1 = WebApiHandler(status, "localhost", PORT)
        handler1.run_in_background()
        time.sleep(0.001)

        response = requests.get(f"http://localhost:{PORT}/api/status")
        assert response.status_code == 200

        with pytest.raises(requests.exceptions.ConnectionError):
            requests.get(f"http://localhost:{PORT+1}/api/status")

        handler2 = WebApiHandler(status, "localhost", PORT)
        handler2.run_in_background()
        time.sleep(0.001)
        response = requests.get(f"http://localhost:{PORT+1}/api/status")
        assert response.status_code == 200
