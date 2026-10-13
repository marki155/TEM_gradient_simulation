import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time

import pytest

from temsim.service.jobs import Clock
from temsim.service.server import Server


@pytest.fixture(scope="session")
def qapp():
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def pump(qapp):
    def run(predicate, timeout=60.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            qapp.processEvents()
            if predicate():
                return
            time.sleep(0.01)
        raise AssertionError("condition not reached")
    return run


@pytest.fixture
def server():
    s = Server("tcp://127.0.0.1:*", device="cpu", clock=Clock(frame_interval=0.0, save_interval=0))
    s.start()
    yield s
    s.stop()
