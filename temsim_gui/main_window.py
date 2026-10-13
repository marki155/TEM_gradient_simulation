r"""Part V, Section 11: the main window.

Theory
======

The window is a composition and nothing else: the plot area on the left, a scrollable
column of panels on the right, the editors below.  The proportions are those of the
old interface (splitter 4:1, window 1600 x 950).  All behaviour lives in
:class:`~temsim_gui.controller.Controller`.

Implementation
==============
"""

from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

from .canvas import PlotCanvas
from .connection import Connection, SpawnedCore
from .controller import Controller
from .panels import ControlPanel, MinimaPanel, SeedingPanel, StrainPanel
from .train_editor import TrainEditor
from .values_editor import ValuesEditor


def is_local(endpoint: str) -> bool:
    host = endpoint.split('://', 1)[-1].rsplit(':', 1)[0]
    return host in ("127.0.0.1", "localhost", "[::1]", "::1") or endpoint.startswith(("ipc://", "inproc://"))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, connection: Connection, core: Optional[SpawnedCore] = None):
        super().__init__()
        self.setWindowTitle("TEM gradient simulation")
        self.resize(1600, 950)
        self.connection, self.core = connection, core

        self.canvas = PlotCanvas()
        self.control = ControlPanel()
        self.minima = MinimaPanel()
        self.seeding = SeedingPanel()
        self.strain = StrainPanel()
        self.values = ValuesEditor(connection)
        self.train = TrainEditor(connection)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.values)
        left_layout.addWidget(NavigationToolbar2QT(self.canvas, left))
        left_layout.addWidget(self.canvas, 1)

        column = QtWidgets.QWidget()
        column_layout = QtWidgets.QVBoxLayout(column)
        for widget in (self.control, self.minima, self.seeding, self.strain):
            column_layout.addWidget(widget)
        column_layout.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(400)
        scroll.setWidget(column)

        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right.addWidget(scroll)
        editor_scroll = QtWidgets.QScrollArea()
        editor_scroll.setWidgetResizable(True)
        editor_scroll.setMinimumHeight(260)
        editor_scroll.setWidget(self.train)
        right.addWidget(editor_scroll)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self.controller = Controller(connection, self.canvas, self.control, self.minima, self.seeding,
                                     self.strain, self.values, self.train, self)
        self.controller.remote = core is None and not is_local(connection.endpoint)
        self.controller.notified.connect(lambda text: self.statusBar().showMessage(text, 8000))
        self.controller.refresh_model()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.controller.finish(self.control.save_on_close.isChecked())
        self.connection.close()
        if self.core is not None:
            self.core.stop()
        super().closeEvent(event)
