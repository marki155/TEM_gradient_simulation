r"""Part V, Section 11: the main window.

Theory
======

The window is a composition and nothing else: the plot area on the left, a scrollable
column of panels on the right, the training editor below it.  The proportions are those of the
old interface (splitter 4:1, window 1600 x 950).  The one piece of logic is the *mode*
(Section 9): in the insertion mode the column shows the insertion panel and the training editor,
the strain grid and the training-only options are hidden, and vice versa.  All other behaviour lives in
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
from .panels import INSERT, ControlPanel, InsertPanel, StrainPanel
from .train_editor import TrainEditor
from .values_editor import ValuesEditor
from .widgets import CollapsibleSection


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
        self.insert = InsertPanel()
        self.minima, self.seeding = self.insert.minima, self.insert.seeding
        self.strain = StrainPanel()
        self.values = ValuesEditor(connection)
        self.train = TrainEditor(connection)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.values)
        toolbar = NavigationToolbar2QT(self.canvas, left)
        self.canvas.toolbar_mode = lambda: str(toolbar.mode)
        left_layout.addWidget(toolbar)
        left_layout.addWidget(self.canvas, 1)

        column = QtWidgets.QWidget()
        column_layout = QtWidgets.QVBoxLayout(column)
        self.strain_section = CollapsibleSection("Strain-Kontrollgitter", expanded=False)
        self.strain_section.content_layout.addWidget(self.strain)
        self.control.add_mode_panel(self.insert)
        for widget in (self.control, self.strain_section):
            column_layout.addWidget(widget)
        column_layout.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(400)
        scroll.setWidget(column)

        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right.addWidget(scroll)
        self.editor_scroll = QtWidgets.QScrollArea()
        self.editor_scroll.setWidgetResizable(True)
        self.editor_scroll.setMinimumHeight(260)
        self.editor_scroll.setWidget(self.train)
        right.addWidget(self.editor_scroll)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self.control.mode_changed.connect(self.apply_mode)
        self.insert.method_changed.connect(self.apply_mode)
        self.apply_mode()
        self.controller = Controller(connection, self.canvas, self.control, self.insert,
                                     self.strain, self.values, self.train, self)
        self.controller.remote = core is None and not is_local(connection.endpoint)
        self.controller.notified.connect(lambda text: self.statusBar().showMessage(text, 8000))
        self.controller.refresh_model()

    def apply_mode(self, *_args) -> None:
        """Show what the chosen mode (and insertion method) needs, nothing else."""

        inserting = self.control.mode == INSERT
        self.insert.set_active(inserting)
        self.canvas.click_mode = inserting and self.insert.method == "manual"
        self.control.show_actions(inserting, self.insert.method == "seeding")
        self.strain_section.setVisible(not inserting)
        self.editor_scroll.setVisible(not inserting)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.controller.finish(self.control.save_on_close.isChecked(), stop_job=self.core is not None)
        self.connection.close()
        if self.core is not None:
            self.core.stop()
        super().closeEvent(event)
