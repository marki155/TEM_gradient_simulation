r"""Part V, Section 7 (continued): choosing a run or a data folder of the core's workspace.

Theory
======

A core started with ``--workspace`` lists what it holds (:meth:`temsim.service.client.Client.workspace`):
runs and data folders with their relative path.  The person picks one from a list, never types
a path on the machine of the core.  A data folder with an autosave of an earlier session is
continued from it unless "neu beginnen" is ticked.

Implementation
==============
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from PySide6 import QtCore, QtWidgets

KINDS = {"run": "Lauf", "dataset": "Daten"}


class OpenDialog(QtWidgets.QDialog):
    """List of workspace entries, newest first; double click opens."""

    def __init__(self, listing: Dict[str, Any], parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Laden aus dem Workspace")
        self.resize(560, 420)
        self.entries: List[Dict[str, Any]] = sorted(listing.get("entries", []), key=lambda e: -e.get("mtime", 0))
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(str(listing.get("root", ""))))
        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText("Filter ...")
        self.filter.textChanged.connect(self._apply_filter)
        layout.addWidget(self.filter)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Art", "Geaendert"])
        self.tree.setRootIsDecorated(False)
        self.tree.setSortingEnabled(False)
        for entry in self.entries:
            kind = KINDS.get(entry["kind"], entry["kind"]) + (" (Autosave)" if entry.get("autosave") else "")
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(entry.get("mtime", 0)))
            item = QtWidgets.QTreeWidgetItem([entry["path"], kind, when])
            item.setData(0, QtCore.Qt.UserRole, entry)
            self.tree.addTopLevelItem(item)
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.tree.itemDoubleClicked.connect(lambda *_: self.accept())
        self.tree.itemSelectionChanged.connect(self._update_resume)
        layout.addWidget(self.tree)
        self.fresh_check = QtWidgets.QCheckBox("neu beginnen (Autosave ignorieren)")
        self.fresh_check.setEnabled(False)
        layout.addWidget(self.fresh_check)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Open | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.open_button = buttons.button(QtWidgets.QDialogButtonBox.Open)
        self.open_button.setEnabled(False)

    def _apply_filter(self, text: str) -> None:
        text = text.strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            item.setHidden(bool(text) and text not in item.text(0).lower())

    def _update_resume(self) -> None:
        entry = self.selected()
        self.open_button.setEnabled(entry is not None)
        self.fresh_check.setEnabled(bool(entry and entry.get("autosave")))

    def selected(self) -> Optional[Dict[str, Any]]:
        items = self.tree.selectedItems()
        return items[0].data(0, QtCore.Qt.UserRole) if items else None

    @property
    def resume(self) -> bool:
        return not self.fresh_check.isChecked()
