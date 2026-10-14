r"""Part V, Section 12: starting the interface.

The interface connects to a running core (``--endpoint``) or starts one of its own.
Data is opened by a command after the connection, never by the core's command line
(Part IV, Section 2).

Implementation
==============
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from PySide6 import QtWidgets

from temsim.service.protocol import endpoint_argument

from .connection import Connection, SpawnedCore, spawn_core
from .main_window import MainWindow


def parse(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="temsim_gui", description="Interface of the TEM gradient simulation")
    parser.add_argument("--endpoint", type=endpoint_argument, help="connect to a running core instead of starting one")
    parser.add_argument("--remote", action="store_true",
                        help="the core runs on another machine (also through a tunnel): paths are typed in, not browsed")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="device of a started core")
    parser.add_argument("--run", help="run directory to open")
    parser.add_argument("--target", help="target image of the run (optional)")
    parser.add_argument("--dataset", help="dataset file to open (instead of --run)")
    parser.add_argument("--fresh", action="store_true", help="with --dataset: ignore an existing autosave")
    return parser.parse_args(argv)


def launch(argv: Optional[List[str]] = None) -> int:
    args = parse(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    core: Optional[SpawnedCore] = None
    endpoint = args.endpoint
    if endpoint is None:
        core = spawn_core(args.device)
        endpoint = core.endpoint
    connection = Connection(endpoint)
    window = MainWindow(connection, core)
    window.controller.remote = window.controller.remote or args.remote
    window.show()
    if args.run:
        window.controller.open_run(args.run, args.target)
    elif args.dataset:
        window.controller.open_dataset(args.dataset, resume=not args.fresh)
    return app.exec()
