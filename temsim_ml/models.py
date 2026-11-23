r"""Part V, Section 13: where trained detectors live.

Two places, both outside any single run:

* the **global** directory ``~/data/models`` (override with ``TEMSIM_MODELS``) --
  models here are always available, in every run and workspace;
* a **local** staging directory ``~/data/models_local`` -- a model freshly
  trained on one run lands here and is only copied to the global directory once
  it has proven itself ("Als global setzen").

A model is a directory with ``detector.pt`` and ``meta.json`` (Part II,
Section 8.3).
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List

from temsim.learning import detection as det

GLOBAL_MODELS = os.path.expanduser(os.environ.get("TEMSIM_MODELS") or "~/data/models")
LOCAL_MODELS = os.path.expanduser("~/data/models_local")


def list_models(directory: str) -> List[Dict[str, Any]]:
    """The detectors in ``directory``, newest first: ``{name, path, meta}``."""

    if not os.path.isdir(directory):
        return []
    found = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if det.is_detector(path):
            found.append({"name": name, "path": path, "meta": det.read_meta(path)})
    found.sort(key=lambda m: m["meta"].get("created", ""), reverse=True)
    return found


def promote(path: str, name: str = "", target_dir: str = GLOBAL_MODELS) -> str:
    """Copy the model directory ``path`` into ``target_dir`` (default: global); returns the new path."""

    if not det.is_detector(path):
        raise ValueError(f"no detector in {path}")
    os.makedirs(target_dir, exist_ok=True)
    destination = os.path.join(target_dir, name or os.path.basename(path.rstrip("/")))
    if os.path.abspath(destination) == os.path.abspath(path):
        return destination
    if os.path.exists(destination):
        shutil.rmtree(destination)
    shutil.copytree(path, destination)
    return destination
