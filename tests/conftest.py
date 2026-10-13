import pathlib
import sys

import pytest
import torch


LEGACY = pathlib.Path(__file__).resolve().parents[1] / "archive" / "legacy_core"
if str(LEGACY) not in sys.path:
    sys.path.append(str(LEGACY))


@pytest.fixture
def legacy():
    """Import a legacy-core module by name (some set a CUDA default device on import)."""

    def load(name):
        module = pytest.importorskip(name)
        torch.set_default_device("cpu")
        return module

    return load
