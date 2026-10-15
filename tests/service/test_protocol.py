"""Part IV, Section 1: framing and data classes on the wire."""

import json
import subprocess
import sys

import numpy as np
import pytest

from temsim.data.schema import ModelState, TrainSettings
from temsim.service import protocol

from ..learning.helpers import make_experiment


def test_pack_unpack_roundtrip_keeps_dtype_and_shape():
    arrays = {"a": np.arange(12, dtype=np.float32).reshape(3, 4), "b": np.array([1, 0, 1], dtype=np.uint8),
              "c": np.zeros((0, 2)), "d": np.array([[True, False]]), "e": np.array(3.5)}
    frames = protocol.pack({"kind": "event", "name": "x"}, arrays)
    assert len(frames) == 1 + len(arrays)
    header, out = protocol.unpack(frames)
    assert header == {"kind": "event", "name": "x"} and set(out) == set(arrays)
    for key, value in arrays.items():
        assert out[key].dtype == value.dtype and out[key].shape == value.shape
        np.testing.assert_array_equal(out[key], value)


def test_non_contiguous_and_unusual_dtypes_are_converted():
    a = np.arange(20, dtype=np.float64).reshape(4, 5).T
    header, out = protocol.unpack(protocol.pack({}, {"a": a, "h": np.arange(3, dtype=np.float16), "i": np.arange(3, dtype=np.int16), "u": np.arange(3, dtype=np.uint16)}))
    np.testing.assert_array_equal(out["a"], a)
    assert out["h"].dtype == np.float16 and out["i"].dtype == np.int64 and out["u"].dtype == np.int64


def test_unpack_rejects_inconsistent_messages():
    frames = protocol.pack({"k": 1}, {"a": np.zeros(4, np.float32)})
    with pytest.raises(ValueError, match="buffers"):
        protocol.unpack(frames[:1])
    head = json.loads(frames[0])
    head["arrays"][0]["shape"] = [5]
    with pytest.raises(ValueError, match="values for shape"):
        protocol.unpack([json.dumps(head).encode(), frames[1]])
    head["arrays"][0].update(shape=[4], dtype="object")
    with pytest.raises(ValueError, match="not allowed"):
        protocol.unpack([json.dumps(head).encode(), frames[1]])


def test_message_constructors():
    assert protocol.command(3, "frame", {"x": 1}) == {"kind": "command", "id": 3, "name": "frame", "args": {"x": 1}}
    assert protocol.reply(3)["ok"] is True
    f = protocol.failure(3, "boom", "trace")
    assert f["ok"] is False and f["error"] == "boom"
    assert protocol.event("log")["kind"] == "event"


def test_experiment_state_train_travel():
    exp = make_experiment()
    wire = json.loads(json.dumps(protocol.experiment_to_wire(exp)))
    back = protocol.experiment_from_wire(wire)
    assert [l.name for l in back.layers] == ["top", "bottom"]
    np.testing.assert_allclose(back.structure.positions[0], exp.structure.positions[0])

    state = protocol.state_from_wire(json.loads(json.dumps(protocol.state_to_wire(ModelState()))))
    assert isinstance(state, ModelState)

    settings = TrainSettings().ensure_for(exp)
    settings.masks = {"g1": np.arange(12).reshape(3, 4) % 2 == 0}
    settings.targets = {"g1": np.arange(12, dtype=np.float32).reshape(3, 4)}
    wire, arrays = protocol.train_to_wire(settings)
    assert set(arrays) == {"mask.g1", "target.g1"} and arrays["mask.g1"].dtype == np.uint8
    header, arrays = protocol.unpack(protocol.pack({"train": json.loads(json.dumps(wire))}, arrays))
    back = protocol.train_from_wire(header["train"], arrays)
    np.testing.assert_array_equal(back.masks["g1"], settings.masks["g1"])
    np.testing.assert_array_equal(back.targets["g1"], settings.targets["g1"])
    assert back.masks["g1"].dtype == bool


def test_every_command_has_a_description():
    assert all(isinstance(v, str) and v for v in protocol.COMMANDS.values())


@pytest.mark.parametrize("module", ["temsim.service.protocol", "temsim.service.client"])
def test_client_side_needs_no_torch(module):
    code = (f"import sys, {module};"
            "bad = [m for m in ('torch', 'temsim.physics', 'temsim.learning', 'temsim.service.session') if m in sys.modules];"
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.dirname(__file__))))
    assert r.returncode == 0, r.stdout + r.stderr


def test_endpoint_check():
    import pytest
    from temsim.service.protocol import check_endpoint
    for good in ("tcp://127.0.0.1:5555", "tcp://gpu-box:9097", "tcp://127.0.0.1:*", "ipc:///tmp/x"):
        assert check_endpoint(good) == good
    for bad in ("tcp:\\\\127.0.0.1:9097", "127.0.0.1:5555", "tcp://127.0.0.1", "http://x:1"):
        with pytest.raises(ValueError, match="invalid endpoint"):
            check_endpoint(bad)
