"""Part IV, Section 3: the frame thread and the flow control."""

import threading
import time

from temsim.service import protocol
from temsim.service.sender import Flow, FrameSender, package

import numpy as np


def compose(capture):
    return {"loss": float(capture), "layers": []}, {"sim": np.full((4, 4), capture, np.float32)}


def wait_for(sender, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        frames = sender.take()
        if frames is not None:
            return protocol.unpack(frames.frames)
        time.sleep(0.005)
    raise AssertionError("no message")


def test_message_carries_a_sequence_number_and_the_arrays():
    sender = FrameSender(compose)
    try:
        sender.submit(2.0)
        header, arrays = wait_for(sender)
        assert header["name"] == "frame" and header["data"]["seq"] >= 1 and header["data"]["loss"] == 2.0
        assert arrays["sim"].shape == (4, 4) and sender.take() is None
    finally:
        sender.close()


def test_latest_capture_wins():
    gate = threading.Event()

    def slow(capture):
        gate.wait(5.0)
        return compose(capture)

    sender = FrameSender(slow)
    try:
        sender.submit(1.0)
        time.sleep(0.05)                                   # the thread now holds 1.0
        sender.submit(2.0)
        sender.submit(3.0)                                 # replaces 2.0
        gate.set()
        seen = []
        end = time.monotonic() + 1.0
        while time.monotonic() < end:
            frames = sender.take()
            if frames is not None:
                seen.append(protocol.unpack(frames.frames)[0]["data"]["loss"])
            time.sleep(0.005)
        assert seen[-1] == 3.0 and 2.0 not in seen
    finally:
        sender.close()


def test_cancel_drops_what_is_not_sent_yet():
    gate = threading.Event()

    def slow(capture):
        gate.wait(5.0)
        return compose(capture)

    sender = FrameSender(slow)
    try:
        sender.submit(1.0)
        time.sleep(0.05)
        sender.cancel()
        gate.set()
        time.sleep(0.2)
        assert sender.take() is None
    finally:
        sender.close()


def test_a_failing_compose_is_kept_not_raised():
    def broken(capture):
        raise RuntimeError("boom")

    sender = FrameSender(broken)
    try:
        sender.submit(1.0)
        time.sleep(0.2)
        assert isinstance(sender.error, RuntimeError) and sender.take() is None
    finally:
        sender.close()


def test_flow_allows_two_unanswered_and_recovers_after_the_timeout():
    t = [0.0]
    flow = Flow(limit=2, timeout=10.0, now=lambda: t[0])
    assert flow.can_send(b"a")
    flow.sent(b"a"), flow.sent(b"a")
    assert not flow.can_send(b"a") and flow.can_send(b"b")
    flow.acknowledged(b"a")
    assert flow.can_send(b"a")
    flow.sent(b"a")
    assert not flow.can_send(b"a")
    t[0] = 11.0
    assert flow.can_send(b"a")                              # a client that never answers is not stalled for ever


def test_package_sends_images_as_float16_and_the_target_in_its_own_message():
    rng = np.random.default_rng(0)
    sim, target = rng.normal(size=(8, 8)).astype(np.float32), rng.normal(size=(8, 8)).astype(np.float32)
    atoms = rng.random((5, 2)).astype(np.float32) * 1000
    mask = rng.random((8, 8)) > 0.5
    packet = package({"loss": 1.0}, {"sim": sim, "target": target, "mask": mask.astype(np.uint8), "atoms.0.pos": atoms})
    header, arrays = protocol.unpack(packet.frames)
    assert set(arrays) == {"sim", "atoms.0.pos"} and arrays["sim"].dtype == np.float16
    assert arrays["atoms.0.pos"].dtype == np.float32 and header["data"]["target_key"] == packet.key
    np.testing.assert_allclose(arrays["sim"], sim, rtol=1e-3, atol=1e-3)
    t_header, t_arrays = protocol.unpack(packet.target_frames)
    assert t_header["name"] == "target" and t_header["data"]["key"] == packet.key
    np.testing.assert_allclose(t_arrays["target"], target, rtol=1e-3, atol=1e-3)
    np.testing.assert_array_equal(t_arrays["mask"], mask)
    assert package({}, {"sim": sim}).target_frames is None
    assert package({}, {"target": target * 2}).key != packet.key
