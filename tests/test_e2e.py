"""End-to-end: real server, real socket client, real frames."""

import json
import threading
import time
import urllib.request

import numpy as np
import pytest

from mlx_visualizer import Visualizer
from mlx_visualizer.protocol import decode_snapshot

from ws_client import WSClient


@pytest.fixture()
def viz():
    v = Visualizer(port=0, interval=0.05)
    v.start()
    yield v
    v.stop()


def _host_port(viz):
    return viz._server.host, viz._server.port


def test_serves_viewer_over_http(viz):
    host, port = _host_port(viz)
    html = urllib.request.urlopen(f"http://{host}:{port}/", timeout=5).read()
    assert b"MLX Visualizer" in html
    js = urllib.request.urlopen(f"http://{host}:{port}/app.js", timeout=5).read()
    assert b"drawArraysInstanced" in js
    assert b"labelStyleForZoom" in js
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(f"http://{host}:{port}/nope.txt", timeout=5)


def test_hello_snapshots_and_pick(viz):
    data = np.arange(20000, dtype=np.float32).reshape(100, 200)
    viz.watch("m/a", lambda: data, group="g1")
    viz.watch("m/b", np.linspace(0, 1, 64))
    viz.connect("m/a", "m/b")

    host, port = _host_port(viz)
    client = WSClient(host, port)
    try:
        opcode, payload = client.recv_message()
        hello = json.loads(payload)
        assert opcode == 1
        assert hello["type"] == "hello"
        names = {w["name"] for w in hello["watches"]}
        assert names == {"m/a", "m/b"}
        assert hello["edges"] == [["m/a", "m/b"]]

        # Collect snapshot frames for both watches.
        seen = {}
        deadline = time.time() + 10
        while len(seen) < 2 and time.time() < deadline:
            msg = client.recv_message()
            assert msg is not None
            opcode, payload = msg
            if opcode != 2:
                continue
            meta, img = decode_snapshot(payload)
            seen[meta["name"]] = (meta, img)
        assert set(seen) == {"m/a", "m/b"}

        meta_a, img_a = seen["m/a"]
        assert meta_a["shape"] == [100, 200]
        np.testing.assert_allclose(img_a, data)
        meta_b, img_b = seen["m/b"]
        assert meta_b["h"] == 1 and meta_b["w"] == 64

        # Exact-value pick round trip.
        client.send_text(json.dumps({"type": "pick", "id": meta_a["id"], "row": 3, "col": 7}))
        deadline = time.time() + 10
        while time.time() < deadline:
            msg = client.recv_message()
            assert msg is not None
            opcode, payload = msg
            if opcode == 1:
                obj = json.loads(payload)
                if obj.get("type") == "pickResult":
                    assert obj["value"] == float(data[3, 7])
                    break
        else:
            pytest.fail("no pickResult received")
    finally:
        client.close()


def test_unchanged_data_is_not_resent(viz):
    data = np.ones((8, 8))
    viz.watch("static", data)
    host, port = _host_port(viz)
    client = WSClient(host, port)
    try:
        client.recv_message()  # hello
        # First snapshot arrives...
        opcode = None
        deadline = time.time() + 10
        while time.time() < deadline:
            opcode, payload = client.recv_message()
            if opcode == 2:
                break
        assert opcode == 2
        # ...then several capture intervals pass with no duplicate frame.
        client.sock.settimeout(0.5)
        import socket as _socket
        with pytest.raises((_socket.timeout, TimeoutError)):
            while True:
                msg = client.recv_message()
                assert msg is not None
                opcode, _ = msg
                assert opcode != 2, "unchanged snapshot was resent"
    finally:
        client.close()


def test_reconnected_client_receives_unchanged_initial_snapshot(viz):
    viz.watch("static", np.ones((8, 8)))
    host, port = _host_port(viz)

    first = WSClient(host, port)
    try:
        first.recv_message()  # hello
        while True:
            opcode, _payload = first.recv_message()
            if opcode == 2:
                break
    finally:
        first.close()

    second = WSClient(host, port)
    second.sock.settimeout(5)
    try:
        second.recv_message()  # hello
        while True:
            opcode, payload = second.recv_message()
            if opcode == 2:
                meta, image = decode_snapshot(payload)
                assert meta["name"] == "static"
                np.testing.assert_allclose(image, 1)
                break
    finally:
        second.close()


def test_large_matrix_is_downsampled(viz):
    big = np.zeros((4096, 4096), dtype=np.float32)
    big[0, 0] = 100.0
    viz.watch("big", big)
    host, port = _host_port(viz)
    client = WSClient(host, port)
    try:
        client.recv_message()  # hello
        deadline = time.time() + 15
        while time.time() < deadline:
            opcode, payload = client.recv_message()
            if opcode == 2:
                meta, img = decode_snapshot(payload)
                assert meta["w"] <= 1024 and meta["h"] <= 1024
                assert meta["rows"] == 4096 and meta["cols"] == 4096
                # Block mean of the hot corner block: 100 / 16.
                assert img[0, 0] == pytest.approx(100.0 / 16.0)
                return
        pytest.fail("no snapshot received")
    finally:
        client.close()


def test_metric_streams_as_scalar_snapshot(viz):
    state = {"loss": 2.5}
    viz.metric("training/loss", lambda: state["loss"], history=25)
    watch = viz.registry.structure_message()["watches"][0]
    assert watch["kind"] == "metric"
    assert watch["history"] == 25

    host, port = _host_port(viz)
    client = WSClient(host, port)
    try:
        client.recv_message()  # hello
        deadline = time.time() + 10
        while time.time() < deadline:
            opcode, payload = client.recv_message()
            if opcode == 2:
                meta, image = decode_snapshot(payload)
                assert meta["name"] == "training/loss"
                assert meta["shape"] == []
                assert image.shape == (1, 1)
                assert image[0, 0] == pytest.approx(2.5)
                return
        pytest.fail("no metric snapshot received")
    finally:
        client.close()


def test_staged_provider_is_only_resolved_on_refreshing_thread(viz):
    owner_thread = threading.get_ident()
    provider_calls = []
    data = np.arange(12, dtype=np.float32).reshape(3, 4)

    def provider():
        provider_calls.append(threading.get_ident())
        assert threading.get_ident() == owner_thread
        return data

    viz.watch("gpu-safe", provider, staged=True)
    viz.refresh()

    host, port = _host_port(viz)
    client = WSClient(host, port)
    try:
        client.recv_message()  # hello
        deadline = time.time() + 10
        while time.time() < deadline:
            opcode, payload = client.recv_message()
            if opcode == 2:
                meta, image = decode_snapshot(payload)
                assert meta["name"] == "gpu-safe"
                np.testing.assert_allclose(image, data)
                break
        else:
            pytest.fail("no staged snapshot received")
        assert provider_calls == [owner_thread]
    finally:
        client.close()


def test_mlx_gpu_staged_snapshot_never_enters_worker_stream(viz):
    mx = pytest.importorskip("mlx.core")
    owner_thread = threading.get_ident()
    data = mx.arange(12, dtype=mx.float32).reshape(3, 4)

    def provider():
        assert threading.get_ident() == owner_thread
        return data

    viz.watch("mlx-gpu-safe", provider, staged=True)
    viz.refresh()

    host, port = _host_port(viz)
    client = WSClient(host, port)
    try:
        client.recv_message()  # hello
        deadline = time.time() + 10
        while time.time() < deadline:
            opcode, payload = client.recv_message()
            if opcode == 2:
                meta, image = decode_snapshot(payload)
                assert meta["name"] == "mlx-gpu-safe"
                np.testing.assert_allclose(image, np.arange(12).reshape(3, 4))
                return
        pytest.fail("no MLX staged snapshot received")
    finally:
        client.close()
