import numpy as np
import pytest

from mlx_visualizer.protocol import decode_snapshot, encode_snapshot
from mlx_visualizer.registry import Registry
from mlx_visualizer.snapshot import take_snapshot
from mlx_visualizer.visualizer import Visualizer


def test_snapshot_roundtrip():
    reg = Registry()
    a = np.random.default_rng(1).normal(size=(37, 53))
    w = reg.watch("layer/w", a, group="encoder", colormap="magma")
    snap = take_snapshot(a, a.shape)
    buf = encode_snapshot(w, snap)
    meta, img = decode_snapshot(buf)
    assert meta["name"] == "layer/w"
    assert meta["group"] == "encoder"
    assert meta["cmap"] == "magma"
    assert meta["shape"] == [37, 53]
    assert (meta["h"], meta["w"]) == img.shape
    np.testing.assert_allclose(img, snap.image)
    assert meta["vmin"] == snap.vmin and meta["vmax"] == snap.vmax


def test_registry_watch_update_and_graph():
    reg = Registry()
    v0 = reg.structure_version
    reg.watch("a", np.zeros((2, 2)))
    reg.watch("b", np.zeros((2, 2)))
    reg.connect("a", "b")
    assert reg.structure_version > v0
    msg = reg.structure_message()
    assert [w["name"] for w in msg["watches"]] == ["a", "b"]
    assert msg["edges"] == [["a", "b"]]
    # Re-watching the same name keeps the id, marks dirty.
    id_a = reg.watch("a", np.ones((2, 2))).id
    assert reg.watch("a", np.ones((2, 2))).id == id_a
    reg.unwatch("a")
    msg = reg.structure_message()
    assert [w["name"] for w in msg["watches"]] == ["b"]
    assert msg["edges"] == []


def test_metric_metadata_and_validation():
    reg = Registry()
    metric = reg.watch(
        "training/loss", lambda: 1.25, kind="metric", history=100,
        colormap="turbo",
    )
    watch = reg.structure_message()["watches"][0]
    assert metric.kind == "metric"
    assert watch == {
        "id": metric.id,
        "name": "training/loss",
        "group": "",
        "colormap": "turbo",
        "kind": "metric",
        "history": 100,
    }
    with pytest.raises(ValueError):
        reg.watch("bad", 0.0, kind="histogram")


def test_duplicate_edges_are_ignored():
    reg = Registry()
    reg.connect("a", "b")
    reg.connect("a", "b")
    assert reg.structure_message()["edges"] == [["a", "b"]]


def test_refresh_stages_an_isolated_float32_copy():
    data = np.arange(6, dtype=np.float64).reshape(2, 3)
    viz = Visualizer(port=0)
    viz.watch("staged", lambda: data, staged=True)

    viz.refresh()
    watch = viz.registry.items()[0]
    matrix, shape = viz.registry.staged_data(watch.id)
    assert shape == (2, 3)
    assert matrix.dtype == np.float32
    assert matrix.flags.c_contiguous
    np.testing.assert_allclose(matrix, data)

    data[:] = -1
    assert matrix[0, 0] == 0  # the worker's current copy is immutable by convention
    viz.refresh()
    updated, _ = viz.registry.staged_data(watch.id)
    np.testing.assert_allclose(updated, data)


def test_refresh_stages_scalar_metrics():
    viz = Visualizer(port=0)
    viz.metric("staged-loss", lambda: np.asarray(2.5), staged=True)
    viz.refresh()

    watch = viz.registry.items()[0]
    matrix, shape = viz.registry.staged_data(watch.id)
    assert shape == ()
    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == pytest.approx(2.5)
