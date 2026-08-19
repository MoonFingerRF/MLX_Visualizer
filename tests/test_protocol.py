import numpy as np

from mlx_visualizer.protocol import decode_snapshot, encode_snapshot
from mlx_visualizer.registry import Registry
from mlx_visualizer.snapshot import take_snapshot


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


def test_duplicate_edges_are_ignored():
    reg = Registry()
    reg.connect("a", "b")
    reg.connect("a", "b")
    assert reg.structure_message()["edges"] == [["a", "b"]]
