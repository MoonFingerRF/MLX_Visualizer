"""Binary wire protocol between the visualizer and the web view.

Snapshot frame layout (little-endian):

    magic   u32   0x4D4C5856 ("MLXV")
    version u16   1
    kind    u16   1 = snapshot
    metalen u32   byte length of UTF-8 JSON metadata
    meta    bytes JSON: {id, name, shape, rows, cols, w, h, vmin, vmax,
                         mean, std, nan, cmap, group}
    payload bytes w*h float32 values, row-major

Control messages (structure updates, picks) travel as JSON text frames.
"""

from __future__ import annotations

import json
import struct
from typing import Tuple

import numpy as np

from .registry import Watch
from .snapshot import Snapshot

MAGIC = 0x4D4C5856
VERSION = 1
KIND_SNAPSHOT = 1

_HEADER = struct.Struct("<IHHI")


def encode_snapshot(watch: Watch, snap: Snapshot) -> bytes:
    img = snap.image
    meta = {
        "id": watch.id,
        "name": watch.name,
        "group": watch.group,
        "cmap": watch.colormap,
        "shape": list(snap.shape),
        "rows": snap.rows,
        "cols": snap.cols,
        "w": int(img.shape[1]),
        "h": int(img.shape[0]),
        "vmin": snap.vmin,
        "vmax": snap.vmax,
        "mean": snap.mean,
        "std": snap.std,
        "nan": snap.nan_count,
    }
    meta_bytes = json.dumps(meta, allow_nan=False).encode("utf-8")
    # Pad metadata to a 4-byte boundary so the float payload is aligned
    # for zero-copy Float32Array views in the browser.
    meta_bytes += b" " * (-len(meta_bytes) % 4)
    header = _HEADER.pack(MAGIC, VERSION, KIND_SNAPSHOT, len(meta_bytes))
    return b"".join((header, meta_bytes, img.tobytes()))


def decode_snapshot(buf: bytes) -> Tuple[dict, np.ndarray]:
    """Inverse of :func:`encode_snapshot`. Used by tests and Python clients."""
    magic, version, kind, metalen = _HEADER.unpack_from(buf, 0)
    if magic != MAGIC:
        raise ValueError("bad magic")
    if version != VERSION:
        raise ValueError(f"unsupported protocol version {version}")
    if kind != KIND_SNAPSHOT:
        raise ValueError(f"unknown frame kind {kind}")
    off = _HEADER.size
    meta = json.loads(buf[off:off + metalen].decode("utf-8"))
    payload = np.frombuffer(buf, dtype="<f4", offset=off + metalen)
    img = payload.reshape(meta["h"], meta["w"])
    return meta, img
