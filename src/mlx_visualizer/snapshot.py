"""Snapshot pipeline: memory-bounded, batched downsampling and statistics.

The heavy lifting for extremely large matrices happens here. A matrix is
reduced to at most ``max_side`` pixels per axis with block means computed
in fixed-size row bands, so peak extra memory stays bounded no matter how
large the input is. NumPy releases the GIL for these reductions, so the
user's compute thread keeps running while a snapshot is taken.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Output rows processed per band. Bounds peak temporary memory to roughly
# band_rows * block_y * width * 4 bytes regardless of total matrix size.
DEFAULT_BAND_ROWS = 512


@dataclass(frozen=True)
class Snapshot:
    """An immutable, render-ready reduction of one array."""

    image: np.ndarray  # float32, C-contiguous, shape (oh, ow), oh/ow <= max_side
    shape: Tuple[int, ...]  # original tensor shape
    rows: int  # logical 2-D rows (after collapsing leading dims)
    cols: int  # logical 2-D cols
    vmin: float
    vmax: float
    mean: float
    std: float
    nan_count: int
    fingerprint: int  # crc32 of the image bytes — cheap change detection


def block_mean(a: np.ndarray, fy: int, fx: int, band_rows: int = DEFAULT_BAND_ROWS) -> np.ndarray:
    """Mean-pool ``a`` with block size (fy, fx), batched over row bands.

    Handles non-divisible edges exactly (edge blocks average fewer cells).
    """
    h, w = a.shape
    oh = -(-h // fy)
    ow = -(-w // fx)
    col_starts = np.arange(0, w, fx)
    col_counts = np.diff(np.append(col_starts, w)).astype(np.float32)
    out = np.empty((oh, ow), dtype=np.float32)
    for ob0 in range(0, oh, band_rows):
        ob1 = min(ob0 + band_rows, oh)
        r0 = ob0 * fy
        r1 = min(ob1 * fy, h)
        band = np.asarray(a[r0:r1], dtype=np.float32)
        csum = np.add.reduceat(band, col_starts, axis=1)
        row_starts = np.arange(0, band.shape[0], fy)
        row_counts = np.diff(np.append(row_starts, band.shape[0])).astype(np.float32)
        rsum = np.add.reduceat(csum, row_starts, axis=0)
        out[ob0:ob1] = rsum / (row_counts[:, None] * col_counts[None, :])
    return out


def downsample(a: np.ndarray, max_side: int, band_rows: int = DEFAULT_BAND_ROWS) -> np.ndarray:
    """Reduce a 2-D matrix to at most ``max_side`` per axis via block means."""
    h, w = a.shape
    fy = -(-h // max_side)
    fx = -(-w // max_side)
    if fy == 1 and fx == 1:
        return np.ascontiguousarray(a, dtype=np.float32)
    return block_mean(a, fy, fx, band_rows=band_rows)


def take_snapshot(matrix: np.ndarray, original_shape: Tuple[int, ...], max_side: int = 1024) -> Snapshot:
    """Downsample + compute display statistics for one matrix.

    Statistics are computed on the reduced image (not the full matrix) so
    the cost of a snapshot is O(elements visited once) for the reduction
    and O(max_side^2) for everything else. For display normalization this
    is the right trade: block means bound the visible dynamic range.
    """
    img = downsample(matrix, max_side)
    finite = np.isfinite(img)
    nan_count = int(img.size - int(finite.sum()))
    if nan_count == img.size:
        vmin = vmax = mean = std = 0.0
    else:
        vals = img[finite]
        vmin = float(vals.min())
        vmax = float(vals.max())
        mean = float(vals.mean())
        std = float(vals.std())
    return Snapshot(
        image=img,
        shape=tuple(int(s) for s in original_shape),
        rows=int(matrix.shape[0]),
        cols=int(matrix.shape[1]),
        vmin=vmin,
        vmax=vmax,
        mean=mean,
        std=std,
        nan_count=nan_count,
        fingerprint=zlib.crc32(img.tobytes()),
    )
