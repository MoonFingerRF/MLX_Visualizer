"""Array backend adapter.

Accepts MLX arrays, NumPy arrays, torch tensors, or anything exposing
``__array__`` / ``tolist``. Normal watches convert on the visualizer worker;
staged MLX watches convert cooperatively on their owning compute thread.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple, Union

import numpy as np

ArrayLike = Any
Provider = Union[ArrayLike, Callable[[], ArrayLike]]


def resolve(provider: Provider) -> ArrayLike:
    """Resolve a watch target: call it if it is a provider callable."""
    if callable(provider) and not hasattr(provider, "shape"):
        return provider()
    return provider


def _ensure_evaluated(x: ArrayLike) -> None:
    """Force evaluation of MLX lazy arrays before touching their buffer.

    Evaluating a foreign thread's graph raises a catchable RuntimeError from
    ``mx.eval`` on platforms where compute streams are thread-local, while
    converting it with ``np.asarray`` directly can hard-abort the process.
    Calling ``mx.eval`` first turns the dangerous case into a skippable error.
    """
    if type(x).__module__.split(".")[0] == "mlx":
        import mlx.core as mx

        try:
            mx.eval(x)
        except RuntimeError as exc:
            raise RuntimeError(
                "cannot access this MLX array from the visualizer worker; "
                "register it with staged=True and call viz.refresh() on "
                "the MLX compute thread"
            ) from exc


def to_numpy_2d(x: ArrayLike) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Convert an array-like object to a 2-D float NumPy view.

    Returns ``(matrix, original_shape)``. Vectors become a 1-row matrix,
    N-D tensors (N > 2) are collapsed to ``(prod(leading dims), last dim)``.
    For MLX arrays, ``np.asarray`` forces evaluation and copies the buffer,
    which is exactly the isolation we want: the snapshot is decoupled from
    the live computation graph.
    """
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
    _ensure_evaluated(x)
    a = np.asarray(x)
    original_shape = a.shape
    if a.ndim == 0:
        a = a.reshape(1, 1)
    elif a.ndim == 1:
        a = a.reshape(1, -1)
    elif a.ndim > 2:
        a = a.reshape(-1, a.shape[-1])
    if not np.issubdtype(a.dtype, np.floating):
        a = a.astype(np.float32)
    return a, original_shape


def pick_value(provider: Provider, row: int, col: int) -> float:
    """Read a single element from the live array without materializing it.

    Used by the inspector tooltip. Indexing one element is cheap for both
    NumPy and MLX arrays.
    """
    x = resolve(provider)
    _ensure_evaluated(x)
    a = np.asarray(x) if not hasattr(x, "shape") else x
    shape = tuple(int(s) for s in a.shape)
    if len(shape) == 0:
        v = a
    elif len(shape) == 1:
        v = a[min(col, shape[0] - 1)]
    else:
        # Collapse leading dims exactly like to_numpy_2d does.
        rows = 1
        for s in shape[:-1]:
            rows *= s
        r = min(row, rows - 1)
        c = min(col, shape[-1] - 1)
        idx = []
        for s in reversed(shape[:-1]):
            idx.append(r % s)
            r //= s
        idx = tuple(reversed(idx)) + (c,)
        v = a[idx]
    _ensure_evaluated(v)  # indexing an MLX array yields a new lazy array
    if hasattr(v, "item"):
        return float(v.item())
    return float(v)
