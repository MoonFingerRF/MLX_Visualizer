"""MLX Visualizer — asynchronous, batched, GPU-rendered views of large
matrices and vectors, without slowing down the computation that owns them.
"""

from .snapshot import Snapshot, block_mean, downsample, take_snapshot
from .visualizer import Visualizer

__version__ = "0.1.0"
__all__ = [
    "Visualizer",
    "Snapshot",
    "take_snapshot",
    "downsample",
    "block_mean",
    "watch",
    "metric",
    "watch_module",
    "connect",
    "start",
    "stop",
]

_default: Visualizer = Visualizer()


def watch(name, provider, **kwargs):
    """Watch an array on the shared default visualizer."""
    return _default.watch(name, provider, **kwargs)


def metric(name, provider, **kwargs):
    """Plot a scalar on the shared default visualizer."""
    return _default.metric(name, provider, **kwargs)


def watch_module(name, module, **kwargs):
    """Watch a whole module tree (auto-captures its architecture) on the
    shared default visualizer."""
    return _default.watch_module(name, module, **kwargs)


def connect(src: str, dst: str):
    """Add an architecture edge on the shared default visualizer."""
    return _default.connect(src, dst)


def start(**kwargs) -> str:
    """Start the shared default visualizer; returns the viewer URL."""
    return _default.start(**kwargs)


def stop() -> None:
    _default.stop()
