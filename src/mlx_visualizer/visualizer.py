"""Public entry point: the :class:`Visualizer`.

The visualizer owns one background thread running an asyncio event loop.
That loop hosts the HTTP/WebSocket server and the capture task. Normal
conversion, reduction, encoding, and network I/O happen on the worker.
MLX GPU watches use :meth:`Visualizer.refresh` to stage a CPU copy on the
owning compute thread because MLX streams are thread-local.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import webbrowser
from typing import Optional

import numpy as np

from . import adapter
from .adapter import Provider
from .protocol import encode_snapshot
from .registry import Registry
from .server import Client, VizServer
from .snapshot import take_snapshot

log = logging.getLogger("mlx_visualizer")

# Wall-clock budget for snapshot work per capture tick. Watches that don't
# fit in the budget are carried over to the next tick (round-robin), so a
# single enormous matrix can never starve the event loop or the network.
DEFAULT_TICK_BUDGET_S = 0.030


class Visualizer:
    """Watches arrays and streams live visualizations to a web view.

    Example::

        viz = Visualizer()
        viz.watch("layers/w1", lambda: model.w1)
        viz.watch("layers/w2", lambda: model.w2)
        viz.connect("layers/w1", "layers/w2")
        viz.start()          # non-blocking; open the printed URL
        ...training loop...  # runs at full speed
        viz.stop()

    Parameters
    ----------
    host, port:
        Bind address for the built-in server. ``port=0`` picks a free port.
    interval:
        Seconds between capture ticks (default 0.25 = 4 Hz).
    max_side:
        Maximum texture side per array; larger matrices are mean-pooled
        down in memory-bounded bands.
    tick_budget:
        Max seconds of snapshot work per tick before yielding.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8791,
        *,
        interval: float = 0.25,
        max_side: int = 1024,
        tick_budget: float = DEFAULT_TICK_BUDGET_S,
    ) -> None:
        self.registry = Registry()
        self.interval = interval
        self.max_side = max_side
        self.tick_budget = tick_budget
        self._server = VizServer(host, port, on_message=self._on_message, hello=self._hello)
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._started = threading.Event()
        self._rr_cursor = 0  # round-robin start index across ticks
        self._sent_structure_version = -1

    # -- registration (any thread) ------------------------------------------
    def watch(self, name: str, provider: Provider, *, group: str = "",
              colormap: str = "viridis", every: int = 1,
              staged: bool = False) -> "Visualizer":
        """Track an array or a zero-argument callable returning one.

        Set ``staged=True`` for MLX GPU arrays, then call :meth:`refresh`
        from the compute thread after updates. The worker will only touch
        the resulting CPU copy, avoiding cross-thread GPU stream access.
        """
        self.registry.watch(
            name, provider, group=group, colormap=colormap, every=every,
            staged=staged,
        )
        return self

    def metric(self, name: str, provider: Provider, *, group: str = "",
               colormap: str = "turbo", every: int = 1,
               history: int = 512, staged: bool = False) -> "Visualizer":
        """Plot a scalar or zero-argument scalar provider as a live series.

        ``history`` controls the maximum number of samples retained by each
        connected browser. Metric capture uses the same asynchronous,
        change-aware pipeline as tensor watches. Use ``staged=True`` when the
        provider returns an MLX GPU scalar, then update it with :meth:`refresh`.
        """
        self.registry.watch(
            name, provider, group=group, colormap=colormap, every=every,
            kind="metric", history=history, staged=staged,
        )
        return self

    def unwatch(self, name: str) -> None:
        self.registry.unwatch(name)

    def connect(self, src: str, dst: str) -> "Visualizer":
        """Declare an architecture edge from watch ``src`` to watch ``dst``.

        Only needed for custom flows — :meth:`watch_module` discovers
        edges automatically for module trees.
        """
        self.registry.connect(src, dst)
        return self

    def watch_module(self, name: str, module, *, sample_input=None,
                     trace=None, every: int = 1, param_filter=None,
                     staged: bool = False) -> "Visualizer":
        """Watch every parameter of an ``mlx.nn.Module`` tree and capture
        its architecture automatically by tracing a forward pass.

        With ``sample_input`` (or a ``trace`` callable) the graph is
        discovered immediately; otherwise the first forward pass the
        user's own code runs is traced and the instrumentation removes
        itself.
        """
        from .introspect import watch_module as _watch_module
        _watch_module(self, name, module, sample_input=sample_input,
                      trace=trace, every=every, param_filter=param_filter,
                      staged=staged)
        if staged:
            self.refresh()
        return self

    def refresh(self) -> "Visualizer":
        """Stage fresh CPU copies for all watches registered as staged.

        Call this method on the thread that owns the watched MLX arrays,
        normally immediately after ``mx.eval(model.parameters(), ...)``.
        Copies are swapped atomically, so the background snapshot worker can
        safely finish reading the previous version while training continues.
        """
        for watch in self.registry.items():
            if not watch.staged:
                continue
            try:
                raw = adapter.resolve(watch.provider)
                matrix, original_shape = adapter.to_numpy_2d(raw)
                staged_matrix = np.array(
                    matrix, dtype=np.float32, order="C", copy=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"failed to stage visualizer watch {watch.name!r} on "
                    "the calling thread"
                ) from exc
            self.registry.set_staged(watch.name, staged_matrix, original_shape)
        return self

    # -- lifecycle ------------------------------------------------------------
    def start(self, open_browser: bool = False, timeout: float = 10.0) -> str:
        """Start the worker thread and server. Returns the viewer URL."""
        if self._thread is not None:
            return self.url
        self._thread = threading.Thread(target=self._run, name="mlx-visualizer", daemon=True)
        self._thread.start()
        if not self._started.wait(timeout):
            raise RuntimeError("MLX Visualizer failed to start")
        if open_browser:
            webbrowser.open(self.url)
        return self.url

    def stop(self) -> None:
        loop, stop_event = self._loop, self._stop_event
        if loop is None or stop_event is None:
            return
        loop.call_soon_threadsafe(stop_event.set)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None
        self._loop = None
        self._started.clear()

    @property
    def url(self) -> str:
        return self._server.url

    def __enter__(self) -> "Visualizer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- worker thread ----------------------------------------------------------
    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._main())
        finally:
            loop.close()

    async def _main(self) -> None:
        self._stop_event = asyncio.Event()
        await self._server.start()
        self._started.set()
        capture = asyncio.ensure_future(self._capture_loop())
        await self._stop_event.wait()
        capture.cancel()
        await self._server.stop()

    async def _capture_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.interval)
                if not self._server.has_clients():
                    continue
                self._sync_structure()
                await self._capture_tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("capture tick failed")

    def _sync_structure(self) -> None:
        version = self.registry.structure_version
        if version != self._sent_structure_version:
            self._server.broadcast_json(self.registry.structure_message())
            self._sent_structure_version = version

    async def _capture_tick(self) -> None:
        watches = self.registry.items()
        if not watches:
            return
        n = len(watches)
        start = self._rr_cursor % n
        deadline = time.monotonic() + self.tick_budget
        processed = 0
        for k in range(n):
            w = watches[(start + k) % n]
            w.tick += 1
            if not w.dirty and w.tick % w.every != 0:
                continue
            loop = asyncio.get_event_loop()
            frame = await loop.run_in_executor(None, self._snapshot_one, w)
            if frame is not None:
                self._server.broadcast_binary(frame, w.id)
            processed += 1
            if time.monotonic() > deadline:
                # Budget exhausted: remember where to resume next tick.
                self._rr_cursor = (start + k + 1) % n
                return
        self._rr_cursor = (start + processed) % n

    def _snapshot_one(self, w) -> Optional[bytes]:
        """Runs on the default executor: resolve → reduce → encode."""
        try:
            if w.staged:
                matrix, original_shape = self.registry.staged_data(w.id)
                if matrix is None or original_shape is None:
                    return None
            else:
                raw = adapter.resolve(w.provider)
                matrix, original_shape = adapter.to_numpy_2d(raw)
            snap = take_snapshot(matrix, original_shape, max_side=self.max_side)
        except Exception:
            if not w.failing:  # log once per failure streak, retry next tick
                w.failing = True
                log.exception("snapshot failed for %r", w.name)
            return None
        w.failing = False
        if not w.dirty and snap.fingerprint == w.last_fingerprint:
            return None  # unchanged; save bandwidth
        w.last_fingerprint = snap.fingerprint
        w.last_snapshot = snap
        w.dirty = False
        return encode_snapshot(w, snap)

    # -- client messages ---------------------------------------------------------
    def _hello(self) -> dict:
        # Fingerprints are global, but a newly joined/reloaded browser has no
        # textures yet. Force one frame per watch so it can reconstruct them.
        self.registry.mark_all_dirty()
        msg = self.registry.structure_message()
        msg["type"] = "hello"
        msg["interval"] = self.interval
        msg["maxSide"] = self.max_side
        return msg

    async def _on_message(self, client: Client, obj: dict) -> None:
        kind = obj.get("type")
        if kind == "pick":
            watch = self.registry.get(int(obj.get("id", -1)))
            if watch is None:
                return
            provider = watch.provider
            if watch.staged:
                provider, _shape = self.registry.staged_data(watch.id)
                if provider is None:
                    return
            loop = asyncio.get_event_loop()
            try:
                value = await loop.run_in_executor(
                    None, adapter.pick_value, provider,
                    int(obj.get("row", 0)), int(obj.get("col", 0)))
            except Exception:
                return
            import json as _json
            from .websocket import OP_TEXT
            client.send(OP_TEXT, _json.dumps({
                "type": "pickResult", "id": watch.id,
                "row": obj.get("row"), "col": obj.get("col"),
                "value": value,
            }).encode("utf-8"))
        elif kind == "refresh":
            for w in self.registry.items():
                w.dirty = True
            self._sent_structure_version = -1
