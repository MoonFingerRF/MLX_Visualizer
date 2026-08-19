"""Public entry point: the :class:`Visualizer`.

The visualizer owns one background thread running an asyncio event loop.
That loop hosts the HTTP/WebSocket server and the capture task. The
user's compute thread only ever touches the registry (a dict update
behind a lock), so watching arrays adds essentially zero overhead to the
computation itself; all conversion, reduction, encoding, and network I/O
happen on the worker.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import webbrowser
from typing import Optional

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
              colormap: str = "viridis", every: int = 1) -> "Visualizer":
        """Track an array or a zero-argument callable returning one."""
        self.registry.watch(name, provider, group=group, colormap=colormap, every=every)
        return self

    def unwatch(self, name: str) -> None:
        self.registry.unwatch(name)

    def connect(self, src: str, dst: str) -> "Visualizer":
        """Declare an architecture edge from watch ``src`` to watch ``dst``."""
        self.registry.connect(src, dst)
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
            raw = adapter.resolve(w.provider)
            matrix, original_shape = adapter.to_numpy_2d(raw)
            snap = take_snapshot(matrix, original_shape, max_side=self.max_side)
        except Exception:
            log.exception("snapshot failed for %r", w.name)
            return None
        if not w.dirty and snap.fingerprint == w.last_fingerprint:
            return None  # unchanged; save bandwidth
        w.last_fingerprint = snap.fingerprint
        w.last_snapshot = snap
        w.dirty = False
        return encode_snapshot(w, snap)

    # -- client messages ---------------------------------------------------------
    def _hello(self) -> dict:
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
            loop = asyncio.get_event_loop()
            try:
                value = await loop.run_in_executor(
                    None, adapter.pick_value, watch.provider,
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
