"""Registry of watched arrays and the architecture graph connecting them."""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .adapter import Provider
from .snapshot import Snapshot

_id_counter = itertools.count(1)


@dataclass
class Watch:
    id: int
    name: str
    provider: Provider
    group: str = ""
    colormap: str = "viridis"
    every: int = 1  # capture on every Nth tick
    tick: int = 0
    last_fingerprint: Optional[int] = None
    last_snapshot: Optional[Snapshot] = None
    dirty: bool = True  # force first send


@dataclass
class Graph:
    edges: List[Tuple[str, str]] = field(default_factory=list)


class Registry:
    """Thread-safe store of watches and graph edges.

    The user's thread registers/updates; the worker thread iterates. All
    mutation is guarded by one lock; iteration takes a shallow copy so the
    capture loop never holds the lock while doing heavy work.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._watches: Dict[str, Watch] = {}
        self._graph = Graph()
        self._structure_version = 0

    # -- mutation (user thread) -------------------------------------------
    def watch(self, name: str, provider: Provider, *, group: str = "",
              colormap: str = "viridis", every: int = 1) -> Watch:
        with self._lock:
            existing = self._watches.get(name)
            if existing is not None:
                existing.provider = provider
                existing.group = group or existing.group
                existing.colormap = colormap
                existing.every = max(1, every)
                existing.dirty = True
                self._structure_version += 1
                return existing
            w = Watch(id=next(_id_counter), name=name, provider=provider,
                      group=group, colormap=colormap, every=max(1, every))
            self._watches[name] = w
            self._structure_version += 1
            return w

    def unwatch(self, name: str) -> None:
        with self._lock:
            if self._watches.pop(name, None) is not None:
                self._graph.edges = [
                    e for e in self._graph.edges if name not in e
                ]
                self._structure_version += 1

    def connect(self, src: str, dst: str) -> None:
        """Declare an architecture edge: data flows from ``src`` to ``dst``."""
        with self._lock:
            edge = (src, dst)
            if edge not in self._graph.edges:
                self._graph.edges.append(edge)
                self._structure_version += 1

    # -- access (worker thread) -------------------------------------------
    def items(self) -> List[Watch]:
        with self._lock:
            return list(self._watches.values())

    def get(self, watch_id: int) -> Optional[Watch]:
        with self._lock:
            for w in self._watches.values():
                if w.id == watch_id:
                    return w
        return None

    @property
    def structure_version(self) -> int:
        with self._lock:
            return self._structure_version

    def structure_message(self) -> dict:
        """JSON-serializable description of watches + graph for clients."""
        with self._lock:
            return {
                "type": "structure",
                "version": self._structure_version,
                "watches": [
                    {"id": w.id, "name": w.name, "group": w.group,
                     "colormap": w.colormap}
                    for w in self._watches.values()
                ],
                "edges": [[s, d] for s, d in self._graph.edges],
            }
