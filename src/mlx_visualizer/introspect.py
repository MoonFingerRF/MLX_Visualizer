"""Automatic architecture capture for MLX-style module trees.

``watch_module`` walks an ``mlx.nn.Module`` (any dict-like module tree
with ``named_modules()``), watches every parameter array, and discovers
the architecture edges automatically by tracing a real forward pass:
each submodule call is recorded, and edges are built by matching the
identity of output arrays to the inputs of later calls (exact dataflow),
falling back to call order when functional ops (activations, reshapes)
sit between modules.

No MLX import is required here — everything is duck-typed — so the core
library stays dependency-free and the same mechanism works for any
framework whose modules are dicts of arrays with per-class ``__call__``.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

PARAM_COLORMAPS = {"bias": "coolwarm"}


def _is_array(v: Any) -> bool:
    return hasattr(v, "shape") and hasattr(v, "dtype") and not isinstance(v, dict)


def _direct_params(module: dict) -> List[Tuple[str, Any]]:
    """Parameter arrays stored directly on this module (not descendants)."""
    return [(k, v) for k, v in module.items() if _is_array(v)]


def _named_modules(module: Any) -> List[Tuple[str, Any]]:
    mods = list(module.named_modules())
    # MLX yields (path, module); keep deterministic order: root first,
    # then by path.
    mods.sort(key=lambda item: (item[0] != "", item[0]))
    return mods


def _arrays_in(value: Any) -> List[Any]:
    """Arrays in a call argument/result, flattening one level.

    Records hold the arrays themselves (not their ids): intermediate
    arrays are freed during the forward pass and CPython reuses ids, so
    identity is only meaningful while a strong reference is held.
    """
    if _is_array(value):
        return [value]
    if isinstance(value, (list, tuple)):
        return [v for v in value if _is_array(v)]
    return []


class _Tracer:
    """Temporarily instruments module classes to record the forward pass."""

    def __init__(self, all_modules: Sequence[Any], watched_ids: Dict[int, str],
                 on_done: Callable[[List[Tuple[int, List[Any], List[Any]]]], None]):
        self._watched = watched_ids
        self._on_done = on_done
        self._records: List[Tuple[int, List[Any], List[Any]]] = []
        self._depth = 0
        self._lock = threading.Lock()
        # cls -> (original function, was defined on cls itself)
        self._originals: Dict[type, Tuple[Any, bool]] = {}
        for m in all_modules:
            cls = type(m)
            if cls in self._originals:
                continue
            # Only instrument classes whose instances are callable — the
            # metaclass's __call__ doesn't count.
            defined = any("__call__" in c.__dict__ for c in cls.__mro__)
            if defined:
                self._originals[cls] = (cls.__call__, "__call__" in cls.__dict__)

    def patch(self) -> None:
        for cls, (orig, _own) in self._originals.items():
            cls.__call__ = self._make_wrapper(orig)

    def _restore(self) -> None:
        for cls, (orig, own) in self._originals.items():
            if own:
                cls.__call__ = orig
            else:
                del cls.__call__  # fall back to the inherited method
        self._originals = {}

    def unpatch(self) -> None:
        with self._lock:
            if self._originals:
                self._restore()

    def _make_wrapper(self, orig):
        tracer = self

        def wrapper(mod, *args, **kwargs):
            tracer._depth += 1
            try:
                out = orig(mod, *args, **kwargs)
            finally:
                tracer._depth -= 1
            if id(mod) in tracer._watched:
                inputs = []
                for a in args:
                    inputs.extend(_arrays_in(a))
                for a in kwargs.values():
                    inputs.extend(_arrays_in(a))
                tracer._records.append((id(mod), inputs, _arrays_in(out)))
            if tracer._depth == 0 and tracer._records:
                tracer._finish()
            return out

        return wrapper

    def _finish(self) -> None:
        with self._lock:
            if not self._originals:
                return  # already finished
            self._restore()
        records, self._records = self._records, []
        self._on_done(records)
        records.clear()  # release the kept-alive intermediate arrays


def _edges_from_records(records: List[Tuple[int, List[Any], List[Any]]]) -> List[Tuple[int, int]]:
    """Dataflow edges by array identity, with call-order fallback.

    An edge src→dst means dst consumed an array src produced. When a
    module's inputs match no recorded producer (an activation or other
    functional op sits in between), fall back to the previously called
    module, which is correct for sequential chains. The records keep the
    arrays alive, so id() is a valid identity here.
    """
    producer: Dict[int, int] = {}
    edges: List[Tuple[int, int]] = []
    prev: Optional[int] = None
    for mid, inputs, outputs in records:
        srcs = {producer[id(a)] for a in inputs
                if id(a) in producer and producer[id(a)] != mid}
        if not srcs and prev is not None and prev != mid:
            srcs = {prev}
            # Alias the unmatched inputs to the fallback source: they are
            # (functionally transformed) outputs of it, so later modules
            # consuming the same arrays branch from it too — a diamond
            # like relu(trunk(x)) feeding two heads stays a DAG instead
            # of degenerating into head→head chain edges.
            for a in inputs:
                producer.setdefault(id(a), prev)
        for s in srcs:
            if (s, mid) not in edges:
                edges.append((s, mid))
        for a in outputs:
            producer[id(a)] = mid
        prev = mid
    return edges


def watch_module(
    viz,
    name: str,
    module: Any,
    *,
    sample_input: Any = None,
    trace: Optional[Callable[[], Any]] = None,
    every: int = 1,
    param_filter: Optional[Callable[[str, str], bool]] = None,
    staged: bool = False,
) -> Any:
    """Watch all parameters of a module tree and auto-capture its graph.

    Parameters
    ----------
    name:
        Prefix for all watch names (e.g. ``"mlp"``).
    module:
        An ``mlx.nn.Module`` (or compatible dict-like module tree).
    sample_input:
        If given, ``module(sample_input)`` is traced immediately to
        discover the architecture.
    trace:
        Alternative to ``sample_input``: a zero-argument callable that
        runs one forward pass (for modules taking several arguments).
    every:
        Capture cadence forwarded to every created watch.
    param_filter:
        Optional ``(module_path, param_name) -> bool`` to select which
        parameters get watched.
    staged:
        Register parameters for caller-thread CPU staging. Use this for MLX
        GPU modules and call ``viz.refresh()`` after parameter updates.

    If neither ``sample_input`` nor ``trace`` is given, the module tree
    is instrumented lazily: the first forward pass the user's own code
    runs is traced, edges are registered, and the instrumentation is
    removed. Until then the tensors still stream — only the edges wait.
    """
    if not hasattr(module, "named_modules"):
        raise TypeError(
            "watch_module expects an mlx.nn.Module-like object with "
            "named_modules(); for other data use viz.watch() + viz.connect()")

    mods = _named_modules(module)
    watched_ids: Dict[int, str] = {}   # id(module) -> representative watch name
    for path, mod in mods:
        params = _direct_params(mod)
        if not params:
            continue
        group = f"{name}/{path}" if path else name
        rep = None
        for key, _ in params:
            if param_filter is not None and not param_filter(path, key):
                continue
            watch_name = f"{group}/{key}"
            cmap = PARAM_COLORMAPS.get(key, "viridis")
            viz.watch(watch_name, (lambda m=mod, k=key: m[k]),
                      group=group, colormap=cmap, every=every, staged=staged)
            if rep is None or key == "weight":
                rep = watch_name
        if rep is not None:
            watched_ids[id(mod)] = rep

    # Evaluate MLX parameters once on the caller's thread so the initial
    # weights are immediately snapshottable (fresh modules hold lazy
    # arrays, which cannot be forced from the worker on every platform).
    arrays = [v for _, mod in mods for _, v in _direct_params(mod)]
    if arrays and type(arrays[0]).__module__.split(".")[0] == "mlx":
        import mlx.core as mx

        mx.eval(*arrays)

    if len(watched_ids) < 2:
        return viz  # nothing to connect

    def on_done(records):
        for src_id, dst_id in _edges_from_records(records):
            viz.connect(watched_ids[src_id], watched_ids[dst_id])

    tracer = _Tracer([m for _, m in mods], watched_ids, on_done)
    tracer.patch()
    if sample_input is not None or trace is not None:
        try:
            if trace is not None:
                trace()
            else:
                module(sample_input)
        finally:
            tracer.unpatch()
    # else: lazy — the tracer finishes and unpatches itself after the
    # first top-level forward pass.
    return viz
