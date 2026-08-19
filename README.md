# MLX Visualizer

Live, asynchronous visualization of large matrices and vectors — MLX arrays,
NumPy arrays, or torch tensors — in a beautiful, space-efficient web view that
also shows the architecture connecting them. Designed so that watching your
data costs your computation essentially nothing.

![grid view](docs/grid.png)

## Highlights

- **Asynchronous by construction.** The visualizer owns a background thread
  with its own event loop. Your compute thread only registers watches (a dict
  insert behind a lock); snapshotting, reduction, encoding, and network I/O
  all happen off your critical path, and NumPy releases the GIL for the heavy
  reductions.
- **Extremely large matrices.** A 100,000×100,000 matrix is reduced with
  exact block means computed in fixed-size row bands, so peak extra memory is
  bounded regardless of input size. Each capture tick has a wall-clock budget;
  work that doesn't fit is carried over round-robin, so one huge tensor can
  never starve the stream.
- **Batched, low-level rendering.** Every tensor lives on one layer of a
  single `R32F TEXTURE_2D_ARRAY`; the entire scene — any number of panels —
  is drawn with **one instanced WebGL2 draw call**, with colormapping done in
  the fragment shader via a LUT atlas. Rendering cost is independent of how
  many matrices are on screen.
- **Space-efficient, beautiful layout.** Shelf-packed grid sized by log-scale
  of tensor dimensions, or an **Architecture** mode that lays tensors out as a
  layered dataflow graph with edges you declare via `viz.connect(...)`.
- **Change-aware streaming.** Frames are fingerprinted (CRC32 of the reduced
  image); unchanged tensors are never re-sent. Slow clients get per-connection
  queues that drop stale frames instead of back-pressuring the pipeline.
- **Zero dependencies** beyond NumPy: the HTTP + WebSocket server and the
  binary protocol are built in.

## Install

```bash
pip install -e .
```

## Quick start

```python
from mlx_visualizer import Visualizer

viz = Visualizer()                       # or Visualizer(port=0) for a random port
viz.watch("weights/w1", lambda: model.w1)          # callables are re-resolved live
viz.watch("weights/b1", lambda: model.b1, colormap="coolwarm")
viz.connect("weights/w1", "weights/b1")            # architecture edge
url = viz.start()                        # non-blocking; open the printed URL

... your training / compute loop runs at full speed ...

viz.stop()
```

Watch anything: MLX arrays, NumPy arrays, torch tensors, or zero-argument
callables returning one. Vectors render as strips, N-D tensors are collapsed
to 2-D, and matrices larger than `max_side` (default 1024) are shown at a
level-of-detail with exact block means (the label shows `LOD`, and hovering
requests the exact element value from the live array over the socket).

![architecture view](docs/graph.png)

## API

```python
Visualizer(host="127.0.0.1", port=8791, *, interval=0.25, max_side=1024,
           tick_budget=0.030)
```

| Method | Description |
| --- | --- |
| `watch(name, provider, *, group="", colormap="viridis", every=1)` | Track an array or provider callable. `every=N` samples on every Nth tick. |
| `unwatch(name)` | Stop tracking. |
| `connect(src, dst)` | Declare a dataflow edge for the Architecture view. |
| `start(open_browser=False)` | Start the worker thread + server; returns the URL. |
| `stop()` | Shut down. Also usable as a context manager. |

A module-level default instance is available for quick use:
`mlx_visualizer.watch(...)`, `mlx_visualizer.connect(...)`,
`mlx_visualizer.start()`.

Colormaps: `viridis`, `magma`, `turbo`, `coolwarm`, `gray`. NaNs render
magenta and are counted in the panel stats.

## How it stays fast

1. **Capture** (worker thread, budgeted): resolve provider → convert once →
   banded block-mean reduction to ≤ `max_side`² → stats + CRC32 fingerprint.
2. **Transport**: binary frames (12-byte header + JSON meta padded to 4 bytes
   + raw float32 payload) over a built-in WebSocket; unchanged fingerprints
   are skipped, and nothing at all runs while no client is connected.
3. **Render** (browser): float payloads go straight into a texture-array
   layer (zero-copy `Float32Array` view); the whole scene is one
   `drawArraysInstanced` call; normalization and colormapping run in the
   fragment shader; labels/edges are DOM/SVG positioned per frame.

## Examples

- `examples/basic.py` — live simulation of a 2048² field, its velocity, and a
  4096-element signal.
- `examples/mlx_training.py` — an MLX MLP with weights, biases, and gradients
  watched per layer and connected in the Architecture view.

## Development

```bash
pip install -e .[dev]
pytest
```

The test suite covers the reduction math (banded vs. unbanded equivalence,
non-divisible edges, NaN handling), the wire protocol round trip, the
registry/graph, and full end-to-end tests over a real socket: hello →
snapshots → exact-value pick, LOD for a 4096² matrix, and the
no-resend-when-unchanged guarantee.
