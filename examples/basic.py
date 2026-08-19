"""Minimal example: watch a few live arrays.

Run, open the printed URL, and watch the matrices evolve while the
compute loop below runs at full speed.
"""

import time

import numpy as np

from mlx_visualizer import Visualizer

# Works identically with MLX arrays:
#   import mlx.core as mx
#   w = mx.random.normal((2048, 2048))
rng = np.random.default_rng(0)
state = rng.normal(size=(2048, 2048)).astype(np.float32)
velocity = np.zeros_like(state)
signal = np.zeros(4096, dtype=np.float32)

viz = Visualizer(interval=0.2)
viz.watch("sim/state", lambda: state, colormap="viridis")
viz.watch("sim/velocity", lambda: velocity, colormap="coolwarm")
viz.watch("sim/signal", lambda: signal, colormap="turbo")
viz.connect("sim/state", "sim/velocity")
url = viz.start()
print(f"open {url}")

t = 0
try:
    while True:
        # The "computation" — the visualizer never blocks this loop.
        velocity = 0.99 * velocity + 0.01 * rng.normal(size=state.shape).astype(np.float32)
        state = state + 0.1 * velocity
        x = np.linspace(0, 20 * np.pi, signal.size, dtype=np.float32)
        signal = np.sin(x + t * 0.3) * np.cos(x * 0.1 + t * 0.05)
        t += 1
        time.sleep(0.05)
except KeyboardInterrupt:
    viz.stop()
