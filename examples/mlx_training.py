"""Visualize an MLX MLP's weights and gradients while it trains.

Requires: pip install mlx

One `watch_module` call watches every parameter of the model AND
captures the architecture automatically by tracing the first forward
pass — no manual `connect` calls needed. The Architecture view then
shows the real dataflow through the network, updating live as training
progresses.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_visualizer import Visualizer

BATCH, DIN, HID, DOUT = 256, 784, 512, 10


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(DIN, HID)
        self.l2 = nn.Linear(HID, HID)
        self.l3 = nn.Linear(HID, DOUT)

    def __call__(self, x):
        x = nn.relu(self.l1(x))
        x = nn.relu(self.l2(x))
        return self.l3(x)


model = MLP()
optimizer = optim.Adam(learning_rate=1e-3)
last_grads = {}

viz = Visualizer(interval=0.25)
# Watches l1/l2/l3 weight+bias and auto-discovers l1 → l2 → l3 by
# tracing this sample forward pass. Omit sample_input and the first
# real training step is traced instead.
viz.watch_module("mlp", model, sample_input=mx.zeros((1, DIN)))
# Gradients are not module parameters, so watch them explicitly.
for i in (1, 2, 3):
    viz.watch(f"mlp/l{i}/dW",
              (lambda n: lambda: last_grads.get(n, mx.zeros((1, 1))))(f"l{i}"),
              group=f"mlp/l{i}", colormap="magma")
print("open", viz.start())


def loss_fn(model, x, y):
    return nn.losses.cross_entropy(model(x), y).mean()


loss_and_grad = nn.value_and_grad(model, loss_fn)

step = 0
while True:
    x = mx.random.normal((BATCH, DIN))
    y = mx.random.randint(0, DOUT, (BATCH,))
    loss, grads = loss_and_grad(model, x, y)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    for i in (1, 2, 3):
        last_grads[f"l{i}"] = grads[f"l{i}"]["weight"]
    step += 1
    if step % 100 == 0:
        print(f"step {step}  loss {loss.item():.4f}")
