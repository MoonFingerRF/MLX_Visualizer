"""Visualize an MLX MLP's weights, activations and gradients while it trains.

Requires: pip install mlx  (Apple silicon)

The Architecture view shows the data flow through the network; each
watched tensor updates live as training progresses, with zero changes to
the training loop itself beyond registering the watches.
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
for i, layer in enumerate((model.l1, model.l2, model.l3), start=1):
    name = f"mlp/l{i}"
    viz.watch(f"{name}/W", (lambda l: lambda: l.weight)(layer), group=name)
    viz.watch(f"{name}/b", (lambda l: lambda: l.bias)(layer), group=name, colormap="coolwarm")
    viz.watch(f"{name}/dW", (lambda n: lambda: last_grads.get(n, mx.zeros((1, 1))))(f"l{i}"),
              group=name, colormap="magma")
viz.connect("mlp/l1/W", "mlp/l2/W")
viz.connect("mlp/l2/W", "mlp/l3/W")
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
