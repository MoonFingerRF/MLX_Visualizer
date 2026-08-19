import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from mlx_visualizer import Visualizer
from mlx_visualizer.introspect import _edges_from_records


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 8)
        self.l2 = nn.Linear(8, 6)
        self.l3 = nn.Linear(6, 3)

    def __call__(self, x):
        x = nn.relu(self.l1(x))
        x = nn.relu(self.l2(x))
        return self.l3(x)


class Branchy(nn.Module):
    """Two heads reading the same trunk output — a real DAG, not a chain."""

    def __init__(self):
        super().__init__()
        self.trunk = nn.Linear(4, 8)
        self.head_a = nn.Linear(8, 2)
        self.head_b = nn.Linear(8, 3)

    def __call__(self, x):
        h = self.trunk(x)
        return self.head_a(h), self.head_b(h)


def edge_set(viz):
    return {tuple(e) for e in viz.registry.structure_message()["edges"]}


def watch_names(viz):
    return {w["name"] for w in viz.registry.structure_message()["watches"]}


def test_watch_module_registers_all_params():
    viz = Visualizer(port=0)
    viz.watch_module("mlp", MLP(), sample_input=mx.zeros((2, 4)))
    assert watch_names(viz) == {
        "mlp/l1/weight", "mlp/l1/bias",
        "mlp/l2/weight", "mlp/l2/bias",
        "mlp/l3/weight", "mlp/l3/bias",
    }


def test_chain_edges_discovered_through_activations():
    viz = Visualizer(port=0)
    viz.watch_module("mlp", MLP(), sample_input=mx.zeros((2, 4)))
    assert edge_set(viz) == {
        ("mlp/l1/weight", "mlp/l2/weight"),
        ("mlp/l2/weight", "mlp/l3/weight"),
    }


def test_branching_dataflow_is_a_dag_not_a_chain():
    viz = Visualizer(port=0)
    viz.watch_module("net", Branchy(), sample_input=mx.zeros((2, 4)))
    assert edge_set(viz) == {
        ("net/trunk/weight", "net/head_a/weight"),
        ("net/trunk/weight", "net/head_b/weight"),
    }


class ReluDiamond(nn.Module):
    """Both heads read the SAME activation output — the functional relu
    hides the trunk as direct producer, but the graph must stay a DAG."""

    def __init__(self):
        super().__init__()
        self.trunk = nn.Linear(4, 8)
        self.head_a = nn.Linear(8, 2)
        self.head_b = nn.Linear(8, 3)

    def __call__(self, x):
        h = nn.relu(self.trunk(x))
        return self.head_a(h), self.head_b(h)


def test_diamond_through_activation_stays_a_dag():
    viz = Visualizer(port=0)
    viz.watch_module("net", ReluDiamond(), sample_input=mx.zeros((2, 4)))
    assert edge_set(viz) == {
        ("net/trunk/weight", "net/head_a/weight"),
        ("net/trunk/weight", "net/head_b/weight"),
    }


def test_lazy_trace_on_first_user_forward():
    viz = Visualizer(port=0)
    model = MLP()
    original_call = nn.Linear.__call__
    viz.watch_module("mlp", model)  # no sample input
    assert edge_set(viz) == set()
    assert nn.Linear.__call__ is not original_call  # instrumented
    model(mx.zeros((1, 4)))  # user's own forward pass
    assert edge_set(viz) == {
        ("mlp/l1/weight", "mlp/l2/weight"),
        ("mlp/l2/weight", "mlp/l3/weight"),
    }
    assert nn.Linear.__call__ is original_call  # instrumentation removed


def test_instrumentation_restored_after_eager_trace():
    original_call = nn.Linear.__call__
    viz = Visualizer(port=0)
    viz.watch_module("mlp", MLP(), sample_input=mx.zeros((1, 4)))
    assert nn.Linear.__call__ is original_call


def test_other_instances_unaffected_during_lazy_trace():
    viz = Visualizer(port=0)
    watched = MLP()
    bystander = MLP()
    viz.watch_module("mlp", watched)
    out = bystander(mx.zeros((1, 4)))  # runs through wrappers, records nothing...
    assert out.shape == (1, 3)
    # ...but a bystander forward at depth 0 must not consume the trace:
    # the watched model's own forward still produces the edges.
    watched(mx.zeros((1, 4)))
    assert edge_set(viz) == {
        ("mlp/l1/weight", "mlp/l2/weight"),
        ("mlp/l2/weight", "mlp/l3/weight"),
    }


def test_trace_callable_for_multi_arg_modules():
    class TwoArg(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(4, 4)
            self.k = nn.Linear(4, 4)

        def __call__(self, a, b):
            return self.q(a) @ self.k(b).T

    viz = Visualizer(port=0)
    m = TwoArg()
    viz.watch_module("attn", m, trace=lambda: m(mx.zeros((1, 4)), mx.zeros((1, 4))))
    # q and k both read raw inputs; the only relation is call order fallback.
    assert ("attn/q/weight", "attn/k/weight") in edge_set(viz)


def test_param_filter():
    viz = Visualizer(port=0)
    viz.watch_module("mlp", MLP(), sample_input=mx.zeros((1, 4)),
                     param_filter=lambda path, key: key == "weight")
    assert watch_names(viz) == {"mlp/l1/weight", "mlp/l2/weight", "mlp/l3/weight"}


def test_staged_watch_module_builds_initial_cpu_copies():
    viz = Visualizer(port=0)
    viz.watch_module(
        "mlp", MLP(), sample_input=mx.zeros((1, 4)),
        param_filter=lambda _path, key: key == "weight", staged=True,
    )
    watches = viz.registry.items()
    assert len(watches) == 3
    assert all(watch.staged for watch in watches)
    for watch in watches:
        matrix, shape = viz.registry.staged_data(watch.id)
        assert matrix is not None
        assert shape is not None


def test_rejects_non_modules():
    viz = Visualizer(port=0)
    with pytest.raises(TypeError):
        viz.watch_module("x", mx.zeros((2, 2)))


def test_edges_from_records_prefers_dataflow_over_sequence():
    # a produces t1; b and c both consume t1 (branch), even though b runs
    # between a and c in sequence.
    x, t1, t2, t3 = object(), object(), object(), object()
    records = [
        (1, [x], [t1]),
        (2, [t1], [t2]),
        (3, [t1], [t3]),
    ]
    assert _edges_from_records(records) == [(1, 2), (1, 3)]
