"""Regression tests for the viewer's pure JavaScript layout decisions."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest


APP_JS = Path(__file__).parents[1] / "src/mlx_visualizer/viewer/app.js"


def run_layout_js(body: str):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for viewer layout tests")
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("const PANEL_MIN_SIDE")
    end = source.index("function relayout()")
    pure_layout = source[start:end]
    result = subprocess.run(
        [node, "-e", pure_layout + "\n" + body],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_matrix_display_size_preserves_and_caps_aspect_ratio():
    sizes = run_layout_js(
        "console.log(JSON.stringify(["
        "matrixDisplaySize(128, 128),"
        "matrixDisplaySize(64, 256),"
        "matrixDisplaySize(1, 100000),"
        "matrixDisplaySize(100000, 1)"
        "]));"
    )
    square, rectangular, very_wide, very_tall = sizes
    assert square["w"] / square["h"] == pytest.approx(1)
    assert rectangular["w"] / rectangular["h"] == pytest.approx(4)
    assert very_wide["w"] / very_wide["h"] == pytest.approx(8)
    assert very_tall["w"] / very_tall["h"] == pytest.approx(1 / 8)


def test_architecture_items_follow_dataflow_and_keep_groups_together():
    ordered = run_layout_js(
        "const panelOrder = [3, 1, 4, 2];"
        "const panels = new Map(["
        "[1, {name:'input', group:'input', kind:'tensor', rows:8, cols:8}],"
        "[2, {name:'block/weight', group:'block', kind:'tensor', rows:8, cols:8}],"
        "[3, {name:'output', group:'output', kind:'tensor', rows:8, cols:8}],"
        "[4, {name:'block/bias', group:'block', kind:'tensor', rows:1, cols:8}]"
        "]);"
        "const edges = [['input','block/weight'],['block/weight','output']];"
        "console.log(JSON.stringify(architectureItems(panelOrder).map((item) => "
        "({name:item.p.name, depth:item.depth}))));"
    )
    assert ordered == [
        {"name": "input", "depth": 0},
        {"name": "block/weight", "depth": 1},
        {"name": "block/bias", "depth": 1},
        {"name": "output", "depth": 2},
    ]


def test_busy_architecture_depth_packs_into_multiple_lanes():
    positions = run_layout_js(
        "function panelReady() { return true; }"
        "const panelOrder = Array.from({length:12}, (_, i) => i);"
        "const panels = new Map(panelOrder.map((id) => [id, "
        "{name:`head/${id}`, group:`head-${id}`, kind:'tensor', rows:64, cols:64}]));"
        "const edges = [];"
        "layoutGraph();"
        "console.log(JSON.stringify(panelOrder.map((id) => "
        "({x:panels.get(id).x, y:panels.get(id).y}))));"
    )
    assert len({position["x"] for position in positions}) > 1
