import numpy as np
import pytest

from mlx_visualizer.adapter import pick_value, to_numpy_2d
from mlx_visualizer.snapshot import block_mean, downsample, take_snapshot


def test_block_mean_exact_divisible():
    a = np.arange(16, dtype=np.float64).reshape(4, 4)
    out = block_mean(a, 2, 2)
    expected = np.array([[2.5, 4.5], [10.5, 12.5]], dtype=np.float32)
    np.testing.assert_allclose(out, expected)


def test_block_mean_non_divisible_edges():
    a = np.arange(15, dtype=np.float64).reshape(3, 5)
    out = block_mean(a, 2, 2)
    # Edge blocks average only the cells that exist.
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out[0, 0], np.mean(a[0:2, 0:2]))
    np.testing.assert_allclose(out[0, 2], np.mean(a[0:2, 4:5]))
    np.testing.assert_allclose(out[1, 0], np.mean(a[2:3, 0:2]))
    np.testing.assert_allclose(out[1, 2], np.mean(a[2:3, 4:5]))


def test_block_mean_banded_matches_unbanded():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(1000, 700))
    small_bands = block_mean(a, 3, 5, band_rows=7)
    one_band = block_mean(a, 3, 5, band_rows=10**9)
    np.testing.assert_allclose(small_bands, one_band, rtol=1e-5)


def test_downsample_caps_size():
    a = np.ones((5000, 3000))
    out = downsample(a, 1024)
    assert out.shape[0] <= 1024 and out.shape[1] <= 1024
    np.testing.assert_allclose(out, 1.0)


def test_downsample_small_passthrough_is_contiguous_f32():
    a = np.arange(12).reshape(3, 4).astype(np.float64)[:, ::2]
    out = downsample(a, 1024)
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(out, a)


def test_snapshot_stats_and_fingerprint():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    s1 = take_snapshot(a, a.shape)
    assert s1.vmin == 1.0 and s1.vmax == 4.0
    assert s1.mean == pytest.approx(2.5)
    s2 = take_snapshot(a, a.shape)
    assert s1.fingerprint == s2.fingerprint
    s3 = take_snapshot(a + 1, a.shape)
    assert s3.fingerprint != s1.fingerprint


def test_snapshot_handles_nan_and_inf():
    a = np.array([[np.nan, 1.0], [np.inf, 3.0]])
    s = take_snapshot(a, a.shape)
    assert s.nan_count == 2
    assert s.vmin == 1.0 and s.vmax == 3.0


def test_snapshot_all_nan():
    a = np.full((3, 3), np.nan)
    s = take_snapshot(a, a.shape)
    assert s.vmin == 0.0 and s.vmax == 0.0
    assert s.nan_count == 9


def test_to_numpy_2d_shapes():
    m, shape = to_numpy_2d(np.zeros((3, 4)))
    assert m.shape == (3, 4) and shape == (3, 4)
    v, shape = to_numpy_2d(np.zeros(7))
    assert v.shape == (1, 7) and shape == (7,)
    t, shape = to_numpy_2d(np.zeros((2, 3, 5)))
    assert t.shape == (6, 5) and shape == (2, 3, 5)
    s, shape = to_numpy_2d(np.float64(3.0))
    assert s.shape == (1, 1)


def test_to_numpy_2d_casts_ints():
    m, _ = to_numpy_2d(np.arange(6).reshape(2, 3))
    assert np.issubdtype(m.dtype, np.floating)


def test_pick_value_matrix_vector_and_3d():
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert pick_value(a, 1, 2) == 6.0
    assert pick_value(lambda: a, 2, 3) == 11.0
    v = np.arange(5, dtype=np.float64)
    assert pick_value(v, 0, 3) == 3.0
    t = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    flat = t.reshape(6, 4)
    assert pick_value(t, 4, 1) == flat[4, 1]
