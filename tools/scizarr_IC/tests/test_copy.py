"""copy_group: layout/attr fidelity + banded copy correctness."""
from __future__ import annotations

import numpy as np
import zarr

from scizarr_ic.copy import copy_group


def test_copy_preserves_nested_layout_and_data(tmp_path):
    src = zarr.open_group(str(tmp_path / "s.zarr"), mode="w")
    src.attrs["root"] = "meta"
    a = src.create_array("X", shape=(20, 4), dtype="float64", chunks=(7, 4))
    a[...] = np.arange(80, dtype="float64").reshape(20, 4)
    a.attrs["encoding-type"] = "array"
    sub = src.create_group("layers")
    b = sub.create_array("counts", shape=(20, 4), dtype="int32", chunks=(5, 2))
    b[...] = np.ones((20, 4), dtype="int32")

    dst = zarr.open_group(str(tmp_path / "d.zarr"), mode="w")
    n = copy_group(src, dst)

    assert n == 2
    assert dst.attrs["root"] == "meta"
    assert dst["X"].chunks == (7, 4)
    assert dst["X"].attrs["encoding-type"] == "array"
    np.testing.assert_array_equal(dst["X"][...], a[...])
    assert dst["layers"]["counts"].chunks == (5, 2)
    np.testing.assert_array_equal(dst["layers"]["counts"][...], b[...])


def test_banded_copy_matches_full(tmp_path):
    src = zarr.open_group(str(tmp_path / "s.zarr"), mode="w")
    x = np.random.default_rng(0).random((100, 8))
    a = src.create_array("X", shape=x.shape, dtype=x.dtype, chunks=(9, 8))
    a[...] = x

    dst = zarr.open_group(str(tmp_path / "d.zarr"), mode="w")
    # tiny band_bytes forces multiple chunk-aligned bands
    copy_group(src, dst, band_bytes=512)

    np.testing.assert_array_equal(dst["X"][...], x)
