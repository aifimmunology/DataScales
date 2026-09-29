from __future__ import annotations

import annizarr._layout as _layout
from annizarr._layout import DenseLayout, band_plan, dense_shards


def test_dense_shards_unsharded() -> None:
    layout = dense_shards(row_chunk=8, col_chunk=4, n_rows=100, n_cols=50, factor=1)
    assert layout == DenseLayout(chunks=(8, 4), shards=None, block=(8, 4))


def test_dense_shards_sharded() -> None:
    layout = dense_shards(row_chunk=2, col_chunk=2, n_rows=6, n_cols=3, factor=2)
    # 6 rows / chunk 2 -> 3 chunks; factor 2 fits -> shard_row = 4
    # 3 cols / chunk 2 -> 2 chunks; factor 2 fits -> shard_col = 4
    assert layout.chunks == (2, 2)
    assert layout.shards == (4, 4)
    assert layout.block == layout.shards


def test_dense_shards_factor_capped_at_array_extent() -> None:
    # only 2 row-chunks exist (4 rows / chunk 2); factor 8 is capped to 2 -> shard_row = 4
    layout = dense_shards(row_chunk=2, col_chunk=2, n_rows=4, n_cols=2, factor=8)
    assert layout.shards == (4, 2)


def test_band_plan_exact_division() -> None:
    assert band_plan(10, 5) == ((0, 5), (5, 10))


def test_band_plan_clips_last_band() -> None:
    assert band_plan(7, 3) == ((0, 3), (3, 6), (6, 7))


def test_band_plan_single_band_larger_than_rows() -> None:
    assert band_plan(3, 10) == ((0, 3),)


def test_band_plan_empty() -> None:
    assert band_plan(0, 5) == ()
    assert band_plan(5, 0) == ()
    assert band_plan(-1, 5) == ()


def test_batch_bytes_is_monkeypatchable_module_attribute(monkeypatch) -> None:
    monkeypatch.setattr(_layout, "BATCH_BYTES", 1024)
    assert _layout.BATCH_BYTES == 1024
