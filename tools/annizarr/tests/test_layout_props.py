from __future__ import annotations

import math
from itertools import pairwise

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from annizarr._layout import band_plan, dense_shards
from annizarr._sorting import compute_sort
from annizarr._storage._uri import bucket_prefix, canonical_location, is_remote, scheme, store_name

_DIM = st.integers(min_value=1, max_value=500)
_CHUNK = st.integers(min_value=1, max_value=600)
_FACTOR = st.integers(min_value=2, max_value=8)

# path/URI text kept to characters that survive urlparse/realpath unambiguously (no "/",
# "@", ":", "?", "#" — those are URL/netloc-meaningful and would break the round trip).
_SEGMENT = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="-_."),
    min_size=1,
    max_size=20,
)
_REMOTE_SCHEMES = ("s3", "gs", "gcs")


@st.composite
def _prefix_text(draw: st.DrawFn) -> str:
    segments = draw(st.lists(_SEGMENT, min_size=0, max_size=4))
    return "/".join(segments)


# ---- dense_shards ----------------------------------------------------------------------


@given(row_chunk=_CHUNK, col_chunk=_CHUNK, n_rows=_DIM, n_cols=_DIM, factor=st.integers(min_value=-5, max_value=1))
@settings(max_examples=200, deadline=None)
def test_dense_shards_factor_le_one_disables_sharding(
    row_chunk: int, col_chunk: int, n_rows: int, n_cols: int, factor: int
) -> None:
    layout = dense_shards(row_chunk, col_chunk, n_rows, n_cols, factor)
    assert layout.shards is None
    assert layout.chunks == (row_chunk, col_chunk)
    assert layout.block == layout.chunks


@given(row_chunk=_CHUNK, col_chunk=_CHUNK, n_rows=_DIM, n_cols=_DIM, factor=_FACTOR)
@settings(max_examples=200, deadline=None)
def test_dense_shards_sharded_invariants(row_chunk: int, col_chunk: int, n_rows: int, n_cols: int, factor: int) -> None:
    layout = dense_shards(row_chunk, col_chunk, n_rows, n_cols, factor)
    assert layout.chunks == (row_chunk, col_chunk)
    assert layout.shards is not None
    shard_row, shard_col = layout.shards

    # shard is always an integer multiple of the inner chunk (zarr's own requirement)
    assert shard_row % row_chunk == 0
    assert shard_col % col_chunk == 0

    # "capped at the number of chunks the array actually spans" (docstring) means capped at
    # the chunk-GRID extent (n_chunks * chunk), which can itself overshoot n_rows/n_cols when
    # the chunk doesn't divide the extent evenly -- shard is never bounded by the raw n_rows
    # /n_cols, only by that grid extent, with equality exactly when the factor is capped.
    n_row_chunks = math.ceil(n_rows / row_chunk)
    n_col_chunks = math.ceil(n_cols / col_chunk)
    grid_row = row_chunk * n_row_chunks
    grid_col = col_chunk * n_col_chunks
    assert shard_row <= grid_row
    assert shard_col <= grid_col
    assert (shard_row == grid_row) == (factor >= n_row_chunks)
    assert (shard_col == grid_col) == (factor >= n_col_chunks)

    # block is exactly one shard, so a block-aligned write can never straddle a shard
    assert layout.block == layout.shards


@given(row_chunk=_CHUNK, col_chunk=_CHUNK, n_rows=_DIM, n_cols=_DIM, factor=st.integers(min_value=1, max_value=8))
@settings(max_examples=200, deadline=None)
def test_dense_shards_block_tiles_cover_array_exactly(
    row_chunk: int, col_chunk: int, n_rows: int, n_cols: int, factor: int
) -> None:
    layout = dense_shards(row_chunk, col_chunk, n_rows, n_cols, factor)
    block_row, block_col = layout.block
    row_bands = band_plan(n_rows, block_row)
    col_bands = band_plan(n_cols, block_col)
    assert row_bands[0][0] == 0
    assert row_bands[-1][1] == n_rows
    assert col_bands[0][0] == 0
    assert col_bands[-1][1] == n_cols


# ---- band_plan --------------------------------------------------------------------------


@given(n_rows=st.integers(min_value=1, max_value=200_000), band_rows=st.integers(min_value=1, max_value=200_000))
@settings(max_examples=200, deadline=None)
def test_band_plan_partitions_range_exactly(n_rows: int, band_rows: int) -> None:
    bands = band_plan(n_rows, band_rows)
    assert bands
    assert bands[0][0] == 0
    assert bands[-1][1] == n_rows
    for (_, end), (next_start, _) in pairwise(bands):
        assert end == next_start  # contiguous, no gap or overlap
    for start, end in bands:
        assert start < end
        assert end - start <= band_rows
    starts = [s for s, _ in bands]
    assert starts == sorted(starts)


@given(n_rows=st.integers(min_value=-1000, max_value=0), band_rows=st.integers(min_value=1, max_value=1000))
@settings(max_examples=200, deadline=None)
def test_band_plan_empty_for_non_positive_n_rows(n_rows: int, band_rows: int) -> None:
    assert band_plan(n_rows, band_rows) == ()


@given(n_rows=st.integers(min_value=1, max_value=1000), band_rows=st.integers(min_value=-1000, max_value=0))
@settings(max_examples=200, deadline=None)
def test_band_plan_empty_for_non_positive_band_rows(n_rows: int, band_rows: int) -> None:
    assert band_plan(n_rows, band_rows) == ()


# ---- _uri ---------------------------------------------------------------------------------


@given(
    sch=st.sampled_from((*_REMOTE_SCHEMES, "file", "http", "https", "")),
    bucket=_SEGMENT,
    prefix=_prefix_text(),
)
@settings(max_examples=200, deadline=None)
def test_is_remote_matches_scheme_membership(sch: str, bucket: str, prefix: str) -> None:
    uri = f"{sch}://{bucket}/{prefix}" if prefix else f"{sch}://{bucket}/"
    assert is_remote(uri) == (scheme(uri) in {"s3", "gs", "gcs"})


@given(sch=st.sampled_from(_REMOTE_SCHEMES), bucket=_SEGMENT, prefix=_prefix_text())
@settings(max_examples=200, deadline=None)
def test_bucket_prefix_round_trips(sch: str, bucket: str, prefix: str) -> None:
    uri = f"{sch}://{bucket}/{prefix}" if prefix else f"{sch}://{bucket}/"
    got_bucket, got_prefix = bucket_prefix(uri)
    assert got_bucket == bucket
    assert got_prefix == (prefix or None)


@st.composite
def _realistic_path(draw: st.DrawFn) -> tuple[str, str]:
    # a path guaranteed to end in a non-slash tail segment, with 0..4 leading dirs and 0..4
    # trailing slashes -- the population store_name is actually meant to be called on.
    dirs = draw(st.lists(_SEGMENT, min_size=0, max_size=4))
    tail = draw(_SEGMENT)
    trailing = draw(st.integers(min_value=0, max_value=4))
    path = "/".join((*dirs, tail)) + "/" * trailing
    return path, tail


@given(data=_realistic_path())
@settings(max_examples=200, deadline=None)
def test_store_name_returns_tail_and_strips_trailing_slashes(data: tuple[str, str]) -> None:
    path, tail = data
    assert store_name(path) == tail
    assert store_name(path) != ""


def test_store_name_all_slashes_falls_back_to_input() -> None:
    assert store_name("/") == "/"
    assert store_name("///") == "///"


@given(local=st.lists(st.one_of(_SEGMENT, st.just("."), st.just("..")), min_size=1, max_size=6).map("/".join))
@settings(max_examples=200, deadline=None)
def test_canonical_location_idempotent_local(local: str) -> None:
    once = canonical_location(local)
    twice = canonical_location(once)
    assert once == twice


@given(sch=st.sampled_from(_REMOTE_SCHEMES), bucket=_SEGMENT, prefix=_prefix_text())
@settings(max_examples=200, deadline=None)
def test_canonical_location_idempotent_remote(sch: str, bucket: str, prefix: str) -> None:
    uri = f"{sch}://{bucket}/{prefix}" if prefix else f"{sch}://{bucket}/"
    once = canonical_location(uri)
    twice = canonical_location(once)
    assert once == twice


# ---- compute_sort -------------------------------------------------------------------------

_CATEGORY_VALUE = st.sampled_from(("a", "b", "c", "d", "e"))
_COLUMN_NAME = st.sampled_from(("batch", "donor", "sample", "tissue", "cohort"))


@st.composite
def _obs_frame(draw: st.DrawFn) -> tuple[pd.DataFrame, tuple[str, ...]]:
    n_key_cols = draw(st.integers(min_value=1, max_value=3))
    names = draw(st.lists(_COLUMN_NAME, min_size=n_key_cols, max_size=n_key_cols, unique=True))
    n_rows = draw(st.integers(min_value=1, max_value=200))
    data = {name: draw(st.lists(_CATEGORY_VALUE, min_size=n_rows, max_size=n_rows)) for name in names}
    return pd.DataFrame(data), tuple(names)


@given(obs_and_keys=_obs_frame())
@settings(max_examples=200, deadline=None)
def test_compute_sort_groups_are_contiguous_ordered_and_bijective(
    obs_and_keys: tuple[pd.DataFrame, tuple[str, ...]],
) -> None:
    obs, sort_by = obs_and_keys
    n = len(obs)
    perm, ranges = compute_sort(obs, sort_by)

    # perm is a bijection on range(n)
    assert sorted(perm.tolist()) == list(range(n))

    sorted_keys = list(obs.iloc[perm][list(sort_by)].itertuples(index=False, name=None))

    assert int(ranges["start"].iloc[0]) == 0
    assert int(ranges["end"].iloc[-1]) == n

    prev_end = 0
    prev_key: tuple[str, ...] | None = None
    for row in ranges.itertuples(index=False):
        start, end = int(row.start), int(row.end)
        assert start == prev_end  # contiguous, no gap
        assert start < end  # non-empty group
        key = tuple(getattr(row, c) for c in sort_by)
        # every row in this block carries exactly this key tuple
        assert all(k == key for k in sorted_keys[start:end])
        if prev_key is not None:
            assert prev_key < key  # blocks ordered ascending by the key tuple
        prev_key = key
        prev_end = end
    assert prev_end == n
