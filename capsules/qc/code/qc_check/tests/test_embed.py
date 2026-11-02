import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr

import embed
from conftest import CELLS, GENES, build_store, cohort


def chunks_touched(indptr: np.ndarray, rows: np.ndarray, chunk: int) -> set[int]:
    first, last = embed._chunk_span(indptr, rows, chunk)
    return {int(c) for a, b in zip(first, last) for c in range(a, b + 1)}


def test_reads_at_most_two_chunks_per_sample():
    rng = np.random.default_rng(1)
    sizes = [3000, 2500, 4000]
    sample = pd.Series(pd.Categorical(np.repeat(["A", "B", "C"], sizes)))
    nnz = rng.integers(50, 150, size=sum(sizes))
    indptr = np.concatenate([[0], np.cumsum(nnz)])
    chunk = 120 * 1000
    rows = embed.choose_rows(sample, indptr, chunk, per_sample=1000, max_cells=200_000, seed=3)
    codes = sample.cat.codes.to_numpy()
    for code in range(3):
        mine = rows[codes[rows] == code]
        assert len(mine) == 1000 and len(np.unique(mine)) == 1000
        assert len(chunks_touched(indptr, mine, chunk)) <= 2
        lo, hi = indptr[mine.min()], indptr[mine.max() + 1]
        assert len(set(range(lo // chunk, (hi - 1) // chunk + 1))) <= 2


def test_quota_and_total_cap():
    sample = pd.Series(pd.Categorical(np.repeat(["A", "B", "C", "D"], [10, 500, 500, 500])))
    indptr = np.arange(len(sample) + 1) * 10
    rows = embed.choose_rows(sample, indptr, 1000, per_sample=300, max_cells=600, seed=0)
    counts = pd.Series(sample.to_numpy()[rows]).value_counts()
    assert counts["A"] == 10 and counts["B"] == 150 and len(rows) == 10 + 3 * 150
    assert np.all(np.diff(rows) > 0)


def test_non_contiguous_samples_are_sampled_anyway():
    sample = pd.Series(pd.Categorical(["A", "B"] * 200))
    indptr = np.arange(401) * 5
    rows = embed.choose_rows(sample, indptr, 50, per_sample=50, max_cells=1000, seed=0)
    assert len(rows) == 100


def test_read_rows_returns_exactly_those_rows(tmp_path):
    path = build_store(tmp_path / "s.zarr", cohort(3), "zarr")
    root = zarr.open_group(path, mode="r")
    from anndata.io import read_elem
    full = read_elem(root["X"])
    indptr = root["X/indptr"][:].astype(np.int64)
    rows = np.array([0, 1, 5, 130, 131, 300, CELLS * 3 - 1])
    got = embed.read_rows(root["X"], indptr, rows)
    assert (got != full[rows]).nnz == 0


def test_embedding_is_deterministic_and_skips_tiny_inputs():
    rng = np.random.default_rng(0)
    counts = sp.random(300, len(GENES), density=0.3, format="csr", random_state=0, dtype=np.float32)
    counts.data = rng.integers(1, 20, size=counts.nnz).astype(np.float32)
    a = embed.embed(counts, np.array(GENES), seed=7)
    b = embed.embed(counts, np.array(GENES), seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert a.shape == (300, 3) and a["leiden"].nunique() > 1
    assert embed.embed(counts[:10], np.array(GENES), seed=7) is None
