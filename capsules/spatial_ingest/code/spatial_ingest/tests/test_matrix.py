"""matrix: CSR zarr written and read a row band at a time; coordinate streams bucketed into CSR rows;
memory bounded by the band size, not the matrix."""
import tracemalloc

import numpy as np
import pytest
import scipy.sparse as sp
import zarr

import matrix
from fixtures_bundles import counts


def write(tmp_path, x, band_rows=7):
    group = zarr.open_group(str(tmp_path / "m.zarr"), mode="w")
    group.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})
    bands = (x[a:a + band_rows] for a in range(0, x.shape[0], band_rows))
    matrix.write_csr(group, bands, x.shape[1])
    return tmp_path / "m.zarr"


def gather(path, **kwargs):
    starts, bands = zip(*matrix.iter_rows(path, **kwargs))
    return list(starts), sp.vstack(bands).tocsr()


def test_csr_written_in_bands_reads_back_identical(tmp_path):
    x = counts(53, 17, seed=1)
    path = write(tmp_path, x)
    assert matrix.shape(path) == (53, 17)
    for kwargs in ({}, {"rows": 5}, {"budget": 20}):
        starts, back = gather(path, **kwargs)
        assert (back != x).nnz == 0 and back.dtype == np.float32
        assert starts[0] == 0 and starts == sorted(starts)
    node = zarr.open_group(str(path / "X"), mode="r")
    assert node.attrs["encoding-type"] == "csr_matrix" and node.attrs["shape"] == [53, 17]


def test_bands_respect_the_nonzero_budget(tmp_path):
    x = counts(200, 30, seed=2, density=0.5)
    path = write(tmp_path, x, band_rows=200)
    sizes = [band.nnz for _, band in matrix.iter_rows(path, budget=100)]
    assert max(sizes) <= 100 + 30 and len(sizes) > 10


def test_empty_rows_and_empty_matrix(tmp_path):
    x = sp.csr_matrix((np.array([3.0]), (np.array([2]), np.array([1]))), shape=(6, 4), dtype=np.float32)
    path = write(tmp_path, x, band_rows=2)
    assert (gather(path)[1] != x).nnz == 0


def test_dense_x_is_read_as_csr_bands(tmp_path):
    group = zarr.open_group(str(tmp_path / "d.zarr"), mode="w")
    group.attrs.update({"encoding-type": "anndata"})
    dense = counts(9, 4).toarray()
    group.create_array("X", data=dense)
    assert np.array_equal(sp.vstack([b for _, b in matrix.iter_rows(tmp_path / "d.zarr", rows=4)]).toarray(), dense)


def test_row_buckets_turn_shuffled_coordinates_into_csr(tmp_path, monkeypatch):
    monkeypatch.setattr(matrix, "BAND_NNZ", 37)
    x = counts(80, 25, seed=3).tocoo()
    order = np.random.default_rng(0).permutation(x.nnz)
    rows, cols, vals = x.row[order], x.col[order], x.data[order]
    buckets = matrix.RowBuckets(tmp_path / "b", 80, 25, x.nnz)
    assert buckets.n_bands > 3
    for a in range(0, x.nnz, 50):
        buckets.add(rows[a:a + 50], cols[a:a + 50], vals[a:a + 50])
    bands = list(buckets.bands())
    assert sum(b.shape[0] for b in bands) == 80
    assert (sp.vstack(bands).tocsr() != x.tocsr()).nnz == 0
    assert not (tmp_path / "b").exists()


def test_row_buckets_sum_duplicates_and_reject_bad_rows(tmp_path):
    buckets = matrix.RowBuckets(tmp_path / "b", 3, 2, 4)
    buckets.add(np.array([0, 0, 2]), np.array([1, 1, 0]), np.array([1.0, 2.0, 5.0]))
    assert sp.vstack(list(buckets.bands())).toarray().tolist() == [[0, 3], [0, 0], [5, 0]]
    with pytest.raises(ValueError):
        matrix.RowBuckets(tmp_path / "c", 3, 2, 4).add(np.array([3]), np.array([0]), np.array([1.0]))


def test_streaming_a_matrix_holds_one_band_not_the_matrix(tmp_path, monkeypatch):
    """Peak traced (numpy) memory while copying a matrix band by band stays a small fraction of the matrix:
    it is bounded by one band plus one zarr chunk (both shrunk here so a test-sized matrix shows it)."""
    monkeypatch.setattr(matrix, "ARRAY_CHUNK", 20_000)
    x = counts(20_000, 300, seed=4, density=0.2)                   # 1.2 M nonzeros ≈ 14 MB as float32/int32+indptr
    path = write(tmp_path, x, band_rows=1000)
    full_bytes = x.data.astype(np.float32).nbytes + x.indices.astype(np.int32).nbytes
    del x
    target = zarr.open_group(str(tmp_path / "copy.zarr"), mode="w")
    tracemalloc.start()
    matrix.write_csr(target, (band for _, band in matrix.iter_rows(path, budget=20_000)), 300)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < full_bytes / 2, (peak, full_bytes)       # /2: other tests leave traced allocations around
