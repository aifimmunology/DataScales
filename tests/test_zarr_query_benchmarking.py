"""Tests for the zarr query benchmark.

Builds tiny dense / CSR / CSC zarr v3 stores in tmp_path (matching the AnnData
encoding conventions the readers expect) so the tests don't depend on the large
fixtures in zarr_dbs/.
"""

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import zarr

from zarr_query_benchmarking.benchmark import benchmark_request
from zarr_query_benchmarking.query import (
    QueryError,
    QueryRequest,
    inspect_store,
    run_query,
    validate_request,
)

# A small, deterministic matrix with explicit zeros so sparse round-trips matter.
DENSE = np.array(
    [
        [1.0, 0.0, 2.0, 0.0, 5.0],
        [0.0, 3.0, 0.0, 0.0, 0.0],
        [4.0, 0.0, 5.0, 6.0, 0.0],
        [0.0, 0.0, 0.0, 7.0, 8.0],
    ],
    dtype=np.float32,
)


def _write_dense(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    arr = root.create_array("X", shape=DENSE.shape, chunks=(2, 2), dtype="float32")
    arr[:] = DENSE
    arr.attrs["encoding-type"] = "array"
    arr.attrs["encoding-version"] = "0.2.0"


def _write_sparse(path: Path, fmt: str) -> None:
    mat = sp.csr_matrix(DENSE) if fmt == "csr" else sp.csc_matrix(DENSE)
    root = zarr.open_group(str(path), mode="w")
    g = root.create_group("X")
    g.attrs["encoding-type"] = f"{fmt}_matrix"
    g.attrs["encoding-version"] = "0.1.0"
    g.attrs["shape"] = list(DENSE.shape)
    for name, data in (("data", mat.data), ("indices", mat.indices), ("indptr", mat.indptr)):
        a = g.create_array(name, shape=data.shape, chunks=data.shape, dtype=data.dtype)
        a[:] = data


@pytest.fixture
def stores(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "dense": tmp_path / "dense.zarr",
        "csr": tmp_path / "csr.zarr",
        "csc": tmp_path / "csc.zarr",
    }
    _write_dense(paths["dense"])
    _write_sparse(paths["csr"], "csr")
    _write_sparse(paths["csc"], "csc")
    return paths


# --------------------------------------------------------------------------- #
# inspect / validate                                                          #
# --------------------------------------------------------------------------- #


def test_inspect_detects_formats(stores: dict[str, Path]) -> None:
    assert inspect_store(stores["dense"]).storage_format == "dense"
    csr = inspect_store(stores["csr"])
    assert csr.storage_format == "csr"
    assert csr.shape == DENSE.shape
    assert csr.nnz == int((DENSE != 0).sum())
    assert inspect_store(stores["csc"]).storage_format == "csc"


def test_validate_rejects_bad_requests(stores: dict[str, Path]) -> None:
    with pytest.raises(QueryError):  # count beyond axis length
        validate_request(QueryRequest(store=stores["csr"], axis="obs", count=999))
    with pytest.raises(QueryError):  # contiguous range out of bounds
        validate_request(
            QueryRequest(store=stores["dense"], axis="obs", count=3, offset=2)
        )
    with pytest.raises(QueryError):  # missing store
        validate_request(QueryRequest(store=stores["csr"].parent / "nope.zarr"))
    with pytest.raises(QueryError):  # unsupported final format
        validate_request(QueryRequest(store=stores["csr"], final_format="csc"))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# run_query — correctness across formats, axes, modes, and final formats      #
# --------------------------------------------------------------------------- #


def _expected_slice(axis: str, idx: np.ndarray) -> np.ndarray:
    return DENSE[idx, :] if axis == "obs" else DENSE[:, idx]


@pytest.mark.parametrize("mode,kw", [("contiguous", {"offset": 1}), ("random", {"seed": 1})])
@pytest.mark.parametrize("axis", ["obs", "var"])
def test_csr_output_matches_known_data(
    stores: dict[str, Path], axis: str, mode: str, kw: dict
) -> None:
    """The csr final format gives back the exact selected data, for every source."""
    count = 2 if axis == "obs" else 3
    from zarr_query_benchmarking.query import _select_indices  # noqa: PLC0415

    idx = _select_indices(QueryRequest(store=stores["dense"], axis=axis, count=count, mode=mode, **kw), DENSE.shape[0 if axis == "obs" else 1])
    expected = _expected_slice(axis, idx)
    for fmt, p in stores.items():
        res = run_query(QueryRequest(store=p, axis=axis, count=count, mode=mode, final_format="csr", **kw))
        assert res.matrix is not None, fmt
        assert np.allclose(res.matrix.toarray(), expected), fmt


@pytest.mark.parametrize("mode,kw", [("contiguous", {"offset": 1}), ("random", {"seed": 1})])
@pytest.mark.parametrize("axis", ["obs", "var"])
def test_dense_and_csr_summaries_agree(
    stores: dict[str, Path], axis: str, mode: str, kw: dict
) -> None:
    """Dense (materialised then summarised) and csr outputs touch the same data."""
    count = 2 if axis == "obs" else 3
    for p in stores.values():
        base = dict(store=p, axis=axis, count=count, mode=mode, **kw)
        dense = run_query(QueryRequest(final_format="dense", **base))
        csr = run_query(QueryRequest(final_format="csr", **base))
        assert dense.matrix is None  # dense output is summarised, not retained
        assert dense.shape == csr.shape
        assert dense.nnz == csr.nnz
        assert np.isclose(dense.checksum, csr.checksum)


@pytest.mark.parametrize("threads", [1, 4])
def test_threads_do_not_change_results(stores: dict[str, Path], threads: int) -> None:
    """Parallel read + conversion must produce identical data regardless of threads."""
    for p in stores.values():
        for fmt in ("dense", "csr"):
            res = run_query(
                QueryRequest(store=p, axis="obs", count=4, final_format=fmt, threads=threads)
            )
            assert res.threads == threads
            if fmt == "csr":
                assert np.allclose(res.matrix.toarray(), DENSE[0:4, :])
            else:
                assert res.shape == (4, DENSE.shape[1])
                assert np.isclose(res.checksum, DENSE[0:4, :].sum())


def test_result_orientation(stores: dict[str, Path]) -> None:
    obs = run_query(QueryRequest(store=stores["csr"], axis="obs", count=2, offset=0))
    assert obs.shape == (2, DENSE.shape[1])  # (count, n_vars)
    var = run_query(QueryRequest(store=stores["csc"], axis="var", count=3, offset=0))
    assert var.shape == (DENSE.shape[0], 3)  # (n_obs, count)


def test_contiguous_matches_known_slice(stores: dict[str, Path]) -> None:
    res = run_query(
        QueryRequest(store=stores["csr"], axis="obs", count=2, offset=1, final_format="csr")
    )
    assert res.matrix is not None
    assert np.allclose(res.matrix.toarray(), DENSE[1:3, :])


def test_parallel_read_matches_direct_across_chunks(tmp_path: Path) -> None:
    """A selection spanning several native chunks reads correctly in one parallel call."""
    big = np.arange(60, dtype=np.float32).reshape(20, 3)
    root = zarr.open_group(str(tmp_path / "big.zarr"), mode="w")
    arr = root.create_array("X", shape=big.shape, chunks=(4, 3), dtype="float32")
    arr[:] = big
    arr.attrs["encoding-type"] = "array"
    res = run_query(
        QueryRequest(store=tmp_path / "big.zarr", axis="obs", count=20, final_format="csr", threads=4)
    )
    assert np.allclose(res.matrix.toarray(), big)
    assert np.isclose(res.checksum, big.sum())


# --------------------------------------------------------------------------- #
# benchmark_request                                                           #
# --------------------------------------------------------------------------- #


def test_benchmark_returns_timings(stores: dict[str, Path]) -> None:
    res = benchmark_request(
        QueryRequest(store=stores["csr"], axis="obs", count=2), repeats=3, warmup=1
    )
    assert res.ok
    assert len(res.timings_s) == 3
    assert res.result_shape == (2, DENSE.shape[1])
    assert res.min_s <= res.median_s


def test_benchmark_captures_error_without_raising(stores: dict[str, Path]) -> None:
    res = benchmark_request(QueryRequest(store=stores["csr"], axis="obs", count=999))
    assert not res.ok
    assert res.error is not None
    assert res.timings_s == []
