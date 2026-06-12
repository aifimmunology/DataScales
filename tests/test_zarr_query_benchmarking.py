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
        validate_request(QueryRequest(store=stores["csr"], final_format="csr"))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# run_query — correctness across formats, axes, and modes                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode,kw", [("contiguous", {"offset": 1}), ("random", {"seed": 1})])
@pytest.mark.parametrize("axis", ["obs", "var"])
def test_all_formats_agree(stores: dict[str, Path], axis: str, mode: str, kw: dict) -> None:
    count = 2 if axis == "obs" else 3
    out = {
        fmt: run_query(QueryRequest(store=p, axis=axis, count=count, mode=mode, **kw))
        for fmt, p in stores.items()
    }
    assert np.allclose(out["csr"], out["dense"])
    assert np.allclose(out["csc"], out["dense"])


def test_result_orientation(stores: dict[str, Path]) -> None:
    obs = run_query(QueryRequest(store=stores["csr"], axis="obs", count=2, offset=0))
    assert obs.shape == (2, DENSE.shape[1])  # (count, n_vars)
    var = run_query(QueryRequest(store=stores["csc"], axis="var", count=3, offset=0))
    assert var.shape == (DENSE.shape[0], 3)  # (n_obs, count)


def test_contiguous_matches_known_slice(stores: dict[str, Path]) -> None:
    out = run_query(QueryRequest(store=stores["csr"], axis="obs", count=2, offset=1))
    assert np.allclose(out, DENSE[1:3, :])


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
