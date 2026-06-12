"""Locate, validate, and pull data from a single-cell AnnData zarr store.

The reading path is intentionally low-level: we open the zarr arrays directly
and do the index math ourselves rather than going through anndata. This isolates
the cost of the storage layout / chunking strategy from AnnData overhead, which
is the whole point of the benchmark.

Supported ``X`` encodings (zarr v3, AnnData conventions):
  - ``array``       — dense 2D array
  - ``csr_matrix``  — group with ``data`` / ``indices`` / ``indptr``, row-major
  - ``csc_matrix``  — same, column-major

A query selects ``count`` items along one axis (``obs`` = rows/cells,
``var`` = cols/genes), either as a contiguous block or a random sample, and the
result is materialised in the requested ``final_format``. The non-selected axis
is always returned in full, so a row query yields ``(count, n_cols)`` and a
column query yields ``(n_rows, count)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import scipy.sparse as sp
import zarr

Axis = Literal["obs", "var"]
Mode = Literal["contiguous", "random"]
FinalFormat = Literal["dense"]

# Encodings we know how to read. CSC is included because it is the exact
# symmetric counterpart of CSR (major axis = columns instead of rows).
_SPARSE_ENCODINGS = {"csr_matrix": "csr", "csc_matrix": "csc"}


class QueryError(ValueError):
    """Raised when a request is invalid or a store cannot be read as expected."""


@dataclass
class QueryRequest:
    """A single thing to pull from a store, before any timing happens."""

    store: Path
    axis: Axis = "obs"
    count: int = 1000
    mode: Mode = "contiguous"
    final_format: FinalFormat = "dense"
    array_path: str = "X"
    offset: int = 0          # start index for contiguous mode
    seed: int = 0            # rng seed for random mode

    def __post_init__(self) -> None:
        self.store = Path(self.store)


@dataclass
class StoreInfo:
    """What we learned about an ``X`` node by inspecting it (cheap, no data read)."""

    path: Path
    array_path: str
    storage_format: str           # "dense" | "csr" | "csc"
    shape: tuple[int, int]        # (n_rows, n_cols) == (n_obs, n_vars)
    dtype: str
    chunks: tuple[int, ...] | None
    nnz: int | None = None        # total stored nonzeros (sparse only)

    @property
    def n_rows(self) -> int:
        return self.shape[0]

    @property
    def n_cols(self) -> int:
        return self.shape[1]


def inspect_store(store: Path, array_path: str = "X") -> StoreInfo:
    """Open ``store[array_path]`` read-only and report its layout. Reads no bulk data."""
    store = Path(store)
    if not store.exists():
        raise QueryError(f"Store does not exist: {store}")

    root = zarr.open_group(str(store), mode="r")
    try:
        node = root[array_path]
    except KeyError as exc:
        raise QueryError(f"'{array_path}' not found in store {store}") from exc

    if isinstance(node, zarr.Array):
        return StoreInfo(
            path=store,
            array_path=array_path,
            storage_format="dense",
            shape=tuple(int(s) for s in node.shape),  # type: ignore[arg-type]
            dtype=str(node.dtype),
            chunks=tuple(int(c) for c in node.chunks),
        )

    # Otherwise it is a group — expect a sparse AnnData encoding.
    enc = node.attrs.get("encoding-type")
    if enc not in _SPARSE_ENCODINGS:
        raise QueryError(
            f"'{array_path}' in {store} is a group with unsupported "
            f"encoding-type={enc!r}; expected one of {sorted(_SPARSE_ENCODINGS)} "
            "or a dense array."
        )
    shape = tuple(int(s) for s in node.attrs["shape"])
    data = node["data"]
    return StoreInfo(
        path=store,
        array_path=array_path,
        storage_format=_SPARSE_ENCODINGS[enc],
        shape=shape,  # type: ignore[arg-type]
        dtype=str(data.dtype),
        chunks=tuple(int(c) for c in data.chunks),
        nnz=int(data.shape[0]),
    )


def _axis_len(info: StoreInfo, axis: Axis) -> int:
    return info.n_rows if axis == "obs" else info.n_cols


def validate_request(req: QueryRequest) -> StoreInfo:
    """Check a request is satisfiable *before* timing. Returns the store info.

    Raises QueryError with a clear message on any problem so the benchmark
    never times a query that would fail or read out of bounds.
    """
    info = inspect_store(req.store, req.array_path)

    if req.axis not in ("obs", "var"):
        raise QueryError(f"axis must be 'obs' or 'var', got {req.axis!r}")
    if req.mode not in ("contiguous", "random"):
        raise QueryError(f"mode must be 'contiguous' or 'random', got {req.mode!r}")
    if req.final_format != "dense":
        raise QueryError(
            f"final_format {req.final_format!r} not supported yet; only 'dense'."
        )

    n = _axis_len(info, req.axis)
    if req.count < 1:
        raise QueryError(f"count must be >= 1, got {req.count}")
    if req.count > n:
        raise QueryError(
            f"count={req.count} exceeds {req.axis} length {n} for store {req.store}"
        )
    if req.mode == "contiguous":
        if req.offset < 0 or req.offset + req.count > n:
            raise QueryError(
                f"contiguous range [{req.offset}:{req.offset + req.count}] "
                f"out of bounds for {req.axis} length {n}"
            )
    return info


def _select_indices(req: QueryRequest, axis_len: int) -> np.ndarray:
    """Resolve a request into a sorted array of integer indices along its axis."""
    if req.mode == "contiguous":
        return np.arange(req.offset, req.offset + req.count)
    rng = np.random.default_rng(req.seed)
    # replace=False guarantees distinct items; sorted for predictable zarr reads.
    return np.sort(rng.choice(axis_len, size=req.count, replace=False))


# --------------------------------------------------------------------------- #
# Readers — each returns a dense 2D numpy array oriented (selected-axis, other) #
# for axis="obs" and (other, selected-axis) for axis="var".                    #
# --------------------------------------------------------------------------- #


def _read_dense(node: zarr.Array, axis: Axis, idx: np.ndarray, contiguous: bool) -> np.ndarray:
    if axis == "obs":
        if contiguous:
            return np.asarray(node[idx[0] : idx[-1] + 1, :])
        return np.asarray(node.oindex[idx, :])
    if contiguous:
        return np.asarray(node[:, idx[0] : idx[-1] + 1])
    return np.asarray(node.oindex[:, idx])


def _read_sparse(
    group: zarr.Group, info: StoreInfo, axis: Axis, idx: np.ndarray, contiguous: bool
) -> np.ndarray:
    """Pull rows/cols from a CSR/CSC store and densify.

    Selecting along the matrix's *major* axis (rows for CSR, cols for CSC) is the
    cheap path: we read only the relevant slices of ``data``/``indices``. Selecting
    along the *minor* axis is the expensive path — the whole matrix is read and
    then sliced — which is exactly the cost this benchmark exists to expose.
    """
    fmt = info.storage_format          # "csr" or "csc"
    major_axis: Axis = "obs" if fmt == "csr" else "var"
    R, C = info.shape

    data = group["data"]
    indices = group["indices"]
    indptr = np.asarray(group["indptr"][:])  # small: one int per major + 1

    if axis == major_axis:
        d_parts, i_parts, lengths = [], [], []
        if contiguous:
            lo, hi = int(indptr[idx[0]]), int(indptr[idx[-1] + 1])
            d_parts.append(np.asarray(data[lo:hi]))
            i_parts.append(np.asarray(indices[lo:hi]))
            lengths = list(np.diff(indptr[idx[0] : idx[-1] + 2]))
        else:
            for j in idx:
                lo, hi = int(indptr[j]), int(indptr[j + 1])
                d_parts.append(np.asarray(data[lo:hi]))
                i_parts.append(np.asarray(indices[lo:hi]))
                lengths.append(hi - lo)
        d = np.concatenate(d_parts) if d_parts else np.empty(0, dtype=data.dtype)
        i = np.concatenate(i_parts) if i_parts else np.empty(0, dtype=indices.dtype)
        new_indptr = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)

        # CSR: indptr indexes rows -> result is (n_selected, C).
        # CSC: indptr indexes cols -> result is (R, n_selected).
        if fmt == "csr":
            sub = sp.csr_matrix((d, i, new_indptr), shape=(len(idx), C))
        else:
            sub = sp.csc_matrix((d, i, new_indptr), shape=(R, len(idx)))
        return sub.toarray()

    # Minor-axis selection: read the full matrix, then slice (expensive on purpose).
    d = np.asarray(data[:])
    i = np.asarray(indices[:])
    cls = sp.csr_matrix if fmt == "csr" else sp.csc_matrix
    full = cls((d, i, indptr), shape=(R, C))
    if axis == "obs":
        return full[idx, :].toarray()
    return full[:, idx].toarray()


def run_query(req: QueryRequest, info: StoreInfo | None = None) -> np.ndarray:
    """Execute a query end-to-end and return a dense numpy array.

    This is the full timed unit of work: read from storage + convert to the
    final (dense) format. ``info`` may be passed in to skip re-inspection.
    """
    if info is None:
        info = inspect_store(req.store, req.array_path)

    axis_len = _axis_len(info, req.axis)
    idx = _select_indices(req, axis_len)
    contiguous = req.mode == "contiguous"

    root = zarr.open_group(str(req.store), mode="r")
    node = root[req.array_path]

    if info.storage_format == "dense":
        return _read_dense(node, req.axis, idx, contiguous)  # type: ignore[arg-type]
    return _read_sparse(node, info, req.axis, idx, contiguous)  # type: ignore[arg-type]
