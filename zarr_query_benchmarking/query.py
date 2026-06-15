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
result is produced in the requested ``final_format`` (``dense`` or ``csr``). The
non-selected axis is always returned in full, so a row query yields
``(count, n_cols)`` and a column query yields ``(n_rows, count)``.

Streaming model
---------------
To mimic a tool that processes a large number of chunks without holding the
whole selection in dense form, reads are **streamed in bands** along the query
axis — only one band is dense in memory at a time, and every chunk in the
selection is touched:

  - dense source -> bands are read at the store's native chunk extent along the
    query axis; each band is materialised (touched) and then released.
  - sparse source -> the selection is first assembled as a *compact* CSR matrix
    (no full densify), then densified one row-band at a time.

``final_format`` decides what is kept:

  - ``csr``   -> bands are accumulated into a single ``scipy.sparse.csr_matrix``
    (dense bands are sparsified as they arrive; a sparse source stays sparse
    end-to-end). The compact matrix is returned.
  - ``dense`` -> each band is touched and discarded; only a lightweight summary
    (shape, nnz, checksum) is returned. The full dense block is never all in RAM.

A *minor-axis* sparse query (e.g. ``var`` on a CSR store) must read every stored
nonzero — that is the inherent cost of the layout — but those stay sparse, and
the dense materialisation is still streamed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import scipy.sparse as sp
import zarr

Axis = Literal["obs", "var"]
Mode = Literal["contiguous", "random"]
FinalFormat = Literal["dense", "csr"]

# Cap on the dense footprint of a single streamed band (bytes). Bounds peak
# memory when densifying; used for sparse sources, which have no native row/col
# chunk grid (their data is chunked along the flat nonzero array).
_MAX_BAND_BYTES = 64 * 1024 * 1024

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
    if req.final_format not in ("dense", "csr"):
        raise QueryError(
            f"final_format {req.final_format!r} not supported; use 'dense' or 'csr'."
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


@dataclass
class QueryResult:
    """Outcome of a single (streamed) query.

    For ``final_format='csr'`` the compact matrix is kept in ``matrix``. For
    ``final_format='dense'`` the block is streamed and discarded, so ``matrix``
    is None and only the summary fields are populated. ``checksum`` is the sum
    of all materialised values — it forces every band to be touched and lets
    callers confirm two formats pulled the same data.
    """

    final_format: FinalFormat
    shape: tuple[int, int]
    dtype: str
    nnz: int
    checksum: float
    n_bands: int            # number of streamed bands processed
    nbytes: int             # logical size of the full result (dense-equivalent for dense)
    matrix: sp.csr_matrix | None = None


def _band_rows(n_cols: int, itemsize: int) -> int:
    """Rows per densified band so one band stays under the memory cap."""
    return max(1, _MAX_BAND_BYTES // max(1, n_cols * itemsize))


# --------------------------------------------------------------------------- #
# Source readers — yield/return work in streamable pieces, never the whole     #
# dense selection at once.                                                     #
# --------------------------------------------------------------------------- #


def _iter_dense_bands(
    node: zarr.Array, info: StoreInfo, axis: Axis, idx: np.ndarray, contiguous: bool
) -> Iterator[np.ndarray]:
    """Yield dense bands of the selection, one native chunk-extent at a time.

    For ``obs`` each band is ``(<=chunk_rows, n_cols)``; for ``var`` it is
    ``(n_rows, <=chunk_cols)``. Random selections read scattered rows/cols via
    orthogonal indexing but stay bounded to one band's worth at a time.
    """
    if axis == "obs":
        step = max(1, info.chunks[0])  # type: ignore[index]
        for a in range(0, len(idx), step):
            sel = idx[a : a + step]
            if contiguous:
                yield np.asarray(node[sel[0] : sel[-1] + 1, :])
            else:
                yield np.asarray(node.oindex[sel, :])
    else:
        step = max(1, info.chunks[1])  # type: ignore[index]
        for a in range(0, len(idx), step):
            sel = idx[a : a + step]
            if contiguous:
                yield np.asarray(node[:, sel[0] : sel[-1] + 1])
            else:
                yield np.asarray(node.oindex[:, sel])


def _select_sparse(
    group: zarr.Group, info: StoreInfo, axis: Axis, idx: np.ndarray, contiguous: bool
) -> sp.csr_matrix:
    """Assemble the selection as a *compact* CSR matrix (no full densify).

    Selecting along the matrix's major axis (rows for CSR, cols for CSC) reads
    only the relevant ``data``/``indices`` slices. Selecting along the minor axis
    reads every stored nonzero — the inherent cost of the layout — but the result
    stays sparse. Result orientation is ``(count, n_cols)`` for ``obs`` and
    ``(n_rows, count)`` for ``var``.
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

        if fmt == "csr":
            sub = sp.csr_matrix((d, i, new_indptr), shape=(len(idx), C))
        else:
            sub = sp.csc_matrix((d, i, new_indptr), shape=(R, len(idx)))
        return sub.tocsr()

    # Minor-axis selection: read the full matrix (sparse, compact), then slice.
    d = np.asarray(data[:])
    i = np.asarray(indices[:])
    cls = sp.csr_matrix if fmt == "csr" else sp.csc_matrix
    full = cls((d, i, indptr), shape=(R, C))
    sliced = full[idx, :] if axis == "obs" else full[:, idx]
    return sliced.tocsr()


# --------------------------------------------------------------------------- #
# Consumers — turn streamed pieces into a QueryResult for each final format.   #
# --------------------------------------------------------------------------- #


def _result_from_csr(matrix: sp.csr_matrix, info: StoreInfo, n_bands: int) -> QueryResult:
    matrix = matrix.tocsr()
    nbytes = int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
    return QueryResult(
        final_format="csr",
        shape=(matrix.shape[0], matrix.shape[1]),
        dtype=info.dtype,
        nnz=int(matrix.nnz),
        checksum=float(matrix.data.sum()),
        n_bands=n_bands,
        nbytes=nbytes,
        matrix=matrix,
    )


def _summary_from_dense_bands(
    bands: Iterator[np.ndarray], shape: tuple[int, int], info: StoreInfo
) -> QueryResult:
    """Touch every band (sum + nnz), then discard it — never holds the whole block."""
    total, nnz, n_bands = 0.0, 0, 0
    for band in bands:
        total += float(band.sum())
        nnz += int(np.count_nonzero(band))
        n_bands += 1
    itemsize = np.dtype(info.dtype).itemsize
    return QueryResult(
        final_format="dense",
        shape=shape,
        dtype=info.dtype,
        nnz=nnz,
        checksum=total,
        n_bands=n_bands,
        nbytes=int(shape[0]) * int(shape[1]) * itemsize,
        matrix=None,
    )


def run_query(req: QueryRequest, info: StoreInfo | None = None) -> QueryResult:
    """Execute a query end-to-end and return a :class:`QueryResult`.

    This is the full timed unit of work: stream the selection out of storage and
    convert to ``final_format``. ``info`` may be passed in to skip re-inspection.
    """
    if info is None:
        info = inspect_store(req.store, req.array_path)

    idx = _select_indices(req, _axis_len(info, req.axis))
    contiguous = req.mode == "contiguous"
    itemsize = np.dtype(info.dtype).itemsize

    root = zarr.open_group(str(req.store), mode="r")
    node = root[req.array_path]

    if info.storage_format == "dense":
        bands = _iter_dense_bands(node, info, req.axis, idx, contiguous)  # type: ignore[arg-type]
        if req.final_format == "csr":
            # Sparsify each dense band as it arrives; never hold all dense at once.
            parts = [sp.csr_matrix(b) for b in bands]
            combine = sp.vstack if req.axis == "obs" else sp.hstack
            matrix = combine(parts, format="csr")
            return _result_from_csr(matrix, info, n_bands=len(parts))
        shape = (
            (len(idx), info.n_cols) if req.axis == "obs" else (info.n_rows, len(idx))
        )
        return _summary_from_dense_bands(bands, shape, info)

    # Sparse source: assemble a compact CSR result first (no full densify).
    result = _select_sparse(node, info, req.axis, idx, contiguous)  # type: ignore[arg-type]
    if req.final_format == "csr":
        return _result_from_csr(result, info, n_bands=1)

    # Dense output from a sparse source: densify one row-band at a time.
    def _bands() -> Iterator[np.ndarray]:
        step = _band_rows(result.shape[1], itemsize)
        for a in range(0, result.shape[0], step):
            yield result[a : a + step, :].toarray()

    return _summary_from_dense_bands(_bands(), (result.shape[0], result.shape[1]), info)
