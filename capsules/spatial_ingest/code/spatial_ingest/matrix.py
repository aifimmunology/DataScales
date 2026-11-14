"""Count matrices on disk, a row band at a time: CSR AnnData zarr groups written and read chunk-wise,
and coordinate (row, col, value) streams turned into CSR rows through per-band bucket files on scratch.
Nothing here holds a whole matrix in memory: the largest object is one row band (bounded by `BAND_NNZ`
nonzeros) plus the row pointer (8 bytes per row)."""
from __future__ import annotations

import math
import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import zarr

BAND_NNZ = 20_000_000          # nonzeros per row band read or bucketed (≈ 160–240 MB in memory)
ARRAY_CHUNK = 4_000_000        # zarr chunk length of data/indices
CSR_ATTRS = {"encoding-type": "csr_matrix", "encoding-version": "0.1.0"}


class CSRWriter:
    """Appends CSR row bands to `group[key]` (an AnnData `csr_matrix` element) without holding earlier
    bands: data/indices are written to disk in whole zarr chunks (the remainder waits for the next band),
    the row pointer is kept (n_rows + 1 int64) and written on `close`."""

    def __init__(self, group: zarr.Group, n_cols: int, dtype=np.float32, key: str = "X"):
        self.n_cols, self.dtype = int(n_cols), np.dtype(dtype)
        self.node = group.create_group(key, overwrite=True)
        index_dtype = np.int32 if n_cols < np.iinfo(np.int32).max else np.int64
        self.chunk = ARRAY_CHUNK
        self.data = self.node.create_array("data", shape=(0,), chunks=(self.chunk,), dtype=self.dtype)
        self.indices = self.node.create_array("indices", shape=(0,), chunks=(self.chunk,), dtype=index_dtype)
        self._indptr: list[np.ndarray] = [np.zeros(1, dtype=np.int64)]
        self._pending: list[tuple[np.ndarray, np.ndarray]] = []
        self._pending_n = 0
        self.written = 0
        self.nnz = 0
        self.n_rows = 0

    def append(self, rows: sp.csr_matrix) -> None:
        rows = sp.csr_matrix(rows)
        if rows.shape[1] != self.n_cols:
            raise ValueError(f"row band has {rows.shape[1]} columns, the matrix {self.n_cols}")
        rows.sum_duplicates()
        if rows.nnz:
            self._pending.append((rows.data.astype(self.dtype, copy=False),
                                  rows.indices.astype(self.indices.dtype, copy=False)))
            self._pending_n += rows.nnz
        self._indptr.append(rows.indptr[1:].astype(np.int64) + self.nnz)
        self.nnz += rows.nnz
        self.n_rows += rows.shape[0]
        if self._pending_n >= self.chunk:
            self._flush(whole_chunks=True)

    def _flush(self, whole_chunks: bool) -> None:
        if not self._pending_n:
            return
        data = np.concatenate([d for d, _ in self._pending])
        indices = np.concatenate([i for _, i in self._pending])
        n = (len(data) // self.chunk) * self.chunk if whole_chunks else len(data)
        if n:
            end = self.written + n
            self.data.resize((end,))
            self.indices.resize((end,))
            self.data[self.written:end] = data[:n]
            self.indices[self.written:end] = indices[:n]
            self.written = end
        self._pending = [(data[n:], indices[n:])] if n < len(data) else []
        self._pending_n = len(data) - n

    def close(self) -> None:
        self._flush(whole_chunks=False)
        indptr = np.concatenate(self._indptr)
        self.node.create_array("indptr", data=indptr, chunks=(max(1, min(len(indptr), 1 << 22)),))
        self.node.attrs.update({**CSR_ATTRS, "shape": [self.n_rows, self.n_cols]})
        for name in ("data", "indices", "indptr"):
            self.node[name].attrs.update({"encoding-type": "array", "encoding-version": "0.2.0"})


def write_csr(group: zarr.Group, bands: Iterable[sp.csr_matrix], n_cols: int, dtype=np.float32,
              key: str = "X") -> tuple[int, int]:
    """Every band of `bands` appended as `group[key]`; returns (n_rows, nnz)."""
    writer = CSRWriter(group, n_cols, dtype, key)
    for band in bands:
        writer.append(band)
    writer.close()
    return writer.n_rows, writer.nnz


def x_node(group_path: Path | zarr.Group):
    """The X element of an AnnData zarr group (or the matrix node itself when given one)."""
    if not isinstance(group_path, (str, Path)) and not isinstance(group_path, zarr.Group):
        return group_path                                   # a matrix node already (zarr array or CSR-like)
    root = group_path if isinstance(group_path, zarr.Group) else zarr.open(str(group_path), mode="r")
    if isinstance(root, zarr.Group) and "X" in root and root.attrs.get("encoding-type") == "anndata":
        return root["X"]
    if isinstance(root, zarr.Group) and "X" in root and "indptr" not in root:
        return root["X"]
    return root


def shape(group_path: Path | zarr.Group) -> tuple[int, int]:
    node = x_node(group_path)
    if isinstance(node, zarr.Array):
        return tuple(node.shape)
    return tuple(int(v) for v in node.attrs["shape"])


def read_indptr(group_path: Path | zarr.Group) -> np.ndarray:
    node = x_node(group_path)
    if isinstance(node, zarr.Array):
        raise ValueError("dense X has no row pointer")
    return np.asarray(node["indptr"][:], dtype=np.int64)


def rows_for(indptr: np.ndarray, budget: int | None = None, max_rows: int | None = None) -> list[tuple[int, int]]:
    """Row ranges whose nonzeros stay near `budget` (default `BAND_NNZ`; at least one row each)."""
    budget = budget or BAND_NNZ
    n = len(indptr) - 1
    out, start = [], 0
    while start < n:
        stop = int(np.searchsorted(indptr, indptr[start] + budget, side="right")) - 1
        stop = max(stop, start + 1)
        if max_rows:
            stop = min(stop, start + max_rows)
        stop = min(stop, n)
        out.append((start, stop))
        start = stop
    return out


def iter_rows(group_path: Path | zarr.Group, rows: int | None = None,
              budget: int | None = None) -> Iterator[tuple[int, sp.csr_matrix]]:
    """(first row, CSR band) over a CSR (or dense) X; bands hold at most `rows` rows and about
    `budget` nonzeros. `group_path` is an AnnData zarr group or its X element."""
    node = x_node(group_path)
    n_rows, n_cols = shape(node)
    if isinstance(node, zarr.Array):
        step = rows or max(1, (budget or BAND_NNZ) // max(n_cols, 1))
        for start in range(0, n_rows, step):
            yield start, sp.csr_matrix(np.asarray(node[start:start + step]))
        return
    if node.attrs.get("encoding-type") != "csr_matrix":
        raise ValueError(f"X is {node.attrs.get('encoding-type')}, not csr_matrix")
    indptr = read_indptr(node)
    data, indices = node["data"], node["indices"]
    for start, stop in rows_for(indptr, budget, rows):
        a, b = int(indptr[start]), int(indptr[stop])
        yield start, sp.csr_matrix((data[a:b], indices[a:b], indptr[start:stop + 1] - a), shape=(stop - start, n_cols))


class RowBuckets:
    """Coordinate triples → CSR rows, out of core: `add` spools (row, col, value) chunks into one
    binary file per row band under `folder`; `bands` then yields each band as CSR, in row order.
    Duplicate coordinates are summed."""

    def __init__(self, folder: Path, n_rows: int, n_cols: int, nnz_hint: int, dtype=np.float32):
        self.folder, self.n_rows, self.n_cols, self.dtype = Path(folder), int(n_rows), int(n_cols), np.dtype(dtype)
        shutil.rmtree(self.folder, ignore_errors=True)
        self.folder.mkdir(parents=True)
        n_bands = max(1, math.ceil(max(nnz_hint, 1) / BAND_NNZ))
        self.band_rows = max(1, math.ceil(self.n_rows / n_bands))
        self.n_bands = max(1, math.ceil(self.n_rows / self.band_rows))
        self._files = {}

    def _path(self, band: int, field: str) -> Path:
        return self.folder / f"{band:06d}.{field}"

    def add(self, rows: np.ndarray, cols: np.ndarray, values: np.ndarray) -> None:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            return
        if rows.min() < 0 or rows.max() >= self.n_rows:
            raise ValueError(f"row index out of range 0..{self.n_rows - 1}")
        band = rows // self.band_rows
        order = np.argsort(band, kind="stable")
        band, rows = band[order], rows[order]
        cols = np.asarray(cols, dtype=np.int32)[order]
        values = np.asarray(values)[order].astype(self.dtype, copy=False)
        cuts = np.flatnonzero(np.diff(band)) + 1
        for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, len(band)]):
            b = int(band[lo])
            for field, arr in (("r", (rows[lo:hi] - b * self.band_rows).astype(np.int32)), ("c", cols[lo:hi]),
                               ("v", values[lo:hi])):
                with open(self._path(b, field), "ab") as f:
                    arr.tofile(f)

    def bands(self) -> Iterator[sp.csr_matrix]:
        try:
            for b in range(self.n_bands):
                n = min(self.band_rows, self.n_rows - b * self.band_rows)
                if not self._path(b, "r").exists():
                    yield sp.csr_matrix((n, self.n_cols), dtype=self.dtype)
                    continue
                r = np.fromfile(self._path(b, "r"), dtype=np.int32)
                c = np.fromfile(self._path(b, "c"), dtype=np.int32)
                v = np.fromfile(self._path(b, "v"), dtype=self.dtype)
                for field in "rcv":
                    self._path(b, field).unlink()
                band = sp.csr_matrix((v, (r, c)), shape=(n, self.n_cols))
                band.sum_duplicates()
                band.sort_indices()
                yield band
        finally:
            shutil.rmtree(self.folder, ignore_errors=True)
