"""A bundle's count matrix and cells table, read as they are stored and streamed into one CSR AnnData zarr
group on scratch (`contracts.TableSource`): Gene Expression features only, symbol-indexed var
(`genes.symbol_var`), rows = the matrix's cells that have a position in the cells table, in matrix order.

Matrix formats: 10x `cell_feature_matrix.h5` (cells × features CSR on disk, read a row band at a time),
an MTX triplet (`matrix.mtx(.gz)` + `features.tsv(.gz)` + `barcodes.tsv(.gz)`, bucketed by cell on
scratch), Xenium Explorer `cell_feature_matrix.zarr.zip` (cell-major `csc/` when present, else the
feature-major arrays bucketed by cell) and Atera's AnnData `cell_feature_matrix.zarr.zip`.
Cells tables: `cells.parquet` / `cells.csv(.gz)` (gzip sniffed), Xenium `cells.zarr.zip` (`cell_summary`),
or Atera's matrix obs. Only obs/var (one row per cell / feature) are held in memory."""
from __future__ import annotations

import gzip
import logging
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
from anndata.io import write_elem

import genes
import matrix
from contracts import TableSource

logger = logging.getLogger(__name__)

X_DTYPE = np.float32
POSITION = ("x_centroid", "y_centroid")
ATERA_POSITION = ("centroid_column", "centroid_row")
CELL_ID = "cell_id"
EXPLORER_TYPES = {"gene": genes.GEX}
UNASSIGNED = {"UNASSIGNED", "-1", "", "nan", "None", "0"}


class TableError(ValueError):
    """A count matrix or cells table that can't be read as it is."""


# --------------------------------------------------------------------------- small helpers

def is_gzip(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(2) == b"\x1f\x8b"


def open_text(path: Path):
    """A binary stream of the file's bytes, gunzipped when it is gzip (whatever its name says)."""
    return gzip.open(path, "rb") if is_gzip(path) else open(path, "rb")


def plain(path: Path, scratch: Path) -> Path:
    """`path`, or a gunzipped copy on scratch when it is gzip (parquet needs a seekable file)."""
    if not is_gzip(path):
        return path
    target = scratch / "gunzip" / (path.name[:-3] if path.name.endswith(".gz") else path.name)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".partial")
        with gzip.open(path, "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 24)
        tmp.rename(target)
    return target


def text(values) -> np.ndarray:
    """Strings from str/bytes/number values (XOA 1.x stores some ids as bytes, 1.0 as integers)."""
    arr = np.asarray(values)
    if arr.dtype.kind == "S":
        return np.char.decode(arr, "utf-8").astype(object)
    if arr.dtype == object:
        return np.array([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in arr], dtype=object)
    return arr.astype(str).astype(object)


def cell_keys(values) -> np.ndarray:
    """Cell IDs as the table's instance key: integers stay int64 (XOA 1.0; spatialdata joins on integer keys
    are fast, on text keys quadratic), anything else becomes text (bytes decoded)."""
    arr = np.asarray(values)
    if arr.dtype.kind in "iu":
        return arr.astype(np.int64)
    return text(arr)


def is_parquet(path: Path) -> bool:
    with open_text(path) as f:
        return f.read(4) == b"PAR1"


def read_frame(path: Path, scratch: Path, columns: list[str] | None = None, nrows: int | None = None) -> pd.DataFrame:
    """A parquet or csv(.gz) table (format sniffed), optionally only some columns / the first rows."""
    if is_parquet(path):
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(plain(path, scratch))
        if nrows is None:
            return pf.read(columns=columns).to_pandas()
        batches, n = [], 0
        for batch in pf.iter_batches(batch_size=min(nrows, 1 << 20), columns=columns):
            batches.append(batch.to_pandas())
            n += len(batches[-1])
            if n >= nrows:
                break
        return pd.concat(batches, ignore_index=True) if batches else pf.schema_arrow.empty_table().to_pandas()
    with open_text(path) as f:
        return pd.read_csv(f, usecols=columns, nrows=nrows)


def frame_columns(path: Path, scratch: Path) -> list[str]:
    if is_parquet(path):
        import pyarrow.parquet as pq

        return list(pq.read_schema(plain(path, scratch)).names)
    with open_text(path) as f:
        return list(pd.read_csv(f, nrows=0).columns)


# --------------------------------------------------------------------------- count matrices

@dataclass
class MatrixFile:
    """A count matrix as stored: its cell barcodes and raw features (all types), and a row-band reader
    over every feature in barcode order."""
    kind: str
    barcodes: pd.Index
    var: pd.DataFrame
    nnz: int
    _bands: Callable[[Path], Iterator[sp.csr_matrix]] = field(repr=False)
    obs: pd.DataFrame | None = None          # Atera: the matrix's own cell table
    obsm: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.barcodes), len(self.var)

    def bands(self, scratch: Path) -> Iterator[sp.csr_matrix]:
        return self._bands(scratch)


def _h5(path: Path) -> MatrixFile:
    import h5py

    with h5py.File(path, "r") as f:
        if "matrix" not in f:
            raise TableError(f"{path.name} is not a 10x feature-barcode matrix (no 'matrix' group).")
        m = f["matrix"]
        feat = m["features"]
        var = pd.DataFrame({"gene_ids": text(feat["id"][:]), "feature_types": text(feat["feature_type"][:])},
                           index=pd.Index(text(feat["name"][:])))
        if "genome" in feat:
            var["genome"] = text(feat["genome"][:])
        barcodes = pd.Index(text(m["barcodes"][:]))
        n_features, n_cells = (int(v) for v in m["shape"][:])
        nnz = int(m["indptr"][-1])
    if (n_cells, n_features) != (len(barcodes), len(var)):
        raise TableError(f"{path.name}: shape {n_features}×{n_cells} doesn't match its barcodes/features.")

    def bands(_: Path) -> Iterator[sp.csr_matrix]:
        with h5py.File(path, "r") as f:
            m = f["matrix"]
            indptr = m["indptr"][:].astype(np.int64)
            for start, stop in matrix.rows_for(indptr):
                a, b = int(indptr[start]), int(indptr[stop])
                yield sp.csr_matrix((m["data"][a:b], m["indices"][a:b], indptr[start:stop + 1] - a),
                                    shape=(stop - start, n_features))
    return MatrixFile("h5", barcodes, var, nnz, bands)


def _mtx_header(path: Path) -> tuple[int, int, int, int, str]:
    """(rows, cols, nnz, header lines, field) of a MatrixMarket coordinate file."""
    with open_text(path) as f:
        first = f.readline().decode().strip().lower()
        if not first.startswith("%%matrixmarket matrix coordinate"):
            raise TableError(f"{path.name} is not a MatrixMarket coordinate matrix (header: {first[:60]!r}).")
        n_lines = 1
        line = f.readline()
        while line.startswith(b"%"):
            n_lines += 1
            line = f.readline()
        n_lines += 1
        r, c, nnz = (int(v) for v in line.split()[:3])
    return r, c, nnz, n_lines, first.split()[3]


def _mtx(path: Path, features: Path, barcodes: Path) -> MatrixFile:
    with open_text(features) as f:
        feats = pd.read_csv(f, sep="\t", header=None, dtype=str, keep_default_na=False)
    with open_text(barcodes) as f:
        cells = pd.Index(pd.read_csv(f, sep="\t", header=None, dtype=str, keep_default_na=False)[0].to_numpy(object))
    name = feats[1] if feats.shape[1] > 1 else feats[0]
    var = pd.DataFrame({"gene_ids": feats[0].to_numpy(object),
                        "feature_types": feats[2].to_numpy(object) if feats.shape[1] > 2 else genes.GEX},
                       index=pd.Index(name.to_numpy(object)))
    r, c, nnz, skip, kind = _mtx_header(path)
    if (r, c) == (len(var), len(cells)):
        by_feature = True
    elif (r, c) == (len(cells), len(var)):
        by_feature = False
    else:
        raise TableError(f"{path.name} is {r}×{c}, which fits neither {len(var)} features × {len(cells)} barcodes "
                         "nor the transpose: the MTX triplet's files don't belong together.")
    if kind not in ("integer", "real"):
        raise TableError(f"{path.name} holds {kind} values; a count matrix is integer or real.")

    def bands(scratch: Path) -> Iterator[sp.csr_matrix]:
        buckets = matrix.RowBuckets(scratch / "mtx_buckets", len(cells), len(var), nnz, X_DTYPE)
        with open_text(path) as f:
            reader = pd.read_csv(f, sep=r"\s+", header=None, skiprows=skip, chunksize=5_000_000,
                                 dtype={0: np.int64, 1: np.int64, 2: np.float64}, engine="c")
            for chunk in reader:
                i, j, v = chunk[0].to_numpy() - 1, chunk[1].to_numpy() - 1, chunk[2].to_numpy()
                buckets.add(j if by_feature else i, i if by_feature else j, v)
        yield from buckets.bands()
    return MatrixFile("mtx", cells, var, nnz, bands)


def _zip_group(path: Path) -> tuple[zarr.Group, object]:
    store = zarr.storage.ZipStore(path, mode="r", read_only=True)
    return zarr.open_group(store, mode="r"), store


def is_explorer_matrix(path: Path) -> bool:
    group, store = _zip_group(path)
    try:
        return "cell_features" in group
    finally:
        store.close()


def _explorer(path: Path) -> MatrixFile:
    """Xenium Explorer `cell_feature_matrix.zarr.zip`: `cell_features/{data,indices,indptr}` feature-major
    (indptr over features), `cell_features/csc/` cell-major (major version ≥ 4), feature names / ids /
    types in the attrs, cell IDs as (prefix, suffix) uint32 pairs."""
    from spatialdata_io.readers.xenium import cell_id_str_from_prefix_suffix_uint32

    group, store = _zip_group(path)
    try:
        cf = group["cell_features"]
        attrs = dict(cf.attrs)
        raw_ids = cf["cell_id"][:]
        n_cells, n_features = int(attrs["number_cells"]), int(attrs["number_features"])
        cell_major = "csc" in cf and cf["csc"]["indptr"].shape[0] == n_cells + 1
        nnz = int(cf["indptr"][-1])
    finally:
        store.close()
    barcodes = pd.Index(text(cell_id_str_from_prefix_suffix_uint32(raw_ids[:, 0], raw_ids[:, 1]))) \
        if raw_ids.ndim == 2 else pd.Index(text(raw_ids))
    types = [EXPLORER_TYPES.get(t, t) for t in attrs["feature_types"]]
    var = pd.DataFrame({"gene_ids": np.asarray(attrs["feature_ids"], dtype=object), "feature_types": types},
                       index=pd.Index(np.asarray(attrs["feature_keys"], dtype=object)))

    def bands(scratch: Path) -> Iterator[sp.csr_matrix]:
        group, store = _zip_group(path)
        try:
            cf = group["cell_features"]
            if cell_major:
                yield from (band for _, band in matrix.iter_rows(_CSR(cf["csc"], n_cells, n_features)))
                return
            buckets = matrix.RowBuckets(scratch / "explorer_buckets", n_cells, n_features, nnz, X_DTYPE)
            indptr = cf["indptr"][:].astype(np.int64)
            for f0, f1 in matrix.rows_for(indptr):
                a, b = int(indptr[f0]), int(indptr[f1])
                counts = np.diff(indptr[f0:f1 + 1])
                feature = np.repeat(np.arange(f0, f1, dtype=np.int64), counts)
                buckets.add(cf["indices"][a:b], feature, cf["data"][a:b])
            yield from buckets.bands()
        finally:
            store.close()
    return MatrixFile("xenium_zarr", barcodes, var, nnz, bands)


class _CSR:
    """A bare data/indices/indptr zarr group seen as a CSR matrix node by `matrix.iter_rows`."""

    def __init__(self, group: zarr.Group, n_rows: int, n_cols: int):
        self._group, self.attrs = group, {**matrix.CSR_ATTRS, "shape": [n_rows, n_cols]}

    def __getitem__(self, key: str):
        return self._group[key]

    def __contains__(self, key: str) -> bool:
        return key in self._group


def _atera(path: Path) -> MatrixFile:
    """Atera `cell_feature_matrix.zarr.zip`: an AnnData zarr (CSR X); obs carries the cells' positions."""
    from anndata.io import read_elem

    from vendor._atera_common import _patched_ragged_vlen_chunk_decode
    from vendor.atera import _add_label_id, _read_dataframe

    group, store = _zip_group(path)
    try:
        with _patched_ragged_vlen_chunk_decode():
            var = _read_dataframe(group, "var")
            obs = _read_dataframe(group, "obs")
            obsm = dict(read_elem(group["obsm"])) if "obsm" in group else {}
        _add_label_id(obs)
        x = group["X"]
        if x.attrs.get("encoding-type") != "csr_matrix":
            raise TableError(f"{path.name}: X is {x.attrs.get('encoding-type')}, expected csr_matrix.")
        nnz = int(x["indptr"][-1])
    finally:
        store.close()

    def bands(_: Path) -> Iterator[sp.csr_matrix]:
        group, store = _zip_group(path)
        try:
            for _, band in matrix.iter_rows(group["X"]):
                yield band
        finally:
            store.close()
    return MatrixFile("atera_zarr", pd.Index(obs.index.astype(str)), var, nnz, bands, obs=obs, obsm=obsm)


def open_matrix(files: dict[str, Path], kind: str) -> MatrixFile:
    """The count matrix among a bundle's staged files: role `matrix` (+ `matrix_features`,
    `matrix_barcodes` for MTX)."""
    path = files["matrix"]
    name = path.name.lower()
    try:
        if name.endswith(".h5"):
            return _h5(path)
        if name.endswith((".mtx", ".mtx.gz")):
            return _mtx(path, files["matrix_features"], files["matrix_barcodes"])
        if name.endswith(".zarr.zip"):
            return _explorer(path) if is_explorer_matrix(path) else _atera(path)
    except TableError:
        raise
    except (OSError, KeyError, ValueError) as exc:
        raise TableError(f"{path.name} can't be read as a count matrix ({exc}).") from exc
    raise TableError(f"{path.name}: unknown count matrix format.")


# --------------------------------------------------------------------------- cells tables

def read_cells(path: Path, scratch: Path) -> pd.DataFrame:
    """A cells.parquet / cells.csv(.gz) table with `cell_id` as strings."""
    cells = read_frame(path, scratch)
    if CELL_ID not in cells:
        raise TableError(f"{path.name} has no cell_id column.")
    cells[CELL_ID] = cell_keys(cells[CELL_ID].to_numpy())
    return cells


def xenium_cells_zarr(path: Path) -> pd.DataFrame:
    """Xenium `cells.zarr.zip`: cell IDs and `cell_summary` (centroids, areas, z_level, nucleus_count),
    named like cells.parquet. Requires XOA ≥ 1.3 (string cell IDs)."""
    from spatialdata_io.readers.xenium import cell_id_str_from_prefix_suffix_uint32

    group, store = _zip_group(path)
    try:
        raw = group["cell_id"][:]
        summary = group["cell_summary"][:]
        names = list(group["cell_summary"].attrs["column_names"])
    finally:
        store.close()
    if raw.ndim != 2:
        raise TableError(f"{path.name} stores integer cell IDs (XOA < 1.3) that can't be matched to the matrix.")
    df = pd.DataFrame(summary, columns=names).rename(columns={"cell_centroid_x": "x_centroid",
                                                              "cell_centroid_y": "y_centroid"})
    df.insert(0, CELL_ID, text(cell_id_str_from_prefix_suffix_uint32(raw[:, 0], raw[:, 1])))
    return df


# --------------------------------------------------------------------------- the TableSource

def _new_group(out: Path) -> zarr.Group:
    shutil.rmtree(out, ignore_errors=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(out), mode="w")
    group.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})
    return group


def dropped_features(var: pd.DataFrame, keep: np.ndarray) -> str:
    """'4 Protein Expression, 61 Negative Control Probe' for the features left out."""
    types = var["feature_types"] if "feature_types" in var else var.get("feature_type")
    if types is None or keep.all():
        return ""
    dropped = np.asarray(types)[~keep]
    counts = pd.Series(dropped).value_counts()
    protein = [t for t in counts.index if "protein" in str(t).lower()]
    parts = []
    for t in protein:                                   # protein first, by name: easy to miss among controls
        names = list(np.asarray(var.index)[~keep][dropped == t])
        parts.append(f"{counts[t]} {t} ({', '.join(map(str, names[:10]))}{', …' if len(names) > 10 else ''})")
    parts += [f"{n} {t}" for t, n in counts.items() if t not in protein]
    return ", ".join(parts)


def write_source(m: MatrixFile, rows: np.ndarray | None, obs: pd.DataFrame, obsm: dict, sid: str,
                 out: Path, scratch: Path) -> TableSource:
    """The matrix's rows `rows` (positions in matrix order; None = all), Gene Expression columns only,
    streamed into an AnnData zarr at `out` with `obs` (already in that row order), symbol var and obsm."""
    keep, var = genes.symbol_var(m.var, sid)
    dropped = dropped_features(m.var, keep)
    if dropped:
        logger.info("%s: only Gene Expression features kept; left out: %s", sid, dropped)
    if not keep.any():
        raise TableError(f"{sid}: the count matrix has no Gene Expression features.")
    columns = np.flatnonzero(keep)
    group = _new_group(out)
    row_mask = None
    if rows is not None:
        row_mask = np.zeros(m.shape[0], dtype=bool)
        row_mask[rows] = True
        if row_mask.all():
            row_mask = None

    def bands() -> Iterator[sp.csr_matrix]:
        start = 0
        for band in m.bands(scratch):
            stop = start + band.shape[0]
            if row_mask is not None:
                band = band[row_mask[start:stop]]
            start = stop
            yield band[:, columns] if len(columns) < m.shape[1] else band
        if start != m.shape[0]:
            raise TableError(f"{sid}: the count matrix ended after {start} of {m.shape[0]} cells.")

    n_obs, _ = matrix.write_csr(group, bands(), len(columns), X_DTYPE)
    if n_obs != len(obs):
        raise TableError(f"{sid}: {n_obs} matrix rows written but obs has {len(obs)} rows.")
    write_elem(group, "obs", obs)
    write_elem(group, "var", var)
    write_elem(group, "obsm", {k: np.asarray(v) for k, v in obsm.items()})
    return TableSource(out, obs, n_obs, len(columns))


def align(barcodes: pd.Index, cells: pd.DataFrame, id_column: str = CELL_ID) -> tuple[np.ndarray, pd.DataFrame, int]:
    """Matrix rows that have a cells-table row: (their positions, the cells rows in that order indexed by
    barcode, number of matrix cells without a row)."""
    ids = pd.Index(cells[id_column].astype(str))
    if not ids.is_unique:
        cells = cells.loc[~ids.duplicated()]
        ids = pd.Index(cells[id_column].astype(str))
    where = ids.get_indexer(barcodes.astype(str))
    rows = np.flatnonzero(where >= 0)
    obs = cells.iloc[where[rows]].copy()
    obs.index = pd.Index(barcodes[rows].astype(str))
    return rows, obs, int(len(barcodes) - len(rows))


def xenium_table(m: MatrixFile, cells: pd.DataFrame, sid: str, out: Path, scratch: Path,
                 notes: list[str]) -> TableSource:
    """Xenium: obs = the cells table (centroids moved to obsm['spatial'], as spatialdata-io does)."""
    missing = [c for c in POSITION if c not in cells]
    if missing:
        raise TableError(f"{sid}: the cells table has no {', '.join(missing)} column.")
    rows, obs, n_unplaced = align(m.barcodes, cells)
    if not len(rows):
        raise TableError(f"{sid}: no cell of the count matrix is in the cells table (the files belong to "
                         "different cell sets).")
    if n_unplaced:
        notes.append(f"{n_unplaced} matrix cells without a position left out")
    spatial = obs[list(POSITION)].to_numpy(dtype=np.float64)
    obs = obs.drop(columns=list(POSITION))
    return write_source(m, rows, obs, {"spatial": spatial}, sid, out, scratch)


def atera_table(m: MatrixFile, sid: str, out: Path, scratch: Path) -> TableSource:
    """Atera: obs = the matrix's own obs (with label_id), obsm['spatial'] from its centroids."""
    obs = m.obs
    missing = [c for c in ATERA_POSITION if c not in obs]
    if missing:
        raise TableError(f"{sid}: the matrix obs has no {', '.join(missing)} column (cell positions).")
    obsm = dict(m.obsm)
    obsm["spatial"] = obs[list(ATERA_POSITION)].to_numpy()
    return write_source(m, None, obs, obsm, sid, out, scratch)
