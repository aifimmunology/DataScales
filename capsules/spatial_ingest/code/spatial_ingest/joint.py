"""The joint single-cell table `tables/table`: X, obs and var of every sample, no spatial elements
(spec §3S.8), and the CSR table files every stage passes around.

Genes and obs columns are the UNION over samples. A sample bringing new genes extends the joint var
at its END, so the CSR rows already stored stay valid (only var and X's shape attribute change);
its own rows are remapped onto the union order chunk by chunk and appended with annizarr
(`include_new_obs=True` adds obs columns the store lacks, missing on the earlier rows). Genes a
sample did not measure are zero in its rows: `uns/spatial_ingest/measured_genes/<sid>` lists the
genes each sample measured and `var["n_samples_measured"]` counts them per gene, so filled zeros
can be told from measured ones.

Genes are matched by gene_id when both sides have one (else by symbol), so two loci sharing a symbol
never merge (`MATR3-ENSG01` / `MATR3-ENSG02`). `check_rows` runs everything `add` would refuse before
a sample is installed; `repair` rebuilds the table from the per-sample tables after a killed run and
takes out of the store a sample it can't append (one WARNING) instead of failing every later run.

Nothing reads a whole count matrix: tables are CSR AnnData zarr groups read in row chunks
(`matrix.iter_rows`) and written chunk by chunk (`TableWriter`). Every change to the joint table is either
a rename into place (the first sample) or leaves a state `sample_counts` reports as broken (a
killed append, var swap or record write), which the repair rebuilds from the per-sample tables
with `add`."""
from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
from anndata.io import read_elem, write_elem

import genes
from matrix import iter_rows
import store as stores

logger = logging.getLogger(__name__)

NAME = "table"
UNS_KEY = "spatial_ingest"
MEASURED = "measured_genes"
N_MEASURED = "n_samples_measured"
LEADING = ("sample_id", "cell_id", "n_genes_by_counts", "total_counts", "pct_counts_mt")
EXCLUDED = {"region", "cell_labels", "label_id", "barcode", "total_counts_mt"}
TABLE_ATTRS = {"spatialdata-encoding-type": "ngff:regions_table", "version": "0.2"}
ANNDATA_ATTRS = {"encoding-type": "anndata", "encoding-version": "0.1.0"}
CSR_ATTRS = {"encoding-type": "csr_matrix", "encoding-version": "0.1.0"}
OBS_CHUNK = 1 << 16                       # zarr chunk of obs arrays (rows): appends touch few files
FLAT_CHUNK = 1 << 20                      # zarr chunk of X's data/indices (elements)
INDPTR_CHUNK = 1 << 20                    # zarr chunk of X's indptr: fixed, so a grown indptr keeps old chunks' layout
# Every array of the joint table keeps one chunk length for life (OBS_CHUNK for obs/var/record arrays,
# FLAT_CHUNK / INDPTR_CHUNK for X). Growing an array then only appends to its last chunk, so a chunk
# uploaded by a sync before the new metadata still reads correctly under the old metadata (the old,
# shorter shape is a prefix of it): an S3 reader never sees a half-synced joint table (s3.sync order).
STRINGLIKE = ("categorical", "string-array", "nullable-string-array")


class JointError(ValueError):
    """Rows the joint table can't take."""


# --------------------------------------------------------------------------- CSR tables on disk

def x_info(group_path: Path) -> tuple[tuple[int, int], np.dtype, int]:
    """Shape, data dtype and nonzeros of the CSR X at `group_path`, from metadata and indptr's end."""
    x = zarr.open_group(Path(group_path) / "X", mode="r", use_consolidated=False)
    shape = (int(x.attrs["shape"][0]), int(x.attrs["shape"][1]))
    nnz = int(x["indptr"][-1]) if x["indptr"].shape[0] else 0
    return shape, x["data"].dtype, nnz


class TableWriter:
    """An AnnData zarr group (zarr v3, CSR X) written row chunk by row chunk: obs and var first, then
    X's data/indices grown in whole zarr chunks; indptr is int64 from the start (so no table ever
    needs widening past 2^31 nonzeros) and is written by `close`, which checks every row arrived."""

    def __init__(self, path: Path, obs: pd.DataFrame, var: pd.DataFrame, dtype, *,
                 attrs: dict | None = None, uns: dict | None = None, obsm: dict | None = None):
        stores.harden_local_writes()
        self.path, self.n_obs, self.n_vars = Path(path), len(obs), len(var)
        shutil.rmtree(self.path, ignore_errors=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        root = zarr.open_group(self.path, mode="w", zarr_format=3)
        root.attrs.update({**ANNDATA_ATTRS, **(attrs or {})})
        write_elem(root, "obs", obs, dataset_kwargs={"chunks": (OBS_CHUNK,)})
        write_elem(root, "var", var)
        for key, value in (("uns", uns), ("obsm", obsm), ("varm", None), ("obsp", None), ("varp", None),
                           ("layers", None)):
            write_elem(root, key, value or {})
        self.x = root.create_group("X")
        self.x.attrs.update({**CSR_ATTRS, "shape": [self.n_obs, self.n_vars]})
        self.data = self.x.create_array("data", shape=(0,), dtype=np.dtype(dtype), chunks=(FLAT_CHUNK,))
        self.indices = self.x.create_array("indices", shape=(0,), dtype=np.int32, chunks=(FLAT_CHUNK,))
        self.row_nnz: list[np.ndarray] = []
        self.buffer: list[tuple[np.ndarray, np.ndarray]] = []
        self.buffered = 0

    def add(self, chunk: sp.csr_matrix) -> None:
        """Appends the next rows (CSR, the table's columns)."""
        if chunk.shape[1] != self.n_vars:
            raise ValueError(f"chunk has {chunk.shape[1]} columns, the table {self.n_vars}")
        chunk = sp.csr_matrix(chunk)
        chunk.sort_indices()
        self.row_nnz.append(np.diff(chunk.indptr).astype(np.int64))
        self.buffer.append((np.asarray(chunk.data, dtype=self.data.dtype), chunk.indices.astype(np.int32)))
        self.buffered += chunk.nnz
        if self.buffered >= FLAT_CHUNK:
            self._flush(final=False)

    def _flush(self, final: bool) -> None:
        if not self.buffer:
            return
        data = np.concatenate([d for d, _ in self.buffer])
        indices = np.concatenate([i for _, i in self.buffer])
        cut = len(data) if final else len(data) - len(data) % FLAT_CHUNK
        if cut:
            self.data.append(data[:cut])
            self.indices.append(indices[:cut])
        self.buffer = [(data[cut:], indices[cut:])] if cut < len(data) else []
        self.buffered = len(data) - cut

    def close(self) -> Path:
        self._flush(final=True)
        counts = np.concatenate(self.row_nnz) if self.row_nnz else np.zeros(0, dtype=np.int64)
        if len(counts) != self.n_obs:
            raise ValueError(f"{self.path}: {len(counts)} rows written, obs has {self.n_obs}")
        indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.x.create_array("indptr", data=indptr, chunks=(INDPTR_CHUNK,))
        return self.path


# --------------------------------------------------------------------------- schema

@dataclass
class Schema:
    """What the joint table holds: its genes (var order), X dtype, obs column kinds, the CellTypist
    model and species, the samples in table order, and the genes each sample measured."""
    genes: list[str]
    gene_ids: list[str]
    x_dtype: str
    obs: dict[str, str]
    panel_source: str
    celltypist_model: str = ""
    species: str = ""
    samples: list[str] = field(default_factory=list)
    measured: dict[str, list[str]] = field(default_factory=dict)
    intact: bool = True                       # False: the record is missing (a killed rewrite)

    def var(self) -> pd.DataFrame:
        """The joint var: gene_ids, feature_types and how many samples measured each gene (a sample
        recorded before measured genes were kept counts for every gene)."""
        index = pd.Index(np.asarray(self.genes, dtype=object))
        counts = np.zeros(len(index), dtype=np.int64)
        for sid in self.samples:
            measured = self.measured.get(sid)
            counts += 1 if measured is None else index.isin(measured)
        return pd.DataFrame({"gene_ids": np.asarray(self.gene_ids, dtype=object), "feature_types": genes.GEX,
                             N_MEASURED: counts}, index=index)

    def record(self) -> dict:
        """The schema as stored in `uns/spatial_ingest` (genes and ids live in var)."""
        out = {k: v for k, v in asdict(self).items() if k not in ("genes", "gene_ids", "measured", "intact")}
        out["obs"] = json.dumps(self.obs)
        out["samples"] = np.asarray(self.samples, dtype=object)
        out[MEASURED] = {sid: np.asarray(g, dtype=object) for sid, g in self.measured.items()}
        return out


def table_path(store: Path) -> Path:
    return Path(store) / "tables" / NAME


def _group(path: Path, mode: str = "r") -> zarr.Group:
    return zarr.open_group(path, mode=mode, use_consolidated=False)


def read_schema(store: Path) -> Schema | None:
    """The joint table's schema, or None when the store has none. Never raises: a table whose record
    or var is missing or unreadable (a rewrite or an S3 sync killed midway) comes back with
    `intact=False` and no samples; `sample_counts` reports it broken, so the repair rebuilds it."""
    path = table_path(store)
    if not (path / "zarr.json").exists():
        return None
    try:
        return _read_schema(path)
    except Exception as exc:
        logger.debug("joint table record unreadable: %s", exc)
        return Schema([], [], "float32", {}, "", intact=False)


def _read_schema(path: Path) -> Schema:
    group = _group(path)
    var = read_elem(group["var"]) if "var" in group else pd.DataFrame(index=pd.Index([], dtype=object))
    ids = list(var["gene_ids"].astype(str)) if "gene_ids" in var else [""] * len(var)
    if "uns" not in group or UNS_KEY not in group["uns"]:
        dtype = str(group["X"]["data"].dtype) if "X" in group else "float32"
        return Schema(list(var.index), ids, dtype, {}, "", intact=False)
    record = dict(read_elem(group["uns"][UNS_KEY]))
    measured = {str(k): [str(g) for g in v] for k, v in dict(record.get(MEASURED, {})).items()}
    return Schema(list(var.index), ids, str(record["x_dtype"]), json.loads(record.get("obs", "{}")),
                  str(record.get("panel_source", "")), str(record.get("celltypist_model", "")),
                  str(record.get("species", "")), [str(s) for s in np.atleast_1d(record.get("samples", []))],
                  measured)


def _replace(parent: Path, name: str, value, staging: Path) -> None:
    """Rewrites element `name` of the zarr group at `parent`: written whole in staging, then swapped
    in by two renames (a kill between them leaves the element missing, which `read_schema` and
    `sample_counts` report as broken)."""
    tmp = staging / f".{parent.name}.{name}.new"
    old = staging / f".{parent.name}.{name}.old"
    for p in (tmp, old):
        shutil.rmtree(p, ignore_errors=True)
    _write(zarr.open_group(tmp, mode="w", zarr_format=3), name, value)
    if (parent / name).exists():
        os.rename(parent / name, old)
    os.rename(tmp / name, parent / name)
    shutil.rmtree(old, ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)


def _write(group: zarr.Group, name: str, value) -> None:
    """write_elem with every 1-d array in OBS_CHUNK-long chunks (see INDPTR_CHUNK); dicts recursively."""
    if isinstance(value, dict):
        write_elem(group, name, {})
        for key, item in value.items():
            _write(group[name], key, item)
    elif isinstance(value, pd.DataFrame) or (isinstance(value, np.ndarray) and value.ndim == 1):
        write_elem(group, name, value, dataset_kwargs={"chunks": (OBS_CHUNK,)})
    else:
        write_elem(group, name, value)


def write_schema(path: Path, schema: Schema, staging: Path) -> None:
    """Writes the joint var (with n_samples_measured) and the uns record of the table at `path`."""
    _replace(path, "var", schema.var(), staging)
    if not (path / "uns").exists():
        _group(path, "r+").require_group("uns")
    _replace(path / "uns", UNS_KEY, schema.record(), staging)


# --------------------------------------------------------------------------- gene union

def union(store_genes: Sequence[str], new_vars: dict[str, Sequence[str]]) -> tuple[list[str], dict[str, dict]]:
    """The joint genes after adding samples (the store's first, then each sample's new genes in input
    order) and, per sample: genes in its panel, genes it shares with the other panels (the store's
    and the other new samples'), that as a % of its panel, the genes it adds to the union, and the
    union genes it lacks (zero in its joint rows)."""
    out = list(store_genes)
    seen = set(out)
    added: dict[str, int] = {}
    for sid, names in new_vars.items():
        fresh = [g for g in dict.fromkeys(names) if g not in seen]
        out += fresh
        seen.update(fresh)
        added[sid] = len(fresh)
    stats = {}
    for sid, names in new_vars.items():
        own = set(names)
        others = set(store_genes).union(*(set(n) for s, n in new_vars.items() if s != sid))
        shared = len(own & others) if others else len(own)
        stats[sid] = {"genes_in_panel": len(own), "genes_shared": shared,
                      "pct_shared": round(100.0 * shared / len(own), 1) if own else 0.0,
                      "genes_added": added[sid], "genes_missing": len(out) - len(own)}
    return out, stats


def _note(n_new: int, n_union: int, n_missing: int) -> str:
    parts = []
    if n_new:
        parts.append(f"{n_new} new gene(s) added to the joint table (now {n_union})")
    if n_missing:
        parts.append(f"{n_missing} joint gene(s) not in its panel (zero in its rows, not counted in "
                     f"var['{N_MEASURED}'])")
    return "; ".join(parts)


# --------------------------------------------------------------------------- obs union

def kinds(obs: pd.DataFrame) -> dict[str, str]:
    """Each obs column's kind: category, str, bool, int or float."""
    out = {}
    for c, dtype in obs.dtypes.items():
        if isinstance(dtype, pd.CategoricalDtype):
            out[c] = "category"
        elif pd.api.types.is_bool_dtype(dtype):
            out[c] = "bool"
        elif pd.api.types.is_integer_dtype(dtype):
            out[c] = "int"
        elif pd.api.types.is_float_dtype(dtype):
            out[c] = "float"
        else:
            out[c] = "str"
    return out


def _as_str(values: pd.Series) -> pd.Series:
    """Values as nullable strings (missing stays missing)."""
    objects = values.astype(object)
    return pd.Series([None if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v) for v in objects],
                     index=values.index, dtype="str")


def normalized(obs: pd.DataFrame) -> pd.DataFrame:
    """obs in encodings an in-place append can extend: unordered categoricals with string categories,
    nullable strings, bool/boolean, int64/Int64 and float64."""
    out = {}
    for c in obs.columns:
        v = obs[c]
        if isinstance(v.dtype, pd.CategoricalDtype):
            out[c] = pd.Categorical(_as_str(v))
        elif pd.api.types.is_bool_dtype(v.dtype):
            out[c] = v.astype("boolean") if v.isna().any() else v.astype(bool)
        elif pd.api.types.is_integer_dtype(v.dtype):
            out[c] = v.astype("Int64") if v.isna().any() else v.astype(np.int64)
        elif pd.api.types.is_float_dtype(v.dtype):
            out[c] = v.astype(np.float64)
        else:
            out[c] = _as_str(v)
    return pd.DataFrame(out, index=obs.index)


def store_obs(path: Path) -> dict[str, tuple[str, np.dtype | None]]:
    """The joint obs columns by (family, dtype), from their encodings alone: str (any string-like),
    bool, int, float or other."""
    obs = _group(path / "obs")
    out = {}
    for c in obs.attrs.get("column-order", []):
        node = obs[c]
        if node.attrs.get("encoding-type") in STRINGLIKE:
            out[c] = ("str", None)
            continue
        values = node if isinstance(node, zarr.Array) else node["values"]
        out[c] = ({"b": "bool", "i": "int", "u": "int", "f": "float"}.get(values.dtype.kind, "other"), values.dtype)
    return out


def _fit_column(v: pd.Series, family: str, dtype: np.dtype | None):
    """A sample's column in the store column's family and dtype, or None when that would change its values."""
    mine = kinds(v.to_frame(name="c"))["c"]
    if family == "str":
        return v if mine in ("category", "str") else pd.Categorical(_as_str(v))
    if mine in ("category", "str") or family == "other":
        return None
    if family == "float":
        return v.astype(dtype or np.float64)
    if family == "bool":
        return (v.astype("boolean") if v.isna().any() else v.astype(bool)) if mine == "bool" else None
    finite = v.dropna().astype(np.float64)
    if mine == "float" and not np.all(np.mod(finite, 1) == 0):
        return None
    dtype = np.dtype(dtype or np.int64)
    return v.astype(f"Int{dtype.itemsize * 8}") if v.isna().any() else v.astype(dtype)


def rows_obs(obs: pd.DataFrame, sid: str, region_key: str | None) -> pd.DataFrame:
    """A per-sample table's obs as joint rows: indexed `<sid>:<cell_id>` (with the region when cell
    IDs repeat across regions), `sample_id` and `cell_id` first, region links and per-sample
    clusters left out, names spatialdata accepts, encodings an append can extend."""
    import existing

    cell_ids = (obs["cell_id"] if "cell_id" in obs else pd.Series(obs.index, index=obs.index)).astype(str)
    names = pd.Index([f"{sid}:{c}" for c in cell_ids], dtype=object)
    if not names.is_unique and region_key in obs:
        names = pd.Index([f"{sid}:{r}:{c}" for r, c in zip(obs[region_key].astype(str), cell_ids)], dtype=object)
    if not names.is_unique:
        raise JointError(f"{sid}: cell IDs repeat within the sample, so its joint cell names wouldn't be unique.")
    keep = [c for c in obs.columns if c not in EXCLUDED and c != region_key and not str(c).startswith("leiden")
            and c not in ("sample_id", "cell_id")]
    rest = normalized(obs[keep].rename(columns=existing.safe_names(keep, reserved=LEADING)))
    head = pd.DataFrame({"sample_id": pd.Categorical([sid] * len(obs)), "cell_id": _as_str(cell_ids)},
                        index=obs.index)
    out = pd.concat([head, rest], axis=1)
    out = out[[c for c in LEADING if c in out] + [c for c in out.columns if c not in LEADING]]
    out.index = names
    return out


def fit_obs(obs: pd.DataFrame, target: Path, sid: str, warn: bool = True) -> pd.DataFrame:
    """Joint rows' obs fitted to the joint table's columns: a shared column takes the store's
    encoding family and dtype; one that can't is left out of the joint table for this sample (one
    warning). A store string column of the oldest kind (no missing values) the sample lacks gets ""."""
    out, dropped = obs.copy(), []
    obs_group = _group(target / "obs")
    for c, (family, dtype) in store_obs(target).items():
        if c not in out:
            if obs_group[c].attrs.get("encoding-type") == "string-array":
                out[c] = ""
            continue
        fitted = _fit_column(out[c], family, dtype)
        if fitted is None:
            out = out.drop(columns=c)
            dropped.append(c)
        else:
            out[c] = fitted
    if dropped and warn:
        logger.warning("%s: obs columns %s are left out of the joint table: their values don't fit the type the joint "
                       "table already holds them in (the sample's own table keeps them).", sid, ", ".join(dropped))
    return out


# --------------------------------------------------------------------------- adding a sample

def _stage_rows(staged: Path, table_path: Path, obs: pd.DataFrame, union_genes: Sequence[str],
                targets: Sequence[str], dtype: np.dtype, sid: str) -> None:
    """The sample's joint rows as a CSR table on the union genes (its gene i goes to `targets[i]`),
    written chunk by chunk."""
    var = pd.DataFrame(index=pd.Index(np.asarray(union_genes, dtype=object)))
    columns = var.index.get_indexer(pd.Index(np.asarray(targets, dtype=object))).astype(np.int32)
    writer = TableWriter(staged, obs, var, dtype)
    for _, chunk in iter_rows(table_path):
        if not genes.holds_counts(chunk):
            raise JointError(f"{sid}: its table's X holds non-count values; the joint table takes raw counts only.")
        moved = sp.csr_matrix((chunk.data, columns[chunk.indices], chunk.indptr), shape=(chunk.shape[0], len(var)))
        try:
            writer.add(genes.as_dtype(moved, dtype))
        except genes.PanelError as exc:
            raise JointError(f"{sid}: {exc}") from None
    writer.close()


def _extend_var(target: Path, schema: Schema, staging: Path) -> None:
    """Grows the joint table to the schema's genes: var rewritten (new genes at the end), then X's
    shape attribute; CSR rows already stored stay valid since their column indices don't move."""
    _replace(target, "var", schema.var(), staging)
    x = _group(target / "X", "r+")
    x.attrs["shape"] = [int(x.attrs["shape"][0]), len(schema.genes)]


def _fix_chunks(group_path: Path, chunk: int, staging: Path) -> None:
    """Rewrites every 1-d array under `group_path` whose chunk length isn't `chunk` (written while the
    table was small, by an older version, or by annizarr for a new column / a grown indptr) with
    `chunk`-long chunks, each swapped in by rename. Appends then touch a few chunks per column
    instead of one file per few rows (a 13-cell first sample made a 7K-cell append write ~9000
    files, 97 s), and grown arrays stay readable under older metadata (see INDPTR_CHUNK)."""
    group = zarr.open_array(group_path, mode="r") if (group_path / "zarr.json").is_file() and \
        json.loads((group_path / "zarr.json").read_text()).get("node_type") == "array" else _group(group_path)
    nodes = [group] if isinstance(group, zarr.Array) else [n for _, n in group.members(max_depth=None)]
    for node in nodes:
        if not isinstance(node, zarr.Array) or node.ndim != 1 or (node.shards or node.chunks)[0] == chunk:
            continue
        where = group_path / node.path[len(group.path):].lstrip("/") if not isinstance(group, zarr.Array) \
            else group_path
        tmp, old = staging / ".rechunk.new", staging / ".rechunk.old"
        for p in (tmp, old):
            shutil.rmtree(p, ignore_errors=True)
        zarr.create_array(store=str(tmp), data=node[:], chunks=(chunk,), fill_value=node.fill_value,
                          attributes=dict(node.attrs), zarr_format=3)
        os.rename(where, old)
        os.rename(tmp, where)
        shutil.rmtree(old, ignore_errors=True)


def _commit(table: Path) -> None:
    """Consolidates the joint table's own metadata into its `zarr.json`. Readers that open the table by
    itself (anndata, spatialdata's table reader) then see it whole as of its last commit, so an S3 sync
    that uploads this file after its arrays' files (s3.sync: deepest first) flips it in one step."""
    zarr.consolidate_metadata(str(table))


def config(cpus: int):
    from annizarr import AppConfig
    from annizarr.config import ChunkConfig, IOConfig

    return AppConfig(io=IOConfig(overwrite=True), chunks=ChunkConfig(cpus=cpus))


def _targets(schema: Schema, sample_genes: pd.Index, gene_ids: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Where each of a sample's genes goes in the joint var, and the joint genes it adds (name, gene_id).
    Genes match by gene_id when both sides have one (so `MATR3`/ENSG02 meets `MATR3-ENSG02`), else by
    symbol; a symbol the joint table holds for another gene_id becomes `<symbol>-<gene_id>`."""
    by_id = {i: g for g, i in zip(schema.genes, schema.gene_ids) if i}
    ids = dict(zip(schema.genes, schema.gene_ids))
    suffixed = {g.removesuffix(f"-{i}") for g, i in ids.items() if i and g.endswith(f"-{i}")}
    out, fresh = [], []
    for g, i in zip(sample_genes, gene_ids):
        if i and i in by_id:
            t = by_id[i]
        elif i and ((g in ids and ids[g] and ids[g] != i) or g in suffixed):
            t = f"{g}-{i}"
        else:
            t = g
        if t not in ids:
            fresh.append((t, i))
            ids[t] = i
            if i:
                by_id[i] = t
        out.append(t)
    if len(set(out)) != len(out):
        dup = pd.Index(out)[pd.Index(out).duplicated()][:5]
        raise JointError(f"several of the sample's genes map to the same joint gene ({', '.join(map(str, dup))}); "
                         "check its gene names and gene_ids.")
    return out, fresh


@dataclass
class _Rows:
    """A sample's joint rows as planned (nothing written): its schema after the add, obs, gene targets."""
    schema: Schema
    rows: pd.DataFrame
    targets: list[str]
    n_new: int
    n_missing: int
    seed: bool


def _plan_rows(local: Path, sid: str, table_path: Path, celltypist_model: str, warn: bool = True) -> _Rows:
    """Everything `add` decides before writing; raises JointError for anything it would refuse."""
    group = _group(table_path)
    var = read_elem(group["var"])
    obs = read_elem(group["obs"])
    (n_obs, n_vars), x_dtype, _ = x_info(table_path)
    if (n_obs, n_vars) != (len(obs), len(var)):
        raise JointError(f"{sid}: its table's X is {n_obs} x {n_vars}, but obs/var hold {len(obs)} x {len(var)}.")
    if not var.index.is_unique:
        raise JointError(f"{sid}: its table's gene names repeat; they must be unique to place them in the joint table.")
    target = local / "tables" / NAME
    schema = read_schema(local)
    if schema is not None and not schema.intact:
        raise JointError(f"{target} is incomplete (a killed run); the repair rebuilds it before samples are added.")
    if schema is not None and sid in schema.samples:
        raise JointError(f"{sid} is already in the joint table.")
    sample_genes = pd.Index(np.asarray(var.index, dtype=object))
    gene_ids = [str(i) if not pd.isna(i) else "" for i in var["gene_ids"]] if "gene_ids" in var else [""] * len(var)
    rows = rows_obs(obs, sid, group.attrs.get("region_key"))
    if schema is None:
        schema = Schema(list(sample_genes), gene_ids, str(x_dtype), kinds(rows), sid, celltypist_model,
                        genes.species(var) or "")
        return _Rows(schema, rows, list(sample_genes), 0, 0, True)
    targets, fresh = _targets(schema, sample_genes, gene_ids)
    n_missing = len(schema.genes) - (len(targets) - len(fresh))
    schema.genes += [g for g, _ in fresh]
    schema.gene_ids += [i for _, i in fresh]
    rows = fit_obs(rows, target, sid, warn)
    return _Rows(schema, rows, targets, len(fresh), n_missing, False)


def _encodable(obs: pd.DataFrame, sid: str) -> None:
    """Writes obs to memory the way the joint table stores it: a column anndata can't encode fails here."""
    try:
        write_elem(zarr.open_group(zarr.storage.MemoryStore(), mode="w", zarr_format=3), "obs", obs)
    except Exception as exc:
        raise JointError(f"{sid}: its obs can't be stored in the joint table ({type(exc).__name__}: {exc}).") from None


def check_rows(local: Path, sid: str, table_path: Path) -> None:
    """Raises JointError for anything `add` would refuse for this sample, without writing: shapes,
    repeated gene names or cell IDs, genes mapping onto one joint gene, obs it can't encode, counts
    that aren't counts or don't fit the joint table's dtype. Reads obs/var and streams X once (its
    max only), so it runs before the sample is installed."""
    local, table_path = Path(local), Path(table_path)
    planned = _plan_rows(local, sid, table_path, "", warn=False)
    _encodable(planned.rows, sid)
    dtype = np.dtype(planned.schema.x_dtype)
    for _, chunk in iter_rows(table_path):
        if not genes.holds_counts(chunk):
            raise JointError(f"{sid}: its table's X holds non-count values; the joint table takes raw counts only.")
        try:
            genes.as_dtype(chunk, dtype)
        except genes.PanelError as exc:
            raise JointError(f"{sid}: {exc}") from None


def add(local: Path, sid: str, table_path: Path, staging: Path, cpus: int, celltypist_model: str = "") -> str:
    """Adds the per-sample table at `table_path` (a CSR AnnData zarr group) as sample `sid`'s rows of
    the joint table: seeds the table (staged, renamed into place), or extends var with the sample's
    new genes and appends in place. Also the repair's way to rebuild the table (pass the stored
    `tables/<sid>_table`). Returns a note on how the sample's genes relate to the joint genes."""
    import annizarr

    local, table_path, staging = Path(local), Path(table_path), Path(staging)
    staging.mkdir(parents=True, exist_ok=True)
    planned = _plan_rows(local, sid, table_path, celltypist_model)
    schema, rows, target = planned.schema, planned.rows, local / "tables" / NAME
    staged = staging / f"{sid}.joint.zarr"
    stores.retry_once(f"{sid}'s joint rows", lambda: _stage_rows(staged, table_path, rows, schema.genes,
                                                                 planned.targets, np.dtype(schema.x_dtype), sid),
                      lambda: shutil.rmtree(staged, ignore_errors=True))
    if planned.seed:
        schema.samples, schema.measured = [sid], {sid: list(planned.targets)}
        _group(staged, "r+").attrs.update(TABLE_ATTRS)
        write_schema(staged, schema, staging)
        _commit(staged)
        _group(local, "a").require_group("tables")
        os.rename(staged, target)
    else:
        if planned.n_new:
            _extend_var(target, schema, staging)
        _fix_chunks(target / "obs", OBS_CHUNK, staging)
        _commit(target)                                  # annizarr reads the store through this metadata
        annizarr.append(str(target), cells=str(staged), include_new_obs=True, cfg=config(cpus))
        _fix_chunks(target / "obs", OBS_CHUNK, staging)       # columns annizarr just added or rewrote whole
        _fix_chunks(target / "X" / "indptr", INDPTR_CHUNK, staging)   # annizarr rewrites it in one chunk
        schema.samples.append(sid)
        schema.measured[sid] = list(planned.targets)
        schema.obs = {**kinds(rows), **schema.obs}
        write_schema(target, schema, staging)
        _commit(target)
    shutil.rmtree(staged, ignore_errors=True)
    return _note(planned.n_new, len(schema.genes), planned.n_missing)


def repair(local: Path, present: list[str], staging: Path, cpus: int, models: dict[str, str] | None = None) -> list[str]:
    """Brings the joint table in line with the store's per-sample tables (`present`, in store order):
    rebuilt from them when it is inconsistent or unreadable (a killed append or S3 sync), then every
    present sample missing from it appended. A present sample that can't be appended is taken out
    of the store (`stores.uninstall_sample`) with one WARNING naming it and why, so it never blocks
    later runs; add it again once fixed. Returns what was done (one line each); the caller consolidates."""
    from annizarr.errors import AnzError

    local, staging = Path(local), Path(staging)
    models = models or {}
    actions: list[str] = []
    order: list[str] = []
    has_joint = (table_path(local) / "zarr.json").exists()
    counts = sample_counts(local) if has_joint else pd.Series(dtype=np.int64)
    sizes = {sid: stores.n_obs(_group(local / "tables" / f"{sid}_table")) for sid in present}
    if counts is None or any(sizes.get(sid) != n for sid, n in counts.items()):
        schema = read_schema(local)
        recorded = schema.samples if schema is not None else []
        order = [sid for sid in recorded if sid in sizes]
        drop(local)
        counts = pd.Series(dtype=np.int64)
        actions.append("the joint table was inconsistent (a killed append or sync); rebuilt from the per-sample tables")
    missing = [sid for sid in present if sid not in counts.index and sid not in order]
    for sid in order + missing:
        try:
            add(local, sid, local / "tables" / f"{sid}_table", staging, cpus, models.get(sid, ""))
        except (JointError, genes.PanelError, stores.StoreError, AnzError) as exc:
            if sample_counts(local) is None and table_path(local).exists():
                return actions + _quarantine_and_rebuild(local, sid, exc, present, staging, cpus, models)
            stores.uninstall_sample(local, sid)
            logger.warning("%s is taken out of the store: its rows can't go into the joint table (%s). Add it "
                           "again once that is fixed.", sid, exc)
            actions.append(f"took {sid} out of the store (its rows can't go into the joint table)")
            continue
        if sid in missing:
            actions.append(f"appended present sample {sid}, missing from the joint table")
    return actions


def _quarantine_and_rebuild(local: Path, sid: str, exc: Exception, present: list[str], staging: Path, cpus: int,
                            models: dict[str, str]) -> list[str]:
    """A failed append left the joint table broken: take the sample out and rebuild from the rest."""
    stores.uninstall_sample(local, sid)
    logger.warning("%s is taken out of the store: its rows can't go into the joint table (%s). Add it again once "
                   "that is fixed.", sid, exc)
    drop(local)
    rest = [s for s in present if s != sid]
    return [f"took {sid} out of the store (its rows can't go into the joint table)"] + \
        repair(local, rest, staging, cpus, models)


def sample_counts(store: Path) -> pd.Series | None:
    """Cells per sample in the joint table, or None when it is inconsistent: obs, X's shape and
    indptr disagree, or an obs column is missing or of another length (a killed append or column
    rewrite), var and X disagree (a killed var swap), or the samples in obs aren't the ones recorded
    (a killed record write)."""
    path = table_path(store)
    try:
        group = _group(path)
        sample = read_elem(group["obs"]["sample_id"])
        n_index = stores.n_obs(group)
        shape = group["X"].attrs["shape"]
        indptr = group["X"]["indptr"].shape[0]
        var = group["var"]
        n_var = stores.column_length(var[var.attrs["_index"]])
        obs = group["obs"]
        lengths = {stores.column_length(obs[c]) for c in obs.attrs.get("column-order", [])}
        schema = read_schema(store)
    except Exception:
        return None
    if not (len(sample) == n_index == shape[0] == indptr - 1) or n_var != shape[1] or lengths - {n_index}:
        return None
    counts = pd.Series(np.asarray(sample, dtype=object)).value_counts()
    if schema is None or not schema.intact or set(counts.index) != set(schema.samples):
        return None
    return counts


def drop(store: Path) -> None:
    shutil.rmtree(table_path(store), ignore_errors=True)
