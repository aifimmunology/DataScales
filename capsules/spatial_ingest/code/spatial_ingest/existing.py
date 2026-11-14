"""Existing SpatialData objects, single- or multi-sample, split into samples and read one sample at a time.

A sample is a set of annotated regions with the table rows that annotate them, plus the elements that
belong with them. Regions are grouped into samples by the table's sample column when it has one;
otherwise regions linked by a shared coordinate system are one sample, and samples in separate
systems are named `<object>_<tag>` after the object's folder and their region's tag (what sets its
name apart from the other regions': `cell_labels-<tag>`). A sample annotated by several tables keeps
one (`table` first). Samples sharing one coordinate system need the sample column. Other elements
follow the coordinate systems only their sample uses, or else the sample's tag in their name
(`cells-<tag>`, `<tag>_cells`). The joint table this capsule writes annotates nothing and is left
out; the per-sample tables are the source of truth.

Stores of every SpatialData era are read: zarr v2 or v3, with or without root `spatialdata_attrs`
(pre-Nov-2024), and the pre-0.1 single table at `table/table`. Elements are read one by one, so an
element spatialdata can't read, or an image/labels in no coordinate system, is left out (one warning
in `plan`) without losing the rest. Only tables of RNA counts go in: from X, `layers["counts"]` or
`.raw`; others are refused with the reason. A sample's table rows are streamed in row chunks to a
CSR table on scratch (`load_part`), never read whole."""
from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
from anndata.io import read_elem

import genes
from contracts import Loaded, Part, TableSource

__all__ = ["ELEMENT_KINDS", "Part", "SplitError", "is_spatialdata", "load_part", "plan", "safe_names", "table_var"]

ELEMENT_KINDS = ("images", "labels", "shapes", "points")
TABLE_DIRS = ("tables", "table")
GROUP_FILES = ("zarr.json", ".zgroup", ".zattrs")
SEPARATORS = "_-."
COUNT_SOURCES = (("X", "var"), ("layers/counts", "var"), ("raw/X", "raw/var"))
CHECK_ROWS = 2000                       # rows sampled to judge whether a matrix holds counts
NNZ_PER_CHUNK = 32 * 1024 * 1024        # nonzeros per row chunk streamed (sparse sources)
DENSE_VALUES_PER_CHUNK = 16 * 1024 * 1024
INVALID_CHARS = re.compile(r"[^A-Za-z0-9_.-]")
JOINT_MARKER = "spatial_ingest"          # root attribute of a store this capsule built (store.MARKER)
RESERVED_OBS = ("sample_id", "cell_id", "region", "n_genes_by_counts", "total_counts", "pct_counts_mt",
                "total_counts_mt")

logger = logging.getLogger(__name__)


class SplitError(ValueError):
    """A SpatialData object (or one of its samples) that can't be added, with the reason. The text never
    names the object's path (the caller reports the user's input path once; an archive's staged copy
    on scratch means nothing to the user)."""


# --------------------------------------------------------------------------- layout

def _is_group(path: Path) -> bool:
    return path.is_dir() and any((path / f).is_file() for f in GROUP_FILES)


def _children(path: Path) -> list[str]:
    """Child zarr groups of the folder `path`, by name (hidden entries and metadata files skipped)."""
    if not path.is_dir():
        return []
    return sorted(p.name for p in path.iterdir() if not p.name.startswith(".") and _is_group(p))


def _open(path: Path) -> zarr.Group:
    return zarr.open_group(path, mode="r", use_consolidated=False)


def root_attrs(path: Path) -> dict:
    try:
        return dict(_open(path).attrs)
    except Exception:
        return {}


def _is_anndata(path: Path) -> bool:
    return _is_group(path) and (path / "obs").is_dir() and (path / "X").exists()


def is_spatialdata(path: Path) -> bool:
    """A zarr folder with root `spatialdata_attrs`, or (an older store) element groups
    images/labels/shapes/points/tables, or the pre-0.1 `table`, holding at least one element."""
    path = Path(path)
    if "spatialdata_attrs" in root_attrs(path):
        return True
    return any(_children(path / kind) for kind in (*ELEMENT_KINDS, *TABLE_DIRS)) or _is_anndata(path / "table")


def tables(path: Path) -> dict[str, Path]:
    """The object's tables by name: `tables/<name>`, or the pre-0.1 `table/table` (or a `table`
    folder that is itself the AnnData)."""
    path = Path(path)
    for folder in TABLE_DIRS:
        found = {n: path / folder / n for n in _children(path / folder) if _is_anndata(path / folder / n)}
        if found:
            return found
    if _is_anndata(path / "table"):
        return {"table": path / "table"}
    return {}


def _table_attrs(group: zarr.Group) -> dict:
    """A table's region link: the group attributes, or (older writers) `uns/spatialdata_attrs`."""
    attrs = dict(group.attrs)
    if "region" not in attrs and "uns" in group and "spatialdata_attrs" in group["uns"]:
        attrs.update(dict(read_elem(group["uns"]["spatialdata_attrs"])))
    region = attrs.get("region")
    if region is not None and not isinstance(region, str):
        attrs["region"] = [str(r) for r in np.atleast_1d(region)]
    return attrs


def table_attrs(path: Path) -> dict[str, dict]:
    return {name: _table_attrs(_open(p)) for name, p in tables(path).items()}


def _regions(attrs) -> list[str]:
    region = attrs.get("region")
    if region is None:
        return []
    return [region] if isinstance(region, str) else list(region)


def _column(path: Path, table: str, column: str) -> np.ndarray | None:
    obs = _open(tables(path)[table] / "obs")
    if column not in obs:
        return None
    values = read_elem(obs[column])
    return np.asarray(values.astype(object) if hasattr(values, "astype") else values, dtype=object)


# --------------------------------------------------------------------------- elements

def _read_element(kind: str, where: Path):
    from spatialdata._io.io_points import _read_points
    from spatialdata._io.io_raster import _read_multiscale
    from spatialdata._io.io_shapes import _read_shapes
    from spatialdata._io.io_zarr import get_raster_format_for_read

    if kind in ("images", "labels"):
        group = _open(where)
        version = "0.2" if "ome" in group.attrs else "0.1"
        return _read_multiscale(str(where), "image" if kind == "images" else "labels",
                                get_raster_format_for_read(group, version))
    return (_read_points if kind == "points" else _read_shapes)(str(where))


def read_elements(path: Path, wanted: set[tuple[str, str]] | None = None
                  ) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    """The object's elements read one by one (rasters and points lazily), by kind and name, and those
    left out with why: unreadable, or in no coordinate system (newer spatialdata refuses those, so
    an object holding one wouldn't open at all)."""
    import warnings

    from spatialdata.transformations import get_transformation

    path = Path(path)
    elements: dict[str, dict[str, object]] = {k: {} for k in ELEMENT_KINDS}
    left_out: dict[str, str] = {}
    for kind in ELEMENT_KINDS:
        for name in _children(path / kind):
            if wanted is not None and (kind, name) not in wanted:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    element = _read_element(kind, path / kind / name)
                systems = get_transformation(element, get_all=True)
            except Exception as exc:
                left_out[f"{kind}/{name}"] = f"unreadable: {type(exc).__name__}: {str(exc)[:120]}"
                continue
            if not systems:
                left_out[f"{kind}/{name}"] = "in no coordinate system"
                continue
            elements[kind][name] = element
    return elements, left_out


def coordinate_systems(elements: dict[str, dict[str, object]]) -> dict[tuple[str, str], set[str]]:
    from spatialdata.transformations import get_transformation

    return {(kind, name): set(get_transformation(element, get_all=True))
            for kind, items in elements.items() for name, element in items.items()}


# --------------------------------------------------------------------------- naming

def _trim(text: str, from_end: bool) -> str:
    """A common prefix (or suffix) cut back to a separator, so it never splits a token."""
    cut = max((text.rfind(s) for s in SEPARATORS), default=-1) if not from_end else \
        min((i for i in (text.find(s) for s in SEPARATORS) if i >= 0), default=len(text))
    return text[: cut + 1] if not from_end else text[cut:]


def tags(names: list[str]) -> dict[str, str]:
    """Each name without the prefix and suffix all names share: `cell_labels-a` -> `a`."""
    if len(names) < 2:
        return {n: n for n in names}
    prefix = _trim(os.path.commonprefix(names), from_end=False)
    suffix = _trim(os.path.commonprefix([n[::-1] for n in names])[::-1], from_end=True)
    out = {}
    for n in names:
        tag = n[len(prefix): len(n) - len(suffix) if suffix else None]
        out[n] = tag.strip(SEPARATORS) or n
    return out


def has_token(name: str, token: str) -> bool:
    return any(name.startswith(token + s) or name.endswith(s + token) or (s + token + s) in name
               for s in SEPARATORS) or name == token


def safe_names(names, reserved=RESERVED_OBS) -> dict[str, str]:
    """old -> new for table column names spatialdata would refuse, so a store always opens: characters
    other than letters, digits and `_-.` become `_`; a leading `__` and `_index` get a `source`
    prefix; a case-variant of a reserved name (`cell_ID` beside the `cell_id` this capsule writes)
    or of an earlier name becomes `source_<name>` (numbered if still taken). Fine names are left
    out of the mapping."""
    reserved_lower = {r.lower(): r for r in reserved}
    seen: set[str] = set()
    out = {}

    def taken(n: str) -> bool:
        return n.lower() in seen or (n.lower() in reserved_lower and n != reserved_lower[n.lower()])

    for name in names:
        new = INVALID_CHARS.sub("_", str(name)) or "unnamed"
        if new.startswith("__") or new == "_index":
            new = f"source{new}"
        if taken(new):
            new = f"source_{new}"
        base, k = new, 1
        while taken(new):
            new, k = f"{base}_{k}", k + 1
        seen.add(new.lower())
        if new != name:
            out[name] = new
    return out


# --------------------------------------------------------------------------- splitting

def _region_table(attrs: dict[str, dict]) -> tuple[dict[str, str], list[str]]:
    """Which table annotates each region (a table named `table` first), and tables left out."""
    chosen, left_out = {}, []
    for name in sorted(attrs, key=lambda n: (n != "table", n)):
        regions = _regions(attrs[name])
        if not regions or all(r in chosen for r in regions):
            left_out.append(name)
            continue
        for r in regions:
            chosen.setdefault(r, name)
    return chosen, left_out


def _groups_by_column(path: Path, region_table: dict[str, str], attrs: dict[str, dict],
                      column: str) -> dict[str, list[str]] | None:
    """Regions grouped by the table's `column`, or None when a table lacks it."""
    groups: dict[str, list[str]] = {}
    for region, table in region_table.items():
        values = _column(path, table, column)
        if values is None:
            return None
        keys = _column(path, table, attrs[table]["region_key"])
        found = {str(v) for v in values[keys.astype(str) == region] if not pd.isna(v)}
        if len(found) > 1:
            raise SplitError(f"region {region} holds cells of several samples in {column!r} "
                             f"({', '.join(sorted(found)[:5])}); write one region per sample.")
        if not found:
            return None
        groups.setdefault(found.pop(), []).append(region)
    return groups


def _groups_by_system(regions: list[str], systems: dict[str, set[str]]) -> dict[str, list[str]]:
    """Regions linked by a shared coordinate system are one sample (cell and nucleus boundaries of
    one section, or several bin sizes); regions in separate systems are separate samples, each
    keyed by its first region's tag."""
    groups: list[list[str]] = []
    for r in regions:
        linked = [g for g in groups if any(systems[r] & systems[o] for o in g)]
        for g in linked:
            groups.remove(g)
        groups.append(sum(linked, []) + [r])
    region_tags = tags(regions)
    return {region_tags[g[0]]: g for g in groups}


def _n_rows(table: Path) -> int:
    from store import n_obs

    return n_obs(_open(table))


def plan(path: Path, default_sid: str, sample_column: str) -> list[Part]:
    """The samples of the SpatialData at `path`, each with its table, regions and elements. Any
    problem with the object (no table, a table that isn't RNA counts, regions it can't split, an
    unreadable store) is a `SplitError` naming the object, so the precheck can collect it and go on
    with the other inputs."""
    path = Path(path)
    notes: list[tuple[int, str]] = []
    try:
        parts = _plan(path, default_sid, sample_column, notes)
        for level, text in notes:              # only for an object that goes in: no noise before a refusal
            logger.log(level, "%s: %s", path.name, text)
        return parts
    except SplitError:
        raise
    except Exception as exc:
        raise SplitError(f"it can't be read as SpatialData ({type(exc).__name__}: {exc}).") from None


def _plan(path: Path, default_sid: str, sample_column: str, notes: list[tuple[int, str]]) -> list[Part]:
    attrs = table_attrs(path)
    if not attrs:
        raise SplitError("it has no table; spatial_ingest needs a table of cells x genes (counts) "
                         "in the SpatialData object.")
    elements, unusable = read_elements(path)
    if unusable:
        notes.append((logging.WARNING, f"{len(unusable)} element(s) left out: " +
                      "; ".join(f"{k} ({why})" for k, why in unusable.items())))
    if "spatialdata_attrs" not in root_attrs(path):
        old = ", pre-0.1 table/table" if (path / "table").is_dir() else ""
        notes.append((logging.INFO, f"older SpatialData layout (no version attributes{old}); its elements are "
                                    "rewritten in the current format"))
    systems = coordinate_systems(elements)
    region_table, left_out = _region_table(attrs)
    if not region_table:
        if len(attrs) != 1:
            raise SplitError(f"it has {len(attrs)} tables ({', '.join(sorted(attrs))}) and none annotates an "
                             "element, so it isn't clear which holds the cells; keep one table, or link each to "
                             "its region (region / region_key / instance_key).")
        table = next(iter(attrs))
        count_source(path, table)
        return [Part(default_sid, table, (), tuple(sorted(systems)), ())]
    missing = [r for r in region_table if not any(k[1] == r for k in systems)]
    if missing:
        why = [f"{r} ({unusable[k]})" for r in missing for k in unusable if k.endswith(f"/{r}")]
        raise SplitError(f"its tables annotate regions that aren't (readable) elements of the object: "
                         f"{', '.join(why or missing)}. Write the object with those elements, or with tables that "
                         "only annotate elements it holds.")
    region_systems = {r: next(cs for (k, n), cs in systems.items() if n == r) for r in region_table}
    groups = _groups_by_column(path, region_table, attrs, sample_column)
    named = groups is not None
    if not named:
        groups = _groups_by_system(list(region_table), region_systems)
    ours = JOINT_MARKER in root_attrs(path)
    for name in left_out:
        n = _n_rows(tables(path)[name])
        if ours and name == "table":
            notes.append((logging.INFO, f"its joint table ({n} rows) is left out; the per-sample tables carry the "
                                        "samples"))
        else:
            notes.append((logging.WARNING if n else logging.INFO,
                          f"table {name} ({n} rows) annotates no further region; left out"))

    single = len(groups) == 1
    region_tags = tags(list(region_table))
    owners: dict[str, set[str]] = {}
    for key, regions in groups.items():
        for r in regions:
            for cs in region_systems[r]:
                owners.setdefault(cs, set()).add(key)
    region_keys = {r: key for key, regions in groups.items() for r in regions}
    assigned: dict[str, list[tuple[str, str]]] = {key: [] for key in groups}
    unplaced = []
    for kind, name in sorted(systems):
        if name in region_keys:
            assigned[region_keys[name]].append((kind, name))
            continue
        if single:
            assigned[next(iter(groups))].append((kind, name))
            continue
        cs_owners = {next(iter(owners[c])) if len(owners.get(c, ())) == 1 else None for c in systems[(kind, name)]}
        if len(cs_owners) == 1 and None not in cs_owners:
            assigned[cs_owners.pop()].append((kind, name))
            continue
        by_name = [key for key, regions in groups.items()
                   if any(has_token(name, t) for t in {key, *(region_tags[r] for r in regions)})]
        if len(by_name) == 1:
            assigned[by_name[0]].append((kind, name))
        else:
            unplaced.append(name)
    if unplaced:
        notes.append((logging.WARNING, f"elements that belong to no single sample are left out: {', '.join(unplaced)}"))
    parts = []
    checked: set[str] = set()
    for key, regions in groups.items():
        sid = key if named else default_sid if single else f"{default_sid}_{key}"
        names = sorted({region_table[r] for r in regions}, key=lambda n: (n != "table", n))
        if len(names) > 1:
            notes.append((logging.WARNING, f"sample {sid} is annotated by several tables; keeping {names[0]}, "
                                           f"leaving out {', '.join(names[1:])}"))
            regions = [r for r in regions if region_table[r] == names[0]]
        if names[0] not in checked:
            count_source(path, names[0])
            checked.add(names[0])
        # tags name what sets a sample apart inside the object; the only sample of an unnamed object has none
        # (its region's own name is no tag: stripping it would rename `cell_boundaries` to the sample ID)
        found = () if single and not named else \
            ((key,) if named or len(regions) == 1 else ()) + tuple(region_tags[r] for r in regions if len(regions) == 1)
        parts.append(Part(sid, names[0], tuple(regions), tuple(assigned[key]),
                          tuple(dict.fromkeys(t for t in found if t != sid))))
    return parts


# --------------------------------------------------------------------------- counts

def _matrix(table: Path, where: str):
    """The matrix node at `where` (X, layers/counts, raw/X) of the table group, or None."""
    node = _open(table)
    for part in where.split("/"):
        if not isinstance(node, zarr.Group) or part not in node:
            return None
        node = node[part]
    if isinstance(node, zarr.Group) and node.attrs.get("encoding-type") not in ("csr_matrix", "csc_matrix"):
        return None
    return node


def _shape(node) -> tuple[int, int]:
    shape = node.shape if isinstance(node, zarr.Array) else node.attrs["shape"]
    return int(shape[0]), int(shape[1])


def _head(node, n: int) -> sp.csr_matrix:
    """Up to the first `n` rows of a matrix node as CSR (for CSC, the first columns' entries among them)."""
    if isinstance(node, zarr.Array):
        return sp.csr_matrix(np.asarray(node[:n]))
    n_obs, n_vars = _shape(node)
    indptr = np.asarray(node["indptr"][:], dtype=np.int64)
    if node.attrs.get("encoding-type") == "csr_matrix":
        n = min(n, n_obs)
        end = int(indptr[n])
        return sp.csr_matrix((np.asarray(node["data"][:end]), np.asarray(node["indices"][:end]), indptr[: n + 1]),
                             shape=(n, n_vars))
    end = min(int(indptr[-1]), NNZ_PER_CHUNK)
    head = sp.csc_matrix((np.asarray(node["data"][:end]), np.asarray(node["indices"][:end]),
                          np.minimum(indptr, end)), shape=(n_obs, n_vars))
    return sp.csr_matrix(head[: min(n, n_obs)])


def _problem(x: sp.csr_matrix) -> str | None:
    """Why the values aren't counts (negative or fractional), or None."""
    data = np.asarray(x.data, dtype=np.float64)
    if data.size and np.nanmin(data) < 0:
        return f"negative values (min {np.nanmin(data):.3g})"
    frac = data[np.mod(data, 1) != 0]
    if frac.size:
        return f"fractional values (e.g. {frac[0]:.3g})"
    return None


def _n_vars(table: Path, var: str) -> int:
    from store import column_length

    group = _open(table / var)
    return column_length(group[group.attrs["_index"]])


def count_source(path: Path, table: str) -> tuple[str, str]:
    """Where the table's RNA counts are, as (matrix, var) paths inside it: X, layers['counts'] or .raw
    (with raw.var), judged from its first rows. A table with none is refused with the reason."""
    where = tables(Path(path))[table]
    problems = {}
    for matrix, var in COUNT_SOURCES:
        node = _matrix(where, matrix)
        if node is None or (matrix == "raw/X" and _n_vars(where, var) != _shape(node)[1]):
            continue
        problem = _problem(_head(node, CHECK_ROWS))
        if problem is None:
            return matrix, var
        problems[matrix] = problem
    var = read_elem(_open(where)["var"])
    features = ", ".join(map(str, list(var.index[:5]))) + (", ..." if len(var) > 5 else "")
    found = "; ".join(f"{m} has {p}" for m, p in problems.items()) or "it has no count matrix"
    raise SplitError(f"table {table!r} doesn't hold RNA counts ({found}; no layers['counts'] or .raw with "
                     f"counts). With features like {features} it looks like protein intensities, metabolite signals "
                     "or normalized expression; spatial_ingest takes tables of cells x genes with raw counts only. If "
                     "the counts are elsewhere in the object, put them in X or layers['counts'].")


def table_var(path: Path, part: Part) -> pd.DataFrame:
    """The raw var of the part's count matrix (raw.var when the counts are in .raw)."""
    _, var = count_source(path, part.table)
    return read_elem(_open(tables(Path(path))[part.table] / var))


# --------------------------------------------------------------------------- one sample, streamed

def _rows(path: Path, part: Part, attrs: dict) -> np.ndarray | None:
    """The part's rows of its table (None: all of them)."""
    if not part.regions:
        return None
    keys = _column(path, part.table, attrs["region_key"]).astype(str)
    if set(np.unique(keys)) <= set(part.regions):
        return None
    return np.flatnonzero(np.isin(keys, part.regions))


def _reader(node, scratch: Path):
    """A row reader over a CSR/CSC group or a dense array; annizarr streams CSC to CSR on scratch."""
    from annizarr import AppConfig
    from annizarr._sources._readers import as_reader

    scratch.mkdir(parents=True, exist_ok=True)
    return as_reader(node, cfg=AppConfig(), tmp_dir=str(scratch))


def _chunks(reader, dense: bool, rows: np.ndarray | None, keep: np.ndarray, sid: str) -> Iterator[sp.csr_matrix]:
    """The selected rows and Gene Expression columns of the matrix, chunk by chunk, checked to hold counts."""
    n_obs, n_vars = reader.shape
    per_row = n_vars if dense else max(1.0, reader.nnz / max(1, n_obs))
    step = max(1, int(min(NNZ_PER_CHUNK / per_row, DENSE_VALUES_PER_CHUNK / max(1, n_vars) if dense else n_obs)))
    columns = np.flatnonzero(keep)
    for r0 in range(0, n_obs, step):
        r1 = min(r0 + step, n_obs)
        local = None
        if rows is not None:
            local = rows[(rows >= r0) & (rows < r1)] - r0
            if not len(local):
                continue
        chunk = sp.csr_matrix(reader.csr_rows(r0, r1))
        if local is not None:
            chunk = chunk[local]
        if len(columns) != n_vars:
            chunk = chunk[:, columns]
        problem = _problem(chunk)
        if problem:
            raise SplitError(f"its table's counts hold {problem} (rows {r0}-{r1}); only raw counts go in.")
        yield chunk


def _clean_obs(obs: pd.DataFrame, keys: dict[str, str | None]) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """obs with names spatialdata accepts (`cell_ID` beside the capsule's `cell_id` -> `source_cell_ID`),
    and the region/instance keys following their columns."""
    renames = safe_names(obs.columns)
    if obs.index.name is not None and (obs.index.name in obs.columns or obs.index.name.lower() in
                                       {c.lower() for c in RESERVED_OBS}):
        obs.index.name = None
    if renames:
        logger.info("obs columns renamed so spatialdata can read them: %s",
                    ", ".join(f"{a} -> {b}" for a, b in renames.items()))
        obs = obs.rename(columns=renames)
    return obs, {k: renames.get(v, v) if v else v for k, v in keys.items()}


def _obsm(table: Path, rows: np.ndarray | None, n: int) -> dict:
    """The table's obsm arrays (coordinates, embeddings) for the part's rows; others left out."""
    group = _open(table)
    out = {}
    if "obsm" not in group:
        return out
    for key in group["obsm"]:
        try:
            value = read_elem(group["obsm"][key])
            if rows is not None:
                value = value.iloc[rows] if isinstance(value, pd.DataFrame) else value[rows]
        except Exception:
            continue
        if getattr(value, "shape", (None,))[0] == n and not sp.issparse(value):
            out[safe_names([key], ()).get(key, key)] = value
    return out


def load_part(path: Path, part: Part, scratch: Path) -> Loaded:
    """One sample of an existing SpatialData: its elements (lazy) and its table rows, streamed as a
    CSR table of RNA counts on Gene Expression symbols to `scratch/tables/<sid>.zarr` (obs, var,
    obsm with the part's rows)."""
    from joint import TableWriter

    path, scratch = Path(path), Path(scratch)
    elements, left_out = read_elements(path, set(part.elements))
    if left_out:
        logger.info("%s: elements left out: %s", part.sid, ", ".join(left_out))
    where = tables(path)[part.table]
    attrs = _table_attrs(_open(where))
    matrix, var_at = count_source(path, part.table)
    if matrix != "X":
        logger.info("%s: X isn't raw counts; using %s", part.sid, "layers['counts']" if matrix != "raw/X" else ".raw")
    rows = _rows(path, part, attrs)
    obs = read_elem(_open(where)["obs"])
    if rows is not None:
        obs = obs.iloc[rows]
    obs, keys = _clean_obs(obs.copy(), {"region_key": attrs.get("region_key") if part.regions else None,
                                        "instance_key": attrs.get("instance_key") if part.regions else None})
    try:
        keep, var = genes.symbol_var(read_elem(_open(where / var_at)), part.sid)
    except genes.PanelError as exc:
        raise SplitError(str(exc)) from None
    if not keep.all():
        logger.info("%s: %d non-Gene-Expression features left out", part.sid, int((~keep).sum()))
    node = _matrix(where, matrix)
    out = scratch / "tables" / f"{part.sid}.zarr"
    reader = _reader(node, scratch / "tmp")
    try:
        writer = TableWriter(out, obs, var, reader.dtype, obsm=_obsm(where, rows, len(obs)))
        for chunk in _chunks(reader, isinstance(node, zarr.Array), rows, keep, part.sid):
            writer.add(chunk)
        writer.close()
    finally:
        reader.close()
    return Loaded(elements, TableSource(out, obs, len(obs), len(var)), list(part.regions), keys["region_key"],
                  keys["instance_key"], part.tags)
