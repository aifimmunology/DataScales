"""A bundle's spatial elements, lazy: images, labels, cell/nucleus shapes and transcripts. Built from the
parts that exist, so an incomplete bundle (GEO uploads, per-file downloads, Explorer-only exports) gives
whatever its files allow.

Xenium: spatialdata-io 0.7.1's own readers for labels and 2D images (`_XeniumCells`, `_get_labels`,
`_get_morphology_focus`, `_get_images`; pinned version; images read by tiles); our builders for the rest, which
give the same elements as `spatialdata_io.xenium()` for a complete bundle: polygons from boundary parquet/csv
built batch by batch (grouped by an integer `label_id`, else by `cell_id`; nuclei indexed by label like
spatialdata-io), polygons from cells.zarr.zip (Explorer), transcripts from parquet/csv (bytes columns decoded,
XOA 1.x) or Explorer transcripts.zarr.zip, the z-stack image, circles from the centroids.
Atera: the vendored reader (`vendor.atera`) when cells.zarr.zip exists, else circles from the centroids.
Every element keeps spatialdata-io's names and its `Scale(1 / pixel_size)` transformation to `global`."""
from __future__ import annotations

import contextlib
import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from readers import table as tables

logger = logging.getLogger(__name__)

DEFAULT_PIXEL_SIZE = 0.2125
REGION = "cell_boundaries"
CIRCLES = "cell_circles"
POINTS_PARTITION_ROWS = 2_000_000    # transcripts per points partition
KINDS = ("images", "labels", "shapes", "points")


@dataclass
class Built:
    """Elements by kind and name, the shapes the table annotates, and what didn't work out."""
    elements: dict[str, dict[str, object]] = field(default_factory=lambda: {k: {} for k in KINDS})
    region: str = REGION
    obs_columns: dict[str, np.ndarray] = field(default_factory=dict)   # extra obs columns (table row order)
    problems: list[str] = field(default_factory=list)                 # optional parts that failed to read


def _size(*paths: Path) -> str:
    """Total size of files / folders for progress lines (GB = 10^9 bytes)."""
    total = 0
    for p in paths:
        p = Path(p)
        items = [q for q in p.rglob("*") if q.is_file()] if p.is_dir() else [p]
        total += sum(q.stat().st_size for q in items if q.exists())
    return f"{total / 1e9:.1f} GB"


def progress(sid: str, what: str, *paths: Path) -> None:
    """One INFO line before a long element read/build, with the size it works through (L7)."""
    logger.info("%s: reading %s (%s)", sid, what, _size(*paths) if paths else "")


def scale(pixel_size: float, axes=("x", "y")):
    from spatialdata.transformations import Scale

    return Scale([1.0 / pixel_size] * len(axes), axes=axes)


def xenium_specs(folder: Path) -> tuple[dict, object]:
    """experiment.xenium's content (pixel_size defaulted) and the analyzer version (None if unknown)."""
    from spatialdata_io.readers.xenium import _parse_version_of_xenium_analyzer

    path = folder / "experiment.xenium"
    specs = {}
    if path.exists():
        try:
            specs = json.loads(path.read_text())
        except (OSError, ValueError):
            specs = {}
    specs.setdefault("pixel_size", DEFAULT_PIXEL_SIZE)
    version = None
    if specs.get("analysis_sw_version") or specs.get("xenium_ranger"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            version = _parse_version_of_xenium_analyzer(specs)
    return specs, version


# --------------------------------------------------------------------------- shapes

def circles(xy: np.ndarray, area: np.ndarray | None, ids, pixel_size: float):
    """Cells as circles at their centroids, radius sqrt(area / π) (spatialdata-io's `cells_as_circles`)."""
    from spatialdata.models import ShapesModel

    radius = np.sqrt(np.asarray(area, dtype=float) / np.pi) if area is not None else np.full(len(xy), 5.0)
    radius = np.where(np.isfinite(radius) & (radius > 0), radius, 1.0)
    return ShapesModel.parse(np.asarray(xy, dtype=float), geometry=0, radius=radius,
                             index=pd.Index(np.asarray(ids), name="cell_id"),
                             transformations={"global": scale(pixel_size)})


def _usable_label_id(path: Path) -> bool:
    """True when a boundary parquet has no label_id, or an integer label_id without nulls (re-exports
    carry it as all-NaN floats: grouping by it would make one polygon per vertex)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    if "label_id" not in schema.names:
        return True
    if not pa.types.is_integer(schema.field("label_id").type):
        return False
    return pq.read_table(path, columns=["label_id"]).column(0).null_count == 0


BOUNDARY_BATCH_ROWS = 1 << 21          # vertices per boundary read batch


def _boundary_batches(path: Path, scratch: Path):
    """(vertex_x, vertex_y, cell_id, label_id or None) as numpy arrays per batch of ~BOUNDARY_BATCH_ROWS
    vertices: parquet by arrow record batches, csv(.gz) by pandas chunks. Only one batch is in memory."""
    if tables.is_parquet(path):
        import pyarrow.parquet as pq

        names = pq.read_schema(tables.plain(path, scratch)).names

        def record_batches():
            with pq.ParquetFile(tables.plain(path, scratch)) as pf:
                for b in pf.iter_batches(batch_size=BOUNDARY_BATCH_ROWS,
                                         columns=[c for c in ("vertex_x", "vertex_y", "cell_id", "label_id")
                                                  if c in names]):
                    yield {c: b.column(c).to_numpy(zero_copy_only=False) for c in b.schema.names}
        batches = record_batches()
    else:
        with tables.open_text(path) as f:
            names = list(pd.read_csv(f, nrows=0).columns)

        def chunks():
            with tables.open_text(path) as f:
                for chunk in pd.read_csv(f, usecols=[c for c in ("vertex_x", "vertex_y", "cell_id", "label_id")
                                                     if c in names], chunksize=BOUNDARY_BATCH_ROWS):
                    yield {c: chunk[c].to_numpy() for c in chunk.columns}
        batches = chunks()
    if not {"vertex_x", "vertex_y", "cell_id"} <= set(names):
        raise tables.TableError(f"{path.name} has no vertex_x / vertex_y / cell_id columns")
    # label_id separates polygons only when it is an integer column without nulls in the whole file (decided
    # once, so batches agree); csv re-exports are grouped by cell_id
    usable = tables.is_parquet(path) and "label_id" in names and _usable_label_id(tables.plain(path, scratch))
    for b in batches:
        yield b["vertex_x"], b["vertex_y"], np.asarray(b["cell_id"]), (b["label_id"] if usable else None)


def polygons_from_table(path: Path, pixel_size: float, scratch: Path, is_nucleus: bool):
    """Polygons from a boundary parquet/csv(.gz), read batch by batch (a polygon cut by a batch end is carried
    into the next batch): vertices grouped by consecutive runs of an integer `label_id` (when usable) or of
    `cell_id`; indexed by cell_id (first polygon of a cell kept). Only the polygons are held."""
    import geopandas as gpd
    import shapely
    from spatialdata.models import ShapesModel

    geometries, ids, labels = [], [], []
    carry = None

    def build(x, y, cid, key, final):
        if not len(key):
            return None
        starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        cut = len(starts) if final else len(starts) - 1          # the last run may continue in the next batch
        stop = len(key) if final else int(starts[-1])
        if cut:
            lengths = np.diff(np.r_[starts[:cut], stop])
            ok = lengths >= 3
            if ok.any():
                rows = np.repeat(ok, lengths)
                group = np.repeat(np.arange(cut), lengths)[rows]
                rings = shapely.linearrings(np.column_stack([x[:stop][rows], y[:stop][rows]]),
                                            indices=np.searchsorted(np.flatnonzero(ok), group))
                geometries.append(shapely.polygons(rings))
                ids.append(tables.cell_keys(cid[starts[:cut][ok]]))
                if by_label:
                    labels.append(np.asarray(key[starts[:cut][ok]], dtype=np.int64))
        return None if final else (x[stop:], y[stop:], cid[stop:], key[stop:])

    by_label = False
    for x, y, cid, label in _boundary_batches(path, scratch):
        by_label = label is not None
        key = label if by_label else cid
        if carry is not None:
            x, y = np.concatenate([carry[0], x]), np.concatenate([carry[1], y])
            cid, key = np.concatenate([carry[2], cid]), np.concatenate([carry[3], key])
        carry = build(x, y, cid, key, final=False)
    if carry is not None:
        build(*carry, final=True)
    geometry = np.concatenate(geometries) if geometries else np.array([], dtype=object)
    cell = np.concatenate(ids) if ids else np.array([], dtype=object)
    if is_nucleus and labels:
        # like spatialdata-io: every nucleus kept (multinucleate cells have several), indexed by its label
        # (= the nucleus_labels raster value), owning cell in a column
        gdf = gpd.GeoDataFrame({"geometry": geometry, "cell_id": cell}, index=pd.Index(np.concatenate(labels)))
    else:
        gdf = gpd.GeoDataFrame({"geometry": geometry}, index=pd.Index(cell, name="cell_id"))
        gdf = gdf[~gdf.index.duplicated()]
    return ShapesModel.parse(gdf, transformations={"global": scale(pixel_size)})


def polygons_from_cells_zarr(path: Path, pixel_size: float, cell_ids: np.ndarray, is_nucleus: bool):
    """Polygons from a cells.zarr.zip `polygon_sets` (Xenium Explorer / Atera layout, µm vertices)."""
    import zarr

    from vendor._atera_common import _get_polygons

    store = zarr.storage.ZipStore(path, mode="r", read_only=True)
    try:
        group = zarr.open_group(store, mode="r")
        return _get_polygons(group, mask_index=0 if is_nucleus else 1, cell_ids=cell_ids, pixel_size=pixel_size,
                             is_nucleus=is_nucleus)
    finally:
        store.close()


# --------------------------------------------------------------------------- points

def _decode(series: pd.Series) -> pd.Series:
    """bytes values as text (a typed string column, so arrow can infer the points' schema)."""
    if series.dtype == object and len(series) and isinstance(series.iloc[0], bytes):
        return series.str.decode("utf-8").astype("str")
    return series


def _feature_names(path: Path, parquet: bool) -> pd.Index:
    """The distinct feature names of a transcripts file, in order of first appearance, read one column one
    batch at a time (dictionary pages when the parquet has them): bounded memory whatever the row count.
    A whole-column `drop_duplicates().compute()` took 22 GB on 740 M rows."""
    seen: dict[str, None] = {}
    if parquet:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        with pq.ParquetFile(path, read_dictionary=["feature_name"]) as pf:
            for batch in pf.iter_batches(batch_size=1 << 21, columns=["feature_name"]):
                column = batch.column(0)
                values = column.dictionary if hasattr(column, "dictionary") else pc.unique(column)
                for v in values.to_pylist():
                    seen.setdefault(v.decode() if isinstance(v, bytes) else str(v), None)
    else:
        with tables.open_text(path) as f:
            for chunk in pd.read_csv(f, usecols=["feature_name"], chunksize=1 << 21):
                for v in chunk["feature_name"].unique():
                    seen.setdefault(v.decode() if isinstance(v, bytes) else str(v), None)
    return pd.Index(list(seen), dtype=object)


def _row_groups_per_partition(path: Path) -> int:
    """Row groups per dask partition so a partition holds about POINTS_PARTITION_ROWS transcripts (row-group
    sizes from the footer's row groups, never its total, which fastparquet 1.0 files get wrong)."""
    import pyarrow.parquet as pq

    meta = pq.read_metadata(path)
    rows = [meta.row_group(i).num_rows for i in range(meta.num_row_groups)]
    typical = int(np.median(rows)) if rows else 1
    return max(1, POINTS_PARTITION_ROWS // max(typical, 1))


def transcripts(path: Path, pixel_size: float, scratch: Path):
    """Transcripts (parquet or csv(.gz)) as lazy points in partitions of ~POINTS_PARTITION_ROWS rows (so a
    parallel write holds a few small partitions, not 8 M-row ones), `feature_name` categorical with categories
    from a bounded scan, bytes columns (XOA 1.x `cell_id`, `fov_name`, `feature_name`) decoded."""
    import dask.dataframe as dd
    import pyarrow as pa
    from spatialdata.models import PointsModel

    parquet = tables.is_parquet(path)
    if parquet:
        import pyarrow.parquet as pq

        path = tables.plain(path, scratch)
        schema = pq.read_schema(path)
        if "feature_name" not in schema.names:
            raise tables.TableError(f"{path.name} has no feature_name column")
        binary = [n for n in schema.names if pa.types.is_binary(schema.field(n).type)
                  or pa.types.is_large_binary(schema.field(n).type)]
        ddf = dd.read_parquet(str(path), split_row_groups=_row_groups_per_partition(path))
    else:
        ddf = dd.read_csv(str(path), blocksize=None if tables.is_gzip(path) else "128MB",
                          compression="gzip" if tables.is_gzip(path) else None)
        if "feature_name" not in ddf.columns:
            raise tables.TableError(f"{path.name} has no feature_name column")
        binary = []
    for col in binary:
        if col != "feature_name":
            ddf[col] = ddf[col].map_partitions(_decode, meta=pd.Series([], dtype="str", name=col))
    dtype = pd.CategoricalDtype(categories=_feature_names(path, parquet))
    ddf["feature_name"] = ddf["feature_name"].map_partitions(lambda s: _decode(s).astype(str).astype(dtype),
                                                             meta=pd.Series([], dtype=dtype, name="feature_name"))
    coords = {"x": "x_location", "y": "y_location"}
    if "z_location" in ddf.columns:
        coords["z"] = "z_location"
    return PointsModel.parse(ddf, coordinates=coords, feature_key="feature_name",
                             instance_key="cell_id" if "cell_id" in ddf.columns else None,
                             transformations={"global": scale(pixel_size)}, sort=True)


def _attr_list(value) -> list:
    """Explorer attrs store some lists as their Python repr."""
    import ast

    return list(ast.literal_eval(value)) if isinstance(value, str) else list(value)


def _explorer_tile(path: Path, key: str, names: list[str], start: int, stop: int) -> pd.DataFrame:
    import zarr

    store = zarr.storage.ZipStore(path, mode="r", read_only=True)
    try:
        tile = zarr.open_group(store, mode="r")[f"grids/0/{key}"]
        location = np.asarray(tile["location"][start:stop])
        gene = np.asarray(tile["gene_identity"][start:stop]).reshape(-1).astype(np.int64)
        quality = np.asarray(tile["quality_score"][start:stop]).reshape(-1)
    finally:
        store.close()
    return pd.DataFrame({"x_location": location[:, 0].astype(np.float32), "y_location": location[:, 1].astype(np.float32),
                         "z_location": location[:, 2].astype(np.float32),
                         "feature_name": pd.Categorical.from_codes(gene, categories=pd.Index(names)),
                         "qv": quality.astype(np.float32)})


def explorer_transcripts(path: Path, pixel_size: float):
    """Xenium Explorer `transcripts.zarr.zip` (RnaDataset v4/v5, zarr v2) as lazy points: the full-resolution
    grid (`grids/0`), one task per ≤ POINTS_PARTITION_ROWS slice of a tile. Locations are µm like
    transcripts.parquet; the format holds no cell assignment, so the points carry no cell_id."""
    import dask.dataframe as dd
    import zarr
    from dask import delayed
    from spatialdata.models import PointsModel

    store = zarr.storage.ZipStore(path, mode="r", read_only=True)
    try:
        root = zarr.open_group(store, mode="r")
        if root.attrs.get("name") != "RnaDataset":
            raise tables.TableError(f"{path.name} is not a Xenium Explorer RnaDataset")
        names = [str(n) for n in _attr_list(root.attrs["gene_names"])]
        grid = root["grids/0"]
        sizes = {key: grid[key]["location"].shape[0] for key in grid.group_keys()}
    finally:
        store.close()
    dtype = pd.CategoricalDtype(categories=pd.Index(names))
    meta = pd.DataFrame({"x_location": pd.array([], dtype="float32"), "y_location": pd.array([], dtype="float32"),
                         "z_location": pd.array([], dtype="float32"), "feature_name": pd.Categorical([], dtype=dtype),
                         "qv": pd.array([], dtype="float32")})
    parts = [delayed(_explorer_tile)(path, key, names, a, min(a + POINTS_PARTITION_ROWS, n))
             for key, n in sorted(sizes.items()) for a in range(0, n, POINTS_PARTITION_ROWS) if n]
    ddf = dd.from_delayed(parts, meta=meta) if parts else dd.from_pandas(meta, npartitions=1)
    return PointsModel.parse(ddf, coordinates={"x": "x_location", "y": "y_location", "z": "z_location"},
                             feature_key="feature_name", transformations={"global": scale(pixel_size)}, sort=False)


# --------------------------------------------------------------------------- images

# Handles lazy elements keep open (TiffFile for tiled images, ZipStore for labels), by sample: they must stay
# open until the elements are written, then `release(sid)` closes them (NFS keeps `.nfs…` files for open,
# deleted files, which would block the scratch cleanup).
OPEN_HANDLES: dict[str, list] = {}


def adopt_vendor_handles(sid: str) -> None:
    """Move the vendored reader's newly opened handles under `sid`."""
    from vendor import _atera_common

    OPEN_HANDLES.setdefault(sid, []).extend(_atera_common.OPENED)
    _atera_common.OPENED.clear()


def release(sid: str) -> int:
    """Close every file handle the sample's lazy elements hold; returns how many."""
    adopt_vendor_handles(sid)
    handles = OPEN_HANDLES.pop(sid, [])
    for handle in handles:
        try:
            handle.close()
        except Exception:                               # noqa: BLE001 — closing is best effort
            logger.debug("closing %r failed", handle, exc_info=True)
    return len(handles)


def tiled_imread(path, **_):
    """An OME-TIFF as a dask array chunked by the file's own tiles (pages as a leading axis, like
    dask_image's imread). dask_image gives each page ONE chunk — a whole 2.7 GB plane for a Prime slide — so
    writing 4 channels in parallel held ~11 GB; tiles keep image writes bounded."""
    from vendor._atera_common import _tiled_imread

    image = _tiled_imread(Path(path))
    return image[None] if image.ndim == 2 else image


@contextlib.contextmanager
def tiled_images():
    """spatialdata-io's Xenium image readers with `tiled_imread` in place of dask_image's imread."""
    import spatialdata_io.readers.xenium as xenium_reader

    original = xenium_reader.imread
    xenium_reader.imread = tiled_imread
    try:
        yield
    finally:
        xenium_reader.imread = original


def ome_z_step(path: Path) -> float | None:
    """PhysicalSizeZ (µm) of an OME-TIFF, or None."""
    import re

    import tifffile

    with tifffile.TiffFile(path) as tf:
        found = re.search(r'PhysicalSizeZ="([0-9.eE+-]+)"', tf.ome_metadata or "")
    return float(found.group(1)) if found else None


def z_scale(z_step: float):
    """z slices → µm, the unit of the transcripts' z (x/y stay in pixels, like every image)."""
    from spatialdata.transformations import Scale

    return Scale([z_step, 1.0, 1.0], axes=("z", "y", "x"))


def morphology_3d(path: Path, scratch: Path, z_step: float | None = None):
    """A z-stack OME-TIFF (Xenium `morphology.ome.tif`, plain or gzipped) as a lazy multiscale (c, z, y, x)
    image read tile by tile; z scaled to µm from the file's PhysicalSizeZ (or `z_step`)."""
    import re

    import tifffile
    from spatialdata.models import Image3DModel

    path = tables.plain(path, scratch)
    with tifffile.TiffFile(path) as tf:
        axes = tf.series[0].axes
        names = re.findall(r'<Channel [^>]*Name="([^"]+)"', tf.ome_metadata or "")
    image = tiled_imread(path)
    if axes.endswith("YX") and "Z" in axes and "C" not in axes:
        image = image[None] if image.ndim == 3 else image
    elif axes not in ("CZYX",):
        raise tables.TableError(f"{path.name} has axes {axes}, not a z-stack")
    step = ome_z_step(path) or z_step or 1.0
    names = names if len(names) == image.shape[0] else [f"ch{i}" for i in range(image.shape[0])]
    return Image3DModel.parse(image, dims=("c", "z", "y", "x"), c_coords=names,
                              transformations={"global": z_scale(step)}, chunks=(1, 1, 4096, 4096),
                              scale_factors=[{"z": 1, "y": 2, "x": 2}] * 3)


# --------------------------------------------------------------------------- Xenium

def _try(built: Built, what: str, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:                           # noqa: BLE001 — an optional part never stops a sample
        built.problems.append(f"{what} ({type(exc).__name__}: {str(exc)[:160]})")
        logger.debug("%s failed", what, exc_info=True)
        return None


def xenium(folder: Path, files: dict[str, Path], obs: pd.DataFrame, spatial: np.ndarray, scratch: Path,
           sid: str = "") -> Built:
    """`folder` holds the bundle's chosen files under canonical names; `files` maps roles to them."""
    from spatialdata_io.readers._utils._utils import _initialize_raster_models_kwargs
    from spatialdata_io.readers.xenium import _get_images, _get_labels, _get_morphology_focus, \
        _XeniumCells
    import packaging.version

    built = Built()
    specs, version = xenium_specs(folder)
    pixel = float(specs["pixel_size"])
    image_kwargs, label_kwargs = _initialize_raster_models_kwargs({}, {})
    cells_zarr = None
    if "cells_zarr" in files:
        cells_zarr = _try(built, "cells.zarr.zip", _XeniumCells.open, folder, version)
        if cells_zarr is not None:
            OPEN_HANDLES.setdefault(sid, []).append(cells_zarr.group.store)
    if cells_zarr is not None:
        for name, index in (("nucleus_labels", 0), ("cell_labels", 1)):
            label = _try(built, name, _get_labels, cells_zarr.group, index, label_kwargs)
            if label is not None:
                built.elements["labels"][name] = label
        mapping = cells_zarr.cell_indices_mapping
        if mapping is not None and "cell_labels" in built.elements["labels"]:
            where = pd.Index(mapping["cell_id"].astype(str)).get_indexer(obs["cell_id"].astype(str))
            if (where >= 0).all():
                built.obs_columns["cell_labels"] = mapping["label_index"].to_numpy()[where]
        summary = _try(built, "cell_summary", tables.xenium_cells_zarr, folder / "cells.zarr.zip") \
            if cells_zarr.cell_id_str is not None else None
        if summary is not None:
            where = pd.Index(summary["cell_id"]).get_indexer(obs["cell_id"].astype(str))
            if (where >= 0).all():
                for col in ("z_level", "nucleus_count"):
                    if col in summary and col not in obs:
                        built.obs_columns[col] = summary[col].to_numpy()[where]

    for role, is_nucleus in (("nucleus_boundaries", True), ("cell_boundaries", False)):
        path = files.get(role)
        if path is None:
            continue
        progress(sid, f"{role.replace('_', ' ')} from {path.name}", path)
        shapes = None
        if path.name.endswith(".zarr.zip"):
            ids = cells_zarr.cell_id_str if cells_zarr is not None else None
            shapes = _try(built, role, polygons_from_cells_zarr, path, pixel, ids, is_nucleus) if ids is not None else None
        else:   # batch-wise builder; same polygons and index as spatialdata-io's reader, a fraction of the memory
            shapes = _try(built, role, polygons_from_table, path, pixel, scratch, is_nucleus)
        if shapes is not None:
            shapes = _try(built, role, _ids_like_table, shapes, obs["cell_id"], is_nucleus)
        if shapes is not None:
            built.elements["shapes"][role] = shapes

    if "transcripts" in files:
        path = files["transcripts"]
        progress(sid, f"transcripts from {path.name}", path)
        points = _try(built, "transcripts", explorer_transcripts, path, pixel) if path.name.endswith(".zarr.zip") \
            else _try(built, "transcripts", transcripts, path, pixel, scratch)
        if points is not None:
            built.elements["points"]["transcripts"] = points

    images = [p for p in (folder / "morphology_focus", folder / "morphology_focus.ome.tif",
                          folder / "morphology_mip.ome.tif") if p.exists()]
    if images:
        progress(sid, "images " + ", ".join(p.name for p in images), *images)
    with tiled_images():
        focus = folder / "morphology_focus"
        if focus.is_dir():
            v = version if version is not None and version >= packaging.version.parse("2.0.0") \
                else packaging.version.parse("2.0.0")
            image = _try(built, "morphology_focus", _get_morphology_focus, folder, v, {}, image_kwargs)
            if image is not None:
                built.elements["images"]["morphology_focus"] = image
        elif (folder / "morphology_focus.ome.tif").exists():
            image = _try(built, "morphology_focus", _get_images, folder, "morphology_focus.ome.tif", {}, image_kwargs)
            if image is not None:
                built.elements["images"]["morphology_focus"] = image
        if (folder / "morphology_mip.ome.tif").exists() and (version is None or version < packaging.version.parse("2.0.0")):
            image = _try(built, "morphology_mip", _get_images, folder, "morphology_mip.ome.tif", {}, image_kwargs)
            if image is not None:
                built.elements["images"]["morphology_mip"] = image
    if "morphology_3d" in files:
        progress(sid, f"z-stack image {files['morphology_3d'].name}", files["morphology_3d"])
        image = _try(built, "morphology.ome.tif (z-stack)", morphology_3d, files["morphology_3d"], scratch,
                     specs.get("z_step_size"))
        if image is not None:
            built.elements["images"]["morphology"] = image

    _set_region(built, obs["cell_id"], spatial, obs.get("cell_area"), pixel)
    adopt_vendor_handles(sid)
    return built


def _ids_like_table(shapes, table_ids: pd.Series, is_nucleus: bool):
    """Shapes keyed by cell IDs of the table's instance-key dtype (int64 for XOA 1.0 integer IDs, else text),
    so the region link is exact and spatialdata's joins stay fast."""
    integer = pd.api.types.is_integer_dtype(table_ids)

    def cast(values):
        return pd.Index(values).astype(np.int64 if integer else str)

    if shapes.index.name == "cell_id":
        shapes.index = pd.Index(cast(shapes.index), name="cell_id")
    if is_nucleus and "cell_id" in shapes:
        shapes["cell_id"] = np.asarray(cast(shapes["cell_id"]))
    return shapes


def _set_region(built: Built, cell_ids: pd.Series, spatial: np.ndarray, area, pixel: float) -> None:
    """The table annotates cell_boundaries when every placed cell has one, else circles at the centroids.
    Cells without a position (obs['has_position'] False) get no shape."""
    placed = np.isfinite(np.asarray(spatial, dtype=float)).all(axis=1)
    ids = pd.Index(np.asarray(cell_ids))[placed]
    shapes = built.elements["shapes"].get(REGION)
    if shapes is not None:
        uncovered = int((~ids.isin(shapes.index)).sum())
        if not uncovered:
            built.region = REGION
            return
        built.problems.append(f"{uncovered} cells have no cell boundary, so the table annotates cell circles")
    area = None if area is None else np.asarray(area, dtype=float)[placed]
    built.elements["shapes"][CIRCLES] = circles(np.asarray(spatial)[placed], area, np.asarray(ids), pixel)
    built.region = CIRCLES


# --------------------------------------------------------------------------- Atera

def atera(folder: Path, files: dict[str, Path], obs: pd.DataFrame, spatial: np.ndarray, sid: str = "") -> Built:
    """Atera elements with the vendored reader (its table is lazy and dropped)."""
    built = Built()
    has_cells = "cells_zarr" in files
    has_2d, has_3d = (folder / "morphology_2d").is_dir(), (folder / "morphology_3d").is_dir()
    pixel = None
    if (folder / "experiment.spatial").exists() or has_2d:
        from vendor._atera_common import _get_pixel_size

        pixel = _try(built, "pixel size", _get_pixel_size, folder)
    if pixel is not None:
        from vendor.atera import atera as read_atera

        parts = [folder / n for n in ("cells.zarr.zip", "transcripts.zarr.zip", "morphology_2d", "morphology_3d")
                 if (folder / n).exists()]
        progress(sid, "Atera elements " + ", ".join(p.name for p in parts), *parts)

        # The reader's own (lazy) table is dropped: ours is streamed. Without cells.zarr.zip it would annotate
        # `cell_labels`, which isn't read, and spatialdata warns about it; that warning is moot here.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            sdata = _try(built, "Atera elements", read_atera, folder, cells_table=has_cells or "transcripts" in files,
                         cells_boundaries=has_cells, nucleus_boundaries=has_cells, cells_labels=has_cells,
                         nucleus_labels=has_cells, transcripts="transcripts" in files, morphology_images=has_2d,
                         morphology_3d_images=has_3d)
        if sdata is not None:
            for kind in KINDS:
                built.elements[kind].update(dict(getattr(sdata, kind)))
        if "morphology_3d" in built.elements["images"]:
            _try(built, "morphology_3d z scale", _scale_atera_z, built.elements["images"]["morphology_3d"], folder)
    else:
        pixel = DEFAULT_PIXEL_SIZE
        built.problems.append(f"pixel size unknown (no experiment.spatial, no images); {pixel} µm assumed")
    _set_region(built, obs["cell_id"], spatial, obs.get("cell_area"), pixel)
    adopt_vendor_handles(sid)
    return built


def _scale_atera_z(image, folder: Path) -> None:
    """The vendored reader leaves the 3D image's z in slices; transcripts' z is in µm: scale z by the z step
    (experiment.spatial `z_step_size`, else the first file's PhysicalSizeZ)."""
    from spatialdata.transformations import set_transformation

    step = None
    spec = folder / "experiment.spatial"
    if spec.exists():
        step = json.loads(spec.read_text()).get("z_step_size")
    if not step:
        first = sorted((folder / "morphology_3d").glob("*.ome.tif"))[0]
        step = ome_z_step(first)
    if step:
        set_transformation(image, z_scale(float(step)), to_coordinate_system="global")
