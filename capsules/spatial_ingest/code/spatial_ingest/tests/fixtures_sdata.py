"""Tiny synthetic SpatialData objects (generated at test time, < 1 MB) for the existing-input, store and
joint-table tests: modern v3 stores, zarr v2 stores with and without root attributes, the pre-0.1
`table/table` layout, an image in no coordinate system, a `cell_ID` instance key, counts only in
`.raw`, protein-like and table-less objects, and per-sample tables with differing panels."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

GENES = [f"GENE{i}" for i in range(30)] + ["MT-CO1"]


def counts(n: int, n_genes: int, seed: int, dtype=np.float32) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    x = sp.random(n, n_genes, density=0.3, format="csr", random_state=seed, dtype=np.float64)
    x.data = rng.integers(1, 20, size=x.nnz).astype(np.float64)
    return x.astype(dtype)


def table(n: int, genes: list[str], seed: int, region: str | list[str], *, instance_key: str = "cell_id",
          region_key: str = "region", dtype=np.float32, extra_obs: dict | None = None) -> ad.AnnData:
    regions = [region] * n if isinstance(region, str) else list(region)
    obs = pd.DataFrame({instance_key: np.arange(n), region_key: pd.Categorical(regions),
                        "cell_area": np.linspace(10, 20, n)}, index=pd.Index([f"c{seed}_{i}" for i in range(n)]))
    for k, v in (extra_obs or {}).items():
        obs[k] = v
    var = pd.DataFrame({"gene_ids": [f"ENSG{i:011d}" for i in range(len(genes))],
                        "feature_types": "Gene Expression"}, index=pd.Index(genes))
    out = ad.AnnData(counts(n, len(genes), seed, dtype), obs=obs, var=var)
    out.obsm["spatial"] = np.random.default_rng(seed).uniform(0, 50, (n, 2))
    return out


def elements(tag: str, n: int, system: str) -> dict:
    """Image, labels, cell circles and points of one sample, named `<kind-name>-<tag>`."""
    import geopandas as gpd
    from shapely.geometry import Point
    from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, ShapesModel
    from spatialdata.transformations import Identity

    rng = np.random.default_rng(n)
    t = {system: Identity()}
    centers = rng.uniform(0, 30, (n, 2))
    shapes = gpd.GeoDataFrame({"geometry": [Point(x, y) for x, y in centers], "radius": np.ones(n)},
                              index=pd.Index(np.arange(n)))
    points = pd.DataFrame({"x": rng.uniform(0, 30, 20), "y": rng.uniform(0, 30, 20),
                           "feature_name": pd.Categorical(rng.choice(GENES[:4], 20))})
    return {
        "images": {f"image-{tag}": Image2DModel.parse(rng.integers(0, 255, (1, 32, 32)).astype(np.uint8),
                                                      dims=("c", "y", "x"), transformations=t)},
        "labels": {f"labels-{tag}": Labels2DModel.parse(rng.integers(0, 4, (32, 32)).astype(np.uint16),
                                                        dims=("y", "x"), transformations=t)},
        "shapes": {f"cells-{tag}": ShapesModel.parse(shapes, transformations=t)},
        "points": {f"points-{tag}": PointsModel.parse(points, coordinates={"x": "x", "y": "y"},
                                                      feature_key="feature_name", transformations=t)},
    }


def write(path: Path, samples: dict[str, dict], *, v2: bool = False, table_name: str = "table",
          instance_key: str = "cell_id", dtype=np.float32, extra_obs: dict | None = None) -> Path:
    """A SpatialData with one coordinate system per sample (`samples`: tag -> {"n", "genes", "seed"})
    and one table annotating every sample's `cells-<tag>`. `v2` writes the container in zarr v2."""
    from spatialdata import SpatialData
    from spatialdata.models import TableModel

    parts, tables_ = {k: {} for k in ("images", "labels", "shapes", "points")}, []
    for tag, spec in samples.items():
        for kind, items in elements(tag, spec["n"], f"global-{tag}" if len(samples) > 1 else "global").items():
            parts[kind].update(items)
        tables_.append(table(spec["n"], spec.get("genes", GENES), spec.get("seed", 0), f"cells-{tag}",
                             instance_key=instance_key, dtype=dtype))
    joined = ad.concat(tables_, merge="same")
    for k, v in (extra_obs or {}).items():
        joined.obs[k] = v
    joined.obs["region"] = pd.Categorical(joined.obs["region"].astype(str))
    t = TableModel.parse(joined, region=[f"cells-{tag}" for tag in samples], region_key="region",
                         instance_key=instance_key)
    sdata = SpatialData(**parts, tables={table_name: t})
    kwargs = {}
    if v2:
        from spatialdata._io.format import SpatialDataContainerFormatV01

        kwargs["sdata_formats"] = SpatialDataContainerFormatV01()
    sdata.write(path, overwrite=True, **kwargs)
    return Path(path)


def strip_root_attrs(path: Path) -> Path:
    """The store as written before root `spatialdata_attrs` existed (zarr v2 `.zattrs` or v3 `zarr.json`)."""
    path = Path(path)
    for meta in (".zattrs", "zarr.json"):
        f = path / meta
        if f.is_file():
            data = json.loads(f.read_text())
            attrs = data.get("attributes", data) if meta == "zarr.json" else data
            attrs.pop("spatialdata_attrs", None)
            data.pop("consolidated_metadata", None)
            f.write_text(json.dumps(data))
    for consolidated in (".zmetadata", "zmetadata"):
        (path / consolidated).unlink(missing_ok=True)
    return path


def pre01(path: Path, samples: dict[str, dict]) -> Path:
    """A pre-0.1 store: zarr v2, no root attributes, the table at `table/table`."""
    write(path, samples, v2=True)
    strip_root_attrs(path)
    shutil.move(path / "tables", path / "table")
    return path


def unplace_image(path: Path, kind: str, name: str) -> Path:
    """Drops every coordinate transformation of a raster element (as some spatialdata 0.7 writers did)."""
    meta = Path(path) / kind / name / "zarr.json"
    data = json.loads(meta.read_text())
    data["attributes"]["ome"]["multiscales"][0]["coordinateTransformations"] = []
    meta.write_text(json.dumps(data))
    for consolidated in ("zmetadata", ".zmetadata"):
        (Path(path) / consolidated).unlink(missing_ok=True)
    root = json.loads((Path(path) / "zarr.json").read_text())
    root.pop("consolidated_metadata", None)
    (Path(path) / "zarr.json").write_text(json.dumps(root))
    return Path(path)


def replace_table(path: Path, new: ad.AnnData, name: str = "table") -> Path:
    """Rewrites a table in place (zarr format of the store), keeping its spatialdata attributes."""
    import zarr
    from anndata.io import write_elem

    folder = Path(path) / "tables" / name
    group = zarr.open_group(folder, mode="r", use_consolidated=False)
    attrs, fmt = dict(group.attrs), group.metadata.zarr_format
    shutil.rmtree(folder)
    parent = zarr.open_group(Path(path) / "tables", mode="a", zarr_format=fmt, use_consolidated=False)
    write_elem(parent, name, new)
    zarr.open_group(folder, mode="r+", use_consolidated=False).attrs.update(attrs)
    return Path(path)


def raw_counts(path: Path) -> Path:
    """X log-normalised, the counts kept only in `.raw`."""
    from anndata.io import read_elem
    import zarr

    t = read_elem(zarr.open_group(Path(path) / "tables" / "table", mode="r", use_consolidated=False))
    t.raw = t.copy()
    t.X = sp.csr_matrix(np.log1p(t.X.toarray()))
    return replace_table(path, t)


def protein_like(path: Path) -> Path:
    """X scaled marker intensities (negative and fractional), as in CODEX/MIBI tables; no counts anywhere."""
    from anndata.io import read_elem
    import zarr

    t = read_elem(zarr.open_group(Path(path) / "tables" / "table", mode="r", use_consolidated=False))
    x = t.X.toarray()
    t.X = (x - x.mean(0)) / (x.std(0) + 1)
    t.var.index = [f"CD{i}" for i in range(t.n_vars)]
    return replace_table(path, t)


def drop_tables(path: Path) -> Path:
    shutil.rmtree(Path(path) / "tables")
    for consolidated in ("zmetadata", ".zmetadata"):
        (Path(path) / consolidated).unlink(missing_ok=True)
    return Path(path)


def sample_table(path: Path, sid: str, genes: list[str], n: int, seed: int, *, dtype=np.int32,
                 extra_obs: dict | None = None) -> Path:
    """A per-sample table as `sample.prepare` stages it: a CSR AnnData zarr (v3) with sample_id, cell_id,
    QC columns, spatialdata attributes linking `<sid>_cells`."""
    import zarr

    t = table(n, genes, seed, f"{sid}_cells", dtype=dtype, extra_obs=extra_obs)
    t.obs["cell_id"] = [f"cell{i}" for i in range(n)]
    t.obs["sample_id"] = pd.Categorical([sid] * n)
    t.obs["total_counts"] = np.asarray(t.X.sum(1)).ravel().astype(np.float64)
    t.obs["n_genes_by_counts"] = np.diff(t.X.indptr).astype(np.int64)
    t.uns["spatialdata_attrs"] = {"region": f"{sid}_cells", "region_key": "region", "instance_key": "cell_id"}
    t.write_zarr(path)
    zarr.open_group(path, mode="r+", use_consolidated=False).attrs.update(
        {"spatialdata-encoding-type": "ngff:regions_table", "version": "0.2", "region": f"{sid}_cells",
         "region_key": "region", "instance_key": "cell_id"})
    return Path(path)
