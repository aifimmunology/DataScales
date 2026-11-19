import json
import os
import shutil
import sys
import warnings
import zipfile
import zlib
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import scipy.sparse as sp
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import config  # noqa: E402
from config import Candidate, Member  # noqa: E402

config.setup_logging()


def pytest_configure(config):
    for category in ("FutureWarning", "UserWarning", "DeprecationWarning"):
        config.addinivalue_line("filterwarnings", f"ignore::{category}")


BUNDLES = Path(os.environ.get("SPATIAL_TEST_DATA", "/scratch/spatial_dev/data"))
GENES = [f"GENE{i}" for i in range(40)] + ["MT-CO1", "MT-ND1"]
CONTROLS = ["NegControlProbe_0", "NegControlProbe_1", "UnassignedCodeword_0"]


def bundle(name: str, zipped: bool = False) -> Path:
    path = BUNDLES / ("zip" if zipped else "extract") / (f"{name}_outs.zip" if zipped else name)
    if not path.exists():
        pytest.skip(f"public test bundle {path} not present (set SPATIAL_TEST_DATA)")
    return path


# Placeholder bundles: the right file names, for discovery.

XENIUM_FILES = ("experiment.xenium", "cell_feature_matrix.h5", "cells.parquet", "cell_boundaries.parquet",
                "nucleus_boundaries.parquet", "transcripts.parquet", "morphology_focus/ch0000_dapi.ome.tif",
                "cell_feature_matrix/matrix.mtx.gz", "cell_feature_matrix/features.tsv.gz",
                "cell_feature_matrix/barcodes.tsv.gz")
ATERA_FILES = ("experiment.spatial", "cell_feature_matrix.zarr.zip", "cells.zarr.zip", "transcripts.zarr.zip",
               "morphology_2d/ch0000_dapi.ome.tif")


def content(name: str) -> bytes:
    return f"content of {name}\n".encode()


def write(path: Path, data: bytes | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content(path.name) if data is None else data)
    return path


def xenium_bundle(folder: Path) -> Path:
    for name in XENIUM_FILES:
        write(folder / name)
    return folder


def atera_bundle(folder: Path, marker: bool = True) -> Path:
    for name in ATERA_FILES:
        if marker or name != "experiment.spatial":
            write(folder / name)
    return folder


def zip_folder(src: Path, out: Path) -> Path:
    with zipfile.ZipFile(out, "w") as zf:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(src).as_posix())
    return out


def sd_v3(folder: Path) -> Path:
    group = {"zarr_format": 3, "node_type": "group"}
    write(folder / "zarr.json", json.dumps({**group, "attributes": {"spatialdata_attrs": {"version": "0.2"}}}).encode())
    for element in ("images", "tables"):
        write(folder / element / "zarr.json", json.dumps(group).encode())
        write(folder / element / "x" / "zarr.json", json.dumps(group).encode())
    return folder


# Readable bundles: real formats with synthetic counts.

def cell_ids(n: int) -> np.ndarray:
    from spatialdata_io.readers.xenium import cell_id_str_from_prefix_suffix_uint32

    return cell_id_str_from_prefix_suffix_uint32(np.arange(1, n + 1, dtype=np.uint32),
                                                 np.ones(n, dtype=np.uint32)).astype(object)


def sparse_counts(n_cells: int, n_features: int, seed: int = 0, dtype=np.float64) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    x = sp.random(n_cells, n_features, density=0.3, format="csr", random_state=seed, dtype=np.float64)
    x.data = rng.integers(1, 20, size=x.nnz).astype(np.float64)
    x.sort_indices()
    return x.astype(dtype)


def features() -> pd.DataFrame:
    names = GENES + CONTROLS
    types = ["Gene Expression"] * len(GENES) + ["Negative Control Probe"] * 2 + ["Unassigned Codeword"]
    return pd.DataFrame({"id": [f"ENSG{i:011d}" for i in range(len(names))], "name": names, "feature_type": types})


def positions(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(10, 200, size=(n, 2))


def write_h5(path: Path, x: sp.csr_matrix, barcodes, feats: pd.DataFrame) -> Path:
    with h5py.File(path, "w") as f:
        m = f.create_group("matrix")
        m["barcodes"] = np.asarray(barcodes, dtype="S")
        m["data"] = x.data.astype(np.int32)
        m["indices"] = x.indices.astype(np.int64)
        m["indptr"] = x.indptr.astype(np.int64)
        m["shape"] = np.array([x.shape[1], x.shape[0]], dtype=np.int32)
        g = m.create_group("features")
        for key in ("id", "name", "feature_type"):
            g[key] = feats[key].to_numpy(dtype="S")
        g["genome"] = np.array([b"GRCh38"] * len(feats))
        g["_all_tag_keys"] = np.array([b"genome"])
    return path


def write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
    return path


def cells_frame(ids, xy: np.ndarray) -> pd.DataFrame:
    n = len(ids)
    return pd.DataFrame({"cell_id": ids, "x_centroid": xy[:, 0], "y_centroid": xy[:, 1],
                         "transcript_counts": np.arange(n), "total_counts": np.arange(n),
                         "cell_area": np.full(n, 30.0), "nucleus_area": np.full(n, 10.0)})


def boundaries_frame(ids, xy: np.ndarray, half: float) -> pd.DataFrame:
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], dtype=float) * half
    vertices = (xy[:, None, :] + square[None]).reshape(-1, 2)
    return pd.DataFrame({"cell_id": np.repeat(np.asarray(ids, dtype=object), 5),
                         "vertex_x": vertices[:, 0].astype(np.float32), "vertex_y": vertices[:, 1].astype(np.float32),
                         "label_id": np.repeat(np.arange(1, len(ids) + 1), 5)})


def transcripts_frame(ids, xy: np.ndarray, seed: int = 0, per_cell: int = 3, unassigned: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cell = np.r_[np.repeat(np.asarray(ids, dtype=object), per_cell), np.array(["UNASSIGNED"] * unassigned, dtype=object)]
    where = np.r_[np.repeat(xy, per_cell, axis=0), rng.uniform(10, 200, size=(unassigned, 2))]
    k = len(cell)
    return pd.DataFrame({"transcript_id": np.arange(k, dtype=np.uint64), "cell_id": cell,
                         "overlaps_nucleus": np.zeros(k, dtype=np.uint8),
                         "feature_name": rng.choice(GENES, size=k).astype(object),
                         "x_location": where[:, 0].astype(np.float32), "y_location": where[:, 1].astype(np.float32),
                         "z_location": rng.uniform(0, 10, size=k).astype(np.float32),
                         "qv": np.full(k, 30.0, dtype=np.float32), "fov_name": np.array(["A1"] * k, dtype=object)})


def xenium_full(folder: Path, n: int = 30, seed: int = 0) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    ids, xy, feats = cell_ids(n), positions(n, seed), features()
    write_h5(folder / "cell_feature_matrix.h5", sparse_counts(n, len(feats), seed), ids, feats)
    (folder / "experiment.xenium").write_text(json.dumps({"pixel_size": 0.2125, "analysis_sw_version": "xenium-2.0.0.10"}))
    write_parquet(cells_frame(ids, xy), folder / "cells.parquet")
    write_parquet(boundaries_frame(ids, xy, 2.5), folder / "cell_boundaries.parquet")
    write_parquet(boundaries_frame(ids, xy, 1.0), folder / "nucleus_boundaries.parquet")
    write_parquet(transcripts_frame(ids, xy, seed), folder / "transcripts.parquet")
    return folder


def atera(folder: Path, n: int = 15, seed: int = 5) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    feats = features()
    x = sparse_counts(n, len(feats), seed)
    xy = positions(n, seed)
    barcodes = [f"bc{i:04d}-1" for i in range(n)]
    obs = pd.DataFrame({"barcode": barcodes, "cell_id": np.arange(1000, 1000 + n, dtype=np.int64),
                        "centroid_column": xy[:, 0].astype(np.float32), "centroid_row": xy[:, 1].astype(np.float32),
                        "cell_area": np.full(n, 30.0, dtype=np.float32),
                        "total_counts": np.asarray(x.sum(1)).ravel().astype(np.int64)},
                       index=pd.Index(barcodes, name="barcode"))
    var = pd.DataFrame({"feature_name": feats["name"].to_numpy(), "filtered": True, "highly_variable": False,
                        "feature_id": feats["id"].to_numpy(), "feature_type": feats["feature_type"].to_numpy(),
                        "genome": "hg38"}, index=pd.Index(feats["id"].to_numpy(), name="feature_id"))
    tmp = folder / "cfm.dir"
    previous = ad.settings.zarr_write_format
    ad.settings.zarr_write_format = 2
    try:
        ad.AnnData(sp.csr_matrix(x.astype(np.int32)), obs=obs, var=var).write_zarr(tmp)
    finally:
        ad.settings.zarr_write_format = previous
    with zipfile.ZipFile(folder / "cell_feature_matrix.zarr.zip", "w", zipfile.ZIP_STORED) as zf:
        for p in sorted(tmp.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(tmp).as_posix())
    shutil.rmtree(tmp)
    (folder / "experiment.spatial").write_text(json.dumps({"pixel_size": 0.2119}))
    return folder


def candidate(folder: Path, kind: str = "xenium") -> Candidate:
    files = {p.relative_to(folder).as_posix(): Member(p, size=p.stat().st_size)
             for p in sorted(folder.rglob("*")) if p.is_file()}
    return Candidate(folder.name, kind, folder, files)


# SpatialData objects and per-sample tables.

def gene_id(symbol: str) -> str:
    return f"ENSG{zlib.crc32(symbol.encode()) % 10**11:011d}"


def sd_table(n: int, genes: list[str], seed: int, region: str, dtype=np.float32,
             extra_obs: dict | None = None) -> ad.AnnData:
    obs = pd.DataFrame({"cell_id": np.arange(n), "region": pd.Categorical([region] * n),
                        "cell_area": np.linspace(10, 20, n)}, index=pd.Index([f"c{seed}_{i}" for i in range(n)]))
    for k, v in (extra_obs or {}).items():
        obs[k] = v
    var = pd.DataFrame({"gene_ids": [gene_id(g) for g in genes], "feature_types": "Gene Expression"},
                       index=pd.Index(genes))
    out = ad.AnnData(sparse_counts(n, len(genes), seed, dtype), obs=obs, var=var)
    out.obsm["spatial"] = np.random.default_rng(seed).uniform(0, 50, (n, 2))
    return out


def sd_elements(tag: str, n: int, system: str) -> dict:
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


def write_sdata(path: Path, samples: dict[str, dict]) -> Path:
    from spatialdata import SpatialData
    from spatialdata.models import TableModel

    parts, tables = {k: {} for k in ("images", "labels", "shapes", "points")}, []
    for tag, spec in samples.items():
        for kind, items in sd_elements(tag, spec["n"], f"global-{tag}" if len(samples) > 1 else "global").items():
            parts[kind].update(items)
        tables.append(sd_table(spec["n"], GENES, spec["seed"], f"cells-{tag}"))
    joined = ad.concat(tables, merge="same")
    joined.obs["region"] = pd.Categorical(joined.obs["region"].astype(str))
    table = TableModel.parse(joined, region=[f"cells-{tag}" for tag in samples], region_key="region",
                             instance_key="cell_id")
    SpatialData(**parts, tables={"table": table}).write(path, overwrite=True)
    return Path(path)


def drop_tables(path: Path) -> Path:
    shutil.rmtree(Path(path) / "tables")
    for consolidated in ("zmetadata", ".zmetadata"):
        (Path(path) / consolidated).unlink(missing_ok=True)
    return Path(path)


def sample_table(path: Path, sid: str, genes: list[str], n: int, seed: int, dtype=np.int32) -> Path:
    t = sd_table(n, genes, seed, f"{sid}_cells", dtype=dtype)
    t.obs["cell_id"] = [f"cell{i}" for i in range(n)]
    t.obs["sample_id"] = pd.Categorical([sid] * n)
    t.obs["total_counts"] = np.asarray(t.X.sum(1)).ravel().astype(np.float64)
    t.obs["n_genes_by_counts"] = np.diff(t.X.indptr).astype(np.int64)
    attrs = {"region": f"{sid}_cells", "region_key": "region", "instance_key": "cell_id"}
    t.uns["spatialdata_attrs"] = attrs
    t.write_zarr(path)
    zarr.open_group(path, mode="r+", use_consolidated=False).attrs.update(
        {"spatialdata-encoding-type": "ngff:regions_table", "version": "0.2", **attrs})
    return Path(path)


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory) -> Path:
    import prepare

    celltypist, _ = prepare._celltypist()
    x = sp.vstack([sparse_counts(60, len(GENES), seed) for seed in range(2)]).astype(np.float32)
    x = sp.csr_matrix(np.log1p(x.multiply(1e4 / np.maximum(x.sum(axis=1), 1)).toarray()))
    train = ad.AnnData(x, var=pd.DataFrame(index=GENES))
    path = tmp_path_factory.mktemp("models") / "tiny_panel.pkl"
    labels = np.resize(["Epithelial", "Immune"], train.n_obs)
    celltypist.train(train, labels=labels, use_SGD=True, max_iter=50, check_expression=False).write(str(path))
    return path
