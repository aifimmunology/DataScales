"""Tiny synthetic Xenium / Atera bundles (generated at test time, a few hundred KB at most) and hand-made
Candidates pointing at their files — one builder per layout the readers must handle: complete Xenium,
GEO-style MTX-only partial, no cell positions, Xenium Explorer zarr matrix, reseg vs canonical transcripts,
XOA 1.x bytes transcripts, Atera."""
from __future__ import annotations

import gzip
import json
import shutil
import zipfile
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.io
import scipy.sparse as sp
import zarr

from contracts import Candidate, Member

GENES = [f"GENE{i}" for i in range(40)] + ["MT-CO1", "MT-ND1"]     # = conftest.HUMAN (the tiny CellTypist model)
CONTROLS = ["NegControlProbe_0", "NegControlProbe_1", "UnassignedCodeword_0"]
PIXEL = 0.2125


def cell_ids(n: int) -> np.ndarray:
    from spatialdata_io.readers.xenium import cell_id_str_from_prefix_suffix_uint32

    return cell_id_str_from_prefix_suffix_uint32(np.arange(1, n + 1, dtype=np.uint32), np.ones(n, dtype=np.uint32)).astype(object)


def counts(n_cells: int, n_features: int, seed: int = 0, density: float = 0.3) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    x = sp.random(n_cells, n_features, density=density, format="csr", random_state=seed, dtype=np.float64)
    x.data = rng.integers(1, 20, size=x.nnz).astype(np.float64)
    x.sort_indices()
    return x


def features(protein: bool = False) -> pd.DataFrame:
    names = GENES + CONTROLS + (["PD-1", "CD45-prot"] if protein else [])
    types = (["Gene Expression"] * len(GENES) + ["Negative Control Probe"] * 2 + ["Unassigned Codeword"]
             + (["Protein Expression"] * 2 if protein else []))
    return pd.DataFrame({"id": [f"ENSG{i:011d}" for i in range(len(names))], "name": names, "feature_type": types})


def write_h5(path: Path, x: sp.csr_matrix, barcodes, feats: pd.DataFrame) -> Path:
    """A 10x v3 feature-barcode h5 (cells × features CSR on disk, as Cell Ranger / XOA write it)."""
    x = sp.csr_matrix(x)
    with h5py.File(path, "w") as f:
        m = f.create_group("matrix")
        m["barcodes"] = np.asarray(barcodes, dtype="S")
        m["data"] = x.data.astype(np.int32)
        m["indices"] = x.indices.astype(np.int64)
        m["indptr"] = x.indptr.astype(np.int64)
        m["shape"] = np.array([x.shape[1], x.shape[0]], dtype=np.int32)
        g = m.create_group("features")
        g["id"] = feats["id"].to_numpy(dtype="S")
        g["name"] = feats["name"].to_numpy(dtype="S")
        g["feature_type"] = feats["feature_type"].to_numpy(dtype="S")
        g["genome"] = np.array([b"GRCh38"] * len(feats))
        g["_all_tag_keys"] = np.array([b"genome"])
    return path


def positions(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(10, 200, size=(n, 2))


def cells_frame(ids, xy: np.ndarray) -> pd.DataFrame:
    n = len(ids)
    return pd.DataFrame({"cell_id": ids, "x_centroid": xy[:, 0], "y_centroid": xy[:, 1],
                         "transcript_counts": np.arange(n), "total_counts": np.arange(n),
                         "cell_area": np.full(n, 30.0), "nucleus_area": np.full(n, 10.0)})


def boundaries_frame(ids, xy: np.ndarray, half: float, label_id: bool = True) -> pd.DataFrame:
    """A closed square (5 vertices) per cell around its centroid."""
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], dtype=float) * half
    vertices = (xy[:, None, :] + square[None]).reshape(-1, 2)
    frame = pd.DataFrame({"cell_id": np.repeat(np.asarray(ids, dtype=object), 5),
                          "vertex_x": vertices[:, 0].astype(np.float32), "vertex_y": vertices[:, 1].astype(np.float32)})
    if label_id:
        frame["label_id"] = np.repeat(np.arange(1, len(ids) + 1), 5)
    return frame


def transcripts_frame(ids, xy: np.ndarray, per_cell: int = 3, seed: int = 0, unassigned: int = 5) -> pd.DataFrame:
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


def write_parquet(frame: pd.DataFrame, path: Path, binary: tuple[str, ...] = ()) -> Path:
    """`frame` as parquet; columns in `binary` stored as bytes (XOA 1.1–1.x transcripts)."""
    table = pa.Table.from_pandas(frame, preserve_index=False)
    for col in binary:
        i = table.schema.get_field_index(col)
        values = pa.array([str(v).encode() for v in frame[col]], type=pa.binary())
        table = table.set_column(i, pa.field(col, pa.binary()), values)
    pq.write_table(table, path)
    return path


def gzip_file(path: Path, target: Path | None = None) -> Path:
    target = target or path.with_name(path.name + ".gz")
    with open(path, "rb") as src, gzip.open(target, "wb") as dst:
        shutil.copyfileobj(src, dst)
    if target != path:
        path.unlink()
    return target


def write_cells_zarr(path: Path, ids_prefix: np.ndarray, xy: np.ndarray, shape=(64, 64)) -> Path:
    """A Xenium `cells.zarr.zip` (v2+ layout: cell_id pairs, cell_summary, masks, polygon_sets)."""
    n = len(ids_prefix)
    tmp = path.with_suffix(".dir")
    shutil.rmtree(tmp, ignore_errors=True)
    root = zarr.open_group(str(tmp), mode="w", zarr_format=2)
    root.attrs.update({"major_version": 6, "minor_version": 2, "number_cells": n})
    root.create_array("cell_id", data=np.stack([ids_prefix, np.ones(n, dtype=np.uint32)], axis=1))
    summary = np.column_stack([xy, np.full(n, 30.0), xy, np.full(n, 10.0), np.zeros(n), np.ones(n)])
    arr = root.create_array("cell_summary", data=summary)
    arr.attrs["column_names"] = ["cell_centroid_x", "cell_centroid_y", "cell_area", "nucleus_centroid_x",
                                 "nucleus_centroid_y", "nucleus_area", "z_level", "nucleus_count"]
    masks = root.create_group("masks")
    for i in (0, 1):
        mask = np.zeros(shape, dtype=np.uint32)
        for c in range(n):
            mask[(c * 2) % shape[0], (c * 3) % shape[1]] = c + 1
        masks.create_array(str(i), data=mask)
    sets = root.create_group("polygon_sets")
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], dtype=np.float32)
    for i, half in ((0, 1.0), (1, 2.5)):
        g = sets.create_group(str(i))
        verts = np.zeros((n, 50), dtype=np.float32)
        verts[:, :10] = (xy[:, None, :] + square[None] * half).reshape(n, 10)
        g.create_array("vertices", data=verts)
        g.create_array("num_vertices", data=np.full(n, 5, dtype=np.int32))
        g.create_array("cell_index", data=np.arange(n, dtype=np.uint32))
        g.create_array("method", data=np.zeros(n, dtype=np.uint32))
    _zip_dir(tmp, path)
    return path


def _zip_dir(folder: Path, target: Path) -> None:
    with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as zf:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(folder).as_posix())
    shutil.rmtree(folder)


def experiment(path: Path, version: str = "xenium-2.0.0.10") -> Path:
    path.write_text(json.dumps({"pixel_size": PIXEL, "analysis_sw_version": version}))
    return path


def candidate(folder: Path, kind: str = "xenium", sid: str | None = None, files: dict[str, Path] | None = None) -> Candidate:
    """A Candidate over every file of `folder` (keys = paths relative to it), or over `files` (key -> path)."""
    folder = Path(folder)
    if files is None:
        files = {p.relative_to(folder).as_posix(): p for p in sorted(folder.rglob("*")) if p.is_file()}
    members = {k: Member(Path(p), gz=Path(p).name.endswith(".gz") and not k.endswith(".gz"),
                         size=Path(p).stat().st_size) for k, p in files.items()}
    return Candidate(sid or folder.name, kind, folder, members)


# --------------------------------------------------------------------------- bundles

def xenium_full(folder: Path, n: int = 30, seed: int = 0, protein: bool = True, labels: bool = True) -> Path:
    """A complete XOA 2.x-like bundle: experiment.xenium, h5 (with protein features), cells/boundaries/
    transcripts parquet, cells.zarr.zip."""
    folder.mkdir(parents=True, exist_ok=True)
    ids = cell_ids(n)
    xy = positions(n, seed)
    feats = features(protein)
    write_h5(folder / "cell_feature_matrix.h5", counts(n, len(feats), seed), ids, feats)
    experiment(folder / "experiment.xenium")
    write_parquet(cells_frame(ids, xy), folder / "cells.parquet")
    write_parquet(boundaries_frame(ids, xy, 2.5), folder / "cell_boundaries.parquet")
    write_parquet(boundaries_frame(ids, xy, 1.0), folder / "nucleus_boundaries.parquet")
    write_parquet(transcripts_frame(ids, xy, seed=seed), folder / "transcripts.parquet")
    if labels:
        write_cells_zarr(folder / "cells.zarr.zip", np.arange(1, n + 1, dtype=np.uint32), xy)
    return folder


def write_mtx(folder: Path, x: sp.csr_matrix, barcodes, feats: pd.DataFrame, prefix: str = "") -> None:
    """A gzipped MTX triplet, features × cells (as Cell Ranger writes it)."""
    folder.mkdir(parents=True, exist_ok=True)
    with gzip.open(folder / f"{prefix}matrix.mtx.gz", "wb") as f:
        scipy.io.mmwrite(f, sp.coo_matrix(x.T.astype(np.int64)), field="integer")
    with gzip.open(folder / f"{prefix}features.tsv.gz", "wt") as f:
        feats[["id", "name", "feature_type"]].to_csv(f, sep="\t", header=False, index=False)
    with gzip.open(folder / f"{prefix}barcodes.tsv.gz", "wt") as f:
        f.write("\n".join(barcodes) + "\n")


def geo_partial(folder: Path, n: int = 25, seed: int = 1, positions_ok: bool = True) -> tuple[Path, Candidate]:
    """A GEO-style per-file upload: `GSM1_s_` prefix, MTX triplet, quoted cells.csv.gz, transcripts.parquet.gz,
    no boundaries / experiment.xenium / cells.zarr.zip. With positions_ok=False the cells.csv.gz is 0 bytes."""
    folder.mkdir(parents=True, exist_ok=True)
    prefix = "GSM1_s_"
    ids = cell_ids(n)
    xy = positions(n, seed)
    feats = features(False)
    write_mtx(folder, counts(n, len(feats), seed), ids, feats, prefix)
    cells_path = folder / f"{prefix}cells.csv.gz"
    if positions_ok:
        with gzip.open(cells_path, "wt") as f:
            cells_frame(ids, xy).to_csv(f, index=False, quoting=1)
    else:
        cells_path.touch()
    gzip_file(write_parquet(transcripts_frame(ids, xy, seed=seed), folder / f"{prefix}transcripts.parquet"))
    files = {"cell_feature_matrix/matrix.mtx.gz": folder / f"{prefix}matrix.mtx.gz",
             "cell_feature_matrix/features.tsv.gz": folder / f"{prefix}features.tsv.gz",
             "cell_feature_matrix/barcodes.tsv.gz": folder / f"{prefix}barcodes.tsv.gz",
             "cells.csv.gz": cells_path, "transcripts.parquet": folder / f"{prefix}transcripts.parquet.gz"}
    return folder, candidate(folder, "xenium", "GSM1_s", files)


def write_explorer_matrix(path: Path, x: sp.csr_matrix, prefixes: np.ndarray, feats: pd.DataFrame,
                          cell_major: bool = True) -> Path:
    """A Xenium Explorer `cell_feature_matrix.zarr.zip`: feature-major arrays (+ `csc/` cell-major ones)."""
    tmp = path.with_suffix(".dir")
    shutil.rmtree(tmp, ignore_errors=True)
    root = zarr.open_group(str(tmp), mode="w", zarr_format=2)
    cf = root.create_group("cell_features")
    xt = sp.csr_matrix(x.T)
    types = {"Gene Expression": "gene", "Negative Control Probe": "negative_control_probe",
             "Unassigned Codeword": "unassigned_codeword", "Protein Expression": "protein"}
    cf.attrs.update({"major_version": 4 if cell_major else 3, "minor_version": 1,
                     "number_cells": x.shape[0], "number_features": x.shape[1],
                     "feature_keys": list(feats["name"]), "feature_ids": list(feats["id"]),
                     "feature_types": [types[t] for t in feats["feature_type"]]})
    cf.create_array("data", data=xt.data.astype(np.uint32))
    cf.create_array("indices", data=xt.indices.astype(np.uint32))
    cf.create_array("indptr", data=xt.indptr.astype(np.uint32))
    cf.create_array("cell_id", data=np.stack([prefixes, np.ones(len(prefixes), dtype=np.uint32)], axis=1))
    if cell_major:
        csc = cf.create_group("csc")
        csc.create_array("data", data=x.data.astype(np.uint32))
        csc.create_array("indices", data=x.indices.astype(np.uint16))
        csc.create_array("indptr", data=x.indptr.astype(np.uint32))
    _zip_dir(tmp, path)
    return path


def xenium_explorer(folder: Path, n: int = 20, seed: int = 2, cell_major: bool = True) -> Path:
    """An Explorer-only export (`_xe`): experiment.xenium + zarr.zip files, no h5, no parquet."""
    folder.mkdir(parents=True, exist_ok=True)
    prefixes = np.arange(1, n + 1, dtype=np.uint32)
    xy = positions(n, seed)
    feats = features(False)
    write_explorer_matrix(folder / "cell_feature_matrix.zarr.zip", counts(n, len(feats), seed), prefixes, feats, cell_major)
    write_cells_zarr(folder / "cells.zarr.zip", prefixes, xy)
    experiment(folder / "experiment.xenium", "xenium-9.9.9.9")
    return folder


def xenium_reseg(folder: Path, n: int = 30, seed: int = 3) -> Path:
    """Matrix/cells/boundaries from a re-segmentation (IDs `gm…`), `transcripts.parquet` with the original
    segmentation's IDs and `reseg_transcripts.parquet` with the matrix's (GSE352490)."""
    folder.mkdir(parents=True, exist_ok=True)
    ids = np.array([f"gm{i:05d}-1" for i in range(n)], dtype=object)
    xy = positions(n, seed)
    feats = features(False)
    write_h5(folder / "cell_feature_matrix.h5", counts(n, len(feats), seed), ids, feats)
    write_parquet(cells_frame(ids, xy), folder / "cells.parquet")
    write_parquet(boundaries_frame(ids, xy, 2.5), folder / "cell_boundaries.parquet")
    write_parquet(transcripts_frame(cell_ids(n), xy, seed=seed), folder / "transcripts.parquet")
    write_parquet(transcripts_frame(ids, xy, seed=seed), folder / "reseg_transcripts.parquet")
    return folder


def xenium_bytes(folder: Path, n: int = 20, seed: int = 4) -> Path:
    """XOA 1.3-like: transcripts' cell_id, fov_name and feature_name stored as bytes; boundaries csv.gz with
    an all-NaN float label_id (re-export pitfall)."""
    folder.mkdir(parents=True, exist_ok=True)
    ids = cell_ids(n)
    xy = positions(n, seed)
    feats = features(False)
    write_h5(folder / "cell_feature_matrix.h5", counts(n, len(feats), seed), ids, feats)
    experiment(folder / "experiment.xenium", "xenium-1.3.0.5")
    write_parquet(cells_frame(ids, xy), folder / "cells.parquet")
    bounds = boundaries_frame(ids, xy, 2.5, label_id=False)
    bounds["label_id"] = np.nan
    with gzip.open(folder / "cell_boundaries.csv.gz", "wt") as f:
        bounds.to_csv(f, index=False)
    write_parquet(transcripts_frame(ids, xy, seed=seed), folder / "transcripts.parquet",
                  binary=("cell_id", "fov_name", "feature_name"))
    return folder


def atera(folder: Path, n: int = 15, seed: int = 5, cells_zarr: bool = False) -> Path:
    """An Atera-like bundle: experiment.spatial + AnnData `cell_feature_matrix.zarr.zip` (zarr v2, obs with
    integer cell_id and centroid_column/row in µm, Ensembl-indexed var with feature_name)."""
    folder.mkdir(parents=True, exist_ok=True)
    feats = features(False)
    x = counts(n, len(feats), seed)
    xy = positions(n, seed)
    barcodes = [f"bc{i:04d}-1" for i in range(n)]
    obs = pd.DataFrame({"barcode": barcodes, "cell_id": np.arange(1000, 1000 + n, dtype=np.int64),
                        "centroid_column": xy[:, 0].astype(np.float32), "centroid_row": xy[:, 1].astype(np.float32),
                        "cell_area": np.full(n, 30.0, dtype=np.float32), "total_counts": np.asarray(x.sum(1)).ravel().astype(np.int64)},
                       index=pd.Index(barcodes, name="barcode"))
    var = pd.DataFrame({"feature_name": feats["name"].to_numpy(), "filtered": True, "highly_variable": False,
                        "feature_id": feats["id"].to_numpy(), "feature_type": feats["feature_type"].to_numpy(),
                        "genome": "hg38"}, index=pd.Index(feats["id"].to_numpy(), name="feature_id"))
    adata = ad.AnnData(sp.csr_matrix(x.astype(np.int32)), obs=obs, var=var)
    tmp = folder / "cfm.dir"
    shutil.rmtree(tmp, ignore_errors=True)
    previous = ad.settings.zarr_write_format
    ad.settings.zarr_write_format = 2
    try:
        adata.write_zarr(tmp)
    finally:
        ad.settings.zarr_write_format = previous
    _zip_dir(tmp, folder / "cell_feature_matrix.zarr.zip")
    (folder / "experiment.spatial").write_text(json.dumps({"pixel_size": 0.2119}))
    return folder


def xenium_xoa10(folder: Path, n: int = 30, seed: int = 6) -> Path:
    """XOA 1.0.x-like: integer cell IDs (h5 barcodes "1".."n", int cell_id in cells / boundaries / transcripts
    with -1 unassigned), boundaries without label_id, no cells.zarr.zip."""
    folder.mkdir(parents=True, exist_ok=True)
    ints = np.arange(1, n + 1, dtype=np.int64)
    xy = positions(n, seed)
    feats = features(False)
    write_h5(folder / "cell_feature_matrix.h5", counts(n, len(feats), seed), [str(i) for i in ints], feats)
    experiment(folder / "experiment.xenium", "xenium-1.0.1")
    write_parquet(cells_frame(ints, xy), folder / "cells.parquet")
    bounds = boundaries_frame(ints, xy, 2.5, label_id=False)
    bounds["cell_id"] = bounds["cell_id"].astype(np.int64)
    write_parquet(bounds, folder / "cell_boundaries.parquet")
    tx = transcripts_frame(ints, xy, seed=seed)
    tx["cell_id"] = np.where(tx["cell_id"] == "UNASSIGNED", -1, tx["cell_id"]).astype(np.int64)
    write_parquet(tx.drop(columns=["fov_name"]), folder / "transcripts.parquet")
    return folder


def write_zstack(path: Path, z: int = 3, size: int = 128, z_step: float = 3.0, gz: bool = False) -> Path:
    """A tiled OME-TIFF z-stack (Xenium `morphology.ome.tif`): ZYX uint16, PhysicalSizeZ = z_step µm."""
    import tifffile

    data = np.random.default_rng(0).integers(0, 1000, size=(z, size, size), dtype=np.uint16)
    tifffile.imwrite(path, data, tile=(64, 64), ome=True,
                     metadata={"axes": "ZYX", "PhysicalSizeX": PIXEL, "PhysicalSizeY": PIXEL,
                               "PhysicalSizeZ": z_step, "Channel": {"Name": ["DAPI"]}})
    return gzip_file(path) if gz else path


def write_explorer_transcripts(path: Path, frame: pd.DataFrame) -> Path:
    """A Xenium Explorer `transcripts.zarr.zip` (RnaDataset, zarr v2): `grids/0/<x,y>/` tiles with location,
    gene_identity, quality_score; gene names in the attrs (as a Python repr, like XOA writes some lists)."""
    tmp = path.with_suffix(".dir")
    shutil.rmtree(tmp, ignore_errors=True)
    root = zarr.open_group(str(tmp), mode="w", zarr_format=2)
    names = sorted(set(frame["feature_name"]))
    root.attrs.update({"name": "RnaDataset", "major_version": 5, "gene_names": repr(names),
                       "number_rnas": len(frame)})
    grid = root.create_group("grids").create_group("0")
    halves = np.array_split(np.arange(len(frame)), 2)
    for k, rows in enumerate(halves):
        part = frame.iloc[rows]
        tile = grid.create_group(f"{k},0")
        tile.create_array("location", data=part[["x_location", "y_location", "z_location"]].to_numpy(np.float32))
        tile.create_array("gene_identity", data=np.array([[names.index(g)] for g in part["feature_name"]], dtype=np.uint16))
        tile.create_array("quality_score", data=part[["qv"]].to_numpy(np.float32))
    _zip_dir(tmp, path)
    return path
