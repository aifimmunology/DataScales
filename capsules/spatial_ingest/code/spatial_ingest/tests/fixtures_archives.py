"""Tiny synthetic look-alikes of the messy input layouts seen in real downloads (for discovery and staging
tests). Files hold a few bytes of placeholder content: discovery only lists names, and staging tests check the
bytes come through. Layouts mirror the robustness reports (t1/t4/t5/t6): two-bundle zip, Xenium Ranger
pipestance tar.gz, macOS zip, 10x per-file prefixes, GEO per-file gzip, `MB263-` folders, mixed parent folders,
SpatialData stores old and new, zipped or not."""
from __future__ import annotations

import gzip
import json
import tarfile
import zipfile
from pathlib import Path

XENIUM_FILES = ("experiment.xenium", "cell_feature_matrix.h5", "cells.parquet", "cells.csv.gz",
                "cells.zarr.zip", "cell_boundaries.parquet", "nucleus_boundaries.parquet", "transcripts.parquet",
                "analysis.zarr.zip", "gene_panel.json", "metrics_summary.csv", "morphology_focus/ch0000_dapi.ome.tif",
                "cell_feature_matrix/matrix.mtx.gz", "cell_feature_matrix/features.tsv.gz",
                "cell_feature_matrix/barcodes.tsv.gz", "analysis/clustering/gene_expression_graphclust/clusters.csv")
ATERA_FILES = ("experiment.spatial", "cell_feature_matrix.zarr.zip", "cells.zarr.zip", "transcripts.zarr.zip",
               "binned_transcripts.zarr.zip", "morphology_2d/ch0000_dapi.ome.tif", "panel_config.json")


def content(name: str) -> bytes:
    """Recognisable bytes per file name (so staging tests can check the right member arrived)."""
    return f"content of {name}\n".encode()


def write(path: Path, data: bytes | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content(path.name) if data is None else data)
    return path


def xenium_bundle(folder: Path, files=XENIUM_FILES, marker: bool = True) -> Path:
    for name in files:
        if marker or name != "experiment.xenium":
            write(folder / name)
    return folder


def atera_bundle(folder: Path, marker: bool = True) -> Path:
    for name in ATERA_FILES:
        if marker or name != "experiment.spatial":
            write(folder / name)
    return folder


def zip_folder(src: Path, out: Path, arcroot: str | None = None, extra: dict[str, bytes] | None = None) -> Path:
    """Zip `src` with its files under `arcroot/` (default: the folder's name; "" for the zip's root)."""
    root = src.name if arcroot is None else arcroot
    with zipfile.ZipFile(out, "w") as zf:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                rel = f.relative_to(src).as_posix()
                zf.write(f, f"{root}/{rel}" if root else rel)
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
    return out


def tar_folder(src: Path, out: Path, arcroot: str | None = None) -> Path:
    mode = "w:gz" if out.name.endswith((".tar.gz", ".tgz")) else "w"
    with tarfile.open(out, mode) as tf:
        tf.add(src, arcname=src.name if arcroot is None else arcroot)
    return out


def two_bundle_zip(tmp: Path) -> Path:
    """A zip with two runs side by side at depth 1 (t5 finding 1: one used to be dropped silently)."""
    src = tmp / "two_src"
    xenium_bundle(src / "Xenium_V1_human_Breast_2fov")
    xenium_bundle(src / "Xenium_V1_human_Lung_2fov")
    out = tmp / "two_runs.zip"
    with zipfile.ZipFile(out, "w") as zf:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(src).as_posix())
    return out


def pipestance_tar(tmp: Path, run: str = "GSM7990532_output-XETG00048__0003392") -> Path:
    """A GEO tar.gz of a Xenium Ranger pipestance: `<run>/outs/` plus martian internals, including a decoy
    `experiment.xenium` inside XENIUM_RANGER_CS."""
    src = tmp / "pipe_src" / run
    xenium_bundle(src / "outs")
    for name in ("_cmdline", "_log", "_versions", "_finalstate", "journal", "pipeline.mro",
                 f"relabel_{run}.mri.tgz", "XENIUM_RANGER_CS/INSITU_COUNTER_CS/fork0/files/experiment.xenium",
                 "XENIUM_RANGER_CS/INSITU_COUNTER_CS/fork0/files/cell_feature_matrix.h5"):
        write(src / name)
    return tar_folder(src, tmp / f"{run}.tar.gz")


def macos_zip(tmp: Path, name: str = "Xenium_V1_human_Breast_2fov") -> Path:
    """A zip made on macOS: `__MACOSX/` twins with `._experiment.xenium` decoys, `.DS_Store`, `._*` files."""
    src = tmp / "mac_src" / name
    xenium_bundle(src)
    write(src / ".DS_Store")
    write(src / "._cells.parquet")
    extra = {f"__MACOSX/{name}/._experiment.xenium": b"\x00\x05\x16\x07",
             f"__MACOSX/{name}/._cell_feature_matrix.h5": b"\x00\x05\x16\x07",
             f"__MACOSX/._{name}": b"\x00\x05\x16\x07"}
    return zip_folder(src, tmp / f"{name}.zip", extra=extra)


def per_file_10x(folder: Path, dataset: str = "Xenium_FFPE_Human_Breast_Cancer_Rep1") -> Path:
    """10x per-file downloads: every file carries the dataset prefix (`<dataset>_experiment.xenium`)."""
    for name in ("experiment.xenium", "cell_feature_matrix.h5", "cells.parquet", "cells.zarr.zip",
                 "cell_boundaries.csv.gz", "transcripts.parquet", "morphology_mip.ome.tif", "analysis.zarr.zip",
                 "cell_feature_matrix.zarr.zip"):
        write(folder / f"{dataset}_{name}")
    return folder


def gz(path: Path, name: str) -> Path:
    return write(path, gzip.compress(content(name)))


def geo_flat(folder: Path) -> Path:
    """An extracted GSE..._RAW.tar: two GSMs flattened into one folder, GEO's per-file gzip, one with an MTX
    triplet and a `-` separator variant, one with a re-segmentation variant and an authors' table, plus an
    unrelated Chromium matrix and series-level files."""
    a = "GSM9829813_CtrlM1_"
    for name in ("experiment.xenium", "cells.parquet", "cell_boundaries.parquet", "transcripts.parquet",
                 "reseg_transcripts.parquet", "morphology.ome.tif"):
        gz(folder / f"{a}{name}.gz", name)
    write(folder / f"{a}cell_feature_matrix.h5")
    write(folder / f"{a}cell_metadata.csv.gz")
    b = "GSM9543314_EX-LN1-"
    for name in ("matrix.mtx", "features.tsv", "barcodes.tsv", "cells.csv"):
        gz(folder / f"{b}{name}.gz", name)
    gz(folder / f"{b}nucleus_boundaries.parquet.gz", "nucleus_boundaries.parquet")
    write(folder / "GSM7782697_3p_count_filtered_feature_bc_matrix.h5")
    write(folder / "GSE336196_README.txt")
    return folder


def mb_flat(folder: Path) -> Path:
    """S-BIAD1093: `MB263-…` and `MB266-…` files in one folder, a stray fused image, a lone prefixed marker
    one level up."""
    for sample in ("MB263", "MB266"):
        for name in ("experiment.xenium", "cell_feature_matrix.h5", "cells.parquet", "cells.zarr.zip",
                     "transcripts.parquet", "analysis.zarr.zip", "gene_panel.json"):
            write(folder / "Files" / "XENIUM_TRANSFER" / f"{sample}-{name}")
    write(folder / "Files" / "XENIUM_TRANSFER" / "MAX_MBEN_ff_Xenium_MB-263B_DAPI_Fused_405nm.tif")
    write(folder / "MB263-experiment.xenium")
    return folder


def sd_v3(folder: Path, root_attrs: bool = True, elements=("images", "tables")) -> Path:
    """A SpatialData store, zarr v3 (`zarr.json`); without root attrs it looks like a pre-Nov-2024 store."""
    attrs = {"spatialdata_attrs": {"version": "0.2"}} if root_attrs else {}
    write(folder / "zarr.json", json.dumps({"zarr_format": 3, "node_type": "group", "attributes": attrs}).encode())
    for element in elements:
        write(folder / element / "zarr.json", json.dumps({"zarr_format": 3, "node_type": "group"}).encode())
        write(folder / element / "x" / "zarr.json", json.dumps({"zarr_format": 3, "node_type": "group"}).encode())
    return folder


def sd_v2_old(folder: Path) -> Path:
    """A pre-0.1 SpatialData store: zarr v2, no root attrs, the table at `table/table`."""
    write(folder / ".zgroup", b'{"zarr_format": 2}')
    for element in ("images", "shapes", "table"):
        write(folder / element / ".zgroup", b'{"zarr_format": 2}')
    write(folder / "table" / "table" / ".zgroup", b'{"zarr_format": 2}')
    write(folder / "table" / "table" / ".zattrs", b'{"encoding-type": "anndata"}')
    write(folder / "images" / "img" / ".zattrs", b'{"multiscales": []}')
    return folder


def plain_zarr(folder: Path) -> Path:
    """A zarr that is not SpatialData (SSAM-like: groups of arrays)."""
    write(folder / ".zgroup", b'{"zarr_format": 2}')
    write(folder / "cell_by_gene_matrix" / ".zarray", b'{"zarr_format": 2}')
    write(folder / "centroids" / ".zarray", b'{"zarr_format": 2}')
    return folder


def anndata_zarr(folder: Path) -> Path:
    write(folder / "zarr.json", json.dumps({"zarr_format": 3, "node_type": "group",
                                            "attributes": {"encoding-type": "anndata"}}).encode())
    write(folder / "X" / "zarr.json", b'{"zarr_format": 3, "node_type": "array"}')
    return folder


def mixed_parent(folder: Path) -> Path:
    """A download folder: Xenium + Atera (with and without marker) + SpatialData + junk + unrelated files."""
    xenium_bundle(folder / "PatientA_run1" / "outs")
    atera_bundle(folder / "AteraKidney")
    atera_bundle(folder / "AteraNoMarker", marker=False)
    sd_v3(folder / "atlas.zarr")
    plain_zarr(folder / "ssam.zarr")
    write(folder / ".DS_Store")
    write(folder / "._PatientA_run1")
    write(folder / "__MACOSX" / "PatientA_run1" / "._experiment.xenium")
    write(folder / "notes.txt")
    with zipfile.ZipFile(folder / "qc_report.zip", "w") as zf:
        zf.writestr("report/index.html", "<html></html>")
    return folder
