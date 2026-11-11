"""The run's inputs: bundle folders, bundle .zip files and SpatialData .zarr folders, each resolved to
the samples it holds, with sample IDs checked before any work (spec §3S.7)."""
from __future__ import annotations

import glob
import logging
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import zarr
from anndata.io import read_elem

import existing
from config import DATA_DIR, in_data, split_list

MARKERS = {"xenium": "experiment.xenium", "atera": "experiment.spatial"}
NAMES = {"xenium": "Xenium", "atera": "Atera", "spatialdata": "SpatialData"}
GENERIC_DIRS = {"outs", "extract", "outputs", "output", "data", "count"}
SAMPLE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
XENIUM_MATRIX = "cell_feature_matrix.h5"
ATERA_MATRIX = "cell_feature_matrix.zarr.zip"


logger = logging.getLogger(__name__)


class InputError(ValueError):
    """Inputs that can't go into the store as given."""


@dataclass(frozen=True)
class Input:
    """One sample to add: a bundle (folder or .zip), or one sample of a SpatialData object."""
    sid: str
    path: Path
    kind: str
    part: existing.Part | None = None

    @property
    def zipped(self) -> bool:
        return self.path.suffix.lower() == ".zip"


def zip_root(path: Path) -> str | None:
    """The folder inside a bundle .zip that holds its files ("" for the zip's root), found by the
    bundle's marker file, or None when the zip holds no bundle."""
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    for marker in MARKERS.values():
        roots = sorted({n[: -len(marker)] for n in names if n == marker or n.endswith("/" + marker)},
                       key=lambda r: r.count("/"))
        if roots and roots[0].count("/") <= 1:
            return roots[0]
    return None


def bundle_kind(path: Path) -> str | None:
    """xenium or atera for a bundle folder or .zip, else None."""
    if path.is_dir():
        return next((kind for kind, marker in MARKERS.items() if (path / marker).is_file()), None)
    if path.suffix.lower() == ".zip" and zipfile.is_zipfile(path):
        root = zip_root(path)
        if root is None:
            return None
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
        return next((kind for kind, marker in MARKERS.items() if root + marker in names), None)
    return None


def is_spatialdata(path: Path) -> bool:
    return path.is_dir() and existing.is_spatialdata(path)


def _is_candidate(path: Path, kind: str) -> bool:
    if path.name.startswith("."):
        return False
    if kind == "spatialdata":
        return is_spatialdata(path) or (path.is_dir() and path.suffix.lower() == ".zarr")
    return bundle_kind(path) is not None or path.suffix.lower() == ".zip"


def _expand(entry: str, kind: str) -> list[Path]:
    what = f"{NAMES[kind]} {'bundles' if kind != 'spatialdata' else '.zarr folders'}"
    if glob.has_magic(entry):
        matches = glob.glob(entry) or ([] if Path(entry).is_absolute() else glob.glob(str(DATA_DIR / entry)))
        found = [p for p in sorted(map(Path, matches)) if _is_candidate(p, kind)]
        if not found:
            raise InputError(f"No {what} match {entry!r}.")
        return found
    path = in_data(Path(entry))
    if not path.exists():
        raise InputError(f"Input not found: {entry}")
    if path.is_file() or bundle_kind(path) or is_spatialdata(path) or path.suffix.lower() == ".zarr":
        return [path]
    children = sorted(p for p in path.iterdir() if not p.name.startswith("."))
    found = [p for p in children if _is_candidate(p, kind)]
    if not found:
        raise InputError(f"The folder {path} holds no {what} (subfolders of subfolders aren't searched).")
    ignored = [p.name for p in children if p not in found]
    if ignored:
        logger.warning("%s: not %s, so ignored: %s", path, what, ", ".join(ignored[:10]) + (" …" if len(ignored) > 10 else ""))
    return found


def list_paths(samples: str, kind: str) -> list[Path]:
    """Every listed path, folder (its direct children) or glob, in order and without repeats."""
    paths: dict[Path, None] = {}
    for entry in split_list(samples):
        for path in _expand(entry, kind):
            paths.setdefault(path, None)
    if not paths:
        raise InputError(f"No inputs given in {samples!r}.")
    return list(paths)


def bundle_id(path: Path) -> str:
    """A bundle folder's name, or a .zip's stem without a trailing `_outs`; generic folder names
    (`outs`, `extract`, ...) are skipped up to the first meaningful parent."""
    if path.suffix.lower() == ".zip":
        return re.sub(r"_outs$", "", path.stem)
    while path.name.lower() in GENERIC_DIRS and path != path.parent:
        path = path.parent
    return path.name


def zarr_id(path: Path) -> str:
    name = path.name
    return name[: -len(".zarr")] if name.lower().endswith(".zarr") else name


def _check_bundle(path: Path, kind: str) -> None:
    found = bundle_kind(path)
    if found == kind:
        return
    if found:
        raise InputError(f"{path} is {NAMES[found]} bundle; choose the input type '{NAMES[found]} bundle' for it.")
    if is_spatialdata(path):
        raise InputError(f"{path} is a SpatialData .zarr; choose the input type 'Existing SpatialData' for it.")
    raise InputError(f"{path} is not {NAMES[kind]} bundle (no {MARKERS[kind]} in it).")


def resolve(settings) -> list[Input]:
    """The run's samples, in input order. Sample IDs must be valid SpatialData names and unique."""
    kind = settings.input_type
    inputs: list[Input] = []
    for path in list_paths(settings.samples, kind):
        if kind == "spatialdata":
            if not is_spatialdata(path):
                raise InputError(f"{path} is not a SpatialData .zarr (no spatialdata_attrs in its zarr.json).")
            inputs += [Input(part.sid, path, kind, part)
                       for part in existing.plan(path, zarr_id(path), settings.sample_column)]
        else:
            _check_bundle(path, kind)
            inputs.append(Input(bundle_id(path), path, kind))
    bad = [i.sid for i in inputs if not SAMPLE_ID.match(i.sid)]
    if bad:
        raise InputError(f"Sample IDs must use only letters, digits, '_', '.' and '-' (SpatialData element names): "
                         f"{bad}. Rename the folders, files or sample IDs.")
    seen: dict[str, Path] = {}
    for item in inputs:
        if item.sid in seen:
            raise InputError(f"Two inputs have the sample ID {item.sid!r}: {seen[item.sid]} and {item.path}.")
        seen[item.sid] = item.path
    return inputs


def _text(values: np.ndarray) -> np.ndarray:
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values], dtype=object)


def _xenium_var(handle) -> pd.DataFrame:
    with h5py.File(handle, "r") as f:
        feat = f["matrix/features"]
        return pd.DataFrame({"gene_ids": _text(feat["id"][:]), "feature_types": _text(feat["feature_type"][:])},
                            index=pd.Index(_text(feat["name"][:])))


def _atera_var(matrix: Path) -> pd.DataFrame:
    store = zarr.storage.ZipStore(matrix, mode="r", read_only=True)
    try:
        return read_elem(zarr.open_group(store, mode="r")["var"])
    finally:
        store.close()


def raw_var(item: Input, scratch: Path) -> pd.DataFrame:
    """A sample's var as its reader will see it, read without reading the sample."""
    if item.kind == "spatialdata":
        return existing.table_var(item.path, item.part)
    member = XENIUM_MATRIX if item.kind == "xenium" else ATERA_MATRIX
    if not item.zipped:
        return _xenium_var(item.path / member) if item.kind == "xenium" else _atera_var(item.path / member)
    target = scratch / "panels" / item.sid
    target.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(item.path) as zf, zf.open(zip_root(item.path) + member) as src, \
                open(target / member, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 24)
        return _xenium_var(target / member) if item.kind == "xenium" else _atera_var(target / member)
    finally:
        shutil.rmtree(target, ignore_errors=True)


def extract(item: Input, scratch: Path) -> Path:
    """A zipped bundle extracted to `<scratch>/extract/<sid>/`; the folder holding its files."""
    target = scratch / "extract" / item.sid
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    with zipfile.ZipFile(item.path) as zf:
        zf.extractall(target)
    return target / zip_root(item.path) if zip_root(item.path) else target
