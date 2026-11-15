"""The run's inputs: reading sample files, the precheck (every problem listed at once, before anything is
written), and picking which inputs are new to the store.

Readable inputs are HISE H5 files (10x matrix plus a per-cell `observations` group, registered as an
annizarr source), other 10x `.h5`, AnnData `.h5ad`/`.zarr`, and the table of a SpatialData `.zarr`.
Both HISE observation layouts are read onto the older HISE column names: tenx-scrna-pipeline's
(`Batch`, `Pool`, `Chip`, `uuid`, ...) is renamed through `_ALIASES`. Only Gene Expression features are
kept, so Xenium control probes and codewords never reach the store."""
from __future__ import annotations

import contextlib
import functools
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
from anndata.io import read_elem
from annizarr import AppConfig
from annizarr.config import IOConfig
from annizarr.sources import Source, open_source, register_source

from config import CELLTYPIST_PRESETS, Settings

logger = logging.getLogger("ingest")

KIND = "hise"
SPATIAL_TABLE = "tables/table"
_GEX = "Gene Expression"
_SAMPLE_ID = re.compile(r"_([A-Z0-9]+)-\d+$")
# 10x/Xenium output names that say nothing about the sample; the sample ID comes from the folder instead.
_GENERIC_STEMS = {"cell_feature_matrix", "filtered_feature_bc_matrix", "raw_feature_bc_matrix", "sample_filtered_feature_bc_matrix"}
_GENERIC_DIRS = {"outs", "extract", "outputs", "output", "data", "count"}
_ALIASES = {
    "sample_name": "pbmc_sample_id", "Batch": "batch_id", "Pool": "pool_id", "Chip": "chip_id",
    "hashtag": "hto_barcode", "uuid": "cell_uuid", "barcodes": "original_barcodes",
    "n_mitochondrial_counts": "n_mito_umis",
}


class IngestError(RuntimeError):
    """The inputs or the store's state don't allow this run."""


class ModelError(ValueError):
    """A CellTypist model file that can't be loaded."""


@dataclass(frozen=True)
class RawInput:
    """A raw sample (HISE/10x .h5, .h5ad, or a SpatialData table) and the sample ID its file name gives."""
    sample_id: str
    path: Path

    @property
    def name(self) -> str:
        return self.sample_id


@dataclass(frozen=True)
class ObjectInput:
    """An existing AnnData taken as it is: a unique label, the obs column naming its samples (None when
    the file name is the one sample), and its sample IDs."""
    label: str
    path: Path
    column: str | None
    sample_ids: tuple[str, ...]

    @property
    def name(self) -> str:
        return self.label


def _text(values: np.ndarray) -> np.ndarray:
    """h5py returns fixed-length strings as bytes ("S") and variable-length ones as an object array of bytes."""
    if values.dtype.kind == "S":
        return values.astype(str)
    if values.dtype.kind == "O":
        return np.array([v.decode() if isinstance(v, bytes) else v for v in values], dtype=object)
    return values


def sniff(path: Path) -> str | None:
    try:
        with h5py.File(path, "r") as f:
            matrix = f.get("matrix")
            return KIND if isinstance(matrix, h5py.Group) and "observations" in matrix else None
    except OSError:
        return None


def load(path: Path | str, cfg: AppConfig) -> Source:
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        counts = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:]))
        names = _text(m["features/name"][:])
        if "feature_type" in m["features"]:
            keep = np.asarray(_text(m["features/feature_type"][:]) == _GEX)
            counts, names = counts[keep], names[keep]
        var = pd.DataFrame(index=ad.utils.make_index_unique(pd.Index(names)))
        obs = pd.DataFrame(
            {col: _text(ds[:]) for col, ds in m["observations"].items()},
            index=pd.Index(_text(m["barcodes"][:]), name="barcodes"),
        )
        obs = obs.rename(columns={k: v for k, v in _ALIASES.items() if k in obs and v not in obs})
    adata = ad.AnnData(counts.T.tocsr(), obs=obs, var=var)
    return Source(adata=adata, kind=KIND, lazy=False, warnings=())


register_source(KIND, load, sniffer=sniff)


def zarr_kind(path: Path | str) -> str | None:
    """What a zarr folder holds: "spatialdata", "anndata", "mudata", "zarr" (anything else), or None."""
    return _zarr_kind(str(path))


@functools.lru_cache(maxsize=None)
def _zarr_kind(path: str) -> str | None:
    path = Path(path)
    if not path.is_dir():
        return None
    try:
        root = zarr.open_group(path, mode="r")
    except Exception:
        return None
    if "spatialdata_attrs" in root.attrs and SPATIAL_TABLE in root:
        return "spatialdata"
    if root.attrs.get("encoding-type") == "MuData" or "mod" in root:
        return "mudata"
    if "X" in root and "obs" in root:
        return "anndata"
    return "zarr"


def read_object(path: Path | str) -> ad.AnnData:
    """An existing AnnData (`.h5ad`, AnnData `.zarr`, or a SpatialData table) as it is, in memory."""
    path = Path(path)
    kind = zarr_kind(path)
    if kind == "spatialdata":
        return ad.io.read_elem(zarr.open_group(path, mode="r")[SPATIAL_TABLE])
    return ad.read_zarr(path) if kind == "anndata" else ad.read_h5ad(path)


@contextlib.contextmanager
def object_obs(path: Path | str):
    """The obs group of an existing AnnData (`.h5ad`, AnnData `.zarr`, or a SpatialData table)."""
    path = Path(path)
    kind = zarr_kind(path)
    group = zarr.open_group(path, mode="r") if kind else h5py.File(path, "r")
    try:
        yield group[f"{SPATIAL_TABLE}/obs" if kind == "spatialdata" else "obs"]
    finally:
        if not kind:
            group.close()


def object_n_obs(path: Path | str) -> int:
    with object_obs(path) as obs:
        return len(read_elem(obs[obs.attrs["_index"]]))


def object_sample_ids(path: Path | str, column: str) -> list[str | None] | None:
    """The distinct values of obs `column` in an existing AnnData (None for a missing value), read from
    obs alone, or None without that column."""
    with object_obs(path) as obs:
        if column not in obs:
            return None
        return sorted({None if pd.isna(v) else str(v) for v in pd.unique(read_elem(obs[column]))},
                      key=lambda v: (v is None, v))


def gene_mask(var: pd.DataFrame) -> np.ndarray:
    """True for the Gene Expression features; everything is a gene when var has no feature type."""
    for col in ("feature_types", "feature_type"):
        if col in var:
            return np.asarray(var[col].astype(str) == _GEX)
    return np.ones(len(var), dtype=bool)


def genes_only(adata: ad.AnnData) -> ad.AnnData:
    keep = gene_mask(adata.var)
    return adata if keep.all() else adata[:, keep].copy()


def read_sample(path: Path | str) -> ad.AnnData:
    """Any input annizarr can open (HISE, 10x, h5ad), or a SpatialData `.zarr`'s table, as an
    in-memory AnnData of raw counts, Gene Expression features only.

    Loaded eagerly: annizarr's lazy default leaves a 10x source's X on a separate reader and an
    h5ad's X backed by a file that close() shuts, and every step after this needs X in memory."""
    if zarr_kind(path) in ("spatialdata", "anndata"):
        return genes_only(read_object(path))
    source = open_source(path, AppConfig(io=IOConfig(lazy=False)))
    try:
        return genes_only(source.adata)
    finally:
        source.close()


def gene_names(path: Path | str) -> np.ndarray:
    """The genes `read_sample` will return, read from var alone when the layout allows, for checking
    panels before any work."""
    path = Path(path)
    try:
        return _gene_names(path)
    except KeyError:
        return np.asarray(read_sample(path).var_names, dtype=object)


def _gene_names(path: Path) -> np.ndarray:
    kind = zarr_kind(path)
    if kind in ("spatialdata", "anndata"):
        var = read_elem(zarr.open_group(path, mode="r")[f"{SPATIAL_TABLE}/var" if kind == "spatialdata" else "var"])
    elif path.suffix.lower() == ".h5ad":
        with h5py.File(path, "r") as f:
            var = read_elem(f["var"])
    else:
        with h5py.File(path, "r") as f:
            feat = f["matrix/features"]
            var = pd.DataFrame({"feature_type": _text(feat["feature_type"][:])} if "feature_type" in feat else {},
                               index=pd.Index(_text(feat["name"][:])))
            if "observations" in f["matrix"]:
                var = var[gene_mask(var)]
                var.index = ad.utils.make_index_unique(var.index)
    return np.asarray(var.index[gene_mask(var)], dtype=object)


def sample_id(path: Path | str) -> str:
    """`B092-P1_SP02425-01.h5 -> SP02425`; a generic 10x/Xenium file name (`cell_feature_matrix.h5`)
    takes its folder's name instead, skipping folders such as `outs` or `extract`."""
    path = Path(path)
    stem = path.stem
    if stem in _GENERIC_STEMS:
        parent = path.parent
        while parent.name.lower() in _GENERIC_DIRS and parent != parent.parent:
            parent = parent.parent
        stem = parent.name or stem
    match = _SAMPLE_ID.search(stem)
    return match.group(1) if match else stem


def load_models(paths: list[Path]) -> list[tuple[str, object]]:
    """Each CellTypist model, loaded once before any sample is read: `(name, model)` pairs."""
    from celltypist.models import Model

    loaded, bad = [], []
    for path in paths:
        try:
            loaded.append((path.stem, Model.load(model=str(path))))
        except Exception as exc:  # celltypist raises plain Exception for an invalid file
            bad.append(f"{path} ({str(exc).strip()})")
    if bad:
        raise ModelError(f"CellTypist model(s) can't be loaded: {'; '.join(bad)}. Give a .pkl written by "
                         "celltypist (e.g. celltypist.train(...).write(path)).")
    return loaded


def raise_problems(problems: list[str]) -> None:
    if len(problems) == 1:
        raise IngestError(problems[0])
    if problems:
        raise IngestError("The precheck found problems, so nothing was written. Fix them, or leave those inputs "
                          "out, and run again:\n  - " + "\n  - ".join(problems))


def input_problems(files: list[Path], existing: bool) -> list[str]:
    """What stops these inputs from being ingested; each unreadable file is named."""
    problems, flagged = [], set()

    def flag(paths: list[Path], message: str) -> None:
        if paths:
            problems.append(message.format(paths=[str(p) for p in paths]))
            flagged.update(paths)

    flag([f for f in files if not f.exists()], "Input files not found: {paths}")
    flag([f for f in files if f.suffix.lower() == ".h5mu" or zarr_kind(f) == "mudata"],
         "{paths} is MuData, which this capsule doesn't take; write the modality you want as an AnnData and "
         "ingest that, e.g. anndata.io.read_elem(zarr.open_group(path)['mod/rna']).write_h5ad('rna.h5ad').")
    flag([f for f in files if f.is_dir() and zarr_kind(f) not in ("anndata", "spatialdata", "mudata")],
         "{paths}: a .zarr input must be an AnnData, or a SpatialData with a " + SPATIAL_TABLE + ".")
    if existing:
        flag([f for f in files if f.suffix.lower() == ".h5"],
             "{paths} are raw .h5 samples; choose the input type 'Raw samples' for them.")
    for path in files:
        if path in flagged:
            continue
        try:
            object_n_obs(path) if existing else gene_names(path)
        except Exception as exc:
            problems.append(f"{path} can't be read ({type(exc).__name__}: {str(exc).strip()[:200]}); it may be "
                            "truncated or not an AnnData/10x file.")
    return problems


def model_files(settings: Settings) -> tuple[list[Path], list[str]]:
    """The CellTypist model files, and a problem when any is missing."""
    models = settings.model_files()
    missing = [str(m) for m in models if not m.is_file()]
    if missing:
        return models, [f"CellTypist models not found: {missing} (models_dir is {settings.models_dir}). The "
                        f"Celltypist Model options are {list(CELLTYPIST_PRESETS)}; a custom model is a .pkl path."]
    if models:
        source = ("custom .pkl, used instead of the Celltypist Model option" if settings.celltypist_pkl
                  else f"Celltypist Model {settings.celltypist_models!r}")
        logger.info("CellTypist models (%s): %s", source, ", ".join(m.stem for m in models))
    return models, []


def precheck(settings: Settings) -> tuple[list[Path], list[tuple[str, object]]]:
    """The input files and the loaded CellTypist models, once nothing stands in the way; every problem
    found is raised together in one IngestError."""
    files = settings.sample_files()
    if not files:
        raise IngestError(f"No .h5/.h5ad files or AnnData/SpatialData .zarr folders found in {settings.samples}.")
    problems = input_problems(files, settings.existing)
    paths, model_problems = ([], []) if settings.existing else model_files(settings)
    problems += model_problems
    models = []
    if paths and not model_problems:
        try:
            models = load_models(paths)
        except ModelError as exc:
            problems.append(str(exc))
    raise_problems(problems)
    raise_problems(model_gene_problems(files, models))
    return files, models


def model_gene_problems(files: list[Path], models: list[tuple[str, object]]) -> list[str]:
    """CellTypist needs the model's gene names in the sample: none shared is a problem (e.g. Ensembl IDs,
    mouse, a targeted panel), under half of them only a warning, since labels from so few genes are weak."""
    problems = []
    for path in files if models else []:
        genes = set(gene_names(path))
        for name, model in models:
            shared = len(genes.intersection(model.features))
            if shared == 0:
                problems.append(f"{path.name} shares no genes with the CellTypist model {name} (it expects gene "
                                f"symbols such as {', '.join(model.features[:3])}); choose Celltypist Model None, "
                                "or a model for this gene panel.")
            elif shared < len(model.features) / 2:
                logger.warning("%s has %d of CellTypist %s's %d genes; its labels rest on few genes.",
                               path.name, shared, name, len(model.features))
    return problems


def select_new_samples(files: list[Path], present: set[str], sources: dict[str, str]) -> list[RawInput]:
    """Raw samples not yet in the store, by the sample ID their file name gives. `sources` maps the
    store's samples to the files they came from: a skipped sample from another file is warned about."""
    chosen: dict[str, Path] = {}
    for path in files:
        sid = sample_id(path)
        if sid in present:
            before = sources.get(sid)
            if before and Path(before).name != path.name:
                logger.warning("%s: skipped, since %s is already in the store (from %s). The sample ID comes from "
                               "the SP number in the file name, so a re-sequenced batch counts as the same sample; "
                               "to keep both, ingest this one into a separate store.", path.name, sid, Path(before).name)
            else:
                logger.info("%s already in the store; skipping", sid)
        elif sid in chosen:
            logger.warning("%s listed twice (%s); keeping the first", sid, path.name)
        else:
            chosen[sid] = path
    return [RawInput(sid, path) for sid, path in chosen.items()]


def object_ids(path: Path, column: str) -> tuple[str | None, list[str]]:
    """The obs column naming an existing AnnData's samples (`column`, else `sample_id`, else none) and
    its sample IDs; without either column the file name is the one sample."""
    for name in dict.fromkeys((column, "sample_id")):
        ids = object_sample_ids(path, name)
        if ids is None:
            if name == column != "sample_id":
                logger.warning("%s has no %r column", path.name, column)
            continue
        if None in ids:
            raise IngestError(f"{path.name}: some cells have no {name}; fill it in before ingesting.")
        return name, ids
    return None, [sample_id(path)]


def select_new_objects(files: list[Path], present: set[str], column: str) -> list[ObjectInput]:
    """Existing AnnData objects whose samples are new; one that is partly in the store, or overlaps an
    earlier input, is refused rather than duplicated."""
    chosen, seen, labels = [], set(present), set()
    for path in files:
        if object_n_obs(path) == 0:
            logger.warning("%s has no cells; skipped", path.name)
            continue
        name, ids = object_ids(path, column)
        overlap = sorted(set(ids) & seen)
        if set(ids) <= present:
            logger.info("%s: its samples are already in the store; skipping", path.name)
            continue
        if overlap:
            raise IngestError(f"{path.name} holds samples that are already in the store or in an earlier input "
                              f"({', '.join(overlap[:5])}{', …' if len(overlap) > 5 else ''}); its cells would be "
                              "duplicated. Remove those samples from it, or leave it out.")
        seen.update(ids)
        label = next(f"{path.stem}{n or ''}" for n in range(len(files) + 1) if f"{path.stem}{n or ''}" not in labels)
        labels.add(label)
        chosen.append(ObjectInput(label, path, name, tuple(ids)))
    return chosen
