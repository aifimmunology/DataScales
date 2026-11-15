"""Checks before anything is written. `run` reads the inputs and models before the store is opened, and
gathers every problem into one IngestError, so a run with several bad inputs lists them all at once.
The `select_*` functions then compare the inputs with the store: which are new, which are already in
it, and which would duplicate cells."""
from __future__ import annotations

import logging
from pathlib import Path

import annotate
import contract
import hise
from config import CELLTYPIST_PRESETS, Settings
from contracts import IngestError, ObjectInput, RawInput

logger = logging.getLogger("ingest")


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
    flag([f for f in files if f.suffix.lower() == ".h5mu" or hise.zarr_kind(f) == "mudata"],
         "{paths} is MuData, which this capsule doesn't take; write the modality you want as an AnnData and "
         "ingest that, e.g. anndata.io.read_elem(zarr.open_group(path)['mod/rna']).write_h5ad('rna.h5ad').")
    flag([f for f in files if f.is_dir() and hise.zarr_kind(f) not in ("anndata", "spatialdata", "mudata")],
         "{paths}: a .zarr input must be an AnnData, or a SpatialData with a " + hise.SPATIAL_TABLE + ".")
    if existing:
        flag([f for f in files if f.suffix.lower() == ".h5"],
             "{paths} are raw .h5 samples; choose the input type 'Raw samples' for them.")
    for path in files:
        if path in flagged:
            continue
        try:
            hise.object_n_obs(path) if existing else hise.gene_names(path)
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


def run(settings: Settings) -> tuple[list[Path], list[tuple[str, object]]]:
    """The input files and the loaded CellTypist models, once nothing stands in the way."""
    files = settings.sample_files()
    if not files:
        raise IngestError(f"No .h5/.h5ad files or AnnData/SpatialData .zarr folders found in {settings.samples}.")
    problems = input_problems(files, settings.existing)
    paths, model_problems = ([], []) if settings.existing else model_files(settings)
    problems += model_problems
    models = []
    if paths and not model_problems:
        try:
            models = annotate.load_models(paths)
        except annotate.ModelError as exc:
            problems.append(str(exc))
    raise_problems(problems)
    return files, models


def select_new_samples(files: list[Path], present: set[str], sources: dict[str, str]) -> list[RawInput]:
    """Raw samples not yet in the store, by the sample ID their file name gives. `sources` maps the
    store's samples to the files they came from: a skipped sample from another file is warned about."""
    chosen: dict[str, Path] = {}
    for path in files:
        sid = hise.sample_id(path)
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
        ids = hise.object_sample_ids(path, name)
        if ids is None:
            if name == column != "sample_id":
                logger.warning("%s has no %r column", path.name, column)
            continue
        if None in ids:
            raise IngestError(f"{path.name}: some cells have no {name}; fill it in before ingesting.")
        return name, ids
    return None, [hise.sample_id(path)]


def select_new_objects(files: list[Path], present: set[str], column: str) -> list[ObjectInput]:
    """Existing AnnData objects whose samples are new; one that is partly in the store, or overlaps an
    earlier input, is refused rather than duplicated."""
    chosen, seen, labels = [], set(present), set()
    for path in files:
        if hise.object_n_obs(path) == 0:
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


def check_raw_schema(store_columns: list[str], schema: contract.Schema, run_scrublet: bool) -> None:
    """A store built from raw samples keeps its exact schema; one seeded from an existing AnnData keeps
    its own columns, and raw samples fill only those."""
    if set(contract.requested_schema(False, [])) <= set(store_columns):
        contract.check_schema(store_columns, schema, run_scrublet)
        return
    missing = [c for c in schema if c not in store_columns]
    if missing:
        logger.warning("The store was built from existing AnnData, so these raw-sample columns aren't kept: %s",
                       ", ".join(missing))
