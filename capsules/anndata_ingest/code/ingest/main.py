"""Append samples to an AnnData store, one sample in memory at a time.

Every new sample is read, annotated (QC metrics, CellTypist, Scrublet), projected onto the store
contract, and staged as a zarr before the store is touched. Then the first sample seeds a new
store, and the rest are appended. On icechunk the appends land as one commit.
"""
from __future__ import annotations

import json
import logging
import resource
import shutil
import sys
from dataclasses import asdict, dataclass
from functools import partial
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import annizarr
from annizarr import AppConfig, Repo
from annizarr.config import ChunkConfig, IOConfig
from annizarr.errors import AnzError

import annotate
import contract
import hise
import objects
import store as stores
from config import ConfigError, Settings, load_settings

logger = logging.getLogger("ingest")

USER_ERRORS = (ConfigError, contract.ContractError, stores.StoreError, AnzError)


class IngestError(RuntimeError):
    """The inputs or the store's state don't allow this run."""


@dataclass(frozen=True)
class Staged:
    sample_id: str
    path: Path
    summary: dict
    source: str = ""
    fitted: str = ""
    samples: tuple[str, ...] = ()

    @property
    def line(self) -> str:
        return f"- {self.sample_id}: {self.source}" + (f" ({self.fitted})" if self.fitted else "")


def annizarr_config(settings: Settings) -> AppConfig:
    return AppConfig(
        io=IOConfig(overwrite=True),
        chunks=ChunkConfig(row_chunk=settings.chunk_cells, cpus=settings.write_cpus),
    )


def input_files(settings: Settings) -> list[Path]:
    files = settings.sample_files()
    if not files:
        raise IngestError(f"No .h5/.h5ad samples found in {settings.samples}.")
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise IngestError(f"Input files not found: {missing}")
    mudata = [str(f) for f in files if f.suffix.lower() == ".h5mu" or hise.zarr_kind(f) == "mudata"]
    if mudata:
        raise IngestError(f"{mudata} is MuData, which this capsule doesn't take; write the modality you want "
                          "as an AnnData and ingest that, e.g. anndata.io.read_elem(zarr.open_group(path)['mod/rna'])"
                          ".write_h5ad('rna.h5ad').")
    other = [str(f) for f in files if f.is_dir() and hise.zarr_kind(f) not in ("anndata", "spatialdata")]
    if other:
        raise IngestError(f"{other}: a .zarr input must be an AnnData, or a SpatialData with a {hise.SPATIAL_TABLE}.")
    if settings.existing:
        raw = [str(f) for f in files if f.suffix.lower() == ".h5"]
        if raw:
            raise IngestError(f"{raw} are raw .h5 samples; choose the input type 'Raw samples' for them.")
    return files


def model_files(settings: Settings) -> list[Path]:
    models = settings.model_files()
    missing = [str(m) for m in models if not m.is_file()]
    if missing:
        raise IngestError(f"CellTypist models not found: {missing} (models_dir is {settings.models_dir}).")
    return models


REFRESHED = {"/scratch": "is cleared when the run ends", "/results": "belongs to this run: it can be saved as a data "
              "asset, but a data asset is read-only, so later runs can't append to it", "/tmp": "is cleared when the run ends"}


def storage_warning(location: str) -> str | None:
    """A warning for a store in a location that doesn't keep it for later runs to append to."""
    if "://" in location:
        return None
    path = Path(location).resolve()
    for root, why in REFRESHED.items():
        if path.is_relative_to(root):
            return (f"The store {location} is in {root}, which {why}. A store you will keep adding to belongs "
                    "on durable S3: give an s3://bucket/prefix, or a data asset linked to it with its Origin of s3.")
    return None


def open_store(settings: Settings) -> stores.IcechunkStore | stores.ZarrStore:
    """Opens the store, creating it when it doesn't exist yet and its location allows that."""
    store = stores.open_store(settings.store, settings.origin)
    fresh = not store.exists()
    if fresh and store.location != settings.store and Repo.exists(settings.store):
        raise IngestError(f"{settings.store} holds a store but its origin {store.location} doesn't; check the "
                          "Origin of s3 (a mistyped origin would start a new, separate store).")
    if fresh and not settings.may_create(store.location):
        raise IngestError(
            f"No store at {store.location}. New stores are created only inside {settings.create_roots} or at an "
            "s3:// prefix (/data is read-only); check the path of an existing store, or set CREATE=true to create one "
            "elsewhere.")
    if fresh:
        logger.warning("No store at %s; creating a new %s store there.", store.location, store.kind)
    warning = storage_warning(store.location)
    if warning:
        logger.warning("%s", warning)
    store.open(fresh)
    return store


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


def select_new_objects(files: list[Path], present: set[str],
                       column: str) -> list[tuple[str, Path, str | None, list[str]]]:
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
        chosen.append((label, path, name, ids))
    return chosen


def select_new_samples(files: list[Path], present: set[str]) -> list[tuple[str, Path]]:
    chosen: dict[str, Path] = {}
    for path in files:
        sid = hise.sample_id(path)
        if sid in present:
            logger.info("%s already in the store; skipping", sid)
        elif sid in chosen:
            logger.warning("%s listed twice (%s); keeping the first", sid, path.name)
        else:
            chosen[sid] = path
    return list(chosen.items())


def check_panels(new: list[tuple[str, Path]], expected_var: np.ndarray | None) -> None:
    """Warns, from var alone and before any sample is read, about each new sample whose genes differ
    from the store's (or, for a new store, the first sample's). Staging fits such samples."""
    reference = expected_var
    diffs = {}
    for sid, path in new:
        genes = hise.gene_names(path)
        if reference is None:
            reference = genes
            continue
        diff = contract.panel_diff(reference, genes)
        if diff:
            diffs[sid] = diff
    if diffs:
        against = "the store's" if expected_var is not None else f"the first sample's ({new[0][0]}, {len(reference)} genes)"
        logger.warning("Gene panel differs from %s for %d sample(s); they are fitted to its genes (missing genes "
                       "zero-filled, extra genes dropped): %s", against, len(diffs),
                       "; ".join(contract.describe_diff(sid, d) for sid, d in diffs.items()))


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


def annotate_sample(adata: ad.AnnData, sid: str, settings: Settings) -> ad.AnnData:
    adata.obs_names = [f"{sid}:{bc}" for bc in adata.obs_names]
    adata.obs["sample_id"] = sid
    annotate.qc_metrics(adata)
    models = settings.model_files()
    if models:
        annotate.run_celltypist(adata, models)
    if settings.run_scrublet:
        annotate.run_scrublet(adata, random_state=settings.random_seed)
    return adata


def summarize(obs: pd.DataFrame) -> dict:
    row = {"n_cells": len(obs), "n_samples": obs["sample_id"].nunique()}
    if "n_genes_by_counts" in obs:
        row["mean_genes"] = obs["n_genes_by_counts"].mean()
        row["mean_pct_mt"] = obs["pct_counts_mt"].mean()
    if "predicted_doublets" in obs:
        row["n_doublets"] = obs["predicted_doublets"].sum(min_count=1)
        row["scrublet_scored"] = bool(obs["scrublet_score"].notna().any())
    for column in obs.columns[obs.columns.str.endswith("_label")]:
        row[f"n_{column.removesuffix('_label')}_types"] = obs[column].nunique()
    return row


@dataclass
class Target:
    """What staged cells must match: the store's genes, X dtype, whether X holds counts, and obs kinds.
    A new store takes them from its first staged sample."""
    var: np.ndarray | None = None
    x_dtype: np.dtype | None = None
    counts: bool = False
    obs: dict[str, str] | None = None

    @classmethod
    def of(cls, root) -> "Target":
        if not stores.has_data(root):
            return cls()
        return cls(stores.var_names(root), stores.x_dtype(root), stores.x_holds_counts(root), stores.obs_kinds(root))

    def fit_genes(self, adata: ad.AnnData) -> tuple[ad.AnnData, str]:
        """The sample on the store's genes, and a note on how they were fitted."""
        diff = None if self.var is None else contract.panel_diff(self.var, adata.var_names)
        return (contract.conform_var(adata, self.var), contract.fit_note(diff)) if diff else (adata, "")

    def fit_cells(self, adata: ad.AnnData, name: str) -> ad.AnnData:
        """The sample's X and obs in the store's dtype and kinds; the first sample of a new store sets them."""
        adata.X = contract.fit_x(adata.X, self.x_dtype, self.counts, name)
        if self.obs is not None:
            adata.obs, failed = contract.match_obs(adata.obs, self.obs)
            if failed:
                logger.warning("%s: obs columns left empty, since their values don't fit the store's types: %s",
                               name, ", ".join(failed))
        if self.var is None:
            self.learn(np.asarray(adata.var_names, dtype=object), adata.X.dtype, contract.holds_counts(adata.X),
                       contract.frame_kinds(adata.obs))
        return adata

    def learn(self, var: np.ndarray, x_dtype: np.dtype, counts: bool, obs: dict[str, str]) -> None:
        self.var, self.x_dtype, self.counts, self.obs = var, x_dtype, counts, obs


def stage_samples(new: list[tuple[str, Path]], settings: Settings, schema: contract.Schema, target: Target,
                  cfg: AppConfig, staging: Path) -> list[Staged]:
    staged = []
    for sid, path in new:
        adata = hise.read_sample(path)
        logger.info("%s: %d cells x %d genes", sid, adata.n_obs, adata.n_vars)
        if adata.n_obs == 0:
            logger.warning("%s has no cells; skipped", sid)
            continue
        adata, note = target.fit_genes(annotate_sample(adata, sid, settings))
        adata = target.fit_cells(contract.project(adata, schema, target.var), sid)
        out = staging / f"{len(staged):04d}.zarr"
        annizarr.convert(adata, output=str(out), cfg=cfg)
        staged.append(Staged(sid, out, summarize(adata.obs), str(path), note, (sid,)))
        del adata
    return staged


def stage_objects(new: list[tuple[str, Path, str | None, list[str]]], target: Target, cfg: AppConfig,
                  staging: Path) -> list[Staged]:
    """Existing AnnData objects, streamed by annizarr. The first object of a new store keeps everything;
    later ones keep X, the store's obs columns and var, since an append can't extend embeddings or layers."""
    staged = []
    for label, path, column, ids in new:
        out = staging / f"{len(staged):04d}.zarr"
        obs, note, left_out = objects.stage_object(path, out, column, ids[0], target, keep_all=target.var is None,
                                                   cfg=cfg, name=label)
        logger.info("%s: %d cells, samples %s", label, len(obs), ids)
        if left_out:
            logger.warning("%s: not added, since the store has no place for them: %s", label, ", ".join(left_out))
        staged.append(Staged(label, out, summarize(obs), str(path), note, tuple(ids)))
    return staged


def appended(staged: list[Staged], dropped: list[str]) -> dict:
    return {
        "appended_samples": [sid for s in staged for sid in s.samples],
        "sources": {s.sample_id: s.source for s in staged},
        "fitted": {s.sample_id: s.fitted for s in staged if s.fitted},
        "dropped": dropped if staged else [],
    }


def record(staged: list[Staged], dropped: list[str], n_cells: int, n_samples: int) -> tuple[str, dict]:
    """A commit's message and metadata: the samples it adds, where each came from, and the totals after it."""
    added = appended(staged, dropped)
    lines = "\n".join(s.line for s in staged) + (f"\nDropped (built for the old cells): {', '.join(dropped)}"
                                                    if dropped else "")
    message = f"Ingest: appended {added['appended_samples']}; {n_cells:,} cells\n{lines}"
    return message, {"pipeline": "ingest", "annizarr": annizarr.__version__, **added,
                     "n_cells": n_cells, "n_samples": n_samples}


def commit(store, staged: list[Staged], dropped: list[str]) -> str | None:
    root = store.working_root()
    return store.commit(*record(staged, dropped, stores.n_obs(root), len(stores.sample_ids(root))))


def write(store, staged: list[Staged], seeding: bool, cfg: AppConfig,
          dropped: list[str]) -> tuple[str | None, str | None, list[str]]:
    """Seeds a new store with the first sample if needed, appends the rest, and commits them once.

    Returns the seed and run snapshot ids (the same id when the seed is the only write; None on a
    plain zarr), and the elements the appends dropped."""
    seed = None
    if seeding:
        first = staged[0]
        seed = store.bootstrap(first.path, cfg, *record([first], [], int(first.summary["n_cells"]),
                                                        int(first.summary["n_samples"])))
    rest = staged[1:] if seeding else staged
    if not rest:
        return seed, seed, dropped
    root = store.working_root()
    if seeding:
        dropped = stores.derived_elements(root)
        if dropped:
            logger.warning("%s's %s can't extend to the other samples of this run; dropped.",
                           staged[0].sample_id, ", ".join(dropped))
    stores.drop_unappendable(root)
    for item in rest:
        store.append(str(item.path), cfg)
    return seed, commit(store, staged, dropped), dropped


def ingest(store, stager, settings: Settings, seeding: bool,
           dropped: list[str]) -> tuple[list[Staged], str | None, str | None, list[str]]:
    cfg = annizarr_config(settings)
    staging = Path(settings.staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        staged = stager(cfg, staging)
        return (staged, *write(store, staged, seeding, cfg, dropped)) if staged else ([], None, None, dropped)
    except BaseException:
        store.discard()
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def write_results(settings: Settings, store, staged: list[Staged], seed: str | None, snapshot: str | None,
                  dropped: list[str]) -> dict:
    root = store.root()
    report = {
        "store": store.location,
        "kind": store.kind,
        "snapshot": snapshot,
        "bootstrap_snapshot": seed,
        **appended(staged, dropped),
        "n_cells": stores.n_obs(root),
        "n_samples": len(stores.sample_ids(root)),
        "annizarr": annizarr.__version__,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "peak_memory_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 2),
        "settings": asdict(settings),
    }
    results = Path(settings.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    (results / "ingest_run.json").write_text(json.dumps(report, indent=2, default=str))
    if staged:
        summary = pd.DataFrame({s.sample_id: s.summary for s in staged}).T.rename_axis("sample_id")
        summary.to_csv(results / "ingest_summary.csv")
    return report


def run(settings: Settings) -> dict:
    files = input_files(settings)
    models = [] if settings.existing else model_files(settings)
    store = open_store(settings)
    logger.info("Store %s (%s)", store.location, store.kind)
    root = store.root()
    seeding = not stores.has_data(root)
    present = set() if seeding else stores.sample_ids(root)
    new = select_new_objects(files, present, settings.sample_column) if settings.existing else select_new_samples(files, present)
    logger.info("%d samples in store, %d new input(s): %s", len(present), len(new), [n[0] for n in new])

    schema = contract.requested_schema(settings.run_scrublet, [m.stem for m in models])
    if not seeding and not settings.existing:
        check_raw_schema(stores.obs_columns(root), schema, settings.run_scrublet)
    target = Target.of(root)
    check_panels([n[:2] for n in new], target.var)
    dropped = stores.derived_elements(root) if new and not seeding else []
    if dropped:
        logger.warning("%s holds elements built for its current cells (%s); appending new cells drops them. "
                       "Re-run the analysis afterwards.", store.location, ", ".join(dropped))
    stager = (partial(stage_objects, new, target) if settings.existing
              else partial(stage_samples, new, settings, schema, target))
    staged, seed, snapshot, dropped = (ingest(store, stager, settings, seeding, dropped) if new
                                       else ([], None, None, dropped))
    if seeding and not staged:
        raise IngestError("The store is empty and none of the samples has cells to ingest.")
    if not staged:
        logger.info("Nothing new to ingest; the store is unchanged.")

    report = write_results(settings, store, staged, seed, snapshot, dropped)
    logger.info("Done: %s holds %d cells from %d sample(s).", store.location, report["n_cells"], report["n_samples"])
    return report


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    sc.settings.verbosity = 0
    try:
        run(load_settings(argv))
    except (IngestError, *USER_ERRORS) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
