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
    not_spatial = [str(f) for f in files if f.is_dir() and not hise.is_spatialdata(f)]
    if not_spatial:
        raise IngestError(f"{not_spatial}: a .zarr sample must be a SpatialData object with a {hise.SPATIAL_TABLE}.")
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


def prepare_sample(adata: ad.AnnData, sid: str, settings: Settings, schema: contract.Schema,
                   expected_var: np.ndarray | None, fit: bool) -> ad.AnnData:
    adata.obs_names = [f"{sid}:{bc}" for bc in adata.obs_names]
    adata.obs["sample_id"] = sid
    annotate.qc_metrics(adata)
    models = settings.model_files()
    if models:
        annotate.run_celltypist(adata, models)
    if settings.run_scrublet:
        annotate.run_scrublet(adata, random_state=settings.random_seed)
    if fit:
        adata = contract.conform_var(adata, expected_var)
    return contract.project(adata, schema, expected_var)


def summarize(obs: pd.DataFrame) -> dict:
    row = {
        "n_cells": len(obs),
        "mean_genes": obs["n_genes_by_counts"].mean(),
        "mean_pct_mt": obs["pct_counts_mt"].mean(),
    }
    if "predicted_doublets" in obs:
        row["n_doublets"] = obs["predicted_doublets"].sum(min_count=1)
        row["scrublet_scored"] = bool(obs["scrublet_score"].notna().any())
    for column in obs.columns[obs.columns.str.endswith("_label")]:
        row[f"n_{column.removesuffix('_label')}_types"] = obs[column].nunique()
    return row


def stage_samples(new: list[tuple[str, Path]], settings: Settings, schema: contract.Schema,
                  expected_var: np.ndarray | None, cfg: AppConfig, staging: Path) -> list[Staged]:
    staged = []
    for sid, path in new:
        adata = hise.read_sample(path)
        logger.info("%s: %d cells x %d genes", sid, adata.n_obs, adata.n_vars)
        if adata.n_obs == 0:
            logger.warning("%s has no cells; skipped", sid)
            continue
        diff = None if expected_var is None else contract.panel_diff(expected_var, adata.var_names)
        adata = prepare_sample(adata, sid, settings, schema, expected_var, fit=diff is not None)
        if expected_var is None:
            expected_var = np.asarray(adata.var_names, dtype=object)
        out = staging / f"{sid}.zarr"
        annizarr.convert(adata, output=str(out), cfg=cfg)
        note = contract.fit_note(diff) if diff else ""
        staged.append(Staged(sid, out, summarize(adata.obs), str(path), note))
        del adata
    return staged


def appended(staged: list[Staged], dropped: list[str]) -> dict:
    return {
        "appended_samples": [s.sample_id for s in staged],
        "sources": {s.sample_id: s.source for s in staged},
        "fitted": {s.sample_id: s.fitted for s in staged if s.fitted},
        "dropped": dropped if staged else [],
    }


def commit(store, staged: list[Staged], dropped: list[str]) -> str | None:
    root = store.working_root()
    samples = [s.sample_id for s in staged]
    n_cells = stores.n_obs(root)
    lines = "\n".join(s.line for s in staged) + (f"\nDropped (built for the old cells): {', '.join(dropped)}"
                                                    if dropped else "")
    return store.commit(
        f"Ingest: appended {samples}; {n_cells:,} cells\n{lines}",
        metadata={
            "pipeline": "ingest",
            "annizarr": annizarr.__version__,
            **appended(staged, dropped),
            "n_cells": n_cells,
            "n_samples": len(stores.sample_ids(root)),
        },
    )


def write(store, staged: list[Staged], seeding: bool, cfg: AppConfig,
          dropped: list[str]) -> tuple[str | None, str | None]:
    """Seeds a new store with the first sample if needed, appends the rest, and commits them once.

    Returns the seed and run snapshot ids (the same id when the seed is the only write; None on a
    plain zarr)."""
    seed = None
    if seeding:
        first = staged[0]
        message = f"Bootstrap: {first.sample_id}, {first.summary['n_cells']:,} cells\n{first.line}"
        seed = store.bootstrap(ad.read_zarr(first.path), cfg, message)
    rest = staged[1:] if seeding else staged
    if dropped:
        stores.drop_result_uns(store.working_root())
    for item in rest:
        store.append(str(item.path), cfg, drop_derived=bool(dropped))
    return seed, commit(store, staged, dropped) if rest else seed


def ingest(store, new: list[tuple[str, Path]], settings: Settings, schema: contract.Schema,
           expected_var: np.ndarray | None, seeding: bool,
           dropped: list[str]) -> tuple[list[Staged], str | None, str | None]:
    cfg = annizarr_config(settings)
    staging = Path(settings.staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        staged = stage_samples(new, settings, schema, expected_var, cfg, staging)
        seed, snapshot = write(store, staged, seeding, cfg, dropped) if staged else (None, None)
        return staged, seed, snapshot
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
    models = model_files(settings)
    store = open_store(settings)
    logger.info("Store %s (%s)", store.location, store.kind)
    root = store.root()
    seeding = not stores.has_data(root)
    present = set() if seeding else stores.sample_ids(root)
    new = select_new_samples(files, present)
    logger.info("%d samples in store, %d new: %s", len(present), len(new), [sid for sid, _ in new])

    schema = contract.requested_schema(settings.run_scrublet, [m.stem for m in models])
    if not seeding:
        contract.check_schema(stores.obs_columns(root), schema, settings.run_scrublet)
    expected_var = None if seeding else stores.var_names(root)
    check_panels(new, expected_var)
    dropped = stores.derived_elements(root) if new and not seeding else []
    if dropped:
        logger.warning("%s holds elements built for its current cells (%s); appending new cells drops them. "
                       "Re-run the analysis afterwards.", store.location, ", ".join(dropped))
    staged, seed, snapshot = (ingest(store, new, settings, schema, expected_var, seeding, dropped)
                              if new else ([], None, None))
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
