"""Append samples to an AnnData store, one sample in memory at a time.

A run is: `precheck` (inputs and models read, every problem listed at once), open the store, select
the new inputs, `stage` each one as a zarr, then write: the first staged input seeds a new store and
the rest are appended, as one commit on icechunk. `contracts` describes the records between stages.
"""
from __future__ import annotations

import importlib.metadata
import json
import logging
import resource
import shutil
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import pandas as pd
import scanpy as sc
import annizarr
from annizarr import AppConfig, Repo
from annizarr.config import ChunkConfig, IOConfig
from annizarr.errors import AnzError

import annotate
import contract
import genes
import hise
import logs
import precheck
import stage
import store as stores
from config import ConfigError, Settings, load_settings
from contracts import ContractError, IngestError, Staged

logger = logging.getLogger("ingest")

USER_ERRORS = (IngestError, ConfigError, ContractError, stores.StoreError, annotate.ModelError, AnzError)
VERSIONED = ("annizarr", "icechunk", "zarr", "anndata", "scanpy", "celltypist")


def annizarr_config(settings: Settings) -> AppConfig:
    return AppConfig(
        io=IOConfig(overwrite=True),
        chunks=ChunkConfig(row_chunk=settings.chunk_cells, cpus=settings.write_cpus),
    )


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
    return seed, commit(store, rest, dropped), dropped


def ingest(store, stager, settings: Settings, seeding: bool,
           dropped: list[str]) -> tuple[list[Staged], str | None, str | None, list[str]]:
    cfg = annizarr_config(settings)
    Path(settings.staging_dir).mkdir(parents=True, exist_ok=True)
    # A folder of this run's own inside staging_dir, so cleanup never touches anything else there.
    staging = Path(tempfile.mkdtemp(prefix="ingest_", dir=settings.staging_dir))
    try:
        staged = stager(cfg, staging)
        return (staged, *write(store, staged, seeding, cfg, dropped)) if staged else ([], None, None, dropped)
    except BaseException:
        store.discard()
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def versions() -> dict[str, str]:
    return {p: importlib.metadata.version(p) for p in VERSIONED}


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
        "versions": versions(),
        "run_at": datetime.now(timezone.utc).isoformat(),
        "peak_memory_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 2),
        "settings": asdict(settings),
    }
    results = Path(settings.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    (results / "ingest_run.json").write_text(json.dumps(report, indent=2, default=str))
    if staged:
        summary = pd.DataFrame({s.sample_id: s.summary for s in staged}).T.rename_axis("sample_id")
        summary = summary.map(lambda v: round(v, 3) if isinstance(v, float) else v)
        summary.to_csv(results / "ingest_summary.csv")
    return report


def run(settings: Settings) -> dict:
    files, models = precheck.run(settings)
    store = open_store(settings)
    logger.info("Store %s (%s)", store.location, store.kind)
    root = store.root()
    seeding = not stores.has_data(root)
    present = set() if seeding else stores.sample_ids(root)
    if settings.existing:
        new = precheck.select_new_objects(files, present, settings.sample_column)
    else:
        new = precheck.select_new_samples(files, present, {} if seeding else store.recorded_sources())
    logger.info("%d samples in store, %d new input(s): %s", len(present), len(new), [n.name for n in new])

    schema = contract.requested_schema(settings.run_scrublet, [name for name, _ in models])
    if new and not seeding and not settings.existing:
        precheck.check_raw_schema(stores.obs_columns(root), schema, settings.run_scrublet)
    target = stage.Target.of(root)
    genes.check_panels(new, target.var, hise.gene_names)
    dropped = stores.derived_elements(root) if new and not seeding else []
    if dropped:
        logger.warning("%s holds elements built for its current cells (%s); appending new cells drops them. "
                       "Re-run the analysis afterwards.", store.location, ", ".join(dropped))
    stager = (partial(stage.existing_objects, new, target) if settings.existing
              else partial(stage.samples, new, settings, models, schema, target))
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
    logs.setup()
    sc.settings.verbosity = 0
    try:
        run(load_settings(argv))
    except USER_ERRORS as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
