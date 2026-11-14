"""Grow one SpatialData `.zarr` from Xenium/Atera bundles and existing SpatialData objects (spec §3S).

Per run:
1. open the store (local, or a working copy of an S3 store) and repair what a killed run left;
2. precheck: find every sample in the inputs (kinds detected from their files), check what each new
   sample needs before anything is written, and report (precheck.py);
3. create the store, only now that there is something to add;
4. one sample at a time: read it (lazy elements, counts streamed to scratch), prepare it (renamed
   elements, its table with QC and labels, written chunk-wise), refuse cells already in the store,
   install its elements and table (one consolidate = present), append its rows to the joint table
   (gene and obs union; consolidate again);
5. verify the store, sync it back to S3, write the run JSON and summary CSV, clear scratch.

A failure keeps every sample finished before it (repaired, verified, synced)."""
from __future__ import annotations

import json
import logging
import resource
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import zarr

import annotate
import discover
import existing
import genes
import inventory
import joint
import layout
import logs
import precheck
import readers
import s3
import sample as prep
import store as stores
from config import ConfigError, Settings, load_settings
from contracts import NAMES, Plan, Sample
from precheck import IngestError

logger = logging.getLogger("spatial_ingest")

REFRESHED = {"/scratch": "is cleared when the run ends",
             "/results": "belongs to this run: saved as a data asset it is read-only, so later runs can't add to it"}
SUMMARY_COLUMNS = ["sample_id", "kind", "source", "status", "n_cells", "genes_in_panel", "genes_shared",
                   "genes_added", "celltypist_genes_shared", "mean_genes", "mean_total_counts",
                   "n_celltypist_labels", "seconds"]


def _module_errors(*modules) -> list[type[Exception]]:
    """The `…Error` classes a module defines (its user-facing errors)."""
    return [v for module in modules for k, v in vars(module).items()
            if k.endswith("Error") and isinstance(v, type) and issubclass(v, Exception)
            and v.__module__ == module.__name__]


# Errors that describe the inputs, the settings or the store: one line in the log, exit code 1.
USER_ERRORS = tuple(dict.fromkeys([
    IngestError, ConfigError, discover.InputError, existing.SplitError, genes.PanelError, joint.JointError,
    stores.StoreError, s3.S3Error, annotate.AnnotateError, *_module_errors(prep, readers, inventory)]))


@dataclass
class Run:
    """What a run did, for the outputs."""
    settings: Settings
    local: Path
    started: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    repairs: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    fitted: dict[str, str] = field(default_factory=dict)
    left_out: list[dict] = field(default_factory=list)
    union_genes: int = 0
    synced: dict | None = None
    downloaded: dict = field(default_factory=dict)
    created: bool = False


# ---------------------------------------------------------------- the store

def storage_notice(location: str) -> tuple[int, str] | None:
    """(level, text) about a store that won't last: a WARNING under /scratch (deleted), INFO under /results."""
    if "://" in location:
        return None
    path = Path(location).resolve()
    for root, why in REFRESHED.items():
        if path.is_relative_to(root):
            level = logging.WARNING if root == "/scratch" else logging.INFO
            return level, (f"The store {location} is in {root}, which {why}. A store you will keep adding to belongs "
                           "on S3: give an s3://bucket/prefix.")
    return None


def no_store_here(settings: Settings, local: Path) -> IngestError:
    """Why a missing local store can't be created at `local`, and where to put it instead."""
    roots = settings.create_root_list()
    if local.resolve() in roots:
        return IngestError(f"{local} is a folder for stores, not a store: give a .zarr path inside it, e.g. "
                           f"{local}/spatialdata.zarr.")
    if local.resolve().is_relative_to("/data"):
        return IngestError(f"{local} is under /data, where data assets are mounted read-only. Create the store at "
                           "/results/<name>.zarr, /scratch/<name>.zarr or an s3:// prefix.")
    where = ", ".join(f"{r}/<name>.zarr" for r in roots)
    return IngestError(f"No store at {local}. Check the path of an existing store, or create a new one at {where} "
                       "or an s3:// prefix (outside the App Panel, CREATE=true allows other local folders).")


def open_store(settings: Settings) -> tuple[Path, bool, dict]:
    """The local path the run works on (the store itself, or a fresh copy of the S3 store in scratch),
    whether the store must still be created, and the S3 copy's file stats. A new store is only
    created once a sample is ready to go in, so a run refused early leaves nothing behind."""
    if not settings.remote:
        if not Path(settings.store).is_absolute():
            raise IngestError(f"Storage Location {settings.store!r} must be an absolute path (e.g. "
                              "/results/spatialdata.zarr) or an s3:// prefix.")
        local = Path(settings.store)
        if local.is_file():
            raise IngestError(f"Storage Location {local} is a file; give a SpatialData .zarr folder (a new one is "
                              "created), e.g. /results/spatialdata.zarr.")
        may_create = settings.may_create(settings.store)
        if stores.describe(local) in ("missing", "empty") and not may_create:
            raise no_store_here(settings, local)
        fresh = stores.check_usable(local, may_create)
        if fresh and local.suffix != ".zarr":
            logger.info("The new store %s has no .zarr extension; it is a SpatialData .zarr all the same.", local)
        notice = storage_notice(settings.store)
        if notice:
            logger.log(*notice)
        return local, fresh, {}
    meta = s3.root_meta(settings.store)
    local = Path(settings.scratch_dir) / "store" / (s3.split(settings.store)[1].rsplit("/", 1)[-1] or "store.zarr")
    shutil.rmtree(local, ignore_errors=True)
    if meta is None:
        return local, True, {}
    if not meta:
        raise IngestError(f"{settings.store} holds objects but no SpatialData store (no root zarr.json); give "
                          "an empty prefix or a store this capsule built.")
    kind = stores.kind_of(meta)
    if kind != "ours":
        stores.refuse(settings.store, kind, True)
    return local, False, s3.download(settings.store, local, settings.s3_workers)


def table_path(local: Path, sid: str) -> Path:
    return local / "tables" / layout.table_name(sid)


def sample_sizes(local: Path, sids: list[str]) -> dict[str, int]:
    """Cells per sample, from the per-sample tables' obs index alone."""
    return {sid: stores.n_obs(zarr.open_group(table_path(local, sid), mode="r")) for sid in sids}


def repair(local: Path, settings: Settings) -> list[str]:
    """Removes what a killed run left, rebuilds a joint table a killed append broke (from the stored
    per-sample tables, in its sample order), appends present samples missing from it, consolidates."""
    actions = stores.remove_uncommitted(local)
    present = precheck.present_samples(local)
    sizes = sample_sizes(local, present)
    has_joint = joint.NAME in stores.committed(local)["tables"]
    counts = joint.sample_counts(local) if has_joint else {}
    order: list[str] = []
    if counts is None or any(sizes.get(sid) != n for sid, n in counts.items()):
        schema = joint.read_schema(local)
        order = [sid for sid in schema.samples if sid in sizes]
        joint.drop(local)
        counts = {}
        actions.append("the joint table was inconsistent (a killed append); rebuilt from the per-sample tables")
    missing = [sid for sid in present if sid not in counts and sid not in order]
    records = precheck.records(local, order + missing)
    for sid in order + missing:
        joint.add(local, sid, table_path(local, sid), stores.staging(local), settings.write_cpus,
                  str(records[sid].get("celltypist_model", "")))
        if sid in missing:
            actions.append(f"appended present sample {sid}, missing from the joint table")
    shutil.rmtree(local / stores.STAGING, ignore_errors=True)
    stores.consolidate(local)
    for action in actions:
        logger.warning("Repair: %s", action)
    return actions


# ---------------------------------------------------------------- one sample

def add_sample(local: Path, item: Sample, settings: Settings, model: Path | None,
               seen: dict[str, str]) -> tuple[dict, str]:
    """One sample into the store: read, prepared (its table staged on scratch), skipped with one warning
    when its cells are already in the store under another ID (`seen`: fingerprint -> sample ID; e.g. a
    bundle and a SpatialData object made from it in one run), installed (present at the next
    consolidate), then its rows appended to the joint table. Returns (summary row, fit note)."""
    started = time.monotonic()
    scratch = Path(settings.scratch_dir)
    loaded = readers.load(item, scratch, settings.n_jobs)
    if loaded.table.n_obs == 0:
        logger.warning("%s has no cells; skipped.", item.sid)
        return {"status": "skipped (no cells)", "n_cells": 0}, ""
    prepared = prep.prepare(loaded, item, model, scratch)
    if prepared.fingerprint in seen:
        other = seen[prepared.fingerprint]
        logger.warning("%s holds the same cells as %s, which is already in the store; %s is skipped (leave it out "
                       "of the inputs to avoid this).", item.sid, other, item.sid)
        shutil.rmtree(prepared.table_path, ignore_errors=True)
        return {"status": f"skipped (same cells as {other})", "n_cells": prepared.stats.get("n_cells")}, ""
    stores.install_sample(local, prepared)
    stores.consolidate(local)
    seen[prepared.fingerprint] = item.sid
    note = joint.add(local, item.sid, table_path(local, item.sid), stores.staging(local), settings.write_cpus,
                     model.stem if model else "")
    stores.consolidate(local)
    stats = prepared.stats
    row = {"status": "appended", "n_cells": stats.get("n_cells"), "mean_genes": stats.get("mean_genes"),
           "mean_total_counts": stats.get("mean_total_counts"),
           "n_celltypist_labels": stats.get("n_celltypist_labels", 0),
           "seconds": round(time.monotonic() - started, 1)}
    logger.info("%s: added, %s cells (%s s)", item.sid, row["n_cells"], row["seconds"])
    return row, note or ""


def release(scratch: Path, item: Sample, rest: list[Sample]) -> None:
    """Deletes what was staged for `item`'s input once no later sample comes from the same input."""
    if any(s.candidate.sid == item.candidate.sid for s in rest):
        return
    shutil.rmtree(scratch / precheck.STAGE / item.candidate.sid, ignore_errors=True)


def clear_scratch(settings: Settings, local: Path) -> None:
    """Deletes the run's scratch files, keeping the S3 working copy (scratch/store) and the store itself."""
    scratch = Path(settings.scratch_dir)
    if not scratch.is_dir():
        return
    for child in scratch.iterdir():
        if child.name == "store" or local.resolve().is_relative_to(child.resolve()):
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


# ---------------------------------------------------------------- verify and outputs

def verify(local: Path) -> dict:
    """A fresh read of the store lists every element, every table opens, and the joint table holds
    every sample's cells."""
    import spatialdata as sd

    sdata = sd.read_zarr(local, selection=existing.ELEMENT_KINDS)
    listed = stores.committed(local)
    for kind in existing.ELEMENT_KINDS:
        if set(getattr(sdata, kind)) != listed[kind]:
            raise IngestError(f"Verification failed: {kind} read back as {sorted(getattr(sdata, kind))}, "
                              f"expected {sorted(listed[kind])}.")
    stores.verify_tables(local)
    present = precheck.present_samples(local)
    sizes = sample_sizes(local, present)
    counts = joint.sample_counts(local) if present else {}
    if counts is None or dict(counts) != sizes:
        raise IngestError(f"Verification failed: the joint table's cells per sample {dict(counts or {})} differ "
                          f"from the per-sample tables' {sizes}.")
    return {"n_samples": len(present), "n_cells": int(sum(sizes.values())),
            "n_elements": sum(len(v) for k, v in listed.items() if k != "tables")}


def store_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def versions() -> dict:
    import importlib.metadata as md

    out = {p: md.version(p) for p in ("spatialdata", "spatialdata-io", "annizarr", "zarr", "anndata", "scanpy")}
    source = json.loads((Path(__file__).parent / "vendor" / "SOURCE.json").read_text())
    out["atera_reader"] = f"{source['repository']}@{source['commit']}"
    return out


def write_results(run: Run, totals: dict) -> dict:
    settings = run.settings
    report = {
        "store": settings.store,
        "local_copy": str(run.local) if settings.remote else None,
        "repairs": run.repairs,
        "appended_samples": [r["sample_id"] for r in run.rows if r["status"] == "appended"],
        "same_cells_skipped": {r["sample_id"]: r["status"].removeprefix("skipped (same cells as ").rstrip(")")
                               for r in run.rows if r["status"].startswith("skipped (same cells")},
        "skipped_samples": [r["sample_id"] for r in run.rows if r["status"] != "appended"],
        "kinds": {r["sample_id"]: r["kind"] for r in run.rows if r.get("kind")},
        "left_out": run.left_out,
        "joint_genes": run.union_genes,
        "fitted": run.fitted,
        **totals,
        "store_size_gb": round(store_size(run.local) / 1024**3, 3),
        "s3_sync": run.synced,
        "versions": versions(),
        "started": run.started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "peak_memory_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 2),
        "settings": asdict(settings),
    }
    results = Path(settings.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    (results / "spatial_ingest_run.json").write_text(json.dumps(report, indent=2, default=str))
    pd.DataFrame(run.rows, columns=SUMMARY_COLUMNS).to_csv(results / "spatial_ingest_summary.csv", index=False)
    return report


def finish(state: Run) -> dict:
    """Verifies the store, syncs it back to S3, and writes the outputs."""
    shutil.rmtree(state.local / stores.STAGING, ignore_errors=True)
    totals = verify(state.local)
    if state.settings.remote:
        state.synced = s3.sync(state.local, state.settings.store, state.settings.s3_workers, state.downloaded)
    return write_results(state, totals)


def keep(state: Run) -> None:
    """After a failed sample: repairs what it left, then saves (and on S3 syncs) the samples finished
    before it, so the next run continues from there."""
    if state.created and not stores.committed(state.local)["tables"]:
        shutil.rmtree(state.local, ignore_errors=True)
        return
    try:
        state.repairs += repair(state.local, state.settings)
        report = finish(state)
    except Exception:
        logger.exception("Couldn't save the samples finished before the failure; the store is repaired on the next run")
        return
    logger.warning("Kept what was finished before the failure: the store holds %d cells from %d sample(s)%s.",
                   report["n_cells"], report["n_samples"], " (synced to S3)" if state.settings.remote else "")


# ---------------------------------------------------------------- the run

def plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def add_all(state: Run, plan: Plan, model: Path | None) -> None:
    """Adds the plan's samples in order; on a failure keeps what was finished, then raises."""
    settings, local = state.settings, state.local
    scratch = Path(settings.scratch_dir)
    records = {} if state.created else precheck.records(local, precheck.present_samples(local))
    seen = {r["fingerprint"]: sid for sid, r in records.items() if r.get("fingerprint")}
    for i, item in enumerate(plan.samples):
        base = {"sample_id": item.sid, "kind": item.kind, "source": precheck.source(item),
                **{k: v for k, v in plan.panels.get(item.sid, {}).items() if k in SUMMARY_COLUMNS}}
        try:
            row, note = add_sample(local, item, settings, model, seen)
        except Exception as exc:
            keep(state)
            if isinstance(exc, USER_ERRORS):
                raise IngestError(f"{item.sid} ({precheck.source(item)}): {exc}") from exc
            raise RuntimeError(f"{item.sid} ({precheck.source(item)}) failed") from exc
        finally:
            release(scratch, item, plan.samples[i + 1:])
        if note:
            state.fitted[item.sid] = note
        state.rows.append({**base, **row})


def run(settings: Settings) -> dict:
    model = settings.model_file()
    if model is not None:
        annotate.load_model(model)
    local, fresh, downloaded = open_store(settings)
    logger.info("Store %s%s", settings.store, f" (working copy {local})" if settings.remote else "")
    state = Run(settings, local, downloaded=downloaded)
    try:
        if not fresh:
            state.repairs = repair(local, settings)
        plan = precheck.run(settings, local, fresh, model, Path(settings.scratch_dir))
        state.left_out = [{"path": x.path, "reason": x.reason} for x in plan.left_out]
        state.union_genes = len(plan.union_genes)
        state.rows += [{"sample_id": sid, "kind": plan.panels.get(sid, {}).get("kind"),
                        "source": plan.panels.get(sid, {}).get("source"), "status": "skipped (present)"}
                       for sid in plan.present]
        if fresh:
            if not plan.samples:
                raise IngestError(f"There is no store at {settings.store} yet and nothing new to add to one.")
            stores.create(local)
            state.created = True
        add_all(state, plan, model)
        report = finish(state)
    finally:
        clear_scratch(settings, local)
    added = pd.Series([NAMES.get(r["kind"], r["kind"]) for r in state.rows if r["status"] == "appended"],
                      dtype=object).value_counts()
    logger.info("Done: %s holds %d cells from %s; this run added %s.", settings.store, report["n_cells"],
                plural(report["n_samples"], "sample"), ", ".join(plural(n, k) for k, n in added.items()) or "nothing")
    return report


def main(argv: list[str] | None = None) -> int:
    logs.setup()
    try:
        run(load_settings(argv))
    except USER_ERRORS as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
