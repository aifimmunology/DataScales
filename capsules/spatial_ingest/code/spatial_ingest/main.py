"""Grows one SpatialData .zarr from Xenium/Atera bundles and existing SpatialData objects, one sample at a time."""
from __future__ import annotations

import json
import logging
import resource
import shutil
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import zarr

import config
import inputs
import prepare
import store as stores
from config import NAMES, Plan, Sample, Settings, UserError, block, load_settings
from prepare import IngestError

logger = logging.getLogger("spatial_ingest")

USAGE = """\
spatial_ingest adds Xenium and Atera bundles and SpatialData objects to one SpatialData .zarr (the store), with
a joint single-cell table of every sample's cells.

No samples were found, so nothing ran: List of paths to samples is empty or left at {default}, and
{default} doesn't exist ({data}).

To run it, attach your data as a data asset, open the App Panel, fill in List of paths to samples (e.g.
/data/<your data asset>) and Storage Location (/results/spatialdata.zarr for a one-off store; an s3:// prefix,
with an AWS secret attached, for a store you keep adding to), and press Run. A Reproducible Run adds the
samples of a data asset mounted at {default}. See code/README.md, "Quick start"."""
REFRESHED = {"/scratch": "is cleared when the run ends",
             "/results": "belongs to this run: saved as a data asset it is read-only, so later runs can't add to it"}
SUMMARY_COLUMNS = ["sample_id", "kind", "source", "status", "n_cells", "genes_in_panel", "genes_shared",
                   "genes_added", "celltypist_genes_shared", "mean_genes", "mean_total_counts",
                   "n_celltypist_cell_types", "n_cells_zero_counts", "n_cells_without_position", "seconds",
                   "note", "error"]
TOTALS = {"n_samples": None, "n_cells": None, "n_elements": None}


@dataclass
class Run:
    """What a run did, for the outputs."""
    settings: Settings
    local: Path | None = None
    started: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    repairs: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    gene_notes: dict[str, str] = field(default_factory=dict)
    left_out: list[dict] = field(default_factory=list)
    union_genes: int = 0
    totals: dict = field(default_factory=lambda: dict(TOTALS))
    synced: dict | None = None
    downloaded: dict = field(default_factory=dict)
    created: bool = False
    written: set[str] = field(default_factory=set)      # kind/name of what this run installed (verified in full)
    failed: list[str] = field(default_factory=list)     # one line per sample that failed and was rolled back
    error: str | None = None


def no_samples(settings: Settings) -> bool:
    """The samples list was left at /data/samples and that folder doesn't exist."""
    return settings.default_samples and not Path(config.DEFAULT_SAMPLES).exists()


def usage() -> str:
    data = config.DATA_DIR
    held = sorted(p.name for p in data.iterdir())[:20] if data.is_dir() else []
    where = f"{data} holds: {', '.join(held)}" if held else f"{data} is empty: no data asset is attached"
    return USAGE.format(default=config.DEFAULT_SAMPLES, data=where)


def one_line(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    return text if isinstance(exc, UserError) else f"{type(exc).__name__}: {text}"


def storage_notice(location: str) -> tuple[int, str] | None:
    """(level, text) about a store that won't last past the run."""
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
    """The local store path (or a copy of the S3 store), whether it must be created, and the copy's file stats."""
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
        return local, fresh, {}
    meta = stores.root_meta(settings.store, settings.s3_lock_stale_minutes)
    local = settings.work_dir / "store" / (stores.split(settings.store)[1].rsplit("/", 1)[-1] or "store.zarr")
    shutil.rmtree(local, ignore_errors=True)
    if meta is None:
        return local, True, {}
    if not meta:
        raise IngestError(f"{settings.store} holds objects but no SpatialData store (no root zarr.json); give "
                          "an empty prefix or a store this capsule built.")
    kind = stores.kind_of(meta)
    if kind != "ours":
        stores.refuse(settings.store, kind, True)
    return local, False, stores.download(settings.store, local, settings.s3_workers)


def table_path(local: Path, sid: str) -> Path:
    return local / "tables" / stores.table_name(sid)


def sample_sizes(local: Path, sids: list[str]) -> dict[str, int]:
    return {sid: inputs.n_obs(zarr.open_group(table_path(local, sid), mode="r")) for sid in sids}


def repair(local: Path, settings: Settings) -> list[str]:
    """Removes what a killed run left and brings the joint table in line with the per-sample tables."""
    actions = stores.remove_uncommitted(local)
    for action in actions:
        logger.warning("Repair: %s", action)
    present = prepare.present_samples(local)
    models = {sid: str(record.get("celltypist_model", "")) for sid, record in prepare.records(local, present).items()}
    joint_actions = stores.repair_joint(local, present, stores.staging(local), settings.write_cpus, models)
    for action in joint_actions:
        logger.warning("Repair: %s", action)
    shutil.rmtree(local / stores.STAGING, ignore_errors=True)
    stores.consolidate(local)
    return actions + list(joint_actions)


def roll_back(local: Path, sid: str, settings: Settings) -> None:
    shutil.rmtree(local / stores.STAGING, ignore_errors=True)
    stores.uninstall_sample(local, sid)
    stores.consolidate(local)
    repair(local, settings)


def add_sample(local: Path, item: Sample, settings: Settings, model: Path | None,
               seen: dict[str, str], written: set[str]) -> tuple[dict, str]:
    """Reads, prepares and installs one sample, then appends its rows to the joint table."""
    started = time.monotonic()
    scratch = settings.work_dir
    loaded = prepare.load(item, scratch)
    if loaded.table.n_obs == 0:
        logger.warning("%s has no cells; skipped.", item.sid)
        return {"status": "skipped (no cells)", "n_cells": 0}, ""
    prepared = prepare.prepare(loaded, item, model, scratch)
    if prepared.fingerprint in seen:
        other = seen[prepared.fingerprint]
        logger.warning("%s holds the same cells as %s, which is already in the store; %s is skipped (leave it out "
                       "of the inputs to avoid this).", item.sid, other, item.sid)
        shutil.rmtree(prepared.table_path, ignore_errors=True)
        return {"status": f"skipped (same cells as {other})", "n_cells": prepared.stats.get("n_cells")}, ""
    stores.check_joint_rows(local, item.sid, prepared.table_path)  # what add_joint_rows would refuse, before any write
    written.update(stores.install_sample(local, prepared))
    stores.consolidate(local)
    try:
        note = stores.add_joint_rows(local, item.sid, table_path(local, item.sid), stores.staging(local),
                                     settings.write_cpus, model.stem if model else "")
    except BaseException:
        roll_back(local, item.sid, settings)
        raise
    stores.consolidate(local)
    seen[prepared.fingerprint] = item.sid
    stats = prepared.stats
    row = {"status": "appended", "n_cells": stats.get("n_cells"), "mean_genes": tenth(stats.get("mean_genes")),
           "mean_total_counts": tenth(stats.get("mean_total_counts")),
           "n_celltypist_cell_types": stats.get("n_celltypist_labels", 0),
           "n_cells_zero_counts": stats.get("n_cells_zero_counts", 0),
           "n_cells_without_position": stats.get("n_cells_without_position", 0),
           "seconds": round(time.monotonic() - started, 1)}
    logger.info("%s: added, %s cells (%s s)", item.sid, row["n_cells"], row["seconds"])
    return row, note or ""


def tenth(value: float | None) -> float | None:
    return None if value is None else round(float(value), 1)


def release(scratch: Path, item: Sample, rest: list[Sample]) -> None:
    """Deletes what was staged for `item`'s input once no later sample comes from it."""
    if any(s.candidate.sid == item.candidate.sid for s in rest):
        return
    shutil.rmtree(inputs.stage_dir(scratch, item.candidate.sid), ignore_errors=True)


def clear_scratch(settings: Settings, local: Path) -> None:
    """Deletes the run's scratch files, keeping the S3 working copy and a store inside that folder."""
    scratch = settings.work_dir
    if not scratch.is_dir():
        return
    for child in scratch.iterdir():
        if child.name == "store" or local.resolve().is_relative_to(child.resolve()):
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def verify(local: Path, written: set[str]) -> dict:
    """Checks the store without loading what earlier runs wrote."""
    listed, found = stores.committed(local), stores.on_disk(local)
    for kind in stores.KINDS:
        if listed[kind] != found[kind]:
            raise IngestError(f"Verification failed: {kind} on disk {sorted(found[kind])} differ from the committed "
                              f"{sorted(listed[kind])}.")
    names: dict[str, str] = {}
    for kind in stores.KINDS:
        for name in sorted(listed[kind]):
            if name.lower() in names:
                raise IngestError(f"Verification failed: {kind}/{name} and {names[name.lower()]} differ only in case "
                                  "(or repeat across kinds), which spatialdata can't read.")
            names[name.lower()] = f"{kind}/{name}"
            if kind != "tables" and not (local / kind / name / "zarr.json").is_file():
                raise IngestError(f"Verification failed: {kind}/{name} has no zarr.json.")
    mine = {tuple(w.split("/", 1)) for w in written if w.split("/", 1)[0] in stores.ELEMENT_KINDS}
    if mine:
        _, unreadable = prepare.read_elements(local, mine)
        if unreadable:
            raise IngestError("Verification failed: " + "; ".join(f"{k}: {v}" for k, v in unreadable.items()))
    stores.verify_tables(local)
    present = prepare.present_samples(local)
    sizes = sample_sizes(local, present)
    counts = stores.sample_counts(local) if present else {}
    held = {str(k): int(v) for k, v in counts.items()} if counts is not None else None
    if held != sizes:
        raise IngestError(f"Verification failed: the joint table's cells per sample {held} differ from the "
                          f"per-sample tables' {sizes}.")
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


def write_results(run: Run) -> dict:
    settings = run.settings
    local = run.local if run.local is not None and run.local.is_dir() else None
    report = {
        "status": "failed" if run.error else "succeeded",
        "error": run.error,
        "store": settings.store,
        "local_copy": str(run.local) if settings.remote and run.local else None,
        "repairs": run.repairs,
        "appended_samples": [r["sample_id"] for r in run.rows if r["status"] == "appended"],
        "failed_samples": {r["sample_id"]: r.get("error") for r in run.rows if r["status"] == "failed"},
        "same_cells_skipped": {r["sample_id"]: r["status"].removeprefix("skipped (same cells as ").rstrip(")")
                               for r in run.rows if r["status"].startswith("skipped (same cells")},
        "skipped_samples": [r["sample_id"] for r in run.rows if r["status"].startswith("skipped")],
        "kinds": {r["sample_id"]: r["kind"] for r in run.rows if r.get("kind")},
        "left_out": run.left_out,
        "joint_genes": run.union_genes,
        "gene_notes": run.gene_notes,
        **run.totals,
        "store_size_gb": round(store_size(local) / 1024**3, 3) if local else None,
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


def finish(state: Run) -> None:
    shutil.rmtree(state.local / stores.STAGING, ignore_errors=True)
    state.totals = verify(state.local, state.written)
    if state.settings.remote:
        state.synced = stores.sync(state.local, state.settings.store, state.settings.s3_workers, state.downloaded)


def plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def add_all(state: Run, plan: Plan, model: Path | None) -> None:
    """A sample that fails is rolled back and noted; the run goes on with the others."""
    settings, local = state.settings, state.local
    scratch = settings.work_dir
    records = {} if state.created else prepare.records(local, prepare.present_samples(local))
    seen = {r["fingerprint"]: sid for sid, r in records.items() if r.get("fingerprint")}
    for i, item in enumerate(plan.samples):
        where = f"{item.sid} ({prepare.source(item)})"
        base = {"sample_id": item.sid, "kind": item.kind, "source": prepare.source(item),
                **{k: v for k, v in plan.panels.get(item.sid, {}).items() if k in SUMMARY_COLUMNS}}
        try:
            row, note = add_sample(local, item, settings, model, seen, state.written)
        except Exception as exc:
            if isinstance(exc, UserError):
                message = " ".join(str(exc).split())
                logger.error("%s failed and was left out: %s", where, message)
            else:
                message = f"{type(exc).__name__}: {' '.join(str(exc).split())}"
                logger.exception("%s failed and was left out (an unexpected error):", where)
            try:
                state.repairs += repair(local, settings)
            except Exception as again:
                raise IngestError(f"{where} failed ({message}), and the store couldn't be repaired afterwards "
                                  f"({type(again).__name__}: {again}); the next run repairs it.") from again
            state.failed.append(f"{where}: {message}")
            state.rows.append({**base, "status": "failed", "error": message})
            continue
        finally:
            prepare.close(item.sid)           # lazy elements' file handles, before their staged files go
            release(scratch, item, plan.samples[i + 1:])
        if note:
            state.gene_notes[item.sid] = note
        state.rows.append({**base, **row})


def take_locks(settings: Settings, stack: ExitStack) -> None:
    if settings.remote:
        stack.callback(stores.release_s3_lock, stores.acquire_s3_lock(settings.store, settings.s3_lock_stale_minutes))


def run(settings: Settings) -> dict:
    """The run; the run JSON and summary CSV are written also when it fails."""
    state = Run(settings)
    try:
        model = settings.model_file()
        if model is not None:
            prepare.load_model(model)
        with ExitStack() as stack:
            take_locks(settings, stack)
            state.local, fresh, state.downloaded = open_store(settings)
            if not settings.remote:
                stack.callback(stores.release_lock, stores.acquire_lock(state.local))
            clear_scratch(settings, state.local)          # what a killed earlier run left
            stack.callback(clear_scratch, settings, state.local)
            run_locked(state, model, fresh)
    except BaseException as exc:
        state.error = one_line(exc)
        try:
            write_results(state)
        except Exception as again:
            logger.error("The run JSON couldn't be written either: %s", one_line(again))
        raise
    return write_results(state)


def summary_rows(state: Run, plan: Plan, fresh: bool) -> None:
    recorded = prepare.records(state.local, plan.present) if plan.present else {}
    for sid in plan.present:
        given = plan.panels.get(sid, {}).get("source")
        stored = recorded.get(sid, {}).get("source") or given
        state.rows.append({"sample_id": sid, "kind": plan.panels.get(sid, {}).get("kind"), "source": stored,
                           "status": "skipped (present)",
                           "note": f"this run's input: {given}" if given and given != stored else None})
    panels = [plan.panels[s.sid] for s in plan.samples if "genes_shared" in plan.panels.get(s.sid, {})]
    if fresh and len(panels) == 1:
        panels[0]["genes_shared"] = 0


def run_locked(state: Run, model: Path | None, fresh: bool) -> None:
    settings, local = state.settings, state.local
    logger.info("Store %s%s", settings.store, f" (working copy {local})" if settings.remote else "")
    if not fresh:
        state.repairs = repair(local, settings)
    plan = prepare.precheck(settings, local, fresh, model, settings.work_dir)
    notice = storage_notice(settings.store)
    if notice and (plan.samples or not fresh):
        logger.log(*notice)
    state.left_out = [{"path": x.path, "reason": x.reason} for x in plan.left_out]
    state.union_genes = len(plan.union_genes)
    summary_rows(state, plan, fresh)
    if fresh:
        if not plan.samples:
            raise IngestError(f"There is no store at {settings.store} yet and nothing new to add to one.")
        stores.create(local)
        state.created = True
    add_all(state, plan, model)
    if state.created and not stores.committed(local)["tables"]:
        shutil.rmtree(local, ignore_errors=True)   # every sample failed: leave no empty store behind
        raise IngestError(block("Every sample failed, so no store was created:", state.failed))
    finish(state)
    added = pd.Series([NAMES.get(r["kind"], r["kind"]) for r in state.rows if r["status"] == "appended"],
                      dtype=object).value_counts()
    logger.info("Done: %s holds %d cells from %s; this run added %s.", settings.store, state.totals["n_cells"],
                plural(state.totals["n_samples"], "sample"),
                ", ".join(plural(n, k) for k, n in added.items()) or "nothing")
    if state.failed:
        synced = " and synced" if settings.remote else ""
        were = "was" if len(state.failed) == 1 else "were"
        raise IngestError(block(f"{plural(len(state.failed), 'sample')} failed and {were} left out; the others are "
                                f"in the store{synced}. Fix these and run again:", state.failed))


def main(argv: list[str] | None = None) -> int:
    config.setup_logging()
    try:
        settings = load_settings(argv)
        if no_samples(settings):
            logger.info("%s", usage())
            return 0
        run(settings)
    except UserError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
