"""Grow one SpatialData `.zarr` from Xenium/Atera bundles or existing SpatialData objects (spec §3S).

Per run: resolve the store (local, or copied from S3), repair what a killed run left, list the inputs,
check every new sample's gene panel from var alone, then add one sample at a time. Each sample's
elements and its own table become present at one consolidate, and its rows are appended to the joint
single-cell table at the next, so a failure keeps every sample finished before it."""
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

import anndata as ad
import pandas as pd
import zarr
from anndata.io import read_elem

import annotate
import existing
import genes
import inputs
import joint
import layout
import s3
import sample
import store as stores
from config import ConfigError, Settings, load_settings

logger = logging.getLogger("spatial_ingest")

REFRESHED = {"/scratch": "is cleared when the run ends",
             "/results": "belongs to this run: saved as a data asset it is read-only, so later runs can't add to it"}


QUIET_LOGGERS = ("ome_zarr", "spatialdata", "spatialdata_io", "numcodecs", "distributed")


class IngestError(RuntimeError):
    """The inputs or the store's state don't allow this run."""


USER_ERRORS = (IngestError, ConfigError, inputs.InputError, existing.SplitError, genes.PanelError,
               sample.SampleError, joint.JointError, stores.StoreError, s3.S3Error)


@dataclass
class Run:
    """What a run did, for the outputs."""
    settings: Settings
    local: Path
    started: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    repairs: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    fitted: dict[str, str] = field(default_factory=dict)
    synced: dict | None = None
    downloaded: dict = field(default_factory=dict)
    created: bool = False


def storage_warning(location: str) -> str | None:
    if "://" in location:
        return None
    path = Path(location).resolve()
    for root, why in REFRESHED.items():
        if path.is_relative_to(root):
            return (f"The store {location} is in {root}, which {why}. A store you will keep adding to belongs on S3: "
                    "give an s3://bucket/prefix.")
    return None


def open_store(settings: Settings) -> tuple[Path, bool, dict]:
    """The local path the run works on (the store itself, or a fresh copy of the S3 store in scratch),
    whether the store must still be created, and the S3 copy's file stats. A new store is only
    created once a sample is ready to go in, so a run refused early leaves nothing behind."""
    if not settings.remote:
        if not Path(settings.store).is_absolute():
            raise IngestError(f"Storage Location {settings.store!r} must be an absolute path or an s3:// prefix.")
        local = Path(settings.store)
        fresh = stores.check_usable(local, settings.may_create(settings.store))
        warning = storage_warning(settings.store)
        if warning:
            logger.warning("%s", warning)
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


def per_sample_tables(local: Path) -> list[str]:
    """Sample IDs of the committed per-sample tables (`<sid>_table`)."""
    return sorted(name.removesuffix("_table") for name in stores.committed(local)["tables"]
                  if name != joint.NAME and name.endswith("_table"))


def read_sample_table(local: Path, sid: str) -> ad.AnnData:
    """A per-sample table as stored, with its region link (kept in the group's attributes) in uns."""
    group = zarr.open_group(local / "tables" / layout.table_name(sid), mode="r")
    table = read_elem(group)
    table.uns["spatialdata_attrs"] = {k: group.attrs.get(k) for k in ("region", "region_key", "instance_key")}
    return table


def record(table: ad.AnnData) -> dict:
    """What this capsule noted on a per-sample table: its CellTypist model, source and fingerprint."""
    return dict(table.uns.get(sample.RECORD, {}))


def sample_records(local: Path, sids: list[str]) -> dict[str, dict]:
    out = {}
    for sid in sids:
        uns = zarr.open_group(local / "tables" / layout.table_name(sid), mode="r").get("uns")
        out[sid] = dict(read_elem(uns[sample.RECORD])) if uns is not None and sample.RECORD in uns else {}
    return out


def joint_rows(table: ad.AnnData, sid: str, schema: joint.Schema | None,
               species: str | None) -> tuple[joint.Schema, ad.AnnData, str]:
    """One sample's joint rows and the schema they fit (the first sample sets it), with a fit note."""
    if schema is None:
        schema = joint.first_schema(table, sid, str(record(table).get("celltypist_model", "")), species)
    rows, note = joint.rows(table, sid, schema)
    return schema, rows, note


def sample_sizes(local: Path, sids: list[str]) -> dict[str, int]:
    """Cells per sample, from the per-sample tables' obs index alone."""
    return {sid: stores.n_obs(zarr.open_group(local / "tables" / layout.table_name(sid), mode="r")) for sid in sids}


def repair(local: Path, settings: Settings) -> list[str]:
    """Removes what a killed run left, rebuilds a joint table a killed append broke, appends present
    samples that are missing from it, and consolidates (spec §3S.9)."""
    actions = stores.remove_uncommitted(local)
    present = per_sample_tables(local)
    sizes = sample_sizes(local, present)
    has_joint = joint.NAME in stores.committed(local)["tables"]
    schema = joint.read_schema(local) if has_joint else None
    counts = joint.sample_counts(local) if has_joint else {}
    order: list[str] = []
    if counts is None or any(sizes.get(sid) != n for sid, n in counts.items()):
        order = [sid for sid in schema.samples if sid in sizes]
        schema.samples = []
        counts = {}
        joint.drop(local)
        actions.append("the joint table was inconsistent (a killed append); rebuilt from the per-sample tables")
    for sid in order + [s for s in present if s not in counts and s not in order]:
        table = read_sample_table(local, sid)
        schema, rows, _ = joint_rows(table, sid, schema, genes.species(table.var))
        joint.add(local, rows, sid, schema, stores.staging(local), settings.write_cpus)
        if not order:
            actions.append(f"appended present sample {sid}, missing from the joint table")
    shutil.rmtree(local / stores.STAGING, ignore_errors=True)
    stores.consolidate(local)
    for action in actions:
        logger.warning("Repair: %s", action)
    return actions


def model_file(settings: Settings) -> Path | None:
    """The custom CellTypist model, loaded once here so a bad file stops the run before any work."""
    model = settings.model_file()
    if model is None:
        return None
    if not model.is_file():
        raise IngestError(f"CellTypist model not found: {model}")
    try:
        from celltypist.models import Model

        Model.load(model=str(model))
    except Exception as exc:
        raise IngestError(f"{model} isn't a CellTypist model ({type(exc).__name__}: {exc}).") from None
    return model


def check_model(schema: joint.Schema | None, model: Path | None) -> None:
    name = model.stem if model else ""
    if schema is not None and schema.celltypist_model != name:
        raise IngestError(f"The store's joint table was built with CellTypist model "
                          f"{schema.celltypist_model or 'none'}; this run asks for {name or 'none'}. Give the same "
                          ".pkl (or none), or start a new store.")


def check_panels(new: list[inputs.Input], schema: joint.Schema | None, scratch: Path) -> dict[str, dict]:
    """Every new sample's genes and species from var alone, before any sample is read: a species
    different from the store's is refused, and a panel different from the joint table's is fitted."""
    reference = (schema.genes, schema.species or None, "the store's") if schema else None
    panels, diffs = {}, {}
    for item in new:
        _, var = genes.symbol_var(inputs.raw_var(item, scratch), item.sid)
        found = genes.species(var)
        if reference is None:
            reference = (list(var.index), found, f"the first sample's ({item.sid}, {len(var)} genes)")
        elif reference[1] and found and found != reference[1]:
            raise IngestError(f"{item.sid} is a {found} panel, but the joint table holds {reference[1]} genes; "
                              "gene symbols collide across species, so start a separate store for it.")
        diff = genes.panel_diff(reference[0], var.index)
        if diff:
            diffs[item.sid] = diff
        shared = len(set(reference[0]) & set(var.index))
        panels[item.sid] = {"genes_in_panel": len(var), "genes_shared": shared, "species": found}
    if diffs:
        logger.warning("Gene panel differs from %s for %d sample(s); they are fitted to its genes in the joint table "
                       "(missing genes zero-filled, extra genes dropped; each sample's own table keeps its panel): %s",
                       reference[2], len(diffs), "; ".join(genes.describe_diff(s, d) for s, d in diffs.items()))
    return panels


def load(item: inputs.Input, settings: Settings) -> sample.Loaded:
    """The sample read from its source; a zipped bundle is extracted to scratch first."""
    if item.kind == "spatialdata":
        return sample.read_part(item.path, item.part)
    folder = inputs.extract(item, Path(settings.scratch_dir)) if item.zipped else item.path
    return sample.read_bundle(item.kind, folder, settings.n_jobs)


def add_sample(local: Path, item: inputs.Input, settings: Settings, model: Path | None,
               schema: joint.Schema | None, panel: dict, seen: dict[str, str]) -> tuple[joint.Schema | None, dict, str]:
    """One sample into the store: its joint rows are built first (so a sample the joint table can't
    take is refused before anything is written), then its elements and table go in (staged, renamed
    into place, consolidated), then its rows (appended, consolidated). `seen` maps the fingerprints of
    the store's samples to their IDs, so the same cells can't go in twice under another name."""
    started = time.monotonic()
    try:
        loaded = load(item, settings)
        if loaded.table.n_obs == 0:
            logger.warning("%s has no cells; skipped", item.sid)
            return schema, {"status": "skipped (no cells)"}, ""
        sdata = sample.build(loaded, item.sid, model, str(item.path))
        table = sdata.tables[layout.table_name(item.sid)]
        fp = record(table)["fingerprint"]
        if fp in seen:
            raise IngestError(f"{item.sid} holds the same cells as {seen[fp]}, which is already in the store.")
        schema, rows, note = joint_rows(table, item.sid, schema, panel["species"])
        staged = stores.staging(local) / f"{item.sid}.zarr"
        sdata.write(staged, overwrite=True)
        stores.install(local, staged)
        shutil.rmtree(staged, ignore_errors=True)
        stores.consolidate(local)
        seen[fp] = item.sid
        joint.add(local, rows, item.sid, schema, stores.staging(local), settings.write_cpus)
        stores.consolidate(local)
    finally:
        shutil.rmtree(Path(settings.scratch_dir) / "extract" / item.sid, ignore_errors=True)
    label = annotate.label_columns(model)[0] if model else None
    row = {"status": "appended", "n_cells": table.n_obs,
           "mean_genes": float(table.obs["n_genes_by_counts"].mean()),
           "mean_total_counts": float(table.obs["total_counts"].mean()),
           "n_celltypist_labels": int(table.obs[label].nunique()) if label else 0,
           "seconds": round(time.monotonic() - started, 1)}
    logger.info("%s: added, %d cells (%s s)", item.sid, row["n_cells"], row["seconds"])
    return schema, row, note


def verify(local: Path) -> dict:
    """A fresh read of the store lists every element, and the joint table holds every sample's cells."""
    import spatialdata as sd

    sdata = sd.read_zarr(local, selection=existing.ELEMENT_KINDS)
    listed = stores.committed(local)
    for kind in existing.ELEMENT_KINDS:
        if set(getattr(sdata, kind)) != listed[kind]:
            raise IngestError(f"Verification failed: {kind} read back as {sorted(getattr(sdata, kind))}, "
                              f"expected {sorted(listed[kind])}.")
    present = per_sample_tables(local)
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
        "input_type": settings.input_type,
        "repairs": run.repairs,
        "appended_samples": [r["sample_id"] for r in run.rows if r["status"] == "appended"],
        "skipped_samples": [r["sample_id"] for r in run.rows if r["status"] != "appended"],
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
    columns = ["sample_id", "input_type", "source", "status", "n_cells", "genes_in_panel", "genes_shared",
               "mean_genes", "mean_total_counts", "n_celltypist_labels", "seconds"]
    pd.DataFrame(run.rows, columns=columns).to_csv(results / "spatial_ingest_summary.csv", index=False)
    return report


def finish(state: Run) -> dict:
    """Verifies the store, syncs it back to S3, and writes the outputs."""
    shutil.rmtree(state.local / stores.STAGING, ignore_errors=True)
    totals = verify(state.local)
    if state.settings.remote:
        state.synced = s3.sync(state.local, state.settings.store, state.settings.s3_workers, state.downloaded)
    return write_results(state, totals)


def skip_present(state: Run, items: list[inputs.Input], records: dict[str, dict]) -> None:
    for item in items:
        if item.sid not in records:
            continue
        source = records[item.sid].get("source")
        if source and source != str(item.path):
            logger.warning("%s is already in the store, added from %s; %s is skipped. If it is a different sample, "
                           "rename it.", item.sid, source, item.path)
        else:
            logger.info("%s is already in the store; skipped", item.sid)
        state.rows.append({"sample_id": item.sid, "input_type": item.kind, "source": str(item.path),
                           "status": "skipped (present)"})


def run(settings: Settings) -> dict:
    model = model_file(settings)
    local, fresh, downloaded = open_store(settings)
    logger.info("Store %s%s", settings.store, f" (working copy {local})" if settings.remote else "")
    state = Run(settings, local, downloaded=downloaded)
    if not fresh:
        state.repairs = repair(local, settings)
    items = inputs.resolve(settings)
    present = per_sample_tables(local) if not fresh else []
    records = sample_records(local, present)
    new = [i for i in items if i.sid not in present]
    skip_present(state, items, records)
    logger.info("%d sample(s) in the store, %d new: %s", len(present), len(new), [i.sid for i in new])
    schema = None if fresh else joint.read_schema(local)
    check_model(schema, model)
    panels = check_panels(new, schema, Path(settings.scratch_dir))
    if fresh:
        if not new:
            raise IngestError(f"There is no store at {settings.store} yet and nothing new to add to one.")
        stores.create(local)
        state.created = True
    seen = {r["fingerprint"]: sid for sid, r in records.items() if r.get("fingerprint")}
    for item in new:
        try:
            schema, row, note = add_sample(local, item, settings, model, schema, panels[item.sid], seen)
        except Exception as exc:
            keep(state)
            if isinstance(exc, USER_ERRORS):
                raise IngestError(f"{item.sid} ({item.path}): {exc}") from exc
            raise RuntimeError(f"{item.sid} ({item.path}) failed") from exc
        if note:
            state.fitted[item.sid] = note
        state.rows.append({"sample_id": item.sid, "input_type": item.kind, "source": str(item.path),
                           **{k: v for k, v in panels[item.sid].items() if k != "species"}, **row})
    report = finish(state)
    logger.info("Done: %s holds %d cells from %d sample(s).", settings.store, report["n_cells"], report["n_samples"])
    return report


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


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    for noisy in QUIET_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        run(load_settings(argv))
    except USER_ERRORS as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
