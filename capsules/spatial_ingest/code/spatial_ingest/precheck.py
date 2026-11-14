"""The precheck: every sample of a run is found and checked before anything is written, and the user
gets one report.

    discover the inputs -> inspect every bundle (inventory) / split every SpatialData object (existing.plan)
    -> skip samples already in the store -> check what the new samples need -> Plan

The run stops (IngestError, one message listing every problem) when any new sample misses an essential
part (a count matrix; a cells table with cell IDs and x/y positions), can't be split or read, has a
species other than the store's or the other samples', shares too few genes with the CellTypist model,
or two samples have the same ID. Everything else is reported and the run goes on:
- items in the inputs that aren't samples: one WARNING listing them;
- a bundle and a SpatialData object that are the only samples in one folder (e.g. a SEA-AD section's raw
  output beside its combined_spatialdata_zarr): one WARNING that they may be the same tissue; both are added;
- a sample missing optional parts (images, transcripts, boundaries, …): one WARNING line per sample;
- gene panels that differ: one WARNING block — the joint table holds the union of the genes, and each
  sample reads 0 for genes it didn't measure;
- a CellTypist model that shares fewer than half of its genes with a panel: one WARNING block.
Samples whose cells are already in the store under another ID can only be told once they are read
(fingerprint): main skips them then, with one WARNING, and goes on.

`check` holds the decisions and works on hand-made samples; `run` gathers them from the inputs."""
from __future__ import annotations

import logging
import os
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import zarr
from anndata.io import read_elem

import annotate
import config
import discover
import existing
import genes
import inventory
import joint
import layout
import stage
import store as stores
from contracts import NAMES, Candidate, LeftOut, Plan, Sample
from logs import block

logger = logging.getLogger("spatial_ingest.precheck")

RECORD = "spatial_ingest"           # a per-sample table's uns record (sample.py writes it)
STAGE = "stage"                     # scratch/<STAGE>/<sid>: files staged for a sample
ESSENTIALS = "a count matrix, and a cells table with cell IDs and x/y positions that match the matrix"
# Errors that describe an input, not a bug: reported per sample, then the run stops.
INPUT_ERRORS = (ValueError, OSError, zipfile.BadZipFile, tarfile.TarError)


class IngestError(RuntimeError):
    """The inputs or the store's state don't allow this run."""


@dataclass
class StoreState:
    """What the precheck needs from the store: its samples (with their uns record), the joint table's
    genes, species and CellTypist model."""
    present: list[str] = field(default_factory=list)
    records: dict[str, dict] = field(default_factory=dict)
    genes: list[str] = field(default_factory=list)
    species: str | None = None
    model: str | None = None              # None: no joint table yet; "" built without a model

    @classmethod
    def read(cls, local: Path, fresh: bool) -> "StoreState":
        if fresh:
            return cls()
        present = present_samples(local)
        schema = joint.read_schema(local)
        return cls(present, records(local, present),
                   list(getattr(schema, "genes", []) or []) if schema else [],
                   (getattr(schema, "species", "") or None) if schema else None,
                   str(getattr(schema, "celltypist_model", "") or "") if schema else None)


def present_samples(local: Path) -> list[str]:
    """Sample IDs of the committed per-sample tables (`<sid>_table`)."""
    return sorted(name.removesuffix("_table") for name in stores.committed(local)["tables"]
                  if name != joint.NAME and name.endswith("_table"))


def records(local: Path, sids: list[str]) -> dict[str, dict]:
    """The uns record (CellTypist model, source, fingerprint, …) of each per-sample table."""
    out = {}
    for sid in sids:
        uns = zarr.open_group(local / "tables" / layout.table_name(sid), mode="r").get("uns")
        out[sid] = dict(read_elem(uns[RECORD])) if uns is not None and RECORD in uns else {}
    return out


def source(sample: Sample) -> str:
    """Where a sample came from, as recorded in its table and compared on later runs."""
    return str(sample.candidate.origin)


# ---------------------------------------------------------------- gathering samples from the inputs

def _no_default_samples() -> IngestError:
    data = config.DATA_DIR
    held = sorted(p.name for p in data.iterdir())[:20] if data.is_dir() else []
    where = f"{data} holds: {', '.join(held)}" if held else f"{data} is empty (no data asset is attached)"
    return IngestError(f"No samples given, and {config.DEFAULT_SAMPLES} doesn't exist. Fill in List of paths to "
                       f"samples (e.g. {data}/<your data asset>), or mount your data asset at "
                       f"{config.DEFAULT_SAMPLES}. {where}.")


def _with_var(sample: Sample, raw_var) -> Sample:
    _, sample.var = genes.symbol_var(raw_var, sample.sid)
    sample.species = genes.species(sample.var)
    return sample


def bundle_sample(candidate: Candidate, scratch: Path) -> Sample:
    """A bundle's sample after its inventory; var only when its count matrix was found."""
    inv = inventory.inspect(candidate, scratch)
    sample = Sample(candidate.sid, candidate.kind, candidate, inventory=inv)
    if inv.var is not None and not inv.missing_essential:
        _with_var(sample, inv.var)
    return sample


def part_cells(root: Path, part, parts: list) -> int | None:
    """Cells of a SpatialData sample, from its table's region column, when its regions are its own
    (None when unknown, e.g. samples told apart by a sample_id column within shared regions)."""
    others = {r for p in parts if p is not part for r in p.regions}
    if not part.table or not part.regions or others & set(part.regions):
        return None
    try:
        attrs = existing.table_attrs(root)[part.table]
        rows = existing._rows(root, part, attrs)
        return len(rows) if rows is not None else existing._n_rows(existing.tables(root)[part.table])
    except Exception:  # a count for the report only; reading the sample reports real problems
        return None


def spatialdata_samples(candidate: Candidate, settings, scratch: Path) -> list[Sample]:
    """The samples of one SpatialData object (staged to scratch when it is inside an archive)."""
    root = stage.spatialdata(candidate, scratch / STAGE / candidate.sid)
    out = []
    parts = existing.plan(root, candidate.sid, settings.sample_column)
    for part in parts:
        sample = Sample(part.sid, "spatialdata", candidate, part=part, path=root)
        n = part_cells(root, part, parts)
        _with_var(sample, existing.table_var(root, part))
        if n is not None:
            sample.var.attrs["n_cells"] = n     # for the report (Sample has no cell count of its own)
        out.append(sample)
    return out


def gather(settings, scratch: Path, present: set[str]) -> tuple[list[Sample], list[LeftOut], list[str]]:
    """Every sample in the inputs (bundles already in the store aren't inspected), the left-out items,
    and the problems found (one line each)."""
    entries = settings.sample_entries()
    if settings.default_samples and not Path(config.DEFAULT_SAMPLES).exists():
        raise _no_default_samples()
    found = discover.discover(entries, scratch)
    samples, problems = [], []
    for candidate in found.candidates:
        try:
            if candidate.kind == "spatialdata":
                samples += spatialdata_samples(candidate, settings, scratch)
            elif candidate.sid in present:
                samples.append(Sample(candidate.sid, candidate.kind, candidate))
            else:
                samples.append(bundle_sample(candidate, scratch))
        except INPUT_ERRORS as exc:
            problems.append(f"{candidate.sid} ({candidate.origin}): {exc}")
    return samples, found.left_out, problems


# ---------------------------------------------------------------- the decisions

def _duplicates(samples: list[Sample]) -> list[str]:
    seen: dict[str, Sample] = {}
    out = []
    for s in samples:
        if s.sid in seen:
            out.append(f"Two samples have the ID {s.sid}: {source(seen[s.sid])} and {source(s)}; rename one "
                       "(sample IDs come from folder or archive names, or a SpatialData's sample_id column).")
        else:
            seen[s.sid] = s
    return out


def broken(s: Sample) -> bool:
    return s.kind != "spatialdata" and s.inventory is not None and bool(s.inventory.missing_essential)


def _essentials(new: list[Sample]) -> list[str]:
    """One line per bundle missing an essential part, naming 0-byte files (broken uploads)."""
    out = []
    for s in new:
        if broken(s):
            empty = [n for n in s.inventory.notes if n.startswith("0-byte")]
            hint = (f" ({'; '.join(empty)}: a broken upload or download — download it again, or ask the authors)"
                    if empty else "")
            out.append(f"{s.sid} ({source(s)}): missing {'; '.join(s.inventory.missing_essential)}{hint}")
    return out


def given_as(s: Sample, samples: list[Sample]) -> str:
    """How to list just this sample: its folder or archive, or a glob over its prefixed files when it
    shares a folder with other samples (flat GEO / per-file downloads)."""
    origin = s.candidate.origin
    if not any(o is not s and o.candidate.origin == origin for o in samples):
        return str(origin)
    names = [m.container.name for m in s.candidate.files.values() if m.inner is None]
    prefix = os.path.commonprefix(names) if names else ""
    return f"{origin}/{prefix}*" if prefix else str(origin)


def leave_out_hint(new: list[Sample]) -> str:
    """When some samples are fine and others miss essentials: how to add the fine ones now."""
    good = [s for s in new if not broken(s)]
    if not good or len(good) == len(new):
        return ""
    paths = list(dict.fromkeys(given_as(s, new) for s in good))
    shown = ", ".join(paths[:6]) + (f", … ({len(paths) - 6} more)" if len(paths) > 6 else "")
    return f"\nTo add the other samples now, list only them in List of paths to samples: {shown}"


def _species(new: list[Sample], state: StoreState) -> list[str]:
    """One problem when the samples (and the store) aren't all one species; symbols collide across species."""
    found = {s.sid: s.species for s in new if s.species}
    kinds = set(found.values()) | ({state.species} if state.species else set())
    if len(kinds) <= 1:
        return []
    listed = ", ".join(f"{sid} ({sp})" for sid, sp in found.items())
    store = f"; the store holds {state.species} genes" if state.species else ""
    return [f"Samples of different species: {listed}{store}. Gene symbols collide across species (F3, Gata1 …), "
            "so one store holds one species: add the others to a separate store."]


def _model(new: list[Sample], state: StoreState, model: Path | None) -> tuple[list[str], list[str], dict[str, int]]:
    """Problems, low-overlap notes and genes shared per sample for the CellTypist model."""
    problems, notes, shared = [], [], {}
    name = model.stem if model else ""
    if state.model is not None and state.model != name:
        if state.model and not name:
            problems.append(f"This store labels every sample with CellTypist model {state.model}, and this run gives "
                            f"no model. Give {state.model}.pkl again, or add these samples to a separate store.")
        elif not state.model:
            problems.append(f"This store was built without CellTypist labels, and this run gives model {name}. "
                            "Leave the model field empty, or add these samples to a new store.")
        else:
            problems.append(f"This store labels every sample with CellTypist model {state.model}; this run gives "
                            f"{name}. Give {state.model}.pkl again, or add these samples to a separate store.")
    if model is None:
        return problems, notes, shared
    pinned = state.model == name and bool(name)
    for s in new:
        if s.var is None:
            continue
        try:
            n, _, note = annotate.check_overlap(model, s.sid, list(s.var.index), store_model=pinned)
        except annotate.AnnotateError as exc:
            problems.append(str(exc))
            continue
        shared[s.sid] = n
        if note:
            notes.append(note)
    return problems, notes, shared


def panels(new: list[Sample], state: StoreState) -> tuple[list[str], dict[str, dict], bool]:
    """The joint table's genes after this run (joint.union: the store's first, then each sample's new
    genes in order), per-sample panel stats, and whether the panels differ."""
    names = {s.sid: [str(g) for g in s.var.index] for s in new if s.var is not None}
    union, stats = joint.union(state.genes, names)
    for s in new:
        if s.sid in stats:
            stats[s.sid]["species"] = s.species
    differ = any(st["genes_shared"] < st["genes_in_panel"] or st["genes_missing"] for st in stats.values())
    return union, stats, differ


def panel_warning(stats: dict[str, dict], state: StoreState, union: list[str]) -> str:
    """The strong warning shown when panels differ: what the joint table will hold, then one line per sample."""
    lines = [f"{sid}: {st['genes_in_panel']} genes = {st['genes_shared']} also in other panels "
             f"({st['pct_shared']:.0f}%) + {st['genes_in_panel'] - st['genes_shared']} only in this panel; "
             f"{st['genes_missing']} of the {len(union)} joint genes not measured" for sid, st in stats.items()]
    held = f"the store's joint table has {len(state.genes)} genes; " if state.genes else ""
    return block(f"GENE PANELS DIFFER: {held}after this run the joint table holds the union of {len(union)} genes. "
                 "A cell reads 0 "
                 "for genes its sample didn't measure: var['n_samples_measured'] and "
                 "uns['spatial_ingest']['measured_genes'] tell measured zeros from filled ones. Each sample's own "
                 "table keeps its panel.", lines)


def check(samples: list[Sample], left_out: list[LeftOut], problems: list[str], state: StoreState,
          model: Path | None) -> Plan:
    """Decides the run: raises IngestError listing every problem, else logs the report and returns the Plan."""
    problems = list(problems) + _duplicates(samples)
    present_ids = set(state.present)
    new = [s for s in samples if s.sid not in present_ids]
    present = [s for s in samples if s.sid in present_ids]
    essentials = _essentials(new)
    model_problems, model_notes, model_shared = _model(new, state, model) if new else ([], [], {})
    problems += essentials + _species(new, state) + model_problems
    if problems:
        needs = f"\nEvery bundle needs {ESSENTIALS}." + leave_out_hint(new) if essentials else ""
        raise IngestError(block("The precheck found problems, so nothing was written. Fix or leave out these "
                                "samples and run again:", problems) + needs)
    union, stats, differ = panels(new, state)
    for sid, n in model_shared.items():
        stats[sid]["celltypist_genes_shared"] = n
    _report(new, present, left_out, state, stats, union, differ, model_notes)
    for s in present:   # main's summary rows for skipped samples
        stats[s.sid] = {"kind": s.kind, "source": source(s)}
    return Plan(samples=new, present=[s.sid for s in present], left_out=list(left_out), union_genes=union,
                panels=stats)


def same_folder(new: list[Sample]) -> list[tuple[Sample, Sample]]:
    """(bundle, SpatialData sample) pairs that are the only new samples in one folder."""
    out, done = [], set()
    for z in new:
        if z.kind != "spatialdata" or z.candidate.sid in done:
            continue
        done.add(z.candidate.sid)
        folder = Path(z.candidate.origin).parent
        under = {s.candidate.sid: s for s in new if Path(s.candidate.origin).is_relative_to(folder)}
        bundles = [s for s in under.values() if s.kind != "spatialdata"]
        if len(bundles) == 1 and len(under) == 2:
            out.append((bundles[0], z))
    return out


def _report(new, present, left_out, state, stats, union, differ, model_notes) -> None:
    for s in present:
        recorded = state.records.get(s.sid, {}).get("source")
        if recorded and recorded != source(s):
            logger.warning("%s is already in the store (added from %s); %s is skipped. If it is a different sample, "
                           "rename it.", s.sid, recorded, source(s))
    if left_out:
        logger.warning("%s", block(f"Left out {len(left_out)} item(s) in the inputs:",
                                   [f"{x.path}: {x.reason}" for x in left_out]))
    for bundle, sdata in same_folder(new):
        logger.warning("%s (%s) and %s (%s) are the only samples in %s and may be the same tissue (raw output and "
                       "a SpatialData object made from it); both are added. If they are, give only one of them.",
                       bundle.sid, NAMES[bundle.kind], sdata.sid, NAMES["spatialdata"], Path(source(sdata)).parent)
    lines = []
    for s in new:
        inv = s.inventory
        n_cells = inv.n_cells if inv is not None else (s.var.attrs.get("n_cells") if s.var is not None else None)
        cells = f", {n_cells} cells" if n_cells is not None else ""
        panel = f", {stats[s.sid]['genes_in_panel']} genes" if s.sid in stats else ""
        lines.append(f"{s.sid}: {NAMES.get(s.kind, s.kind)} from {source(s)}{cells}{panel}")
        for note in (inv.notes if inv is not None else ()):   # discovery already logged the candidate's notes
            logger.info("%s: %s", s.sid, note)
        if inv is not None and inv.missing_optional:
            logger.warning("%s: no %s in the bundle; it is added without them.", s.sid,
                           ", ".join(inv.missing_optional))
    held = (f"the store holds {len(state.present)} sample(s)" if state.present else "the store is new or empty")
    again = (f"; {len(present)} of the inputs are already in it and skipped ({', '.join(s.sid for s in present)})"
             if present else "")
    logger.info("%s", block(f"Precheck: {len(new)} sample(s) to add; {held}{again}.", lines))
    if differ:
        logger.warning("%s", panel_warning(stats, state, union))
    if model_notes:
        logger.warning("%s", block("The CellTypist model shares fewer than half of its genes with these panels; "
                                   "its labels rest on the genes present and may be less reliable "
                                   "(the _prob column gives each label's probability):", model_notes))


def run(settings, local: Path, fresh: bool, model: Path | None, scratch: Path) -> Plan:
    """Finds and checks every sample before any write; see the module docstring."""
    if model is not None:
        annotate.load_model(model)
    state = StoreState.read(local, fresh)
    samples, left_out, problems = gather(settings, scratch, set(state.present))
    return check(samples, left_out, problems, state, model)
