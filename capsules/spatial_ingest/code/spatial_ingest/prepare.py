"""The precheck, reading each sample (its table and spatial elements), and making it ready for the store with QC
metrics and CellTypist labels."""
from __future__ import annotations

import ast
import contextlib
import gc
import hashlib
import json
import logging
import math
import os
import re
import tarfile
import warnings
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import zarr
from anndata.io import read_elem, write_elem

import config
import inputs
import store as stores
from config import (NAMES, Candidate, Inventory, LeftOut, Loaded, Part, Plan, Prepared, Sample, TableSource, UserError,
                    block)

logger = logging.getLogger(__name__)

RECORD = "spatial_ingest"           # a per-sample table's uns record
ESSENTIALS = "a count matrix, and a cells table with cell IDs and x/y positions that match the matrix"
# errors that describe an input, not a bug: reported per sample, then the run stops
INPUT_ERRORS = (ValueError, OSError, zipfile.BadZipFile, tarfile.TarError)


class IngestError(UserError, RuntimeError):
    """The inputs or the store's state don't allow this run."""


@dataclass
class StoreState:
    """What the precheck needs to know about the store."""
    present: list[str] = field(default_factory=list)
    records: dict[str, dict] = field(default_factory=dict)
    genes: list[str] = field(default_factory=list)
    species: str | None = None
    model: str | None = None              # None: no joint table yet; "" built without a model
    gene_ids: list[str] = field(default_factory=list)

    @classmethod
    def read(cls, local: Path, fresh: bool) -> "StoreState":
        if fresh:
            return cls()
        present = present_samples(local)
        schema = stores.read_schema(local)
        if schema is None:
            return cls(present, records(local, present))
        return cls(present, records(local, present), list(schema.genes), schema.species or None,
                   schema.celltypist_model or "", list(schema.gene_ids))


def present_samples(local: Path) -> list[str]:
    return sorted(name.removesuffix("_table") for name in stores.committed(local)["tables"]
                  if name != stores.JOINT_NAME and name.endswith("_table"))


def records(local: Path, sids: list[str]) -> dict[str, dict]:
    """The uns record of each per-sample table."""
    out = {}
    for sid in sids:
        uns = zarr.open_group(local / "tables" / stores.table_name(sid), mode="r").get("uns")
        out[sid] = dict(read_elem(uns[RECORD])) if uns is not None and RECORD in uns else {}
    return out


def source(sample: Sample) -> str:
    """Where a sample came from, as recorded in its table."""
    return str(sample.candidate.origin)


def _no_default_samples() -> IngestError:
    data = config.DATA_DIR
    held = sorted(p.name for p in data.iterdir())[:20] if data.is_dir() else []
    where = f"{data} holds: {', '.join(held)}" if held else f"{data} is empty (no data asset is attached)"
    return IngestError(f"No samples given, and {config.DEFAULT_SAMPLES} doesn't exist. Fill in List of paths to "
                       f"samples (e.g. {data}/<your data asset>), or mount your data asset at "
                       f"{config.DEFAULT_SAMPLES}. {where}.")


def _with_var(sample: Sample, raw_var) -> Sample:
    _, sample.var = inputs.symbol_var(raw_var, sample.sid)
    sample.species = inputs.species(sample.var)
    return sample


def bundle_sample(candidate: Candidate, scratch: Path) -> Sample:
    inv = inputs.inspect(candidate, scratch)
    sample = Sample(candidate.sid, candidate.kind, candidate, inventory=inv)
    if inv.var is not None and not inv.missing_essential:
        _with_var(sample, inv.var)
    return sample


def part_cells(root: Path, part, parts: list) -> int | None:
    """Cells of a SpatialData sample whose regions are its own, else None."""
    others = {r for p in parts if p is not part for r in p.regions}
    if not part.table or not part.regions or others & set(part.regions):
        return None
    try:
        attrs = table_attrs(root)[part.table]
        rows = _rows(root, part, attrs)
        return len(rows) if rows is not None else _n_rows(tables(root)[part.table])
    except Exception:  # a count for the report only; reading the sample reports real problems
        return None


def spatialdata_samples(candidate: Candidate, settings, scratch: Path) -> list[Sample]:
    root = inputs.stage_spatialdata(candidate, inputs.stage_dir(scratch, candidate.sid))
    out = []
    parts = plan(root, candidate.sid, settings.sample_column)
    for part in parts:
        sample = Sample(part.sid, "spatialdata", candidate, part=part, path=root)
        n = part_cells(root, part, parts)
        _with_var(sample, table_var(root, part))
        if n is not None:
            sample.var.attrs["n_cells"] = n     # for the report (Sample has no cell count of its own)
        out.append(sample)
    return out


def gather(settings, scratch: Path, present: set[str]) -> tuple[list[Sample], list[LeftOut], list[str]]:
    """Every sample in the inputs, the left-out items, and the problems found."""
    entries = settings.sample_entries()
    if settings.default_samples and not Path(config.DEFAULT_SAMPLES).exists():
        raise _no_default_samples()
    found = inputs.discover(entries, scratch)
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
    out = []
    for s in new:
        if broken(s):
            empty = [n for n in s.inventory.notes if n.startswith("0-byte")]
            hint = (f" ({'; '.join(empty)}: a broken upload or download — download it again, or ask the authors)"
                    if empty else "")
            out.append(f"{s.sid} ({source(s)}): missing {'; '.join(s.inventory.missing_essential)}{hint}")
    return out


def given_as(s: Sample, samples: list[Sample]) -> str:
    """How to list just this sample: its folder or archive, or a glob over its prefixed files."""
    origin = s.candidate.origin
    if not any(o is not s and o.candidate.origin == origin for o in samples):
        return str(origin)
    names = [m.container.name for m in s.candidate.files.values() if m.inner is None]
    prefix = os.path.commonprefix(names) if names else ""
    return f"{origin}/{prefix}*" if prefix else str(origin)


def leave_out_hint(new: list[Sample]) -> str:
    """How to add the fine samples now when others miss essentials."""
    good = [s for s in new if not broken(s)]
    if not good or len(good) == len(new):
        return ""
    paths = list(dict.fromkeys(given_as(s, new) for s in good))
    shown = ", ".join(paths[:6]) + (f", … ({len(paths) - 6} more)" if len(paths) > 6 else "")
    return f"\nTo add the other samples now, list only them in List of paths to samples: {shown}"


def _species(new: list[Sample], state: StoreState) -> tuple[list[str], str]:
    """Symbols collide across species, so samples (and the store) must be one species."""
    found = {s.sid: s.species for s in new if s.species}
    kinds = set(found.values()) | ({state.species} if state.species else set())
    if len(kinds) <= 1:
        return [], ""
    counts = pd.Series(list(found.values()), dtype=object).value_counts(sort=True)
    main = state.species or str(counts.index[0])
    odd: dict[str, list[str]] = {}
    for sid, sp_ in found.items():
        if sp_ != main:
            odd.setdefault(sp_, []).append(sid)
    named = "; ".join(f"{', '.join(sids)} {'is' if len(sids) == 1 else 'are'} {sp_}" for sp_, sids in odd.items())
    n_main = sum(sp_ == main for sp_ in found.values())
    others = f"the other {n_main} sample{'s' if n_main > 1 else ''}" if n_main else ""
    if not others:
        rest = f"the store holds {main} genes"
    elif state.species:
        rest = f"the store and {others} are {main}"
    else:
        rest = f"{others} {'is' if n_main == 1 else 'are'} {main}"
    problem = (f"Samples of different species: {named}; {rest}. "
               "Gene symbols collide across species (F3, Gata1 …), so one store holds one species: add the "
               f"{'/'.join(odd)} sample(s) to a separate store.")
    keep = [s for s in new if s.species in (main, None) and not broken(s)]
    if not keep:
        return [problem], ""
    paths = list(dict.fromkeys(given_as(s, new) for s in keep))
    return [problem], f"\nTo add the {main} samples now, list only them in List of paths to samples: {', '.join(paths)}"


def _model(new: list[Sample], state: StoreState, model: Path | None) -> tuple[list[str], list[str], dict[str, int]]:
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
            n, _, note = check_overlap(model, s.sid, list(s.var.index), store_model=pinned)
        except AnnotateError as exc:
            problems.append(str(exc))
            continue
        shared[s.sid] = n
        if note:
            notes.append(note)
    return problems, notes, shared


def panels(new: list[Sample], state: StoreState) -> tuple[list[str], dict[str, dict], bool, dict[str, list[str]]]:
    """The joint table's genes after this run, per-sample panel stats, whether they differ, and split symbols."""
    names = {s.sid: ([str(g) for g in s.var.index],
                     [str(i) for i in s.var["gene_ids"]] if "gene_ids" in s.var else [""] * len(s.var))
             for s in new if s.var is not None}
    union, stats, splits = stores.gene_union(state.genes, state.gene_ids, names)
    for s in new:
        if s.sid in stats:
            stats[s.sid]["species"] = s.species
    differ = any(st["genes_shared"] < st["genes_in_panel"] or st["genes_missing"] for st in stats.values())
    return union, stats, differ, splits


def panel_warning(stats: dict[str, dict], state: StoreState, union: list[str]) -> str:
    lines = [f"{sid}: {st['genes_in_panel']} genes = {st['genes_shared']} also in the store or in other samples of "
             f"this run ({st['pct_shared']:.0f}%) + {st['genes_in_panel'] - st['genes_shared']} only in this panel; "
             f"{st['genes_missing']} of the {len(union)} joint genes not measured" for sid, st in stats.items()]
    held = f"the store's joint table has {len(state.genes)} genes; " if state.genes else ""
    return (block(f"Gene panels differ: {held}after this run the joint table holds the union of {len(union)} genes. "
                  "Each sample's own table keeps its panel.", lines) +
            "\nNothing to fix: genes a sample didn't measure read 0 in its joint rows; compare samples on genes "
            "measured in all of them (var['n_samples_measured']).")


def split_note(sid: str, splits: list[str]) -> str:
    shown = "; ".join(splits[:5]) + (f"; … ({len(splits) - 5} more)" if len(splits) > 5 else "")
    return f"{sid}: {len(splits)} gene symbol(s) have another gene_id than the same symbol already planned: {shown}"


def _object_name(s: Sample) -> str:
    """What tells a SpatialData object's samples apart from another object's: its ID, or its parent folder."""
    name = s.candidate.sid
    if name == s.sid:
        name = inputs.sanitize(Path(s.candidate.origin).parent.name)[0] or "object"
    return name


def _clash(s: Sample, claimed: dict[str, Sample], taken: dict[str, tuple[str, str | None]]) -> str | None:
    """Why `s.sid` can't be used as it is, or None."""
    low = s.sid.lower()
    if low in claimed:
        other = claimed[low]
        return f"{other.sid} from {source(other)} in this run"
    if low in taken:
        sid, recorded = taken[low]
        if sid != s.sid:
            return f"{sid}, already in the store"
        if recorded and recorded != source(s):
            return f"{sid}, already in the store (added from {recorded})"
    return None


def resolve_ids(samples: list[Sample], state: StoreState) -> list[str]:
    """A SpatialData sample whose ID is taken becomes `<object>_<sid>`; a bundle keeps its folder's ID."""
    taken = {sid.lower(): (sid, state.records.get(sid, {}).get("source")) for sid in state.present}
    claimed: dict[str, Sample] = {}
    problems = []
    for s in sorted(samples, key=lambda s: s.kind == "spatialdata"):     # bundles keep their ID whatever the order
        why = _clash(s, claimed, taken)
        if why and s.kind == "spatialdata":
            old, new = s.sid, f"{_object_name(s)}_{s.sid}"
            s.sid = new
            s.part = replace(s.part, sid=new) if s.part is not None else None
            again = _clash(s, claimed, taken)
            if again:
                problems.append(f"{old} from {source(s)}: its ID is taken by {why}, and {new} by {again}; give the "
                                "object another name (its folder name) or a different sample_id column.")
            else:
                logger.info("%s from %s is added as %s: the ID %s is taken by %s.", old, source(s), new, old, why)
        elif why:
            low = s.sid.lower()
            other = claimed[low].sid if low in claimed else taken[low][0]
            if other != s.sid:
                problems.append(f"{other} and {s.sid} ({source(s)}) differ only in case; spatialdata can't hold both "
                                "in one store: rename one folder.")
        claimed.setdefault(s.sid.lower(), s)
    return problems


def check(samples: list[Sample], left_out: list[LeftOut], problems: list[str], state: StoreState,
          model: Path | None) -> Plan:
    """Raises IngestError listing every problem, else logs the report and returns the Plan."""
    problems = list(problems) + resolve_ids(samples, state) + _duplicates(samples)
    present_ids = set(state.present)
    new = [s for s in samples if s.sid not in present_ids]
    present = [s for s in samples if s.sid in present_ids]
    essentials = _essentials(new)
    model_problems, model_notes, model_shared = _model(new, state, model) if new else ([], [], {})
    species_problems, species_hint = _species(new, state)
    problems += essentials + species_problems + model_problems
    if problems:
        needs = f"\nEvery bundle needs {ESSENTIALS}." + leave_out_hint(new) if essentials else ""
        raise IngestError(block("The precheck found problems, so nothing was written. Fix or leave out these "
                                "samples and run again:", problems) + needs + (species_hint if not essentials else ""))
    union, stats, differ, splits = panels(new, state)
    for sid, n in model_shared.items():
        stats[sid]["celltypist_genes_shared"] = n
    _report(new, present, left_out, state, stats, union, differ, model_notes)
    for sid, split in splits.items():
        if split:
            logger.info("%s", split_note(sid, split))
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
    big = [s.sid for s in new if s.inventory is not None and (s.inventory.n_cells or 0) > BIG_SAMPLE]
    memory = machine_memory_gb()
    if big and memory is not None and memory < BIG_SAMPLE_GB:
        logger.warning("This machine has %.0f GB of memory; samples with over %d cells (%s) have peaked at 8-11 GB. "
                       "If the run is killed without an error, run it again on the large machine (64 GB).",
                       memory, BIG_SAMPLE, ", ".join(big))
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


BIG_SAMPLE = 100_000
BIG_SAMPLE_GB = 24


def machine_memory_gb() -> float | None:
    """The container's memory limit, else the machine's."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            text = Path(path).read_text().strip()
        except OSError:
            continue
        if text.isdigit() and int(text) < 1 << 50:
            return int(text) / 1e9
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError):
        return None


def precheck(settings, local: Path, fresh: bool, model: Path | None, scratch: Path) -> Plan:
    """Finds and checks every sample before anything is written."""
    if model is not None:
        load_model(model)
    state = StoreState.read(local, fresh)
    samples, left_out, problems = gather(settings, scratch, set(state.present))
    return check(samples, left_out, problems, state, model)


TABLE_DIRS = ("tables", "table")
GROUP_FILES = ("zarr.json", ".zgroup", ".zattrs")
COUNT_SOURCES = (("X", "var"), ("layers/counts", "var"), ("raw/X", "raw/var"))
CHECK_ROWS = 2000                       # rows sampled to judge whether a matrix holds counts
NNZ_PER_CHUNK = 32 * 1024 * 1024        # nonzeros per row chunk streamed (sparse sources)
DENSE_VALUES_PER_CHUNK = 16 * 1024 * 1024


class SplitError(UserError, ValueError):
    """A SpatialData object (or one of its samples) that can't be added; never names the staged path."""


def _is_group(path: Path) -> bool:
    return path.is_dir() and any((path / f).is_file() for f in GROUP_FILES)


def _children(path: Path) -> list[str]:
    if not path.is_dir():
        return []
    return sorted(p.name for p in path.iterdir() if not p.name.startswith(".") and _is_group(p))


def _open(path: Path) -> zarr.Group:
    return zarr.open_group(path, mode="r", use_consolidated=False)


def root_attrs(path: Path) -> dict:
    try:
        return dict(_open(path).attrs)
    except Exception:
        return {}


def _is_anndata(path: Path) -> bool:
    return _is_group(path) and (path / "obs").is_dir() and (path / "X").exists()


def tables(path: Path) -> dict[str, Path]:
    """The object's tables by name: `tables/<name>`, or the pre-0.1 `table/table`."""
    path = Path(path)
    for folder in TABLE_DIRS:
        found = {n: path / folder / n for n in _children(path / folder) if _is_anndata(path / folder / n)}
        if found:
            return found
    if _is_anndata(path / "table"):
        return {"table": path / "table"}
    return {}


def _table_attrs(group: zarr.Group) -> dict:
    """A table's region link, from its attributes or (older writers) `uns/spatialdata_attrs`."""
    attrs = dict(group.attrs)
    if "region" not in attrs and "uns" in group and "spatialdata_attrs" in group["uns"]:
        attrs.update(dict(read_elem(group["uns"]["spatialdata_attrs"])))
    region = attrs.get("region")
    if region is not None and not isinstance(region, str):
        attrs["region"] = [str(r) for r in np.atleast_1d(region)]
    return attrs


def table_attrs(path: Path) -> dict[str, dict]:
    return {name: _table_attrs(_open(p)) for name, p in tables(path).items()}


def _region_names(attrs) -> list[str]:
    region = attrs.get("region")
    if region is None:
        return []
    return [region] if isinstance(region, str) else list(region)


def _column(path: Path, table: str, column: str) -> np.ndarray | None:
    obs = _open(tables(path)[table] / "obs")
    if column not in obs:
        return None
    values = read_elem(obs[column])
    return np.asarray(values.astype(object) if hasattr(values, "astype") else values, dtype=object)


def _read_element(kind: str, where: Path):
    from spatialdata._io.io_points import _read_points
    from spatialdata._io.io_raster import _read_multiscale
    from spatialdata._io.io_shapes import _read_shapes
    from spatialdata._io.io_zarr import get_raster_format_for_read

    if kind in ("images", "labels"):
        group = _open(where)
        version = "0.2" if "ome" in group.attrs else "0.1"
        return _read_multiscale(str(where), "image" if kind == "images" else "labels",
                                get_raster_format_for_read(group, version))
    return (_read_points if kind == "points" else _read_shapes)(str(where))


def read_elements(path: Path, wanted: set[tuple[str, str]] | None = None
                  ) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    """The object's elements read one by one, and those left out with why."""
    import warnings

    from spatialdata.transformations import get_transformation

    path = Path(path)
    elements: dict[str, dict[str, object]] = {k: {} for k in stores.ELEMENT_KINDS}
    left_out: dict[str, str] = {}
    for kind in stores.ELEMENT_KINDS:
        for name in _children(path / kind):
            if wanted is not None and (kind, name) not in wanted:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    element = _read_element(kind, path / kind / name)
                systems = get_transformation(element, get_all=True)
            except Exception as exc:
                left_out[f"{kind}/{name}"] = f"unreadable: {type(exc).__name__}: {str(exc)[:120]}"
                continue
            if not systems:
                left_out[f"{kind}/{name}"] = "in no coordinate system"
                continue
            elements[kind][name] = element
    return elements, left_out


def coordinate_systems(elements: dict[str, dict[str, object]]) -> dict[tuple[str, str], set[str]]:
    from spatialdata.transformations import get_transformation

    return {(kind, name): set(get_transformation(element, get_all=True))
            for kind, items in elements.items() for name, element in items.items()}


def _trim(text: str, from_end: bool) -> str:
    cut = max((text.rfind(s) for s in stores.SEPARATORS), default=-1) if not from_end else \
        min((i for i in (text.find(s) for s in stores.SEPARATORS) if i >= 0), default=len(text))
    return text[: cut + 1] if not from_end else text[cut:]


def tags(names: list[str]) -> dict[str, str]:
    """Each name without the prefix and suffix all names share: `cell_labels-a` -> `a`."""
    if len(names) < 2:
        return {n: n for n in names}
    prefix = _trim(os.path.commonprefix(names), from_end=False)
    suffix = _trim(os.path.commonprefix([n[::-1] for n in names])[::-1], from_end=True)
    out = {}
    for n in names:
        tag = n[len(prefix): len(n) - len(suffix) if suffix else None]
        out[n] = tag.strip(stores.SEPARATORS) or n
    return out


def has_token(name: str, token: str) -> bool:
    return any(name.startswith(token + s) or name.endswith(s + token) or (s + token + s) in name
               for s in stores.SEPARATORS) or name == token


def _region_table(attrs: dict[str, dict]) -> tuple[dict[str, str], list[str]]:
    chosen, left_out = {}, []
    for name in sorted(attrs, key=lambda n: (n != "table", n)):
        regions = _region_names(attrs[name])
        if not regions or all(r in chosen for r in regions):
            left_out.append(name)
            continue
        for r in regions:
            chosen.setdefault(r, name)
    return chosen, left_out


def _groups_by_column(path: Path, region_table: dict[str, str], attrs: dict[str, dict],
                      column: str) -> dict[str, list[str]] | None:
    groups: dict[str, list[str]] = {}
    for region, table in region_table.items():
        values = _column(path, table, column)
        if values is None:
            return None
        keys = _column(path, table, attrs[table]["region_key"])
        found = {str(v) for v in values[keys.astype(str) == region] if not pd.isna(v)}
        if len(found) > 1:
            raise SplitError(f"region {region} holds cells of several samples in {column!r} "
                             f"({', '.join(sorted(found)[:5])}); write one region per sample.")
        if not found:
            return None
        groups.setdefault(found.pop(), []).append(region)
    return groups


def _groups_by_system(regions: list[str], systems: dict[str, set[str]]) -> dict[str, list[str]]:
    """Regions sharing a coordinate system are one sample, keyed by its first region's tag."""
    groups: list[list[str]] = []
    for r in regions:
        linked = [g for g in groups if any(systems[r] & systems[o] for o in g)]
        for g in linked:
            groups.remove(g)
        groups.append(sum(linked, []) + [r])
    region_tags = tags(regions)
    return {region_tags[g[0]]: g for g in groups}


def _n_rows(table: Path) -> int:
    return inputs.n_obs(_open(table))


def _sample_ids(keys: list[str], object_id: str, notes: list[tuple[int, str]]) -> dict[str, str]:
    """Values of the sample column as valid IDs, made unique with the object's name."""
    out = {k: stores.INVALID_CHARS.sub("_", str(k)).strip("_") or "sample" for k in keys}
    taken: dict[str, int] = {}
    for v in out.values():
        taken[v.lower()] = taken.get(v.lower(), 0) + 1
    used: set[str] = set()
    for k in keys:
        v = out[k]
        if taken[v.lower()] > 1:
            v = base = f"{object_id}_{v}"
            n = 1
            while v.lower() in used:
                v, n = f"{base}_{n}", n + 1
        used.add(v.lower())
        out[k] = v
    changed = {k: v for k, v in out.items() if k != v}
    if changed:
        notes.append((logging.INFO, "sample IDs from the sample column, in the characters IDs may hold: " +
                      ", ".join(f"{k!r} -> {v}" for k, v in changed.items())))
    return out


def plan(path: Path, default_sid: str, sample_column: str) -> list[Part]:
    """The samples of a SpatialData object; any problem with it is a SplitError."""
    path = Path(path)
    notes: list[tuple[int, str]] = []
    try:
        parts = _plan(path, default_sid, sample_column, notes)
        for level, text in notes:              # only for an object that goes in: no noise before a refusal
            logger.log(level, "%s: %s", path.name, text)
        return parts
    except SplitError:
        raise
    except Exception as exc:
        raise SplitError(f"it can't be read as SpatialData ({type(exc).__name__}: {exc}).") from None


def _plan(path: Path, default_sid: str, sample_column: str, notes: list[tuple[int, str]]) -> list[Part]:
    attrs = table_attrs(path)
    if not attrs:
        raise SplitError("it has no table; spatial_ingest needs a table of cells x genes (counts) "
                         "in the SpatialData object.")
    elements, unusable = read_elements(path)
    if unusable:
        notes.append((logging.WARNING, f"{len(unusable)} element(s) left out: " +
                      "; ".join(f"{k} ({why})" for k, why in unusable.items())))
    if "spatialdata_attrs" not in root_attrs(path):
        old = ", pre-0.1 table/table" if (path / "table").is_dir() else ""
        notes.append((logging.INFO, f"older SpatialData layout (no version attributes{old}); its elements are "
                                    "rewritten in the current format"))
    systems = coordinate_systems(elements)
    region_table, left_out = _region_table(attrs)
    if not region_table:
        if len(attrs) != 1:
            raise SplitError(f"it has {len(attrs)} tables ({', '.join(sorted(attrs))}) and none annotates an "
                             "element, so it isn't clear which holds the cells; keep one table, or link each to "
                             "its region (region / region_key / instance_key).")
        table = next(iter(attrs))
        count_source(path, table)
        return [Part(default_sid, table, (), tuple(sorted(systems)), ())]
    missing = [r for r in region_table if not any(k[1] == r for k in systems)]
    if missing:
        why = [f"{r} ({unusable[k]})" for r in missing for k in unusable if k.endswith(f"/{r}")]
        raise SplitError(f"its tables annotate regions that aren't (readable) elements of the object: "
                         f"{', '.join(why or missing)}. Write the object with those elements, or with tables that "
                         "only annotate elements it holds.")
    region_systems = {r: next(cs for (k, n), cs in systems.items() if n == r) for r in region_table}
    groups = _groups_by_column(path, region_table, attrs, sample_column)
    named = groups is not None
    if not named:
        groups = _groups_by_system(list(region_table), region_systems)
    ours = stores.MARKER in root_attrs(path)
    for name in left_out:
        n = _n_rows(tables(path)[name])
        if ours and name == "table":
            notes.append((logging.INFO, f"its joint table ({n} rows) is left out; the per-sample tables carry the "
                                        "samples"))
        else:
            notes.append((logging.WARNING if n else logging.INFO,
                          f"table {name} ({n} rows) annotates no further region; left out"))

    single = len(groups) == 1
    region_tags = tags(list(region_table))
    owners: dict[str, set[str]] = {}
    for key, regions in groups.items():
        for r in regions:
            for cs in region_systems[r]:
                owners.setdefault(cs, set()).add(key)
    region_keys = {r: key for key, regions in groups.items() for r in regions}
    assigned: dict[str, list[tuple[str, str]]] = {key: [] for key in groups}
    unplaced = []
    for kind, name in sorted(systems):
        if name in region_keys:
            assigned[region_keys[name]].append((kind, name))
            continue
        if single:
            assigned[next(iter(groups))].append((kind, name))
            continue
        cs_owners = {next(iter(owners[c])) if len(owners.get(c, ())) == 1 else None for c in systems[(kind, name)]}
        if len(cs_owners) == 1 and None not in cs_owners:
            assigned[cs_owners.pop()].append((kind, name))
            continue
        by_name = [key for key, regions in groups.items()
                   if any(has_token(name, t) for t in {key, *(region_tags[r] for r in regions)})]
        if len(by_name) == 1:
            assigned[by_name[0]].append((kind, name))
        else:
            unplaced.append(name)
    if unplaced:
        notes.append((logging.WARNING, f"elements that belong to no single sample are left out: {', '.join(unplaced)}"))
    parts = []
    checked: set[str] = set()
    named_ids = _sample_ids(list(groups), default_sid, notes) if named else {}
    for key, regions in groups.items():
        sid = named_ids[key] if named else default_sid if single else f"{default_sid}_{key}"
        names = sorted({region_table[r] for r in regions}, key=lambda n: (n != "table", n))
        if len(names) > 1:
            notes.append((logging.WARNING, f"sample {sid} is annotated by several tables; keeping {names[0]}, "
                                           f"leaving out {', '.join(names[1:])}"))
            regions = [r for r in regions if region_table[r] == names[0]]
        if names[0] not in checked:
            count_source(path, names[0])
            checked.add(names[0])
        # the only sample of an unnamed object has no tag (its region's name would rename `cell_boundaries`)
        found = () if single and not named else \
            ((key,) if named or len(regions) == 1 else ()) + tuple(region_tags[r] for r in regions if len(regions) == 1)
        parts.append(Part(sid, names[0], tuple(regions), tuple(assigned[key]),
                          tuple(dict.fromkeys(t for t in found if t != sid))))
    return parts


def _matrix(table: Path, where: str):
    node = _open(table)
    for part in where.split("/"):
        if not isinstance(node, zarr.Group) or part not in node:
            return None
        node = node[part]
    if isinstance(node, zarr.Group) and node.attrs.get("encoding-type") not in ("csr_matrix", "csc_matrix"):
        return None
    return node


def _shape(node) -> tuple[int, int]:
    shape = node.shape if isinstance(node, zarr.Array) else node.attrs["shape"]
    return int(shape[0]), int(shape[1])


def _head(node, n: int) -> sp.csr_matrix:
    if isinstance(node, zarr.Array):
        return sp.csr_matrix(np.asarray(node[:n]))
    n_obs, n_vars = _shape(node)
    indptr = np.asarray(node["indptr"][:], dtype=np.int64)
    if node.attrs.get("encoding-type") == "csr_matrix":
        n = min(n, n_obs)
        end = int(indptr[n])
        return sp.csr_matrix((np.asarray(node["data"][:end]), np.asarray(node["indices"][:end]), indptr[: n + 1]),
                             shape=(n, n_vars))
    end = min(int(indptr[-1]), NNZ_PER_CHUNK)
    head = sp.csc_matrix((np.asarray(node["data"][:end]), np.asarray(node["indices"][:end]),
                          np.minimum(indptr, end)), shape=(n_obs, n_vars))
    return sp.csr_matrix(head[: min(n, n_obs)])


def _problem(x: sp.csr_matrix) -> str | None:
    """Why the values aren't counts, or None."""
    data = np.asarray(x.data, dtype=np.float64)
    if data.size and np.nanmin(data) < 0:
        return f"negative values (min {np.nanmin(data):.3g})"
    frac = data[np.mod(data, 1) != 0]
    if frac.size:
        return f"fractional values (e.g. {frac[0]:.3g})"
    return None


def _n_vars(table: Path, var: str) -> int:
    group = _open(table / var)
    return inputs.column_length(group[group.attrs["_index"]])


def count_source(path: Path, table: str) -> tuple[str, str]:
    """Where the table's RNA counts are: X, layers['counts'] or .raw, as (matrix, var) paths."""
    where = tables(Path(path))[table]
    problems = {}
    for matrix, var in COUNT_SOURCES:
        node = _matrix(where, matrix)
        if node is None or (matrix == "raw/X" and _n_vars(where, var) != _shape(node)[1]):
            continue
        problem = _problem(_head(node, CHECK_ROWS))
        if problem is None:
            return matrix, var
        problems[matrix] = problem
    var = read_elem(_open(where)["var"])
    features = ", ".join(map(str, list(var.index[:5]))) + (", ..." if len(var) > 5 else "")
    found = "; ".join(f"{m} has {p}" for m, p in problems.items()) or "it has no count matrix"
    raise SplitError(f"table {table!r} doesn't hold RNA counts ({found}; no layers['counts'] or .raw with "
                     f"counts). With features like {features} it looks like protein intensities, metabolite signals "
                     "or normalized expression; spatial_ingest takes tables of cells x genes with raw counts only. If "
                     "the counts are elsewhere in the object, put them in X or layers['counts'].")


def table_var(path: Path, part: Part) -> pd.DataFrame:
    _, var = count_source(path, part.table)
    return read_elem(_open(tables(Path(path))[part.table] / var))


def _rows(path: Path, part: Part, attrs: dict) -> np.ndarray | None:
    """The part's rows of its table (None: all of them)."""
    if not part.regions:
        return None
    keys = _column(path, part.table, attrs["region_key"]).astype(str)
    if set(np.unique(keys)) <= set(part.regions):
        return None
    return np.flatnonzero(np.isin(keys, part.regions))


def _reader(node, scratch: Path):
    from annizarr import AppConfig
    from annizarr._sources._readers import as_reader

    scratch.mkdir(parents=True, exist_ok=True)
    return as_reader(node, cfg=AppConfig(), tmp_dir=str(scratch))


def _chunks(reader, dense: bool, rows: np.ndarray | None, keep: np.ndarray, sid: str) -> Iterator[sp.csr_matrix]:
    n_obs, n_vars = reader.shape
    per_row = n_vars if dense else max(1.0, reader.nnz / max(1, n_obs))
    step = max(1, int(min(NNZ_PER_CHUNK / per_row, DENSE_VALUES_PER_CHUNK / max(1, n_vars) if dense else n_obs)))
    columns = np.flatnonzero(keep)
    for r0 in range(0, n_obs, step):
        r1 = min(r0 + step, n_obs)
        local = None
        if rows is not None:
            local = rows[(rows >= r0) & (rows < r1)] - r0
            if not len(local):
                continue
        chunk = sp.csr_matrix(reader.csr_rows(r0, r1))
        if local is not None:
            chunk = chunk[local]
        if len(columns) != n_vars:
            chunk = chunk[:, columns]
        problem = _problem(chunk)
        if problem:
            raise SplitError(f"its table's counts hold {problem} (rows {r0}-{r1}); only raw counts go in.")
        yield chunk


def _clean_obs(obs: pd.DataFrame, keys: dict[str, str | None]) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """obs with names spatialdata accepts, and the region/instance keys following their columns."""
    renames = stores.safe_names(obs.columns)
    if obs.index.name is not None and (obs.index.name in obs.columns or obs.index.name.lower() in
                                       {c.lower() for c in stores.RESERVED_OBS}):
        obs.index.name = None
    if renames:
        logger.info("obs columns renamed so spatialdata can read them: %s",
                    ", ".join(f"{a} -> {b}" for a, b in renames.items()))
        obs = obs.rename(columns=renames)
    return obs, {k: renames.get(v, v) if v else v for k, v in keys.items()}


def _obsm(table: Path, rows: np.ndarray | None, n: int) -> dict:
    group = _open(table)
    out = {}
    if "obsm" not in group:
        return out
    for key in group["obsm"]:
        try:
            value = read_elem(group["obsm"][key])
            if rows is not None:
                value = value.iloc[rows] if isinstance(value, pd.DataFrame) else value[rows]
        except Exception:
            continue
        if getattr(value, "shape", (None,))[0] == n and not sp.issparse(value):
            out[stores.safe_names([key], ()).get(key, key)] = value
    return out


def load_part(path: Path, part: Part, scratch: Path) -> Loaded:
    """One sample of a SpatialData object: its lazy elements and its counts streamed to scratch."""
    path, scratch = Path(path), Path(scratch)
    elements, left_out = read_elements(path, set(part.elements))
    if left_out:
        logger.info("%s: elements left out: %s", part.sid, ", ".join(left_out))
    where = tables(path)[part.table]
    attrs = _table_attrs(_open(where))
    matrix, var_at = count_source(path, part.table)
    if matrix != "X":
        logger.info("%s: X isn't raw counts; using %s", part.sid, "layers['counts']" if matrix != "raw/X" else ".raw")
    rows = _rows(path, part, attrs)
    obs = read_elem(_open(where)["obs"])
    if rows is not None:
        obs = obs.iloc[rows]
    obs, keys = _clean_obs(obs.copy(), {"region_key": attrs.get("region_key") if part.regions else None,
                                        "instance_key": attrs.get("instance_key") if part.regions else None})
    try:
        keep, var = inputs.symbol_var(read_elem(_open(where / var_at)), part.sid)
    except inputs.PanelError as exc:
        raise SplitError(str(exc)) from None
    if not keep.all():
        logger.info("%s: %d non-Gene-Expression features left out", part.sid, int((~keep).sum()))
    node = _matrix(where, matrix)
    out = scratch / "tables" / f"{part.sid}.zarr"
    reader = _reader(node, scratch / "tmp")
    try:
        writer = inputs.TableWriter(out, obs, var, reader.dtype, obsm=_obsm(where, rows, len(obs)))
        for chunk in _chunks(reader, isinstance(node, zarr.Array), rows, keep, part.sid):
            writer.add(chunk)
        writer.close()
    finally:
        reader.close()
    return Loaded(elements, TableSource(out, obs, len(obs), len(var)), list(part.regions), keys["region_key"],
                  keys["instance_key"], part.tags)


REGION = "cell_boundaries"
CIRCLES = "cell_circles"
POINTS_PARTITION_ROWS = 2_000_000    # transcripts per points partition


@dataclass
class Built:
    elements: dict[str, dict[str, object]] = field(default_factory=lambda: {k: {} for k in stores.ELEMENT_KINDS})
    region: str = REGION
    obs_columns: dict[str, np.ndarray] = field(default_factory=dict)   # extra obs columns (table row order)
    problems: list[str] = field(default_factory=list)                 # optional parts that failed to read


def _size(*paths: Path) -> str:
    total = 0
    for p in paths:
        p = Path(p)
        items = [q for q in p.rglob("*") if q.is_file()] if p.is_dir() else [p]
        total += sum(q.stat().st_size for q in items if q.exists())
    return f"{total / 1e9:.1f} GB"


def progress(sid: str, what: str, *paths: Path) -> None:
    logger.info("%s: reading %s (%s)", sid, what, _size(*paths) if paths else "")


def scale(pixel_size: float, axes=("x", "y")):
    from spatialdata.transformations import Scale

    return Scale([1.0 / pixel_size] * len(axes), axes=axes)


def xenium_specs(folder: Path) -> tuple[dict, object]:
    """experiment.xenium's content and the analyzer version (None if unknown)."""
    from spatialdata_io.readers.xenium import _parse_version_of_xenium_analyzer

    path = folder / "experiment.xenium"
    specs = {}
    if path.exists():
        try:
            specs = json.loads(path.read_text())
        except (OSError, ValueError):
            specs = {}
    specs.setdefault("pixel_size", inputs.DEFAULT_PIXEL_SIZE)
    version = None
    if specs.get("analysis_sw_version") or specs.get("xenium_ranger"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            version = _parse_version_of_xenium_analyzer(specs)
    return specs, version


def circles(xy: np.ndarray, area: np.ndarray | None, ids, pixel_size: float):
    from spatialdata.models import ShapesModel

    radius = np.sqrt(np.asarray(area, dtype=float) / np.pi) if area is not None else np.full(len(xy), 5.0)
    radius = np.where(np.isfinite(radius) & (radius > 0), radius, 1.0)
    return ShapesModel.parse(np.asarray(xy, dtype=float), geometry=0, radius=radius,
                             index=pd.Index(np.asarray(ids), name="cell_id"),
                             transformations={"global": scale(pixel_size)})


def _usable_label_id(path: Path) -> bool:
    """Re-exports carry label_id as all-NaN floats; grouping by it would give one polygon per vertex."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    if "label_id" not in schema.names:
        return True
    if not pa.types.is_integer(schema.field("label_id").type):
        return False
    return pq.read_table(path, columns=["label_id"]).column(0).null_count == 0


BOUNDARY_BATCH_ROWS = 1 << 21          # vertices per boundary read batch


def _boundary_batches(path: Path, scratch: Path):
    """(x, y, cell_id, label_id or None) per batch of vertices."""
    if inputs.is_parquet(path):
        import pyarrow.parquet as pq

        names = pq.read_schema(inputs.plain(path, scratch)).names

        def record_batches():
            with pq.ParquetFile(inputs.plain(path, scratch)) as pf:
                for b in pf.iter_batches(batch_size=BOUNDARY_BATCH_ROWS,
                                         columns=[c for c in ("vertex_x", "vertex_y", "cell_id", "label_id")
                                                  if c in names]):
                    yield {c: b.column(c).to_numpy(zero_copy_only=False) for c in b.schema.names}
        batches = record_batches()
    else:
        with inputs.open_text(path) as f:
            names = list(pd.read_csv(f, nrows=0).columns)

        def chunks():
            with inputs.open_text(path) as f:
                for chunk in pd.read_csv(f, usecols=[c for c in ("vertex_x", "vertex_y", "cell_id", "label_id")
                                                     if c in names], chunksize=BOUNDARY_BATCH_ROWS):
                    yield {c: chunk[c].to_numpy() for c in chunk.columns}
        batches = chunks()
    if not {"vertex_x", "vertex_y", "cell_id"} <= set(names):
        raise inputs.TableError(f"{path.name} has no vertex_x / vertex_y / cell_id columns")
    # decided once for the whole file, so batches agree
    usable = inputs.is_parquet(path) and "label_id" in names and _usable_label_id(inputs.plain(path, scratch))
    for b in batches:
        yield b["vertex_x"], b["vertex_y"], np.asarray(b["cell_id"]), (b["label_id"] if usable else None)


def polygons_from_table(path: Path, pixel_size: float, scratch: Path, is_nucleus: bool):
    """Polygons from a boundary parquet/csv, batch by batch, carrying a polygon cut by a batch end."""
    import geopandas as gpd
    import shapely
    from spatialdata.models import ShapesModel

    geometries, ids, labels = [], [], []
    carry = None

    def build(x, y, cid, key, final):
        if not len(key):
            return None
        starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        cut = len(starts) if final else len(starts) - 1          # the last run may continue in the next batch
        stop = len(key) if final else int(starts[-1])
        if cut:
            lengths = np.diff(np.r_[starts[:cut], stop])
            ok = lengths >= 3
            if ok.any():
                rows = np.repeat(ok, lengths)
                group = np.repeat(np.arange(cut), lengths)[rows]
                rings = shapely.linearrings(np.column_stack([x[:stop][rows], y[:stop][rows]]),
                                            indices=np.searchsorted(np.flatnonzero(ok), group))
                geometries.append(shapely.polygons(rings))
                ids.append(inputs.cell_keys(cid[starts[:cut][ok]]))
                if by_label:
                    labels.append(np.asarray(key[starts[:cut][ok]], dtype=np.int64))
        return None if final else (x[stop:], y[stop:], cid[stop:], key[stop:])

    by_label = False
    for x, y, cid, label in _boundary_batches(path, scratch):
        by_label = label is not None
        key = label if by_label else cid
        if carry is not None:
            x, y = np.concatenate([carry[0], x]), np.concatenate([carry[1], y])
            cid, key = np.concatenate([carry[2], cid]), np.concatenate([carry[3], key])
        carry = build(x, y, cid, key, final=False)
    if carry is not None:
        build(*carry, final=True)
    geometry = np.concatenate(geometries) if geometries else np.array([], dtype=object)
    cell = np.concatenate(ids) if ids else np.array([], dtype=object)
    if is_nucleus and labels:
        # like spatialdata-io: every nucleus kept, indexed by its label, owning cell in a column
        gdf = gpd.GeoDataFrame({"geometry": geometry, "cell_id": cell}, index=pd.Index(np.concatenate(labels)))
    else:
        gdf = gpd.GeoDataFrame({"geometry": geometry}, index=pd.Index(cell, name="cell_id"))
        gdf = gdf[~gdf.index.duplicated()]
    return ShapesModel.parse(gdf, transformations={"global": scale(pixel_size)})


def polygons_from_cells_zarr(path: Path, pixel_size: float, cell_ids: np.ndarray, is_nucleus: bool):
    from vendor._atera_common import _get_polygons

    store = zarr.storage.ZipStore(path, mode="r", read_only=True)
    try:
        group = zarr.open_group(store, mode="r")
        return _get_polygons(group, mask_index=0 if is_nucleus else 1, cell_ids=cell_ids, pixel_size=pixel_size,
                             is_nucleus=is_nucleus)
    finally:
        store.close()


def _decode(series: pd.Series) -> pd.Series:
    if series.dtype == object and len(series) and isinstance(series.iloc[0], bytes):
        return series.str.decode("utf-8").astype("str")
    return series


def _feature_names(path: Path, parquet: bool) -> pd.Index:
    """Distinct feature names in order of first appearance, one batch at a time."""
    seen: dict[str, None] = {}
    if parquet:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        with pq.ParquetFile(path, read_dictionary=["feature_name"]) as pf:
            for batch in pf.iter_batches(batch_size=1 << 21, columns=["feature_name"]):
                column = batch.column(0)
                values = column.dictionary if hasattr(column, "dictionary") else pc.unique(column)
                for v in values.to_pylist():
                    seen.setdefault(v.decode() if isinstance(v, bytes) else str(v), None)
    else:
        with inputs.open_text(path) as f:
            for chunk in pd.read_csv(f, usecols=["feature_name"], chunksize=1 << 21):
                for v in chunk["feature_name"].unique():
                    seen.setdefault(v.decode() if isinstance(v, bytes) else str(v), None)
    return pd.Index(list(seen), dtype=object)


def _row_groups_per_partition(path: Path) -> int:
    import pyarrow.parquet as pq

    meta = pq.read_metadata(path)
    rows = [meta.row_group(i).num_rows for i in range(meta.num_row_groups)]
    typical = int(np.median(rows)) if rows else 1
    return max(1, POINTS_PARTITION_ROWS // max(typical, 1))


def transcripts(path: Path, pixel_size: float, scratch: Path):
    """Transcripts as lazy points in partitions of about POINTS_PARTITION_ROWS rows."""
    import dask.dataframe as dd
    import pyarrow as pa
    from spatialdata.models import PointsModel

    parquet = inputs.is_parquet(path)
    if parquet:
        import pyarrow.parquet as pq

        path = inputs.plain(path, scratch)
        schema = pq.read_schema(path)
        if "feature_name" not in schema.names:
            raise inputs.TableError(f"{path.name} has no feature_name column")
        binary = [n for n in schema.names if pa.types.is_binary(schema.field(n).type)
                  or pa.types.is_large_binary(schema.field(n).type)]
        ddf = dd.read_parquet(str(path), split_row_groups=_row_groups_per_partition(path))
    else:
        ddf = dd.read_csv(str(path), blocksize=None if inputs.is_gzip(path) else "128MB",
                          compression="gzip" if inputs.is_gzip(path) else None)
        if "feature_name" not in ddf.columns:
            raise inputs.TableError(f"{path.name} has no feature_name column")
        binary = []
    for col in binary:
        if col != "feature_name":
            ddf[col] = ddf[col].map_partitions(_decode, meta=pd.Series([], dtype="str", name=col))
    dtype = pd.CategoricalDtype(categories=_feature_names(path, parquet))
    ddf["feature_name"] = ddf["feature_name"].map_partitions(lambda s: _decode(s).astype(str).astype(dtype),
                                                             meta=pd.Series([], dtype=dtype, name="feature_name"))
    coords = {"x": "x_location", "y": "y_location"}
    if "z_location" in ddf.columns:
        coords["z"] = "z_location"
    return PointsModel.parse(ddf, coordinates=coords, feature_key="feature_name",
                             instance_key="cell_id" if "cell_id" in ddf.columns else None,
                             transformations={"global": scale(pixel_size)}, sort=True)


def _attr_list(value) -> list:
    return list(ast.literal_eval(value)) if isinstance(value, str) else list(value)


def _explorer_tile(path: Path, key: str, names: list[str], start: int, stop: int) -> pd.DataFrame:
    store = zarr.storage.ZipStore(path, mode="r", read_only=True)
    try:
        tile = zarr.open_group(store, mode="r")[f"grids/0/{key}"]
        location = np.asarray(tile["location"][start:stop])
        gene = np.asarray(tile["gene_identity"][start:stop]).reshape(-1).astype(np.int64)
        quality = np.asarray(tile["quality_score"][start:stop]).reshape(-1)
    finally:
        store.close()
    return pd.DataFrame({"x_location": location[:, 0].astype(np.float32), "y_location": location[:, 1].astype(np.float32),
                         "z_location": location[:, 2].astype(np.float32),
                         "feature_name": pd.Categorical.from_codes(gene, categories=pd.Index(names)),
                         "qv": quality.astype(np.float32)})


def explorer_transcripts(path: Path, pixel_size: float):
    """A Xenium Explorer transcripts.zarr.zip as lazy points (no cell assignment)."""
    import dask.dataframe as dd
    from dask import delayed
    from spatialdata.models import PointsModel

    store = zarr.storage.ZipStore(path, mode="r", read_only=True)
    try:
        root = zarr.open_group(store, mode="r")
        if root.attrs.get("name") != "RnaDataset":
            raise inputs.TableError(f"{path.name} is not a Xenium Explorer RnaDataset")
        names = [str(n) for n in _attr_list(root.attrs["gene_names"])]
        grid = root["grids/0"]
        sizes = {key: grid[key]["location"].shape[0] for key in grid.group_keys()}
    finally:
        store.close()
    dtype = pd.CategoricalDtype(categories=pd.Index(names))
    meta = pd.DataFrame({"x_location": pd.array([], dtype="float32"), "y_location": pd.array([], dtype="float32"),
                         "z_location": pd.array([], dtype="float32"), "feature_name": pd.Categorical([], dtype=dtype),
                         "qv": pd.array([], dtype="float32")})
    parts = [delayed(_explorer_tile)(path, key, names, a, min(a + POINTS_PARTITION_ROWS, n))
             for key, n in sorted(sizes.items()) for a in range(0, n, POINTS_PARTITION_ROWS) if n]
    ddf = dd.from_delayed(parts, meta=meta) if parts else dd.from_pandas(meta, npartitions=1)
    return PointsModel.parse(ddf, coordinates={"x": "x_location", "y": "y_location", "z": "z_location"},
                             feature_key="feature_name", transformations={"global": scale(pixel_size)}, sort=False)


# File handles that lazy elements keep open, by sample; closed once the elements are written (on NFS an open,
# deleted file leaves a `.nfs…` file that blocks the scratch cleanup).
OPEN_HANDLES: dict[str, list] = {}


def adopt_vendor_handles(sid: str) -> None:
    from vendor import _atera_common

    OPEN_HANDLES.setdefault(sid, []).extend(_atera_common.OPENED)
    _atera_common.OPENED.clear()


def release(sid: str) -> int:
    """Closes the file handles the sample's lazy elements hold; returns how many."""
    adopt_vendor_handles(sid)
    handles = OPEN_HANDLES.pop(sid, [])
    for handle in handles:
        try:
            handle.close()
        except Exception:                               # noqa: BLE001 — closing is best effort
            logger.debug("closing %r failed", handle, exc_info=True)
    return len(handles)


def tiled_imread(path, **_):
    """An OME-TIFF as a dask array chunked by its own tiles (dask_image gives a whole plane per chunk)."""
    from vendor._atera_common import _tiled_imread

    image = _tiled_imread(Path(path))
    return image[None] if image.ndim == 2 else image


@contextlib.contextmanager
def tiled_images():
    import spatialdata_io.readers.xenium as xenium_reader

    original = xenium_reader.imread
    xenium_reader.imread = tiled_imread
    try:
        yield
    finally:
        xenium_reader.imread = original


def morphology_3d(path: Path, scratch: Path):
    """A z-stack as its max projection over z."""
    import tifffile
    from spatialdata.models import Image2DModel

    path = inputs.plain(path, scratch)
    with tifffile.TiffFile(path) as tf:
        axes = tf.series[0].axes
        names = re.findall(r'<Channel [^>]*Name="([^"]+)"', tf.ome_metadata or "")
    image = tiled_imread(path)
    if axes.endswith("YX") and "Z" in axes and "C" not in axes:
        image = image[None] if image.ndim == 3 else image
    elif axes not in ("CZYX",):
        raise inputs.TableError(f"{path.name} has axes {axes}, not a z-stack")
    image = image.max(axis=1)
    names = names if len(names) == image.shape[0] else [f"ch{i}" for i in range(image.shape[0])]
    return Image2DModel.parse(image, dims=("c", "y", "x"), c_coords=names, chunks=(1, 4096, 4096),
                              scale_factors=[2, 2, 2])


def _try(built: Built, what: str, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:                           # noqa: BLE001 — an optional part never stops a sample
        built.problems.append(f"{what} ({type(exc).__name__}: {str(exc)[:160]})")
        logger.debug("%s failed", what, exc_info=True)
        return None


def xenium_elements(folder: Path, files: dict[str, Path], obs: pd.DataFrame, spatial: np.ndarray, scratch: Path,
                    sid: str = "") -> Built:
    from spatialdata_io.readers._utils._utils import _initialize_raster_models_kwargs
    from spatialdata_io.readers.xenium import _get_images, _get_labels, _get_morphology_focus, \
        _XeniumCells
    import packaging.version

    built = Built()
    specs, version = xenium_specs(folder)
    pixel = float(specs["pixel_size"])
    image_kwargs, label_kwargs = _initialize_raster_models_kwargs({}, {})
    cells_zarr = None
    if "cells_zarr" in files:
        cells_zarr = _try(built, "cells.zarr.zip", _XeniumCells.open, folder, version)
        if cells_zarr is not None:
            OPEN_HANDLES.setdefault(sid, []).append(cells_zarr.group.store)
    if cells_zarr is not None:
        for name, index in (("nucleus_labels", 0), ("cell_labels", 1)):
            label = _try(built, name, _get_labels, cells_zarr.group, index, label_kwargs)
            if label is not None:
                built.elements["labels"][name] = label
        mapping = cells_zarr.cell_indices_mapping
        if mapping is not None and "cell_labels" in built.elements["labels"]:
            where = pd.Index(mapping["cell_id"].astype(str)).get_indexer(obs["cell_id"].astype(str))
            if (where >= 0).all():
                built.obs_columns["cell_labels"] = mapping["label_index"].to_numpy()[where]
        summary = _try(built, "cell_summary", inputs.xenium_cells_zarr, folder / "cells.zarr.zip") \
            if cells_zarr.cell_id_str is not None else None
        if summary is not None:
            where = pd.Index(summary["cell_id"]).get_indexer(obs["cell_id"].astype(str))
            if (where >= 0).all():
                for col in ("z_level", "nucleus_count"):
                    if col in summary and col not in obs:
                        built.obs_columns[col] = summary[col].to_numpy()[where]

    for role, is_nucleus in (("nucleus_boundaries", True), ("cell_boundaries", False)):
        path = files.get(role)
        if path is None:
            continue
        progress(sid, f"{role.replace('_', ' ')} from {path.name}", path)
        shapes = None
        if path.name.endswith(".zarr.zip"):
            ids = cells_zarr.cell_id_str if cells_zarr is not None else None
            shapes = _try(built, role, polygons_from_cells_zarr, path, pixel, ids, is_nucleus) if ids is not None else None
        else:   # same polygons as spatialdata-io's reader, a fraction of the memory
            shapes = _try(built, role, polygons_from_table, path, pixel, scratch, is_nucleus)
        if shapes is not None:
            shapes = _try(built, role, _ids_like_table, shapes, obs["cell_id"], is_nucleus)
        if shapes is not None:
            built.elements["shapes"][role] = shapes

    if "transcripts" in files:
        path = files["transcripts"]
        progress(sid, f"transcripts from {path.name}", path)
        points = _try(built, "transcripts", explorer_transcripts, path, pixel) if path.name.endswith(".zarr.zip") \
            else _try(built, "transcripts", transcripts, path, pixel, scratch)
        if points is not None:
            built.elements["points"]["transcripts"] = points

    images = [p for p in (folder / "morphology_focus", folder / "morphology_focus.ome.tif",
                          folder / "morphology_mip.ome.tif") if p.exists()]
    if images:
        progress(sid, "images " + ", ".join(p.name for p in images), *images)
    with tiled_images():
        focus = folder / "morphology_focus"
        if focus.is_dir():
            v = version if version is not None and version >= packaging.version.parse("2.0.0") \
                else packaging.version.parse("2.0.0")
            image = _try(built, "morphology_focus", _get_morphology_focus, folder, v, {}, image_kwargs)
            if image is not None:
                built.elements["images"]["morphology_focus"] = image
        elif (folder / "morphology_focus.ome.tif").exists():
            image = _try(built, "morphology_focus", _get_images, folder, "morphology_focus.ome.tif", {}, image_kwargs)
            if image is not None:
                built.elements["images"]["morphology_focus"] = image
        if (folder / "morphology_mip.ome.tif").exists() and (version is None or version < packaging.version.parse("2.0.0")):
            image = _try(built, "morphology_mip", _get_images, folder, "morphology_mip.ome.tif", {}, image_kwargs)
            if image is not None:
                built.elements["images"]["morphology_mip"] = image
    if "morphology_3d" in files:
        progress(sid, f"z-stack image {files['morphology_3d'].name} (max projection)", files["morphology_3d"])
        image = _try(built, "morphology.ome.tif (z-stack)", morphology_3d, files["morphology_3d"], scratch)
        if image is not None:
            built.elements["images"]["morphology_mip"] = image

    _set_region(built, obs["cell_id"], spatial, obs.get("cell_area"), pixel)
    adopt_vendor_handles(sid)
    return built


def _ids_like_table(shapes, table_ids: pd.Series, is_nucleus: bool):
    integer = pd.api.types.is_integer_dtype(table_ids)

    def cast(values):
        return pd.Index(values).astype(np.int64 if integer else str)

    if shapes.index.name == "cell_id":
        shapes.index = pd.Index(cast(shapes.index), name="cell_id")
    if is_nucleus and "cell_id" in shapes:
        shapes["cell_id"] = np.asarray(cast(shapes["cell_id"]))
    return shapes


def _set_region(built: Built, cell_ids: pd.Series, spatial: np.ndarray, area, pixel: float) -> None:
    """The table annotates cell_boundaries when every placed cell has one, else circles at the centroids."""
    placed = np.isfinite(np.asarray(spatial, dtype=float)).all(axis=1)
    ids = pd.Index(np.asarray(cell_ids))[placed]
    shapes = built.elements["shapes"].get(REGION)
    if shapes is not None:
        uncovered = int((~ids.isin(shapes.index)).sum())
        if not uncovered:
            built.region = REGION
            return
        built.problems.append(f"{uncovered} cells have no cell boundary, so the table annotates cell circles")
    area = None if area is None else np.asarray(area, dtype=float)[placed]
    built.elements["shapes"][CIRCLES] = circles(np.asarray(spatial)[placed], area, np.asarray(ids), pixel)
    built.region = CIRCLES


def atera_elements(folder: Path, files: dict[str, Path], obs: pd.DataFrame, spatial: np.ndarray, sid: str = "") -> Built:
    built = Built()
    has_cells = "cells_zarr" in files
    has_2d = (folder / "morphology_2d").is_dir()
    pixel = None
    if (folder / "experiment.spatial").exists() or has_2d:
        from vendor._atera_common import _get_pixel_size

        pixel = _try(built, "pixel size", _get_pixel_size, folder)
    if pixel is not None:
        from vendor.atera import atera as read_atera

        parts = [folder / n for n in ("cells.zarr.zip", "transcripts.zarr.zip", "morphology_2d")
                 if (folder / n).exists()]
        progress(sid, "Atera elements " + ", ".join(p.name for p in parts), *parts)

        # the reader's own table is dropped (ours is streamed), so its warning about `cell_labels` is moot
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            sdata = _try(built, "Atera elements", read_atera, folder, cells_table=has_cells or "transcripts" in files,
                         cells_boundaries=has_cells, nucleus_boundaries=has_cells, cells_labels=has_cells,
                         nucleus_labels=has_cells, transcripts="transcripts" in files, morphology_images=has_2d)
        if sdata is not None:
            for kind in stores.ELEMENT_KINDS:
                built.elements[kind].update(dict(getattr(sdata, kind)))
    else:
        pixel = inputs.DEFAULT_PIXEL_SIZE
        built.problems.append(f"pixel size unknown (no experiment.spatial, no images); {pixel} µm assumed")
    _set_region(built, obs["cell_id"], spatial, obs.get("cell_area"), pixel)
    adopt_vendor_handles(sid)
    return built


REGION_KEY = "region"
INSTANCE_KEY = "cell_id"


class ReadError(UserError, ValueError):
    """A sample that can't be read as it is."""


def _obs_with(obs: pd.DataFrame, extra: dict[str, np.ndarray], region: str) -> pd.DataFrame:
    obs = obs.copy()
    for col, values in extra.items():
        obs[col] = values
    obs[REGION_KEY] = pd.Categorical([region] * len(obs))
    if INSTANCE_KEY in obs and obs[INSTANCE_KEY].dtype == object:
        obs[INSTANCE_KEY] = obs[INSTANCE_KEY].astype(str)
    return obs


def invalid_polygons(elements: dict[str, dict[str, object]]) -> list[str]:
    out = []
    for name, shapes in elements.get("shapes", {}).items():
        geometry = getattr(shapes, "geometry", None)
        if geometry is None or not len(geometry) or (geometry.geom_type == "Point").all():
            continue
        bad = int((~geometry.is_valid).sum())
        if bad:
            out.append(f"{name}: {bad:,} of {len(geometry):,} polygons are invalid geometries (kept as read)")
    return out


def warn_once(sample: Sample, inv: Inventory, problems: list[str], table_notes: list[str]) -> None:
    line = inputs.describe(Inventory(inv.sid, inv.kind, inv.chosen, [], list(inv.missing_optional), None,
                              inv.n_cells, list(inv.notes) + table_notes))
    if line:
        logger.info("%s: %s", sample.sid, line)
    if problems:
        logger.warning("%s: %s read without: %s", sample.sid, NAMES.get(sample.kind, sample.kind), "; ".join(problems))


def load_bundle(sample: Sample, scratch: Path) -> Loaded:
    inv = sample.inventory or inputs.inspect(sample.candidate, scratch)
    if inv.missing_essential:
        raise ReadError(f"{sample.sid}: can't be read; missing {'; '.join(inv.missing_essential)}.")
    folder, files = inputs.stage_bundle(sample, inv, scratch)
    work = scratch / "work" / sample.sid
    out = scratch / "read" / sample.sid / "counts.zarr"
    notes: list[str] = []
    try:
        m = inputs.open_matrix({r: files[r] for r in ("matrix", "matrix_features", "matrix_barcodes") if r in files},
                                sample.kind)
        if m.kind == "atera_zarr":
            source = inputs.atera_table(m, sample.sid, out, work)
        else:
            cells_path = files["cells"]
            cells = inputs.xenium_cells_zarr(cells_path) if cells_path.name.endswith(".zarr.zip") \
                else inputs.read_cells(cells_path, work)
            source = inputs.xenium_table(m, cells, sample.sid, out, work, notes)
    except inputs.TableError as exc:
        raise ReadError(str(exc)) from exc
    spatial = np.asarray(zarr.open_group(str(out), mode="r")["obsm"]["spatial"][:])
    if m.kind == "atera_zarr":
        built = atera_elements(folder, files, source.obs, spatial, sample.sid)
    else:
        built = xenium_elements(folder, files, source.obs, spatial, work, sample.sid)
    source.obs = _obs_with(source.obs, built.obs_columns, built.region)
    group = zarr.open_group(str(out), mode="r+")
    write_elem(group, "obs", source.obs)
    warn_once(sample, inv, built.problems, notes + invalid_polygons(built.elements))
    unplaced = int((~source.obs[inputs.HAS_POSITION]).sum()) if inputs.HAS_POSITION in source.obs else 0
    if unplaced:
        logger.warning("%s: %s of %s cells (%.1f%%) have no position; kept, marked obs['%s']=False (no shape, "
                       "NaN in obsm['spatial']; filter on it before spatial analysis)", sample.sid, f"{unplaced:,}",
                       f"{source.n_obs:,}", 100 * unplaced / source.n_obs, inputs.HAS_POSITION)
    return Loaded(built.elements, source, [built.region], REGION_KEY, INSTANCE_KEY)


def load(sample: Sample, scratch: Path) -> Loaded:
    if sample.kind == "spatialdata":
        return load_part(sample.path, sample.part, scratch)
    return load_bundle(sample, scratch)


def close(sid: str) -> None:
    """Call after the sample's elements are written, or when it failed, before its staged files are deleted."""
    n = release(sid)
    gc.collect()
    logger.debug("%s: %d file handle(s) closed", sid, n)


LABEL_ROWS = 10_000            # cells per CellTypist call (it densifies the model's genes)
FINGERPRINT_CHUNK = 1 << 24


class SampleError(UserError, ValueError):
    """A sample that can't be added as it is."""


def renamed(elements: dict[str, dict[str, object]], sid: str, tags: tuple[str, ...]) -> dict[str, dict[str, object]]:
    from spatialdata.transformations import get_transformation, set_transformation

    out: dict[str, dict[str, object]] = {}
    sources: dict[str, str] = {}
    for kind, items in elements.items():
        for name, element in items.items():
            new = stores.element_name(name, sid, tags)
            if new in sources:
                raise SampleError(f"{sid}: elements {sources[new]} and {name} would both be named {new}.")
            sources[new] = name
            systems = get_transformation(element, get_all=True)
            set_transformation(element, {stores.system_name(cs, sid, tags): t for cs, t in systems.items()}, set_all=True)
            out.setdefault(kind, {})[new] = element
    return out


class Fingerprint:
    """sha256 of a table's shape, CSR arrays and cell IDs, fed band by band."""

    def __init__(self, n_obs: int, n_vars: int, indptr: np.ndarray):
        self.digest = hashlib.sha256(f"{n_obs}:{n_vars}".encode())
        self.digest.update(np.ascontiguousarray(indptr, dtype=np.int64).tobytes())

    def indices(self, band: sp.csr_matrix) -> None:
        self.digest.update(np.ascontiguousarray(band.indices, dtype=np.int64).tobytes())

    def finish(self, data, ids: pd.Series) -> str:
        for a in range(0, data.shape[0], FINGERPRINT_CHUNK):
            self.digest.update(np.ascontiguousarray(data[a:a + FINGERPRINT_CHUNK], dtype=np.float64).tobytes())
        self.digest.update("\n".join(ids.astype(str)).encode())
        return self.digest.hexdigest()


def cell_ids(obs: pd.DataFrame) -> pd.Series:
    return obs["cell_id"].astype(str) if "cell_id" in obs else pd.Series(obs.index.astype(str), index=obs.index)


def _source_obs(obs: pd.DataFrame, sid: str) -> pd.DataFrame:
    obs = obs.copy()
    if obs.index.name in obs.columns:          # a column named like the index: anndata refuses to write it
        obs.index.name = None
    obs.index = pd.Index(obs.index.astype(str))
    obs["sample_id"] = pd.Categorical([sid] * len(obs))
    return obs


def _annotations(band: sp.csr_matrix, obs: pd.DataFrame, var: pd.DataFrame, model: Path | None) -> pd.DataFrame:
    chunk = ad.AnnData(band, obs=pd.DataFrame(index=obs.index), var=pd.DataFrame(index=var.index))
    parts = [qc_metrics(chunk)]
    if model is not None:
        labels = [celltypist_labels(chunk[a:a + LABEL_ROWS].copy(), model)
                  for a in range(0, chunk.n_obs, LABEL_ROWS)]
        parts.append(pd.concat(labels))
    return pd.concat(parts, axis=1)


def _categorical(frame: pd.DataFrame) -> pd.DataFrame:
    for col in frame.columns:
        if frame[col].dtype == object or isinstance(frame[col].dtype, pd.CategoricalDtype) \
                or pd.api.types.is_string_dtype(frame[col]):
            frame[col] = pd.Categorical(frame[col].astype(str))
    return frame


def _regions(loaded: Loaded, sid: str, obs: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    regions = {r: stores.element_name(r, sid, loaded.tags) for r in loaded.regions}
    if loaded.region_key and regions:
        if loaded.region_key in obs:
            obs[loaded.region_key] = pd.Categorical(obs[loaded.region_key].astype(str).map(regions))
        else:
            obs[loaded.region_key] = pd.Categorical([next(iter(regions.values()))] * len(obs))
        if obs[loaded.region_key].isna().any():
            raise SampleError(f"{sid}: some cells annotate a region that isn't one of {list(regions)}.")
    return obs, list(regions.values())


def _check_counts(band: sp.csr_matrix, sid: str) -> None:
    if band.nnz and (band.data.min() < 0 or np.any(np.mod(band.data, 1))):
        raise SampleError(f"{sid}: the table holds non-count values (negative or fractional); the store takes "
                          "raw counts only.")


def prepare(loaded: Loaded, sample: Sample, model: Path | None, scratch: Path) -> Prepared:
    """The sample's elements renamed and its table written to scratch, ready to install."""
    sid = sample.sid
    table = loaded.table
    src = zarr.open_group(str(table.path), mode="r")
    var = read_elem(src["var"])
    obsm = dict(read_elem(src["obsm"])) if "obsm" in src else {}
    obs = _source_obs(table.obs, sid)
    n_obs, n_vars = inputs.shape(table.path)
    if len(obs) != n_obs:
        raise SampleError(f"{sid}: obs has {len(obs)} rows but the count table {n_obs}.")
    renames = stores.safe_names(list(obs.columns))       # cell_ID beside cell_id, invalid characters
    if renames:
        logger.info("%s: obs columns renamed so spatialdata can read the table: %s", sid,
                    ", ".join(f"{a} -> {b}" for a, b in renames.items()))
        obs = obs.rename(columns=renames)
        loaded.region_key = renames.get(loaded.region_key, loaded.region_key)
        loaded.instance_key = renames.get(loaded.instance_key, loaded.instance_key)
    obs, regions = _regions(loaded, sid, obs)
    if not inputs.is_mito(var.index).any():
        logger.info("%s: the panel has no mitochondrial genes; pct_counts_mt is left empty", sid)

    out = scratch / "prepared" / sid / stores.table_name(sid)
    group = inputs.new_table_group(out)
    writer = inputs.CSRWriter(group, n_vars, inputs.x_node(table.path)["data"].dtype)
    fp = Fingerprint(n_obs, n_vars, inputs.read_indptr(table.path))
    annotations = []
    for start, band in inputs.iter_rows(table.path):
        _check_counts(band, sid)
        writer.append(band)
        fp.indices(band)
        annotations.append(_annotations(band, obs.iloc[start:start + band.shape[0]], var, model))
    writer.close()
    fingerprint = fp.finish(group["X"]["data"], cell_ids(obs))

    extra = _categorical(pd.concat(annotations)) if annotations else pd.DataFrame(index=obs.index)
    extra.index = obs.index
    obs = obs.drop(columns=[c for c in extra.columns if c in obs]).join(extra)
    write_elem(group, "obs", obs)
    write_elem(group, "var", var)
    write_elem(group, "obsm", {k: np.asarray(v) for k, v in obsm.items()})
    attrs = {"region": (regions[0] if len(regions) == 1 else regions) if regions else None,
             "region_key": loaded.region_key if regions else None,
             "instance_key": loaded.instance_key if regions else None}
    uns = {RECORD: {"celltypist_model": model.stem if model else "", "source": str(sample.candidate.origin),
                    "fingerprint": fingerprint, "genes_in_panel": int(n_vars)}}
    if regions:
        uns["spatialdata_attrs"] = dict(attrs)
    write_elem(group, "uns", uns)
    group.attrs.update({**stores.TABLE_ATTRS, **attrs})

    label = f"{model.stem}_label" if model else None
    stats = {"n_cells": int(n_obs), "genes_in_panel": int(n_vars),
             "mean_genes": float(obs["n_genes_by_counts"].mean()) if n_obs else 0.0,
             "mean_total_counts": float(obs["total_counts"].mean()) if n_obs else 0.0,
             "n_celltypist_labels": int(obs[label].nunique()) if label and label in obs else 0,
             "n_cells_zero_counts": int((obs["total_counts"] == 0).sum()) if n_obs else 0,
             "n_cells_without_position": int((~obs["has_position"].astype(bool)).sum()) if "has_position" in obs else 0}
    elements = renamed(loaded.elements, sid, loaded.tags)
    return Prepared(sid, elements, out, obs, var, fingerprint, loaded.region_key, stats)


QC_COLUMNS = ("n_genes_by_counts", "total_counts", "total_counts_mt", "pct_counts_mt")
MIN_SHARED = 10            # a model must share at least this many genes with a sample's panel ...
MIN_FRACTION = 0.05        # ... and at least this fraction of the model's genes
LOW_FRACTION = 0.5         # below this fraction of the model's genes, labels are less reliable: warn
HOW_TO_MAKE = ("A CellTypist model is the .pkl written by celltypist.train(...).write('model.pkl'), trained on "
               "log1p-normalized counts with the same gene symbols as your panel.")


class AnnotateError(UserError, ValueError):
    """A CellTypist model that can't be used for these samples."""


def _celltypist():
    """Imports celltypist without leaving its global warning filters behind."""
    with warnings.catch_warnings():
        import celltypist
        from celltypist.models import Model
    return celltypist, Model


def qc_metrics(adata: ad.AnnData) -> pd.DataFrame:
    """scanpy's per-cell QC metrics without touching `adata`'s var."""
    probe = ad.AnnData(adata.X, obs=pd.DataFrame(index=adata.obs_names), var=pd.DataFrame(index=adata.var_names))
    probe.var["mt"] = np.asarray(inputs.is_mito(adata.var_names), dtype=bool)
    sc.pp.calculate_qc_metrics(probe, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    if not probe.var["mt"].any():
        probe.obs["pct_counts_mt"] = np.nan
    return probe.obs[list(QC_COLUMNS)]


def label_columns(model: Path) -> tuple[str, str]:
    return f"{model.stem}_label", f"{model.stem}_prob"


@lru_cache(maxsize=4)
def _load(path: str, mtime: float):
    _, Model = _celltypist()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Model.load(model=path)


def _detail(exc: Exception) -> str:
    text = str(exc).replace("🛑", "").strip()
    return text.splitlines()[0][:160] if text else type(exc).__name__


def load_model(model: Path):
    model = Path(model)
    if not model.is_file():
        raise AnnotateError(f"CellTypist model not found: {model}. Give the path of a .pkl file, e.g. "
                            "/data/models/my_panel.pkl, or leave the field empty for no labels.")
    try:
        loaded = _load(str(model), model.stat().st_mtime)
        features = loaded.features
    except Exception as exc:
        logger.debug("Loading %s as a CellTypist model: %s", model, _detail(exc))
        raise AnnotateError(f"{model} isn't a CellTypist model (it could not be loaded as one). {HOW_TO_MAKE}") from None
    if len(features) == 0:
        raise AnnotateError(f"{model} is a CellTypist model with no genes. {HOW_TO_MAKE}")
    return loaded


def model_features(model: Path) -> list[str]:
    return [str(g) for g in load_model(model).features]


def model_overlap(model: Path, var_names) -> tuple[int, int]:
    features = model_features(model)
    return len(set(features) & {str(g) for g in var_names}), len(features)


def overlap_floor(n_model: int) -> int:
    return max(MIN_SHARED, math.ceil(MIN_FRACTION * n_model))


def check_overlap(model: Path, sid: str, var_names, store_model: bool = False) -> tuple[int, int, str | None]:
    """(shared, model genes, note) for one sample; AnnotateError when too few genes are shared."""
    shared, n_model = model_overlap(model, var_names)
    floor = overlap_floor(n_model)
    if shared < floor:
        hint = ("another species, or Ensembl IDs instead of gene symbols?" if shared == 0
                else "a model trained on another panel or species?")
        what = (f"This store labels every sample with it, so add {sid} to a separate store (without a model, or with "
                "a model trained on its panel)." if store_model else
                "Give a model trained on this panel, or leave the model field empty.")
        raise AnnotateError(f"CellTypist model {Path(model).name} shares {shared} of its {n_model} genes with {sid}'s "
                            f"panel ({len(var_names)} genes); at least {floor} are needed ({hint}). {what}")
    if shared < LOW_FRACTION * n_model:
        return shared, n_model, (f"{sid}: {shared} of the model's {n_model} genes "
                                 f"({100 * shared / n_model:.0f}%) are in its panel")
    return shared, n_model, None


def celltypist_labels(adata: ad.AnnData, model: Path) -> pd.DataFrame:
    celltypist, _ = _celltypist()
    loaded = load_model(model)
    normalized = ad.AnnData(adata.X.astype(np.float32), obs=pd.DataFrame(index=adata.obs_names),
                            var=pd.DataFrame(index=adata.var_names))
    with warnings.catch_warnings():   # cells with 0 counts are expected (they stay 0 and get a low-probability label)
        warnings.filterwarnings("ignore", message="Some cells have zero counts", category=UserWarning)
        sc.pp.normalize_total(normalized, target_sum=1e4)
    sc.pp.log1p(normalized)
    try:
        predictions = celltypist.annotate(normalized, model=loaded)
    except Exception as exc:
        raise AnnotateError(f"CellTypist could not label the cells with {Path(model).name}: {_detail(exc)}. "
                            f"{HOW_TO_MAKE}") from None
    labels = predictions.predicted_labels["predicted_labels"].astype(str).to_numpy()
    probs = predictions.probability_matrix
    picked = probs.columns.get_indexer(labels)
    label, prob = label_columns(Path(model))
    logger.debug("CellTypist %s: %d labels in %d cells", Path(model).stem, len(np.unique(labels)), len(labels))
    return pd.DataFrame({label: pd.Categorical(labels),
                         prob: np.where(picked >= 0, probs.to_numpy()[np.arange(len(labels)), picked], np.nan)},
                        index=adata.obs_names)
