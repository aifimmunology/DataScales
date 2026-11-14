"""Shared types passed between the run's stages. Data flow of one run:

    discover.discover(entries, scratch)           -> Discovery          (paths, archives, junk -> candidates)
    inventory.inspect(candidate, scratch)          -> Inventory          (bundle files: chosen / missing; raw var)
    existing.plan(stage.spatialdata(c, ...), ...)  -> list[Part]         (SpatialData object -> its samples)
    precheck.run(...)                              -> Plan               (every sample checked before any write)
    readers.load(sample, scratch, n_jobs)          -> Loaded             (lazy elements + TableSource on scratch)
    sample.prepare(loaded, sample, model, scratch) -> Prepared           (renamed elements + staged per-sample table)
    stores.install_sample(local, prepared)                               (elements + table in, one commit point)
    joint.add(local, prepared, staging, cpus)                            (rows appended; genes/obs union)

No stage holds a sample's whole count matrix in memory: tables travel as CSR AnnData zarr groups on
scratch (`TableSource.path`, `Prepared.table_path`) and are read in row chunks (`matrix.iter_rows`)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

KINDS = ("xenium", "atera", "spatialdata")
NAMES = {"xenium": "Xenium bundle", "atera": "Atera bundle", "spatialdata": "SpatialData object"}

# Canonical roles of bundle files (inventory.Inventory.chosen keys).
ESSENTIAL = ("matrix", "cells")           # count matrix; cells table with cell_id + x/y centroids
OPTIONAL = ("experiment", "cells_zarr", "cell_boundaries", "nucleus_boundaries", "transcripts",
            "morphology", "panel")


@dataclass(frozen=True)
class Member:
    """One file as found: `container` is a plain file or folder on disk, or an archive (.zip/.tar/.tar.gz)
    with `inner` the member's path inside it. `gz` marks a per-file gzip (GEO) to decompress when staged."""
    container: Path
    inner: str | None = None
    gz: bool = False
    size: int = 0

    @property
    def display(self) -> str:
        return f"{self.container}!{self.inner}" if self.inner else str(self.container)


@dataclass(frozen=True)
class Candidate:
    """One bundle or SpatialData object found in the inputs. For bundles, `files` maps canonical 10x file
    names (`experiment.xenium`, `cell_feature_matrix.h5`, `cells.parquet`, `morphology_focus/ch0000.ome.tif`,
    ...; folders as `<dir>/<file>` entries) to where they really are — prefixes stripped, `.gz` noted.
    For SpatialData, `zarr` is the store root (a folder, or an archive member prefix)."""
    sid: str
    kind: str                                # one of KINDS
    origin: Path                             # what the user's entry resolved to (folder, archive, .zarr)
    files: dict[str, Member] = field(default_factory=dict)
    zarr: Member | None = None
    notes: tuple[str, ...] = ()              # e.g. "file names had the prefix 'GSM123_'" (INFO level)


@dataclass(frozen=True)
class LeftOut:
    """An item in the inputs that is not a sample (reported once, as one warning listing all of them)."""
    path: str
    reason: str


@dataclass
class Discovery:
    candidates: list[Candidate]
    left_out: list[LeftOut]


@dataclass
class Inventory:
    """What a bundle holds, decided from file names and small reads only (ids, headers, var)."""
    sid: str
    kind: str
    chosen: dict[str, str]                   # role -> canonical file name in Candidate.files
    missing_essential: list[str]             # human-readable, e.g. "count matrix (cell_feature_matrix.h5/.zarr.zip or matrix.mtx)"
    missing_optional: list[str]              # human-readable, e.g. "morphology images", "transcripts"
    var: pd.DataFrame | None                 # raw features (index = names as in the file; gene_ids, feature_types)
    n_cells: int | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Part:
    """One sample of an existing SpatialData: its table and regions, its elements, and the tags naming it
    inside the object (stripped from element and coordinate-system names)."""
    sid: str
    table: str | None
    regions: tuple[str, ...]
    elements: tuple[tuple[str, str], ...]
    tags: tuple[str, ...]


@dataclass
class Sample:
    """One sample the run will add, after the precheck."""
    sid: str
    kind: str
    candidate: Candidate
    inventory: Inventory | None = None       # bundles
    part: Part | None = None                 # SpatialData samples
    path: Path | None = None                 # SpatialData: the readable (staged) store root
    var: pd.DataFrame | None = None          # symbol-indexed Gene Expression var (genes.symbol_var)
    species: str | None = None


@dataclass
class Plan:
    """The precheck's result: samples to add (input order), samples already present, and the panel report."""
    samples: list[Sample]
    present: list[str]
    left_out: list[LeftOut]
    union_genes: list[str]                   # joint gene order after this run (store's first, new appended)
    panels: dict[str, dict]                  # sid -> {"genes_in_panel", "genes_shared", "species"}


@dataclass
class TableSource:
    """A sample's counts as read: CSR AnnData zarr group on scratch (X = raw counts, Gene Expression only,
    var symbol-indexed with gene_ids/feature_types), plus obs in memory (one row per cell, same order)."""
    path: Path
    obs: pd.DataFrame
    n_obs: int
    n_vars: int


@dataclass
class Loaded:
    """A sample as read: lazy elements by kind and name, its table, and how the table annotates them."""
    elements: dict[str, dict[str, object]]
    table: TableSource
    regions: list[str]
    region_key: str | None
    instance_key: str | None
    tags: tuple[str, ...] = ()


@dataclass
class Prepared:
    """A sample ready to install: elements renamed into its coordinate system, and its per-sample table
    staged as a complete zarr group (obs with sample_id/QC/labels, uns record, spatialdata attrs) that
    `stores.install_sample` moves to `tables/<sid>_table`."""
    sid: str
    elements: dict[str, dict[str, object]]
    table_path: Path
    obs: pd.DataFrame
    var: pd.DataFrame
    fingerprint: str
    region_key: str | None
    stats: dict = field(default_factory=dict)  # n_cells, mean_genes, mean_total_counts, n_celltypist_labels
