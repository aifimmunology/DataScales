"""Finds the samples in the inputs, stages the files they need from folders and archives, works out what each
bundle holds, and reads its count matrix and cells table."""
from __future__ import annotations

import difflib
import glob
import gzip
import json
import logging
import math
import os
import re
import shutil
import tarfile
import threading
import zipfile
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Callable
from zlib import error as zlib_error

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
from anndata.io import write_elem

import config
from config import NAMES, Candidate, Discovery, Inventory, LeftOut, Member, Sample, TableSource, UserError

logger = logging.getLogger(__name__)

MAX_DEPTH = 8                     # folder levels searched below each input (archives count their inner levels)
MAX_LISTED = 8                    # loose files named one by one per folder before they're summarised
META_LIMIT = 1 << 20              # largest zarr metadata file read from an archive

MARKERS = {"experiment.xenium": "xenium", "experiment.spatial": "atera"}
ARCHIVES = (".tar.gz", ".tgz", ".tar", ".zip")
UNSUPPORTED = (".rar", ".7z", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".tar.zst", ".bz2", ".xz")
SEPARATORS = "_-. "

# Known 10x bundle file names (Xenium and Atera), matched at the end of a file name after an optional prefix.
CANON = (
    "experiment.xenium", "experiment.spatial",
    "cell_feature_matrix.h5", "cell_feature_matrix.zarr.zip", "csc_cell_feature_matrix.zarr.zip",
    "cell_feature_matrix.tar.gz", "matrix.mtx", "barcodes.tsv", "features.tsv",
    "cells.parquet", "cells.csv", "cells.zarr.zip",
    "cell_boundaries.parquet", "cell_boundaries.csv", "nucleus_boundaries.parquet", "nucleus_boundaries.csv",
    "transcripts.parquet", "transcripts.csv", "transcripts.zarr.zip", "binned_transcripts.zarr.zip",
    "analysis.zarr.zip", "analysis.tar.gz", "analysis_summary.html", "metrics_summary.csv", "gene_panel.json",
    "morphology.ome.tif", "morphology_focus.ome.tif", "morphology_mip.ome.tif",
)
_CANON_ALT = "|".join([re.escape(c) for c in sorted(CANON, key=len, reverse=True)]
                      + [r"morphology_focus_\d{4}\.ome\.tif"])
CANON_RE = re.compile(rf"^(?P<prefix>.*?)(?P<canon>{_CANON_ALT})(?P<gz>\.gz)?$")
MTX = {"matrix.mtx", "barcodes.tsv", "features.tsv"}
KEEP_GZ = (".csv.gz", ".tsv.gz", ".mtx.gz", ".txt.gz", ".tar.gz")   # read as they are; never decompressed

MATRIX_KEYS = {"cell_feature_matrix.h5", "cell_feature_matrix.zarr.zip", "csc_cell_feature_matrix.zarr.zip",
               "cell_feature_matrix/matrix.mtx.gz", "cell_feature_matrix/matrix.mtx", "cell_feature_matrix.tar.gz"}
CELLS_KEYS = {"cells.parquet", "cells.csv.gz", "cells.csv", "cells.zarr.zip"}
METADATA_KEYS = {"experiment.xenium", "experiment.spatial", "gene_panel.json", "metrics_summary.csv",
                 "analysis_summary.html"}
DATA_DIRS = {"cell_feature_matrix", "morphology_focus", "morphology_2d"}
ATERA_ONLY = {"csc_cell_feature_matrix.zarr.zip", "binned_transcripts.zarr.zip", "panel_config.json"}

ELEMENT_GROUPS = ("images", "labels", "shapes", "points", "tables", "table")
ZARR_META = {"zarr.json", ".zgroup", ".zattrs", ".zarray"}

JUNK = {"__MACOSX", ".DS_Store", "Thumbs.db", "desktop.ini", ".end-of-run", "journal", ".snakemake_timestamp"}
RANGER_DIR = re.compile(r"^[A-Z0-9_]+_CS$")          # XENIUM_RANGER_CS, SPATIAL_RANGER_CS, ...
GENERIC = {"outs", "out", "output", "outputs", "data", "extract", "extracted", "count", "counts", "files",
           "raw", "download", "downloads", "input", "inputs", "results", "result", "sample", "samples",
           "spatialdata", "sdata", "combined_spatialdata_zarr", "xenium_transfer", "xenium", "atera", "zarr",
           "unzipped", "tmp", "temp"}
GENERIC_RE = re.compile(r"^(.*[_-])?spatialdata([_.-]zarr)?$", re.IGNORECASE)
SAMPLE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
INSTRUMENT = re.compile(r"^output-XETG\d+__", re.IGNORECASE)     # Xenium instrument run folder names
GSM = re.compile(r"^GSM\d+")
GEO_TAIL = re.compile(r"([_.\- ](xenium|raw|data|outs?|outputs?|bundle|files|results))+$", re.IGNORECASE)
BROWSER_COPY = re.compile(r"^(?P<stem>.+?) \((?P<n>\d+)\)(?P<ext>(\.[A-Za-z0-9]+)+)$")   # "cells (2).parquet"


TAKES = "spatial_ingest takes Xenium or Atera bundles or SpatialData .zarr"
ANNDATA = f"{TAKES} (AnnData tables go to the anndata_ingest capsule)"
FOREIGN = {   # file types users give that are not spatial bundles: what they are, and where they go instead
    ".h5ad": f"an AnnData .h5ad table, not a spatial bundle: {ANNDATA}",
    ".rds": f"an R .rds file (e.g. a Seurat object), not a spatial bundle: {TAKES}; convert it to AnnData "
            "for the anndata_ingest capsule",
    ".rdata": f"an R data file, not a spatial bundle: {TAKES}",
    ".rda": f"an R data file, not a spatial bundle: {TAKES}",
    ".h5seurat": f"a Seurat .h5seurat file, not a spatial bundle: {TAKES}; convert it to AnnData for the "
                 "anndata_ingest capsule",
    ".loom": f"a .loom table, not a spatial bundle: {ANNDATA}",
    ".h5": f"an .h5 file that is not a Xenium cell_feature_matrix.h5 (a Chromium/Visium count matrix?): a count "
           f"matrix alone can't be placed in space; {TAKES}",
    ".mtx": f"a count matrix without the Xenium cells table beside it: {TAKES}",
    ".csv": f"a table (.csv/.tsv), not part of a bundle; a count table alone can't be placed in space: {TAKES}",
    ".tsv": f"a table (.csv/.tsv), not part of a bundle; a count table alone can't be placed in space: {TAKES}",
}


def foreign_reason(name: str) -> str | None:
    """Why a loose file of a known non-bundle type isn't used, else None."""
    if split_name(name) is not None:            # a 10x bundle file (cells.csv.gz, cell_feature_matrix.h5, …)
        return None
    lower = name.lower()
    lower = lower[:-3] if lower.endswith(".gz") else lower
    return next((why for ext, why in FOREIGN.items() if lower.endswith(ext)), None)


def loose_reason(node: Node) -> str:
    if node.size == 0 and not node.is_dir:
        return "an empty file (0 bytes): download it again"
    return foreign_reason(node.name) or "not part of a bundle or SpatialData object"


class InputError(UserError, ValueError):
    """Inputs that can't be used as given."""


def is_junk(name: str) -> bool:
    """OS litter, hidden files and pipestance internals."""
    return (name in JUNK or name.startswith(".") or name.startswith("_") or bool(RANGER_DIR.match(name))
            or name.endswith(".mro") or name.endswith(".mri.tgz"))


def is_archive(name: str) -> bool:
    return name.lower().endswith(ARCHIVES)


def archive_stem(name: str) -> str:
    lower = name.lower()
    for suffix in ARCHIVES:
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return name


def canonical_key(canon: str, gz: bool) -> tuple[str, bool]:
    """Files 10x ships gzipped keep `.gz`; other `.gz` (GEO's per-file gzip) lose it and are marked to gunzip."""
    name = canon + ".gz" if gz and canon.endswith((".csv", ".tsv", ".mtx")) else canon
    gz = gz and not name.endswith(".gz")
    if canon in MTX:
        name = "cell_feature_matrix/" + name
    elif canon.startswith("morphology_focus_"):
        name = "morphology_focus/" + name
    return name, gz


def plain_key(name: str) -> tuple[str, bool]:
    if name.endswith(".gz") and not name.endswith(KEEP_GZ):
        return name[:-3], True
    return name, False


def clean_name(name: str) -> str:
    """A folder or archive name as a sample ID: `.zarr`/`.sdata` and a trailing `_outs` dropped."""
    for suffix in (".zarr", ".sdata"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    return re.sub(r"[_\- ]outs$", "", name, flags=re.IGNORECASE)


def name_id(names: Iterable[str]) -> str:
    """The first meaningful name walking up from a sample's root (`outs`, `data` and the like skipped)."""
    for name in names:
        name = clean_name(name)
        if name and name.lower() not in GENERIC and not GENERIC_RE.match(name):
            return name
    return "sample"


def geo_id(found_sids: list[str], name: str) -> str | None:
    """The GSM name of a GSM-named archive or folder holding one instrument run folder."""
    if len(found_sids) != 1 or not INSTRUMENT.match(found_sids[0]) or not GSM.match(name):
        return None
    return GEO_TAIL.sub("", clean_name(name)) or None


def sanitize(sid: str) -> tuple[str, str | None]:
    """A valid SpatialData element-name ID, and a note when characters were replaced."""
    if SAMPLE_ID.match(sid):
        return sid, None
    fixed = re.sub(r"[^A-Za-z0-9_.-]+", "_", sid).strip("_") or "sample"
    return fixed, f"sample ID {sid!r} became {fixed!r} (only letters, digits, '_', '.' and '-' are allowed)"


class Node:
    """A file or folder, on disk or inside an archive."""
    name: str
    is_dir: bool
    size: int = 0

    def children(self) -> list["Node"]:
        raise NotImplementedError

    def member(self, gz: bool = False) -> Member:
        raise NotImplementedError

    def read(self) -> bytes | None:
        raise NotImplementedError

    def up_names(self) -> Iterator[str]:
        raise NotImplementedError

    @property
    def display(self) -> str:
        raise NotImplementedError

    @property
    def disk_path(self) -> Path:
        """The path on disk this node is, or lives in (its archive)."""
        raise NotImplementedError

    @property
    def in_archive(self) -> bool:
        return False


class DiskNode(Node):
    def __init__(self, path: Path):
        self.path = path
        self.name = path.name
        self.is_dir = path.is_dir()
        self.size = 0 if self.is_dir else _size(path)

    def children(self) -> list[Node]:
        try:
            paths = sorted(self.path.iterdir())
        except OSError:
            return []
        return [DiskNode(p) for p in paths if p.exists()]      # broken symlinks dropped

    def member(self, gz: bool = False) -> Member:
        return Member(self.path, None, gz, self.size)

    def read(self) -> bytes | None:
        try:
            return self.path.read_bytes() if self.size <= META_LIMIT else None
        except OSError:
            return None

    def up_names(self) -> Iterator[str]:
        yield self.path.name
        for parent in self.path.parents:
            if parent.name:
                yield parent.name

    @property
    def display(self) -> str:
        return str(self.path)

    @property
    def disk_path(self) -> Path:
        return self.path


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


class ArchiveIndex:
    """An archive's members as a folder tree, read once (with small zarr metadata files kept)."""

    def __init__(self, path: Path):
        self.path = path
        self.dirs: dict[str, dict[str, tuple[bool, int]]] = {"": {}}
        self.meta: dict[str, bytes] = {}
        self.zip = is_zip(path)
        if self.zip:
            with zipfile.ZipFile(path) as zf:
                infos = [(i.filename, i.is_dir(), i.file_size) for i in zf.infolist()]
                for name, is_dir, size in infos:
                    if not is_dir and PurePosixPath(name).name in ZARR_META and size <= META_LIMIT:
                        self.meta[_norm(name)] = zf.read(name)
        else:
            infos = []
            with tarfile.open(path, "r:*") as tf:
                for info in tf:
                    if not (info.isfile() or info.isdir()):
                        continue
                    infos.append((info.name, info.isdir(), info.size))
                    if info.isfile() and PurePosixPath(info.name).name in ZARR_META and info.size <= META_LIMIT:
                        stream = tf.extractfile(info)
                        if stream is not None:
                            self.meta[_norm(info.name)] = stream.read()
        self.raw_names: dict[str, str] = {}
        for raw, is_dir, size in infos:
            name = _norm(raw)
            if not name or ".." in name.split("/"):
                continue
            self.raw_names[name] = raw
            self._add(name, is_dir, size)

    def _add(self, name: str, is_dir: bool, size: int) -> None:
        parts = name.split("/")
        for i in range(len(parts)):
            parent, child = "/".join(parts[:i]), parts[i]
            last = i == len(parts) - 1
            self.dirs.setdefault(parent, {})
            if last and not is_dir:
                self.dirs[parent][child] = (False, size)
            else:
                self.dirs[parent][child] = (True, 0)
                self.dirs.setdefault("/".join(parts[: i + 1]), {})


def _norm(name: str) -> str:
    name = name.replace("\\", "/")
    while name.startswith("./"):
        name = name[2:]
    return name.strip("/")


class ArchiveNode(Node):
    def __init__(self, index: ArchiveIndex, inner: str, is_dir: bool = True, size: int = 0):
        self.index = index
        self.inner = inner
        self.name = PurePosixPath(inner).name if inner else archive_stem(index.path.name)
        self.is_dir = is_dir
        self.size = size

    def children(self) -> list[Node]:
        kids = self.index.dirs.get(self.inner, {})
        return [ArchiveNode(self.index, f"{self.inner}/{n}" if self.inner else n, d, s)
                for n, (d, s) in sorted(kids.items())]

    def member(self, gz: bool = False) -> Member:
        return Member(self.index.path, self.index.raw_names.get(self.inner, self.inner), gz, self.size)

    def read(self) -> bytes | None:
        return self.index.meta.get(self.inner)

    def up_names(self) -> Iterator[str]:
        if self.inner:
            yield from reversed(self.inner.split("/"))
        yield archive_stem(self.index.path.name)
        yield from DiskNode(self.index.path.parent).up_names()

    @property
    def display(self) -> str:
        return f"{self.index.path}!{self.inner}" if self.inner else str(self.index.path)

    @property
    def disk_path(self) -> Path:
        return self.index.path

    @property
    def in_archive(self) -> bool:
        return True


def _json(node: Node | None) -> dict:
    if node is None:
        return {}
    raw = node.read()
    try:
        data = json.loads(raw) if raw else {}
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def zarr_verdict(node: Node, kids: dict[str, Node]) -> str | None:
    """"spatialdata" for a SpatialData store, a left-out reason for another zarr, None for no zarr."""
    named = node.name.lower().endswith(".zarr") and not (node.in_archive and not getattr(node, "inner", ""))
    is_zarr = bool(ZARR_META & kids.keys()) or named        # (an archive's own root is named after the archive)
    if not is_zarr:
        return None
    v3 = _json(kids.get("zarr.json"))
    attrs = v3.get("attributes", {}) if v3 else _json(kids.get(".zattrs"))
    attrs = attrs if isinstance(attrs, dict) else {}
    if "spatialdata_attrs" in attrs:
        return "spatialdata"
    for name in ELEMENT_GROUPS:
        group = kids.get(name)
        if group is not None and group.is_dir and {"zarr.json", ".zgroup"} & {c.name for c in group.children()}:
            return "spatialdata"
    if attrs.get("encoding-type") == "anndata":
        return f"an AnnData .zarr (a table with no spatial elements), not a SpatialData object: {ANNDATA}"
    if ".zarray" in kids or v3.get("node_type") == "array":
        return "a zarr array, not a SpatialData object"
    return "a zarr store without SpatialData elements (no images/labels/points/shapes/tables groups)"


@dataclass
class Group:
    """Loose files sharing one prefix in front of known 10x names."""
    prefix: str
    raw: set[str] = field(default_factory=set)
    files: dict[str, tuple[Node, bool]] = field(default_factory=dict)      # key -> (node, gz)

    def add(self, key: str, node: Node, gz: bool) -> None:
        old = self.files.get(key)
        if old is None or (old[1] and not gz):                              # prefer the plain copy
            self.files[key] = (node, gz)

    @property
    def keys(self) -> set[str]:
        return set(self.files)


def _has_matrix(keys: set[str]) -> bool:
    return bool(keys & MATRIX_KEYS)


def _has_data(keys: set[str]) -> bool:
    return any(k not in METADATA_KEYS for k in keys)


def _marker_kind(keys: Iterable[str]) -> str | None:
    return next((MARKERS[k] for k in keys if k in MARKERS), None)


def qualifies(keys: set[str]) -> bool:
    """A marker with some data beside it, or a count matrix with a cells table."""
    if _marker_kind(keys) is not None:
        return _has_data(keys)
    return _has_matrix(keys) and bool(keys & CELLS_KEYS)


def infer_kind(keys: Iterable[str]) -> str:
    keys = set(keys)
    if keys & ATERA_ONLY or any(k.startswith(("morphology_2d/", "morphology_3d/")) for k in keys):
        return "atera"
    return "xenium"


def split_name(name: str) -> tuple[str, str, str, bool] | None:
    """(prefix, raw prefix, canonical key, gz) for a file with a known 10x name, else None."""
    m = CANON_RE.match(name)
    if m is None:
        return None
    raw = m["prefix"]
    if raw and raw[-1] not in SEPARATORS:
        return None
    key, gz = canonical_key(m["canon"], bool(m["gz"]))
    return raw.rstrip(SEPARATORS), raw, key, gz


def group_files(files: list[Node]) -> tuple[dict[str, Group], list[Node]]:
    groups: dict[str, Group] = {}
    unknown: list[Node] = []
    for node in files:
        parts = split_name(node.name)
        if parts is None:
            unknown.append(node)
            continue
        prefix, raw, key, gz = parts
        group = groups.setdefault(prefix, Group(prefix))
        group.raw.add(raw)
        group.add(key, node, gz)
    return groups, unknown


def owner(name: str, prefixes: list[str]) -> tuple[str, str] | None:
    for prefix in sorted(prefixes, key=len, reverse=True):
        if prefix and name.startswith(prefix) and len(name) > len(prefix) and name[len(prefix)] in SEPARATORS:
            rest = name[len(prefix):].lstrip(SEPARATORS)
            if rest:
                return prefix, rest
    return None


@dataclass
class Found:
    candidates: list[tuple[Candidate, str]] = field(default_factory=list)   # (candidate, root identity)
    left_out: list[LeftOut] = field(default_factory=list)

    def extend(self, other: "Found") -> None:
        self.candidates += other.candidates
        self.left_out += other.left_out


def looks_like_root(node: Node) -> bool:
    """A folder that is a sample of its own, so it's never absorbed into the bundle around it."""
    kids = node.children()
    if zarr_verdict(node, {k.name: k for k in kids}) == "spatialdata":
        return True
    files = [k for k in kids if not k.is_dir and not is_junk(k.name)]
    groups, _ = group_files(files)
    plain = groups.get("")
    if plain is not None and _marker_kind(plain.keys):
        return _has_data(plain.keys)
    mex = {"cell_feature_matrix/matrix.mtx.gz"} if any(k.is_dir and k.name == "cell_feature_matrix" for k in kids) \
        else set()
    return qualifies((plain.keys if plain else set()) | mex) or any(p and qualifies(g.keys) for p, g in groups.items())


def _files_under(node: Node, rel: str = "", depth: int = 0,
                 nested: list[Node] | None = None) -> Iterator[tuple[str, Node]]:
    for child in node.children():
        if is_junk(child.name):
            continue
        path = f"{rel}{child.name}"
        if child.is_dir:
            if nested is not None and looks_like_root(child):
                nested.append(child)
            elif depth < MAX_DEPTH:
                yield from _files_under(child, path + "/", depth + 1, nested)
        else:
            yield path, child


def bundle_files(node: Node, skip: set[str] = frozenset(),
                 nested: list[Node] | None = None) -> tuple[dict[str, Member], int]:
    """Every file of a bundle folder by canonical key (nested bundles go to `nested`), and how many to gunzip."""
    group = Group("")
    for rel, child in _files_under(node, nested=nested):
        if rel in skip:
            continue
        folder, _, base = rel.rpartition("/")
        parts = split_name(base)
        if parts is not None and parts[1] == "":
            key, gz = parts[2], parts[3]
            if folder and key.startswith(("cell_feature_matrix/", "morphology_focus/")) and \
                    folder.rsplit("/", 1)[-1] == key.split("/")[0]:
                key = f"{folder}/{key.split('/', 1)[1]}"
            elif folder:
                key = f"{folder}/{key}"
        else:
            key, gz = plain_key(base)
            key = f"{folder}/{key}" if folder else key
        group.add(key, child, gz)
    files = {k: n.member(gz) for k, (n, gz) in sorted(group.files.items())}
    return files, sum(m.gz for m in files.values())


class Search:
    """One discovery run over the user's entries."""

    def __init__(self):
        self.archives: dict[Path, ArchiveIndex] = {}

    def _index(self, path: Path) -> ArchiveIndex:
        if path not in self.archives:
            size = _size(path)
            if size > 10 ** 9:
                logger.info("Listing the archive %s (%.1f GB); this can take a few minutes …", path.name, size / 1e9)
            self.archives[path] = ArchiveIndex(path)
        return self.archives[path]

    @staticmethod
    def _candidate(node: Node, kind: str, sid: str, files: dict[str, Member], notes: list[str],
                   zarr: Member | None = None) -> tuple[Candidate, str]:
        sid, fixed = sanitize(sid)
        notes = list(notes) + ([fixed] if fixed else [])
        if node.in_archive:
            notes.append(f"inside the archive {node.disk_path.name}")
        return Candidate(sid, kind, node.disk_path, files, zarr, tuple(notes)), node.display

    def scan(self, node: Node, depth: int = 0, recurse: bool = True, archives: bool = True) -> Found:
        return self._geo_named(self._scan(node, depth, recurse, archives), node.name)

    @staticmethod
    def _geo_named(found: Found, name: str) -> Found:
        sid = geo_id([c.sid for c, _ in found.candidates], name)
        if sid:
            (cand, root), = found.candidates
            sid, _ = sanitize(sid)
            note = f"sample ID from the GEO name {name!r} (the instrument folder is {cand.sid})"
            found.candidates = [(replace(cand, sid=sid, notes=cand.notes + (note,)), root)]
        return found

    def _scan(self, node: Node, depth: int = 0, recurse: bool = True, archives: bool = True) -> Found:
        found = Found()
        if depth > MAX_DEPTH:
            found.left_out.append(LeftOut(node.display, f"more than {MAX_DEPTH} folder levels below the input; "
                                                         "give a path closer to the data"))
            return found
        kids = node.children()
        by_name = {k.name: k for k in kids}
        has_10x = any(not k.is_dir and split_name(k.name) for k in kids)
        verdict = None if has_10x else zarr_verdict(node, by_name)
        if verdict == "spatialdata":
            inner = getattr(node, "inner", None) or None
            zarr = Member(node.disk_path, inner, False, 0)
            found.candidates.append(self._candidate(node, "spatialdata", name_id(node.up_names()), {}, [], zarr))
            return found
        if verdict:
            found.left_out.append(LeftOut(node.display, verdict))
            return found
        kids = [k for k in kids if not is_junk(k.name)]
        files = [k for k in kids if not k.is_dir]
        dirs = [k for k in kids if k.is_dir]
        groups, unknown = group_files(files)
        plain = groups.get("")
        if plain is not None and _marker_kind(plain.keys):
            return self._marker_folder(node, plain, dirs, depth, recurse, archives)
        copies = browser_copies(files)
        if copies:
            found.left_out.append(LeftOut(node.display, copies))
            return found
        mex = {"cell_feature_matrix/matrix.mtx.gz"} if any(d.name == "cell_feature_matrix" for d in dirs) else set()
        if qualifies((plain.keys if plain else set()) | mex):
            # an unprefixed bundle without a marker: the whole folder (minus other prefixes' samples)
            others = [g for p, g in groups.items() if p and qualifies(g.keys)]
            taken = {n.name for g in others for n, _ in g.files.values()}
            found.extend(self._folder_bundle(node, taken, depth))
            for group in others:
                found.candidates.append(self._group_bundle(node, group, []))
            return found
        return self._loose(node, groups, unknown, dirs, depth, recurse, archives)

    def _marker_folder(self, node: Node, plain: Group, dirs: list[Node], depth: int, recurse: bool,
                       archives: bool) -> Found:
        """One bundle, unless the marker stands alone above bundles."""
        found = Found()
        own = _has_data(plain.keys) or any(d.name in DATA_DIRS for d in dirs)
        if own:
            return self._folder_bundle(node, set(), depth)
        if recurse:
            nested = Found()
            for child in dirs:
                nested.extend(self.scan(child, depth + 1, recurse, archives))
            for child in node.children():
                if archives and not child.is_dir and not is_junk(child.name) and is_archive(child.name) \
                        and split_name(child.name) is None:
                    nested.extend(self.archive(child, depth + 1))
            if nested.candidates:
                marker = next(k for k in plain.keys if k in MARKERS)
                found.extend(nested)
                found.left_out.append(LeftOut(f"{node.display}/{marker}",
                                              "a lone marker file with no data beside it (the bundles below it were used)"))
                return found
        marker = next(k for k in plain.keys if k in MARKERS)
        found.left_out.append(LeftOut(node.display, f"holds {marker} but no data files (no count matrix, cells, "
                                                    "boundaries or transcripts) and no bundle below it"))
        return found

    def _folder_bundle(self, node: Node, taken: set[str], depth: int) -> Found:
        nested: list[Node] = []
        files, n_gz = bundle_files(node, taken, nested)
        kind = _marker_kind(files) or infer_kind(files)
        notes = []
        if not _marker_kind(files):
            notes.append(f"no experiment file: kind inferred from the files ({NAMES[kind]})")
        if n_gz:
            notes.append(f"{n_gz} gzipped file(s) are decompressed when staged")
        found = Found([self._candidate(node, kind, name_id(node.up_names()), files, notes)])
        for child in nested:
            inner = self.scan(child, depth + 1)
            for cand, root in inner.candidates:
                note = f"found inside the folder of {found.candidates[0][0].sid}, kept as a sample of its own"
                found.candidates.append((replace(cand, notes=cand.notes + (note,)), root))
            found.left_out += inner.left_out
        return found

    def _group_bundle(self, node: Node, group: Group, extra: list[tuple[str, Node]]) -> tuple[Candidate, str]:
        files: dict[str, tuple[Node, bool]] = dict(group.files)
        for rest, child in extra:
            parts = split_name(rest)
            if parts is not None and parts[1] == "":
                key, gz = parts[2], parts[3]
            else:
                key, gz = plain_key(rest)
            files.setdefault(key, (child, gz))
        members = {k: n.member(gz) for k, (n, gz) in sorted(files.items())}
        kind = _marker_kind(members) or infer_kind(members)
        raws = ", ".join(repr(r) for r in sorted(group.raw))
        notes = [f"loose files in {node.display} with the prefix {raws} (prefix stripped)"]
        if not _marker_kind(members):
            notes.append(f"no experiment file: kind inferred from the files ({NAMES[kind]})")
        n_gz = sum(m.gz for m in members.values())
        if n_gz:
            notes.append(f"{n_gz} gzipped file(s) are decompressed when staged")
        cand = self._candidate(node, kind, group.prefix, members, notes)
        return cand[0], f"{node.display}/{group.prefix}*"

    def _loose(self, node: Node, groups: dict[str, Group], unknown: list[Node], dirs: list[Node], depth: int,
               recurse: bool, archives: bool) -> Found:
        """A folder that is no bundle itself: prefix groups, archives, zarr files and subfolders inside it."""
        found = Found()
        samples = [p for p, g in groups.items() if p and qualifies(g.keys)]
        extra: dict[str, list[tuple[str, Node]]] = {p: [] for p in samples}
        rest: list[Node] = []
        for prefix, group in groups.items():
            if prefix in samples:
                continue
            for raw_node, _ in group.files.values():
                hit = owner(raw_node.name, samples)
                if hit:
                    extra[hit[0]].append((hit[1], raw_node))
                else:
                    rest.append(raw_node)
        for child in unknown:
            hit = owner(child.name, samples) if not is_archive(child.name) or child.name.endswith(".zarr.zip") \
                else None
            if hit:
                extra[hit[0]].append((hit[1], child))
            else:
                rest.append(child)
        for prefix in sorted(samples):
            found.candidates.append(self._group_bundle(node, groups[prefix], extra[prefix]))
        loose: list[Node] = []
        for child in sorted(rest, key=lambda n: n.name):
            lower = child.name.lower()
            if is_archive(lower) and split_name(child.name) is None:     # not a bundle file like analysis.zarr.zip
                if archives and recurse:
                    found.extend(self.archive(child, depth + 1))
                # else: a shallow look (see _holder) that leaves archives alone
            elif lower.endswith(UNSUPPORTED):
                found.left_out.append(LeftOut(child.display, f"a {Path(lower).suffix} archive; only .zip, .tar, "
                                                             ".tar.gz and .tgz are read. Extract it and give the folder"))
            else:
                loose.append(child)
        for reason, names, stem in _sidecars(found, loose):
            loose = [n for n in loose if n.name not in names]
            found.left_out.append(LeftOut(f"{node.display}/{stem}*", reason))
        found.left_out += _summarise(node, loose, groups, samples)
        if recurse:
            for child in sorted(dirs, key=lambda d: d.name):
                found.extend(self.scan(child, depth + 1, recurse, archives))
        return found

    def archive(self, node: Node, depth: int) -> Found:
        found = Found()
        if node.in_archive:
            found.left_out.append(LeftOut(node.display, "an archive inside an archive; extract the outer one and "
                                                        "give the folder"))
            return found
        path = node.disk_path
        if node.size == 0:
            found.left_out.append(LeftOut(node.display, "an empty file (0 bytes): download it again"))
            return found
        if path.name.lower().endswith(".zip") and not is_zip(path):
            found.left_out.append(LeftOut(node.display, "not a valid .zip file (damaged or incomplete download?)"))
            return found
        try:
            index = self._index(path)
        except (OSError, zipfile.BadZipFile, tarfile.TarError, EOFError, zlib_error) as err:
            first = (str(err).splitlines() or [type(err).__name__])[0]
            found.left_out.append(LeftOut(node.display, f"can't be read as an archive ({first}; damaged or "
                                                        "incomplete download?)"))
            return found
        inner = self.scan(ArchiveNode(index, ""), depth)
        if not inner.candidates:
            found.left_out.append(LeftOut(node.display, archive_reason(path, inner.left_out)))
            return found
        inner = self._geo_named(inner, archive_stem(path.name))
        roots = [c for c, _ in inner.candidates]
        if len(roots) > 1:
            logger.info("%s holds %d samples: %s", node.display, len(roots), ", ".join(c.sid for c in roots))
        found.extend(inner)
        return found


def archive_reason(path: Path, inner: list[LeftOut]) -> str:
    if not inner:
        return "holds no Xenium/Atera bundle and no SpatialData object (no files in it)"
    head = f"{path}!"
    parts = []
    for lo in inner[:3]:
        where = lo.path[len(head):] if lo.path.startswith(head) else ""
        where = where.removesuffix("*").rstrip(SEPARATORS + "/")
        parts.append(f"{where}: {lo.reason}" if where else lo.reason)
    more = f"; … and {len(inner) - 3} more" if len(inner) > 3 else ""
    return "holds no Xenium/Atera bundle and no SpatialData object: " + "; ".join(parts) + more


def _sidecars(found: Found, loose: list[Node]) -> list[tuple[str, set[str], str]]:
    """GEO-style extra files beside an archive that held a sample, named with its prefix."""
    out = []
    taken: set[str] = set()
    for name in sorted({c.origin.name for c, _ in found.candidates if is_archive(c.origin.name)}):
        stem = clean_name(archive_stem(name))
        mine = {n.name for n in loose if n.name not in taken and owner(n.name, [stem])}
        if mine:
            taken |= mine
            shown = ", ".join(sorted(m[len(stem):].lstrip(SEPARATORS) for m in mine)[:4])
            more = " …" if len(mine) > 4 else ""
            out.append((f"{len(mine)} extra file(s) beside {name} ({shown}{more}); not read: the sample comes "
                        "from the archive", mine, stem))
    return out


def browser_copies(files: list[Node]) -> str | None:
    """A reason when a folder holds browser-renamed copies of 10x files (`cells (2).parquet`)."""
    copies = [f.name for f in files if (m := BROWSER_COPY.match(f.name)) and split_name(m["stem"] + m["ext"])]
    if not copies:
        return None
    shown = ", ".join(sorted(copies)[:3])
    return (f"holds {len(copies)} browser-renamed copies of 10x file names ({shown} …): the files of several "
            "samples were saved into one folder and can't be told apart; put each sample's files in its own folder")


def _summarise(node: Node, loose: list[Node], groups: dict[str, Group], samples: list[str]) -> list[LeftOut]:
    out: list[LeftOut] = []
    names = {n.name for n in loose}
    for prefix, group in sorted(groups.items()):
        if prefix in samples:
            continue
        mine = [n.name for n, _ in group.files.values() if n.name in names]
        if not mine:
            continue
        names -= set(mine)
        label = f"{node.display}/{group.raw and sorted(group.raw)[0] or ''}*"
        out.append(LeftOut(label, group_reason(group)))
    by_name = {n.name: n for n in loose}
    rest = sorted(names)
    special = [n for n in rest if foreign_reason(n) or by_name[n].size == 0]   # always named one by one
    plain = [n for n in rest if n not in special]
    out += [LeftOut(f"{node.display}/{n}", loose_reason(by_name[n])) for n in special]
    if len(plain) > MAX_LISTED:
        out.append(LeftOut(node.display, f"{len(plain)} files that belong to no sample (e.g. {', '.join(plain[:3])})"))
    else:
        out += [LeftOut(f"{node.display}/{n}", loose_reason(by_name[n])) for n in plain]
    return out


def group_reason(group: Group) -> str:
    keys = ", ".join(sorted(group.keys))
    empty = sorted(k for k, (n, _) in group.files.items() if n.size == 0)
    full = {k for k, (n, _) in group.files.items() if n.size > 0}
    if _marker_kind(group.keys) and not _has_data(group.keys):
        why = "no data files beside the experiment file"
    else:
        missing = []
        if not _has_matrix(full):
            missing.append("count matrix (cell_feature_matrix.h5 / .zarr.zip or matrix.mtx)")
        if not full & CELLS_KEYS:
            missing.append("cells table (cells.parquet / cells.csv.gz)")
        why = "no " + " and no ".join(missing) if missing else "not enough to read it"
    zero = f"; empty (0-byte) file(s): {', '.join(empty)}, download them again" if empty else ""
    return f"10x files ({keys}) but {why}{zero}"


def _expand(entry: str) -> list[Path]:
    """`~` expanded, globs matched, relative paths also tried under /data."""
    text = os.path.expanduser(entry.strip())
    if glob.has_magic(text):
        matches = sorted(glob.glob(text))
        if not matches and not os.path.isabs(text):
            matches = sorted(glob.glob(str(config.DATA_DIR / text)))
        if not matches:
            raise InputError(f"{entry} (nothing matches this pattern" + ("" if os.path.isabs(text) else
                                                                         f", also under {config.DATA_DIR}") + ")")
        return [Path(m).absolute() for m in matches]
    path = config.in_data(Path(text))
    if not path.exists():
        tried = "" if path.is_absolute() else f" (also looked for {config.DATA_DIR / text})"
        raise InputError(f"{entry}{tried}{close_names(path if path.is_absolute() else config.DATA_DIR / text)}")
    return [path.absolute()]


def close_names(path: Path) -> str:
    """Up to 3 names like the first missing part of `path`."""
    missing = path
    while not missing.parent.exists() and missing.parent != missing:
        missing = missing.parent
    try:
        names = [p.name for p in missing.parent.iterdir()]
    except OSError:
        return ""
    close = difflib.get_close_matches(missing.name, names, n=3, cutoff=0.6)
    gone = f"; {missing} doesn't exist" if missing != path else ""
    return gone + (f"; closest in {missing.parent}: {', '.join(close)}" if close else "")


def _inside(member: Member, path: Path) -> bool:
    return member.container == path or path in member.container.parents


def _holder(search: Search, path: Path) -> Found | None:
    """The bundle (up to 3 folders up) holding a path the user pointed at inside it."""
    for parent in list(path.parents)[:3]:
        found = search.scan(DiskNode(parent), recurse=False, archives=False)
        hits = [(c, r) for c, r in found.candidates
                if any(_inside(m, path) for m in c.files.values()) or (c.zarr and _inside(c.zarr, path))]
        if hits:
            out = Found()
            for cand, root in hits:
                note = f"the input {path} lies inside this sample's folder"
                out.candidates.append((Candidate(cand.sid, cand.kind, cand.origin, cand.files, cand.zarr,
                                                 cand.notes + (note,)), root))
            return out
    return None


def _entry(search: Search, path: Path) -> Found:
    node = DiskNode(path)
    if not node.is_dir:
        if is_archive(path.name):
            found = search.archive(node, 0)
            if found.candidates or not path.name.lower().endswith(".zarr.zip"):
                return found
            return _holder(search, path) or found
        if path.name.lower().endswith(UNSUPPORTED):
            return Found(left_out=[LeftOut(str(path), f"a {path.suffix} archive; only .zip, .tar, .tar.gz and .tgz "
                                                      "are read. Extract it and give the folder")])
        held = _holder(search, path)
        if held:
            return held
        if node.size == 0:
            reason = "an empty file (0 bytes): download it again"
        elif split_name(path.name) is not None:
            reason = ("a single 10x bundle file without the rest of its bundle beside it; give the folder or archive "
                      "that holds the sample's files (count matrix and cells table at least)")
        else:
            reason = foreign_reason(path.name) or ("a single file that is not part of a bundle; give the folder or "
                                                   "archive that holds the sample's files")
        return Found(left_out=[LeftOut(str(path), reason)])
    found = search.scan(node)
    if not found.candidates:
        return _holder(search, path) or found
    return found


def _same_files(a: Candidate, b: Candidate) -> bool:
    if a.kind != b.kind:
        return False
    if a.zarr is not None or b.zarr is not None:
        return False
    return {k: m.size for k, m in a.files.items()} == {k: m.size for k, m in b.files.items()}


def _settle(found: Found) -> Discovery:
    """Drops repeats and refuses sample-ID clashes."""
    kept: dict[str, tuple[Candidate, str]] = {}
    roots: set[str] = set()
    left = list(found.left_out)
    for cand, root in found.candidates:
        if root in roots:
            continue
        roots.add(root)
        other = kept.get(cand.sid)
        if other is None:
            kept[cand.sid] = (cand, root)
            continue
        first, first_root = other
        if _same_files(first, cand):
            archived = [x for x in ((first, first_root), (cand, root)) if "!" in x[1] or is_archive(x[1])]
            if len(archived) == 1:
                drop = archived[0]
                keep = (cand, root) if drop[0] is first else other
                kept[cand.sid] = keep
                logger.info("%s: using the folder %s; the archive %s holds the same bundle", cand.sid, keep[1],
                            drop[0].origin)
                continue
        raise InputError(f"Two inputs have the sample ID {cand.sid!r}: {first_root} and {root}. Give only one of "
                         "them, or put one in a folder with another name.")
    return Discovery([c for c, _ in kept.values()], left)


def discover(entries: list[str], scratch: Path) -> Discovery:
    if not entries:
        raise InputError("No inputs given: fill in the list of paths to samples.")
    paths, missing = [], []
    for entry in entries:
        try:
            paths += _expand(entry)
        except InputError as exc:
            missing.append(str(exc))
    if len(missing) == 1:
        raise InputError(f"Input not found: {missing[0]}")
    if missing:
        raise InputError(config.block(f"{len(missing)} inputs not found:", missing))
    search = Search()
    found = Found()
    for path in dict.fromkeys(paths):
        found.extend(_entry(search, path))
    result = _settle(found)
    if not result.candidates:
        why = "\n".join(f"  - {lo.path}: {lo.reason}" for lo in result.left_out[:20])
        more = f"\n  … and {len(result.left_out) - 20} more" if len(result.left_out) > 20 else ""
        raise InputError("No Xenium bundle, Atera bundle or SpatialData object found in the inputs"
                         + (f":\n{why}{more}" if why else "."))
    for cand in result.candidates:
        where = cand.zarr.display if cand.zarr else cand.origin
        count = "" if cand.kind == "spatialdata" else f" ({len(cand.files)} files)"
        logger.info("Found %s %s at %s%s%s", NAMES[cand.kind], cand.sid, where, count,
                    "".join(f"; {n}" for n in cand.notes))
    return result


CHUNK = 1 << 24
BIG = 10 ** 9                       # archives larger than this get a progress line
ZIP_THREADS = 8
PREFETCH_SKIP = ("analysis/", "aux_outputs/", "analysis.zarr.zip", "analysis.tar.gz", "analysis_summary.html")


def gb(size: int) -> str:
    return f"{size / 1e9:.1f} GB"


def _note_big(container: Path, what: str) -> None:
    size = _size(container)
    if size > BIG:
        logger.info("%s from %s (%s); this can take a few minutes …", what, container.name, gb(size))


def is_zip(path: Path) -> bool:
    """By signature: `zipfile.is_zipfile` also accepts a .tar whose last member is a .zip."""
    if not path.is_file():
        return False
    with open(path, "rb") as f:
        head = f.read(4)
    return head in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08") and zipfile.is_zipfile(path)


def _open_member(member: Member):
    if member.inner is None:
        return open(member.container, "rb")
    if is_zip(member.container):
        zf = zipfile.ZipFile(member.container)
        stream = zf.open(member.inner)
        stream._spatial_ingest_owner = zf  # keep the archive open as long as the stream
        return stream
    tf = tarfile.open(member.container, "r:*")
    stream = tf.extractfile(member.inner)
    if stream is None:
        tf.close()
        raise FileNotFoundError(f"{member.display} is not a regular file")
    stream._spatial_ingest_owner = tf
    return stream


def _write(src, member: Member, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".partial")
    data = gzip.GzipFile(fileobj=src) if member.gz else src
    with open(tmp, "wb") as dst:
        shutil.copyfileobj(data, dst, CHUNK)
    tmp.rename(target)


def _copy(member: Member, target: Path) -> None:
    with _open_member(member) as raw:
        _write(raw, member, target)


def _is_tar_member(member: Member) -> bool:
    return member.inner is not None and not is_zip(member.container)


def fetch_member(member: Member, target: Path) -> Path:
    """A symlink for a plain file, else extracted or decompressed."""
    if target.exists() or target.is_symlink():
        return target
    if member.inner is None and not member.gz:
        target.parent.mkdir(parents=True, exist_ok=True)
        if member.container.is_dir():
            target.symlink_to(member.container.resolve(), target_is_directory=True)
        else:
            target.symlink_to(member.container.resolve())
        return target
    _copy(member, target)
    return target


def _fetch_tar(container: Path, wanted: dict[str, list[tuple[Member, Path]]]) -> None:
    """Every wanted member of one tar in a single streaming pass."""
    _note_big(container, f"Extracting {sum(map(len, wanted.values()))} file(s)")
    left = dict(wanted)
    with tarfile.open(container, "r|*") as tf:
        for info in tf:
            jobs = left.pop(info.name, None)
            if not jobs or not info.isfile():
                continue
            _write(tf.extractfile(info), *jobs[0])
            for member, target in jobs[1:]:          # the same member wanted under a second name
                _copy(member, target)
            if not left:
                break
    if left:
        missing = ", ".join(sorted(left)[:3])
        raise FileNotFoundError(f"{container}: no member {missing} (the archive changed since it was listed?)")


def fetch(candidate: Candidate, names: Iterable[str], dest: Path) -> dict[str, Path]:
    """The named canonical files of a bundle under `dest` (a name ending in `/` is a whole folder)."""
    todo: dict[str, Member] = {}
    for name in names:
        if name.endswith("/"):
            todo.update({k: m for k, m in candidate.files.items() if k.startswith(name)})
        elif name in candidate.files:
            todo[name] = candidate.files[name]
    out: dict[str, Path] = {}
    tars: dict[Path, dict[str, list[tuple[Member, Path]]]] = {}
    for key, member in todo.items():
        target = dest / key
        out[key] = target
        if target.exists() or target.is_symlink():
            continue
        if _is_tar_member(member):
            tars.setdefault(member.container, {}).setdefault(member.inner, []).append((member, target))
        else:
            fetch_member(member, target)
    for container, wanted in tars.items():
        for key, member in candidate.files.items():          # the rest of the sample, in the same pass
            target = dest / key
            if member.container == container and member.inner is not None and not key.startswith(PREFETCH_SKIP) \
                    and not (target.exists() or target.is_symlink()):
                jobs = wanted.setdefault(member.inner, [])
                if all(t != target for _, t in jobs):
                    jobs.append((member, target))
        _fetch_tar(container, wanted)
    return out


def _junk(rel: str) -> bool:
    """Archive litter, and member names that would land outside the target folder."""
    parts = rel.split("/")
    return "__MACOSX" in parts or ".." in parts or any(p.startswith("._") or p == ".DS_Store" for p in parts)


def _in_store(raw: str, prefix: str) -> str | None:
    name = _norm(raw)
    if name.startswith(prefix) and len(name) > len(prefix) and not _junk(name[len(prefix):]):
        return name[len(prefix):]
    return None


def _unzip_store(container: Path, prefix: str, tmp: Path) -> None:
    with zipfile.ZipFile(container) as zf:
        jobs = [(info.filename, rel) for info in zf.infolist()
                if not info.is_dir() and (rel := _in_store(info.filename, prefix)) is not None]
    for folder in {os.path.dirname(rel) for _, rel in jobs}:
        (tmp / folder).mkdir(parents=True, exist_ok=True)
    local = threading.local()
    handles: list[zipfile.ZipFile] = []

    def one(job: tuple[str, str]) -> None:
        if not hasattr(local, "zf"):
            local.zf = zipfile.ZipFile(container)
            handles.append(local.zf)
        raw, rel = job
        with local.zf.open(raw) as src, open(tmp / rel, "wb") as dst:
            shutil.copyfileobj(src, dst, CHUNK)

    try:
        with ThreadPoolExecutor(ZIP_THREADS) as pool:
            list(pool.map(one, jobs))
    finally:
        for handle in handles:
            handle.close()


def _untar_store(container: Path, prefix: str, tmp: Path) -> None:
    with tarfile.open(container, "r|*") as tf:
        for info in tf:
            rel = _in_store(info.name, prefix) if info.isfile() else None
            if rel is None:
                continue
            target = tmp / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)


def stage_spatialdata(candidate: Candidate, dest: Path) -> Path:
    """A readable SpatialData store root: the folder itself, or the store extracted from its archive."""
    member = candidate.zarr
    if member is None:
        raise ValueError(f"{candidate.sid} is not a SpatialData candidate")
    if member.inner is None and member.container.is_dir():
        return member.container
    root = dest / (Path(member.inner.rstrip("/")).name if member.inner else candidate.sid + ".zarr")
    if root.exists():
        return root
    prefix = (_norm(member.inner) + "/") if member.inner else ""
    tmp = root.with_name(root.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    _note_big(member.container, f"Extracting the SpatialData object {candidate.sid}")
    if is_zip(member.container):
        _unzip_store(member.container, prefix, tmp)
    else:
        _untar_store(member.container, prefix, tmp)
    tmp.rename(root)
    return root


SAMPLE_ROWS = 1_000_000          # rows of transcripts / boundaries checked against the matrix barcodes
MIN_ASSIGNED_MATCH = 0.99        # share of a file's assigned cell IDs that must be matrix barcodes
MATRIX_TEXT = "count matrix (cell_feature_matrix.h5, cell_feature_matrix.zarr.zip or matrix.mtx + features.tsv + barcodes.tsv)"
CELLS_TEXT = "cell positions (cells.parquet or cells.csv with cell_id, x_centroid, y_centroid)"
TABULAR = (".parquet", ".csv", ".csv.gz", ".parquet.gz")


def stage_dir(scratch: Path, sid: str) -> Path:
    return scratch / "stage" / sid


def _base(key: str) -> str:
    return key[:-3] if key.endswith(".gz") and not key.endswith((".csv.gz", ".tsv.gz", ".mtx.gz")) else key


def is_empty(member: Member) -> bool:
    """A 0-byte file (a failed GEO upload) counts as missing."""
    if member.inner is None:
        return member.container.is_file() and member.container.stat().st_size == 0
    if member.size:
        return False
    try:
        if zipfile.is_zipfile(member.container):
            with zipfile.ZipFile(member.container) as zf:
                return zf.getinfo(member.inner).file_size == 0
        with tarfile.open(member.container, "r:*") as tf:
            return tf.getmember(member.inner).size == 0
    except (KeyError, OSError, tarfile.TarError):
        return False


class Files:
    def __init__(self, candidate: Candidate):
        self.candidate = candidate
        self.empty = sorted(k for k, m in candidate.files.items() if is_empty(m))
        self.keys = [k for k in candidate.files if k not in self.empty]

    def first(self, *names: str) -> str | None:
        for name in names:
            for key in self.keys:
                if _base(key) == name or key == name:
                    return key
        return None

    def variants(self, stem: str, exts: tuple[str, ...] = TABULAR) -> list[str]:
        """`<stem><ext>` first, then `<anything>[_-.]<stem><ext>` (reseg_transcripts), parquet before csv."""
        pattern = re.compile(rf"^(?:(.+)[_.-])?{re.escape(stem)}({'|'.join(re.escape(e) for e in exts)})$")
        found = []
        for key in self.keys:
            match = pattern.match(_base(key))
            if match and "/" not in key and not key.startswith(("binned_", "._")):
                found.append((match.group(1) is not None, 0 if ".parquet" in match.group(2) else 1, key))
        return [k for *_, k in sorted(found)]

    def folder(self, name: str) -> list[str]:
        return sorted(k for k in self.keys if k.startswith(name.rstrip("/") + "/"))

    def mtx(self) -> dict[str, str] | None:
        for key in self.keys:
            folder, _, name = key.rpartition("/")
            if name not in ("matrix.mtx", "matrix.mtx.gz"):
                continue
            prefix = f"{folder}/" if folder else ""
            feats = self.first(*(prefix + n for n in ("features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv")))
            barcodes = self.first(prefix + "barcodes.tsv.gz", prefix + "barcodes.tsv")
            if feats and barcodes:
                return {"matrix": key, "matrix_features": feats, "matrix_barcodes": barcodes}
        return None


class Inspector:
    """Works out a bundle's inventory, staging the few files it reads."""

    def __init__(self, candidate: Candidate, scratch: Path):
        self.c, self.files, self.dest = candidate, Files(candidate), stage_dir(scratch, candidate.sid)
        self.scratch = scratch / "work" / candidate.sid
        self.chosen: dict[str, str] = {}
        self.missing_essential: list[str] = []
        self.missing_optional: list[str] = []
        self.notes: list[str] = []
        self.var: pd.DataFrame | None = None
        self.n_cells: int | None = None
        self.barcodes: pd.Index | None = None
        self.n_placed: int | None = None

    def path(self, key: str) -> Path:
        return fetch(self.c, [key], self.dest)[key]

    def result(self) -> Inventory:
        if self.files.empty:
            self.notes.append(f"0-byte files ignored: {', '.join(self.files.empty)}")
        return Inventory(self.c.sid, self.c.kind, self.chosen, self.missing_essential, self.missing_optional,
                         self.var, self.n_cells, self.notes)

    def matrix(self) -> MatrixFile | None:
        mtx = self.files.mtx()
        options = [{"matrix": k} for k in (self.files.first("cell_feature_matrix.h5"),) if k]
        options += [mtx] if mtx else []
        options += [{"matrix": k} for k in (self.files.first("cell_feature_matrix.zarr.zip"),) if k]
        errors = []
        for roles in options:
            try:
                m = open_matrix({r: self.path(k) for r, k in roles.items()}, self.c.kind)
            except (TableError, OSError) as exc:
                errors.append(str(exc))
                continue
            self.chosen.update(roles)
            self.var, self.barcodes = m.var, m.barcodes
            return m
        reason = f" ({'; '.join(errors)})" if errors else ""
        self.missing_essential.append(MATRIX_TEXT + reason)
        return None

    def cells(self, m: MatrixFile) -> None:
        if m.kind == "atera_zarr":
            missing = [c for c in ATERA_POSITION if c not in m.obs]
            if missing:
                self.missing_essential.append(f"cell positions ({', '.join(missing)} in the matrix obs)")
                return
            self.chosen["cells"] = self.chosen["matrix"]
            self.n_cells = len(m.barcodes)
            return
        options = self.files.variants("cells")
        zarr_key = self.files.first("cells.zarr.zip")
        reasons, best = [], None
        for key in options + ([zarr_key] if zarr_key else []):
            try:
                ids = self._cell_ids(key)
            except (TableError, OSError, ValueError, KeyError) as exc:
                reasons.append(f"{key}: {exc}")
                continue
            placed = int(m.barcodes.isin(ids).sum())
            if not placed:
                reasons.append(f"{key}: none of the matrix's {len(m.barcodes)} cells are in it")
                continue
            if best is None or placed > best[1]:
                best = (key, placed)
            if placed == len(m.barcodes):
                break
        if best is None:
            why = f" ({'; '.join(reasons)})" if reasons else ""
            self.missing_essential.append(CELLS_TEXT + why)
            return
        key, placed = best
        self.chosen["cells"] = key
        self.n_cells = len(m.barcodes)              # every matrix cell is kept, placed or not
        self.n_placed = placed
        if placed < len(m.barcodes):
            self.notes.append(f"{len(m.barcodes) - placed} of {len(m.barcodes)} matrix cells have no row in {key}")
        if not options:
            self.notes.append(f"cell positions read from {key} (no cells.parquet / cells.csv)")
        elif key != options[0]:
            self.notes.append(f"cell positions read from {key} (not {'; '.join(reasons) or options[0]})")

    def _cell_ids(self, key: str) -> pd.Index:
        path = self.path(key)
        if key.endswith(".zarr.zip"):
            return pd.Index(xenium_cells_zarr(path)[CELL_ID])
        columns = frame_columns(path, self.scratch)
        missing = [c for c in (CELL_ID, *POSITION) if c not in columns]
        if missing:
            raise TableError(f"no {', '.join(missing)} column")
        ids = read_frame(path, self.scratch, columns=[CELL_ID])[CELL_ID]
        return pd.Index(text(ids.to_numpy()))


    def assigned_match(self, key: str) -> tuple[float, int]:
        """Share of the file's assigned cell IDs that are matrix barcodes, and how many were checked."""
        path = self.path(key)
        ids = read_frame(path, self.scratch, columns=[CELL_ID], nrows=SAMPLE_ROWS)[CELL_ID]
        ids = pd.Index(text(ids.to_numpy()))
        ids = ids[~ids.isin(list(UNASSIGNED))]
        if not len(ids):
            return 1.0, 0
        return float(ids.isin(self.barcodes).mean()), len(ids)

    def by_ids(self, role: str, stem: str, label: str) -> None:
        """The first `<…>stem` file whose cell IDs are matrix barcodes."""
        options = self.files.variants(stem)
        if not options:
            self.missing_optional.append(label)
            return
        rejected = []
        for key in options:
            try:
                share, n = self.assigned_match(key)
            except (TableError, OSError, ValueError, KeyError) as exc:
                rejected.append(f"{key} (unreadable: {exc})")
                continue
            if share >= MIN_ASSIGNED_MATCH:
                self.chosen[role] = key
                if key != options[0]:
                    self.notes.append(f"{label}: {key} chosen ({share:.1%} of its cell IDs are matrix cells); "
                                      f"not {', '.join(rejected)}")
                elif n == 0:
                    self.notes.append(f"{label}: {key} has no cell IDs assigned in its first rows; taken as is")
                return
            rejected.append(f"{key} ({share:.1%} of its cell IDs are matrix cells)")
        self.missing_optional.append(f"{label} (cell IDs don't match the count matrix: {', '.join(rejected)})")

    def xenium_optional(self) -> None:
        f = self.files
        experiment = f.first("experiment.xenium")
        if experiment:
            self.chosen["experiment"] = experiment
            try:
                spec = json.loads(self.path(experiment).read_text())
                self.notes.append(f"pixel size {float(spec.get('pixel_size', DEFAULT_PIXEL_SIZE))} µm, "
                                  f"{spec.get('analysis_sw_version', 'unknown version')}")
            except (OSError, ValueError) as exc:
                self.notes.append(f"experiment.xenium unreadable ({exc}); pixel size {DEFAULT_PIXEL_SIZE} µm assumed")
        else:
            self.missing_optional.append(f"experiment.xenium (pixel size {DEFAULT_PIXEL_SIZE} µm assumed)")
        cells_zarr = f.first("cells.zarr.zip")
        if cells_zarr:
            self.chosen["cells_zarr"] = cells_zarr
        else:
            self.missing_optional.append("cell/nucleus labels (cells.zarr.zip)")
        self.by_ids("cell_boundaries", "cell_boundaries", "cell boundaries")
        self.by_ids("nucleus_boundaries", "nucleus_boundaries", "nucleus boundaries")
        for role, label in (("cell_boundaries", "cell boundaries"), ("nucleus_boundaries", "nucleus boundaries")):
            if role not in self.chosen and cells_zarr:       # Explorer-only export: polygons in cells.zarr.zip
                self.chosen[role] = cells_zarr
                self.missing_optional = [t for t in self.missing_optional if not t.startswith(label)]
                self.notes.append(f"{label} read from cells.zarr.zip")
        self.by_ids("transcripts", "transcripts", "transcripts")
        if "transcripts" not in self.chosen and f.first("transcripts.zarr.zip") and not f.variants("transcripts"):
            self.chosen["transcripts"] = f.first("transcripts.zarr.zip")
            self.missing_optional = [t for t in self.missing_optional if t != "transcripts"]
            self.notes.append("transcripts read from transcripts.zarr.zip (Xenium Explorer format: no cell assignment)")
        focus = f.folder("morphology_focus")
        if focus:
            self.chosen["morphology"] = "morphology_focus/"
        elif f.first("morphology_focus.ome.tif"):
            self.chosen["morphology"] = f.first("morphology_focus.ome.tif")
        if f.first("morphology_mip.ome.tif"):
            self.chosen["morphology_mip"] = f.first("morphology_mip.ome.tif")
        stack = f.first("morphology.ome.tif")
        if stack and "morphology" not in self.chosen and "morphology_mip" not in self.chosen:
            self.chosen["morphology_3d"] = stack   # only image (GEO uploads): stored as its max projection
            self.notes.append("image: max projection of the morphology.ome.tif z-stack (no 2D image)")
        elif "morphology" not in self.chosen and "morphology_mip" not in self.chosen:
            self.missing_optional.append("morphology images")
        panel = f.first("gene_panel.json")
        if panel:
            self.chosen["panel"] = panel

    def atera_optional(self) -> None:
        f = self.files
        for role, name in (("experiment", "experiment.spatial"), ("panel", "panel_config.json")):
            if f.first(name):
                self.chosen[role] = f.first(name)
        if "experiment" not in self.chosen:
            self.missing_optional.append("experiment.spatial (pixel size read from the images)")
        if f.first("cells.zarr.zip"):
            self.chosen["cells_zarr"] = f.first("cells.zarr.zip")
        else:
            self.missing_optional.append("cell/nucleus boundaries and labels (cells.zarr.zip)")
        if f.first("transcripts.zarr.zip"):
            self.chosen["transcripts"] = f.first("transcripts.zarr.zip")
        else:
            self.missing_optional.append("transcripts (transcripts.zarr.zip)")
        if f.folder("morphology_2d"):
            self.chosen["morphology"] = "morphology_2d/"
        else:
            self.missing_optional.append("morphology images (morphology_2d/)")
        if "experiment" not in self.chosen and "morphology" not in self.chosen:
            self.notes.append("no experiment.spatial and no images: pixel size unknown")


def inspect(candidate: Candidate, scratch: Path) -> Inventory:
    """Never raises for a bundle's content: what's unusable is listed as missing, with the reason."""
    ins = Inspector(candidate, scratch)
    m = ins.matrix()
    if m is not None:
        ins.cells(m)
    if m is not None:
        ins.atera_optional() if m.kind == "atera_zarr" else ins.xenium_optional()
    if ins.n_cells == 0:
        ins.missing_essential.append("cells: the count matrix has no cells")
    return ins.result()


def describe(inv: Inventory) -> str:
    parts = []
    if inv.missing_optional:
        parts.append("not available: " + "; ".join(inv.missing_optional))
    parts += inv.notes
    return ". ".join(parts)


# roles spatialdata-io and the vendored Atera reader open by file name
BY_NAME = ("experiment", "cells_zarr", "cell_boundaries", "nucleus_boundaries", "transcripts", "morphology",
           "morphology_mip", "morphology_3d", "panel")


def _link(target: Path, link: Path) -> Path:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target.resolve(), target_is_directory=target.is_dir())
    return link


def canonical_name(role: str, key: str, kind: str) -> str:
    """Where a chosen file goes in the staged folder: spatialdata-io's name for it."""
    base = key.rsplit("/", 1)[-1]
    if role == "experiment":
        return "experiment.xenium" if kind == "xenium" else "experiment.spatial"
    if base.endswith(".zarr.zip"):
        return {"cells_zarr": "cells.zarr.zip", "transcripts": "transcripts.zarr.zip"}.get(role, base)
    if role in ("cell_boundaries", "nucleus_boundaries", "transcripts", "cells"):
        ext = ".parquet" if ".parquet" in base else ".csv.gz" if base.endswith(".gz") else ".csv"
        return role + ext
    return base


def stage_bundle(sample: Sample, inv: Inventory, scratch: Path) -> tuple[Path, dict[str, Path]]:
    """The chosen files staged and linked under canonical names; returns the folder and role -> path."""
    cand = sample.candidate
    dest = stage_dir(scratch, sample.sid)
    folder = dest / "bundle"
    files: dict[str, Path] = {}
    staged = fetch(cand, list(dict.fromkeys(inv.chosen.values())), dest)   # one call: one pass per tar
    for role, key in inv.chosen.items():
        if key.endswith("/"):
            for name, path in staged.items():
                if name.startswith(key):
                    _link(path, folder / name)
            files[role] = folder / key.rstrip("/")
            continue
        path = staged[key]
        files[role] = _link(path, folder / canonical_name(role, key, cand.kind)) if role in BY_NAME else path
    if inv.chosen.get("matrix", "").endswith(".zarr.zip") and cand.kind == "atera":
        _link(files["matrix"], folder / "cell_feature_matrix.zarr.zip")
    return folder, files


GEX = "Gene Expression"
MITO_PREFIX = "MT-"
SYMBOL_COLUMNS = ("feature_name", "gene_symbols", "gene_name")
ID_COLUMNS = ("gene_ids", "feature_id", "gene_id")
TYPE_COLUMNS = ("feature_types", "feature_type")
_ENSEMBL = re.compile(r"^ENS[A-Z]*[GT]\d+")
_ENSEMBL_VERSION = re.compile(r"^(ENS[A-Z]*[GT]\d+)\.\d+$")
SPECIES = {"ENSG": "human", "ENST": "human", "ENSMUSG": "mouse", "ENSMUST": "mouse"}


class PanelError(UserError, ValueError):
    """A gene panel the store can't take."""


def gene_mask(var: pd.DataFrame) -> np.ndarray:
    for col in TYPE_COLUMNS:
        if col in var:
            return np.asarray(var[col].astype(str) == GEX)
    return np.ones(len(var), dtype=bool)


def _ensembl(values) -> bool:
    values = [str(v) for v in values]
    return bool(values) and sum(bool(_ENSEMBL.match(v)) for v in values) > len(values) / 2


def symbol_var(var: pd.DataFrame, name: str) -> tuple[np.ndarray, pd.DataFrame]:
    """The Gene Expression features (a mask over var) and their var, indexed by unique gene symbol."""
    keep = gene_mask(var)
    genes = var[keep]
    if not _ensembl(genes.index):
        symbols = genes.index.astype(str)
    else:
        column = next((c for c in SYMBOL_COLUMNS if c in genes), None)
        if column is None:
            raise PanelError(f"{name}: the table is indexed by Ensembl IDs and has no symbol column "
                             f"({', '.join(SYMBOL_COLUMNS)}).")
        symbols = genes[column].astype(str)
    ids = next((genes[c].astype(str).to_numpy() for c in ID_COLUMNS if c in genes), None)
    if ids is None and _ensembl(genes.index):
        ids = genes.index.astype(str).to_numpy()
    if ids is not None:
        ids = np.asarray([plain_id(i) for i in ids], dtype=object)
    out = pd.DataFrame(index=unique_symbols(np.asarray(symbols, dtype=object), ids))
    out["gene_ids"] = ids if ids is not None else ""
    out["feature_types"] = GEX
    return keep, out


def plain_id(gene_id: str) -> str:
    return _ENSEMBL_VERSION.sub(r"\1", str(gene_id))


def unique_symbols(symbols, ids) -> pd.Index:
    """A symbol several genes share becomes `<symbol>-<gene_id>`, so a gene has the same name in every panel."""
    symbols = pd.Index(np.asarray(symbols, dtype=object))
    if ids is not None:
        ids = np.asarray(ids, dtype=object)
        repeated = symbols.duplicated(keep=False)
        named = np.where(repeated & (ids != ""), [f"{s}-{i}" for s, i in zip(symbols, ids)], symbols)
        symbols = pd.Index(named.astype(object))
    if symbols.is_unique:
        return symbols
    return pd.Index(ad.utils.make_index_unique(symbols))


def species(var: pd.DataFrame) -> str | None:
    found = {SPECIES.get(re.sub(r"\d.*$", "", str(g))) for g in var.get("gene_ids", [])} - {None}
    return found.pop() if len(found) == 1 else None


def is_mito(var_names) -> np.ndarray:
    return np.char.startswith(np.char.upper(np.asarray(var_names).astype(str)), MITO_PREFIX.upper())


def holds_counts(x: sp.csr_matrix) -> bool:
    return bool(np.all(x.data >= 0) and np.all(np.mod(x.data, 1) == 0))


def as_dtype(x: sp.csr_matrix, dtype: np.dtype) -> sp.csr_matrix:
    dtype = np.dtype(dtype)
    if x.dtype == dtype:
        return x
    if np.issubdtype(dtype, np.integer) and x.nnz and x.data.max() > np.iinfo(dtype).max:
        raise PanelError(f"counts up to {x.data.max():g} don't fit the joint table's {dtype}.")
    return x.astype(dtype)


BAND_NNZ = 20_000_000          # nonzeros per row band read or bucketed (≈ 160–240 MB in memory)
ARRAY_CHUNK = 4_000_000        # zarr chunk length of data/indices
CSR_ATTRS = {"encoding-type": "csr_matrix", "encoding-version": "0.1.0"}


class CSRWriter:
    """Appends CSR row bands to `group[key]`, writing data/indices in whole zarr chunks."""

    def __init__(self, group: zarr.Group, n_cols: int, dtype=np.float32, key: str = "X"):
        self.n_cols, self.dtype = int(n_cols), np.dtype(dtype)
        self.node = group.create_group(key, overwrite=True)
        index_dtype = np.int32 if n_cols < np.iinfo(np.int32).max else np.int64
        self.chunk = ARRAY_CHUNK
        self.data = self.node.create_array("data", shape=(0,), chunks=(self.chunk,), dtype=self.dtype)
        self.indices = self.node.create_array("indices", shape=(0,), chunks=(self.chunk,), dtype=index_dtype)
        self._indptr: list[np.ndarray] = [np.zeros(1, dtype=np.int64)]
        self._pending: list[tuple[np.ndarray, np.ndarray]] = []
        self._pending_n = 0
        self.written = 0
        self.nnz = 0
        self.n_rows = 0

    def append(self, rows: sp.csr_matrix) -> None:
        rows = sp.csr_matrix(rows)
        if rows.shape[1] != self.n_cols:
            raise ValueError(f"row band has {rows.shape[1]} columns, the matrix {self.n_cols}")
        rows.sum_duplicates()
        if rows.nnz:
            self._pending.append((rows.data.astype(self.dtype, copy=False),
                                  rows.indices.astype(self.indices.dtype, copy=False)))
            self._pending_n += rows.nnz
        self._indptr.append(rows.indptr[1:].astype(np.int64) + self.nnz)
        self.nnz += rows.nnz
        self.n_rows += rows.shape[0]
        if self._pending_n >= self.chunk:
            self._flush(whole_chunks=True)

    def _flush(self, whole_chunks: bool) -> None:
        if not self._pending_n:
            return
        data = np.concatenate([d for d, _ in self._pending])
        indices = np.concatenate([i for _, i in self._pending])
        n = (len(data) // self.chunk) * self.chunk if whole_chunks else len(data)
        if n:
            end = self.written + n
            self.data.resize((end,))
            self.indices.resize((end,))
            self.data[self.written:end] = data[:n]
            self.indices[self.written:end] = indices[:n]
            self.written = end
        self._pending = [(data[n:], indices[n:])] if n < len(data) else []
        self._pending_n = len(data) - n

    def close(self) -> None:
        self._flush(whole_chunks=False)
        indptr = np.concatenate(self._indptr)
        self.node.create_array("indptr", data=indptr, chunks=(max(1, min(len(indptr), 1 << 22)),))
        self.node.attrs.update({**CSR_ATTRS, "shape": [self.n_rows, self.n_cols]})
        for name in ("data", "indices", "indptr"):
            self.node[name].attrs.update({"encoding-type": "array", "encoding-version": "0.2.0"})


def write_csr(group: zarr.Group, bands: Iterable[sp.csr_matrix], n_cols: int, dtype=np.float32,
              key: str = "X") -> tuple[int, int]:
    writer = CSRWriter(group, n_cols, dtype, key)
    for band in bands:
        writer.append(band)
    writer.close()
    return writer.n_rows, writer.nnz


def x_node(group_path: Path | zarr.Group):
    if not isinstance(group_path, (str, Path)) and not isinstance(group_path, zarr.Group):
        return group_path                                   # a matrix node already (zarr array or CSR-like)
    root = group_path if isinstance(group_path, zarr.Group) else zarr.open(str(group_path), mode="r")
    if isinstance(root, zarr.Group) and "X" in root and root.attrs.get("encoding-type") == "anndata":
        return root["X"]
    if isinstance(root, zarr.Group) and "X" in root and "indptr" not in root:
        return root["X"]
    return root


def shape(group_path: Path | zarr.Group) -> tuple[int, int]:
    node = x_node(group_path)
    if isinstance(node, zarr.Array):
        return tuple(node.shape)
    return tuple(int(v) for v in node.attrs["shape"])


def read_indptr(group_path: Path | zarr.Group) -> np.ndarray:
    node = x_node(group_path)
    if isinstance(node, zarr.Array):
        raise ValueError("dense X has no row pointer")
    return np.asarray(node["indptr"][:], dtype=np.int64)


def rows_for(indptr: np.ndarray, budget: int | None = None, max_rows: int | None = None) -> list[tuple[int, int]]:
    """Row ranges of about `budget` nonzeros each."""
    budget = budget or BAND_NNZ
    n = len(indptr) - 1
    out, start = [], 0
    while start < n:
        stop = int(np.searchsorted(indptr, indptr[start] + budget, side="right")) - 1
        stop = max(stop, start + 1)
        if max_rows:
            stop = min(stop, start + max_rows)
        stop = min(stop, n)
        out.append((start, stop))
        start = stop
    return out


def iter_rows(group_path: Path | zarr.Group, rows: int | None = None,
              budget: int | None = None) -> Iterator[tuple[int, sp.csr_matrix]]:
    """(first row, CSR band) over a CSR or dense X."""
    node = x_node(group_path)
    n_rows, n_cols = shape(node)
    if isinstance(node, zarr.Array):
        step = rows or max(1, (budget or BAND_NNZ) // max(n_cols, 1))
        for start in range(0, n_rows, step):
            yield start, sp.csr_matrix(np.asarray(node[start:start + step]))
        return
    if node.attrs.get("encoding-type") != "csr_matrix":
        raise ValueError(f"X is {node.attrs.get('encoding-type')}, not csr_matrix")
    indptr = read_indptr(node)
    data, indices = node["data"], node["indices"]
    for start, stop in rows_for(indptr, budget, rows):
        a, b = int(indptr[start]), int(indptr[stop])
        yield start, sp.csr_matrix((data[a:b], indices[a:b], indptr[start:stop + 1] - a), shape=(stop - start, n_cols))


class RowBuckets:
    """(row, col, value) triples spooled to disk per row band, read back as CSR bands in row order."""

    def __init__(self, folder: Path, n_rows: int, n_cols: int, nnz_hint: int, dtype=np.float32):
        self.folder, self.n_rows, self.n_cols, self.dtype = Path(folder), int(n_rows), int(n_cols), np.dtype(dtype)
        shutil.rmtree(self.folder, ignore_errors=True)
        self.folder.mkdir(parents=True)
        n_bands = max(1, math.ceil(max(nnz_hint, 1) / BAND_NNZ))
        self.band_rows = max(1, math.ceil(self.n_rows / n_bands))
        self.n_bands = max(1, math.ceil(self.n_rows / self.band_rows))
        self._files = {}

    def _path(self, band: int, part: str) -> Path:
        return self.folder / f"{band:06d}.{part}"

    def add(self, rows: np.ndarray, cols: np.ndarray, values: np.ndarray) -> None:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            return
        if rows.min() < 0 or rows.max() >= self.n_rows:
            raise ValueError(f"row index out of range 0..{self.n_rows - 1}")
        band = rows // self.band_rows
        order = np.argsort(band, kind="stable")
        band, rows = band[order], rows[order]
        cols = np.asarray(cols, dtype=np.int32)[order]
        values = np.asarray(values)[order].astype(self.dtype, copy=False)
        cuts = np.flatnonzero(np.diff(band)) + 1
        for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, len(band)]):
            b = int(band[lo])
            for part, arr in (("r", (rows[lo:hi] - b * self.band_rows).astype(np.int32)), ("c", cols[lo:hi]),
                              ("v", values[lo:hi])):
                with open(self._path(b, part), "ab") as f:
                    arr.tofile(f)

    def bands(self) -> Iterator[sp.csr_matrix]:
        try:
            for b in range(self.n_bands):
                n = min(self.band_rows, self.n_rows - b * self.band_rows)
                if not self._path(b, "r").exists():
                    yield sp.csr_matrix((n, self.n_cols), dtype=self.dtype)
                    continue
                r = np.fromfile(self._path(b, "r"), dtype=np.int32)
                c = np.fromfile(self._path(b, "c"), dtype=np.int32)
                v = np.fromfile(self._path(b, "v"), dtype=self.dtype)
                for part in "rcv":
                    self._path(b, part).unlink()
                band = sp.csr_matrix((v, (r, c)), shape=(n, self.n_cols))
                band.sum_duplicates()
                band.sort_indices()
                yield band
        finally:
            shutil.rmtree(self.folder, ignore_errors=True)


ANNDATA_ATTRS = {"encoding-type": "anndata", "encoding-version": "0.1.0"}
OBS_CHUNK = 1 << 16                       # zarr chunk of obs arrays (rows): appends touch few files
FLAT_CHUNK = 1 << 20                      # zarr chunk of X's data/indices (elements)
INDPTR_CHUNK = 1 << 20                    # zarr chunk of X's indptr
# An array keeps its chunk length for life, so growing it only appends: a half-synced S3 store still reads.


def column_length(node) -> int:
    if isinstance(node, zarr.Array):
        return node.shape[0]
    return node["values" if "values" in node else "codes"].shape[0]


def n_obs(table: zarr.Group) -> int:
    obs = table["obs"]
    return column_length(obs[obs.attrs["_index"]])


class TableWriter:
    """An AnnData zarr group written row chunk by row chunk; `close` writes the int64 indptr."""

    def __init__(self, path: Path, obs: pd.DataFrame, var: pd.DataFrame, dtype, *,
                 attrs: dict | None = None, uns: dict | None = None, obsm: dict | None = None):
        harden_local_writes()
        self.path, self.n_obs, self.n_vars = Path(path), len(obs), len(var)
        shutil.rmtree(self.path, ignore_errors=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        root = zarr.open_group(self.path, mode="w", zarr_format=3)
        root.attrs.update({**ANNDATA_ATTRS, **(attrs or {})})
        write_elem(root, "obs", obs, dataset_kwargs={"chunks": (OBS_CHUNK,)})
        write_elem(root, "var", var)
        for key, value in (("uns", uns), ("obsm", obsm), ("varm", None), ("obsp", None), ("varp", None),
                           ("layers", None)):
            write_elem(root, key, value or {})
        self.x = root.create_group("X")
        self.x.attrs.update({**CSR_ATTRS, "shape": [self.n_obs, self.n_vars]})
        self.data = self.x.create_array("data", shape=(0,), dtype=np.dtype(dtype), chunks=(FLAT_CHUNK,))
        self.indices = self.x.create_array("indices", shape=(0,), dtype=np.int32, chunks=(FLAT_CHUNK,))
        self.row_nnz: list[np.ndarray] = []
        self.buffer: list[tuple[np.ndarray, np.ndarray]] = []
        self.buffered = 0

    def add(self, chunk: sp.csr_matrix) -> None:
        if chunk.shape[1] != self.n_vars:
            raise ValueError(f"chunk has {chunk.shape[1]} columns, the table {self.n_vars}")
        chunk = sp.csr_matrix(chunk)
        chunk.sort_indices()
        self.row_nnz.append(np.diff(chunk.indptr).astype(np.int64))
        self.buffer.append((np.asarray(chunk.data, dtype=self.data.dtype), chunk.indices.astype(np.int32)))
        self.buffered += chunk.nnz
        if self.buffered >= FLAT_CHUNK:
            self._flush(final=False)

    def _flush(self, final: bool) -> None:
        if not self.buffer:
            return
        data = np.concatenate([d for d, _ in self.buffer])
        indices = np.concatenate([i for _, i in self.buffer])
        cut = len(data) if final else len(data) - len(data) % FLAT_CHUNK
        if cut:
            self.data.append(data[:cut])
            self.indices.append(indices[:cut])
        self.buffer = [(data[cut:], indices[cut:])] if cut < len(data) else []
        self.buffered = len(data) - cut

    def close(self) -> Path:
        self._flush(final=True)
        counts = np.concatenate(self.row_nnz) if self.row_nnz else np.zeros(0, dtype=np.int64)
        if len(counts) != self.n_obs:
            raise ValueError(f"{self.path}: {len(counts)} rows written, obs has {self.n_obs}")
        indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.x.create_array("indptr", data=indptr, chunks=(INDPTR_CHUNK,))
        return self.path


def harden_local_writes() -> None:
    """Makes zarr's chunk renames survive an NFS RENAME that is replayed after it already succeeded."""
    from zarr.storage import _local

    if getattr(_local._move_with_retry, "spatial_ingest", False):
        return
    original = _local._move_with_retry

    def move_with_retry(tmp_path, path, move):
        def checked(src, dst):
            try:
                return move(src, dst)
            except FileNotFoundError:
                if Path(dst).exists() and not Path(src).exists():
                    logger.debug("rename %s -> %s reported ENOENT but is done (replayed on NFS)", src, dst)
                    return None
                raise
        return original(tmp_path, path, checked)

    move_with_retry.spatial_ingest = True
    _local._move_with_retry = move_with_retry


X_DTYPE = np.float32
POSITION = ("x_centroid", "y_centroid")
ATERA_POSITION = ("centroid_column", "centroid_row")
CELL_ID = "cell_id"
EXPLORER_TYPES = {"gene": GEX}
UNASSIGNED = {"UNASSIGNED", "-1", "", "nan", "None", "0"}
DEFAULT_PIXEL_SIZE = 0.2125


class TableError(ValueError):
    """A count matrix or cells table that can't be read as it is."""


def is_gzip(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(2) == b"\x1f\x8b"


def open_text(path: Path):
    """The file's bytes, gunzipped when it is gzip (whatever its name says)."""
    return gzip.open(path, "rb") if is_gzip(path) else open(path, "rb")


def plain(path: Path, scratch: Path) -> Path:
    """`path`, or a gunzipped copy on scratch (parquet needs a seekable file)."""
    if not is_gzip(path):
        return path
    target = scratch / "gunzip" / (path.name[:-3] if path.name.endswith(".gz") else path.name)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".partial")
        with gzip.open(path, "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 24)
        tmp.rename(target)
    return target


def text(values) -> np.ndarray:
    """Strings from str, bytes or number values."""
    arr = np.asarray(values)
    if arr.dtype.kind == "S":
        return np.char.decode(arr, "utf-8").astype(object)
    if arr.dtype == object:
        return np.array([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in arr], dtype=object)
    return arr.astype(str).astype(object)


def cell_keys(values) -> np.ndarray:
    """Integer IDs stay int64 (spatialdata joins on them are fast), anything else becomes text."""
    arr = np.asarray(values)
    if arr.dtype.kind in "iu":
        return arr.astype(np.int64)
    return text(arr)


def is_parquet(path: Path) -> bool:
    with open_text(path) as f:
        return f.read(4) == b"PAR1"


def read_frame(path: Path, scratch: Path, columns: list[str] | None = None, nrows: int | None = None) -> pd.DataFrame:
    """A parquet or csv(.gz) table, optionally only some columns or the first rows."""
    if is_parquet(path):
        import pyarrow.parquet as pq

        with pq.ParquetFile(plain(path, scratch)) as pf:          # closed: no handle left on staged files
            if nrows is None:
                return pf.read(columns=columns).to_pandas()
            batches, n = [], 0
            for batch in pf.iter_batches(batch_size=min(nrows, 1 << 20), columns=columns):
                batches.append(batch.to_pandas())
                n += len(batches[-1])
                if n >= nrows:
                    break
            return pd.concat(batches, ignore_index=True) if batches else pf.schema_arrow.empty_table().to_pandas()
    with open_text(path) as f:
        return pd.read_csv(f, usecols=columns, nrows=nrows)


def frame_columns(path: Path, scratch: Path) -> list[str]:
    if is_parquet(path):
        import pyarrow.parquet as pq

        return list(pq.read_schema(plain(path, scratch)).names)
    with open_text(path) as f:
        return list(pd.read_csv(f, nrows=0).columns)


@dataclass
class MatrixFile:
    """A count matrix as stored, read in row bands."""
    kind: str
    barcodes: pd.Index
    var: pd.DataFrame
    nnz: int
    _bands: Callable[[Path], Iterator[sp.csr_matrix]] = field(repr=False)
    obs: pd.DataFrame | None = None          # Atera: the matrix's own cell table
    obsm: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.barcodes), len(self.var)

    def bands(self, scratch: Path) -> Iterator[sp.csr_matrix]:
        return self._bands(scratch)


def _h5(path: Path) -> MatrixFile:
    import h5py

    with h5py.File(path, "r") as f:
        if "matrix" not in f:
            raise TableError(f"{path.name} is not a 10x feature-barcode matrix (no 'matrix' group).")
        m = f["matrix"]
        feat = m["features"]
        var = pd.DataFrame({"gene_ids": text(feat["id"][:]), "feature_types": text(feat["feature_type"][:])},
                           index=pd.Index(text(feat["name"][:])))
        if "genome" in feat:
            var["genome"] = text(feat["genome"][:])
        barcodes = pd.Index(text(m["barcodes"][:]))
        n_features, n_cells = (int(v) for v in m["shape"][:])
        nnz = int(m["indptr"][-1])
    if (n_cells, n_features) != (len(barcodes), len(var)):
        raise TableError(f"{path.name}: shape {n_features}×{n_cells} doesn't match its barcodes/features.")

    def bands(_: Path) -> Iterator[sp.csr_matrix]:
        with h5py.File(path, "r") as f:
            m = f["matrix"]
            indptr = m["indptr"][:].astype(np.int64)
            for start, stop in rows_for(indptr):
                a, b = int(indptr[start]), int(indptr[stop])
                yield sp.csr_matrix((m["data"][a:b], m["indices"][a:b], indptr[start:stop + 1] - a),
                                    shape=(stop - start, n_features))
    return MatrixFile("h5", barcodes, var, nnz, bands)


def _mtx_header(path: Path) -> tuple[int, int, int, int, str]:
    """(rows, cols, nnz, header lines, field) of a MatrixMarket coordinate file."""
    with open_text(path) as f:
        first = f.readline().decode().strip().lower()
        if not first.startswith("%%matrixmarket matrix coordinate"):
            raise TableError(f"{path.name} is not a MatrixMarket coordinate matrix (header: {first[:60]!r}).")
        n_lines = 1
        line = f.readline()
        while line.startswith(b"%"):
            n_lines += 1
            line = f.readline()
        n_lines += 1
        r, c, nnz = (int(v) for v in line.split()[:3])
    return r, c, nnz, n_lines, first.split()[3]


def _mtx(path: Path, features: Path, barcodes: Path) -> MatrixFile:
    with open_text(features) as f:
        feats = pd.read_csv(f, sep="\t", header=None, dtype=str, keep_default_na=False)
    with open_text(barcodes) as f:
        cells = pd.Index(pd.read_csv(f, sep="\t", header=None, dtype=str, keep_default_na=False)[0].to_numpy(object))
    name = feats[1] if feats.shape[1] > 1 else feats[0]
    var = pd.DataFrame({"gene_ids": feats[0].to_numpy(object),
                        "feature_types": feats[2].to_numpy(object) if feats.shape[1] > 2 else GEX},
                       index=pd.Index(name.to_numpy(object)))
    r, c, nnz, skip, kind = _mtx_header(path)
    if (r, c) == (len(var), len(cells)):
        by_feature = True
    elif (r, c) == (len(cells), len(var)):
        by_feature = False
    else:
        raise TableError(f"{path.name} is {r}×{c}, which fits neither {len(var)} features × {len(cells)} barcodes "
                         "nor the transpose: the MTX triplet's files don't belong together.")
    if kind not in ("integer", "real"):
        raise TableError(f"{path.name} holds {kind} values; a count matrix is integer or real.")

    def bands(scratch: Path) -> Iterator[sp.csr_matrix]:
        buckets = RowBuckets(scratch / "mtx_buckets", len(cells), len(var), nnz, X_DTYPE)
        with open_text(path) as f:
            reader = pd.read_csv(f, sep=r"\s+", header=None, skiprows=skip, chunksize=5_000_000,
                                 dtype={0: np.int64, 1: np.int64, 2: np.float64}, engine="c")
            for chunk in reader:
                i, j, v = chunk[0].to_numpy() - 1, chunk[1].to_numpy() - 1, chunk[2].to_numpy()
                buckets.add(j if by_feature else i, i if by_feature else j, v)
        yield from buckets.bands()
    return MatrixFile("mtx", cells, var, nnz, bands)


def _zip_group(path: Path) -> tuple[zarr.Group, object]:
    store = zarr.storage.ZipStore(path, mode="r", read_only=True)
    return zarr.open_group(store, mode="r"), store


def is_explorer_matrix(path: Path) -> bool:
    group, store = _zip_group(path)
    try:
        return "cell_features" in group
    finally:
        store.close()


def _explorer(path: Path) -> MatrixFile:
    """A Xenium Explorer cell_feature_matrix.zarr.zip."""
    from spatialdata_io.readers.xenium import cell_id_str_from_prefix_suffix_uint32

    group, store = _zip_group(path)
    try:
        cf = group["cell_features"]
        attrs = dict(cf.attrs)
        raw_ids = cf["cell_id"][:]
        n_cells, n_features = int(attrs["number_cells"]), int(attrs["number_features"])
        cell_major = "csc" in cf and cf["csc"]["indptr"].shape[0] == n_cells + 1
        nnz = int(cf["indptr"][-1])
    finally:
        store.close()
    barcodes = pd.Index(text(cell_id_str_from_prefix_suffix_uint32(raw_ids[:, 0], raw_ids[:, 1]))) \
        if raw_ids.ndim == 2 else pd.Index(text(raw_ids))
    types = [EXPLORER_TYPES.get(t, t) for t in attrs["feature_types"]]
    var = pd.DataFrame({"gene_ids": np.asarray(attrs["feature_ids"], dtype=object), "feature_types": types},
                       index=pd.Index(np.asarray(attrs["feature_keys"], dtype=object)))

    def bands(scratch: Path) -> Iterator[sp.csr_matrix]:
        group, store = _zip_group(path)
        try:
            cf = group["cell_features"]
            if cell_major:
                yield from (band for _, band in iter_rows(_CSR(cf["csc"], n_cells, n_features)))
                return
            buckets = RowBuckets(scratch / "explorer_buckets", n_cells, n_features, nnz, X_DTYPE)
            indptr = cf["indptr"][:].astype(np.int64)
            for f0, f1 in rows_for(indptr):
                a, b = int(indptr[f0]), int(indptr[f1])
                counts = np.diff(indptr[f0:f1 + 1])
                feature = np.repeat(np.arange(f0, f1, dtype=np.int64), counts)
                buckets.add(cf["indices"][a:b], feature, cf["data"][a:b])
            yield from buckets.bands()
        finally:
            store.close()
    return MatrixFile("xenium_zarr", barcodes, var, nnz, bands)


class _CSR:
    def __init__(self, group: zarr.Group, n_rows: int, n_cols: int):
        self._group, self.attrs = group, {**CSR_ATTRS, "shape": [n_rows, n_cols]}

    def __getitem__(self, key: str):
        return self._group[key]

    def __contains__(self, key: str) -> bool:
        return key in self._group


def _atera(path: Path) -> MatrixFile:
    from anndata.io import read_elem

    from vendor._atera_common import _patched_ragged_vlen_chunk_decode
    from vendor.atera import _add_label_id, _read_dataframe

    group, store = _zip_group(path)
    try:
        with _patched_ragged_vlen_chunk_decode():
            var = _read_dataframe(group, "var")
            obs = _read_dataframe(group, "obs")
            obsm = dict(read_elem(group["obsm"])) if "obsm" in group else {}
        _add_label_id(obs)
        x = group["X"]
        if x.attrs.get("encoding-type") != "csr_matrix":
            raise TableError(f"{path.name}: X is {x.attrs.get('encoding-type')}, expected csr_matrix.")
        nnz = int(x["indptr"][-1])
    finally:
        store.close()

    def bands(_: Path) -> Iterator[sp.csr_matrix]:
        group, store = _zip_group(path)
        try:
            for _, band in iter_rows(group["X"]):
                yield band
        finally:
            store.close()
    return MatrixFile("atera_zarr", pd.Index(obs.index.astype(str)), var, nnz, bands, obs=obs, obsm=obsm)


def open_matrix(files: dict[str, Path], kind: str) -> MatrixFile:
    path = files["matrix"]
    name = path.name.lower()
    try:
        if name.endswith(".h5"):
            return _h5(path)
        if name.endswith((".mtx", ".mtx.gz")):
            return _mtx(path, files["matrix_features"], files["matrix_barcodes"])
        if name.endswith(".zarr.zip"):
            return _explorer(path) if is_explorer_matrix(path) else _atera(path)
    except TableError:
        raise
    except (OSError, KeyError, ValueError) as exc:
        raise TableError(f"{path.name} can't be read as a count matrix ({exc}).") from exc
    raise TableError(f"{path.name}: unknown count matrix format.")


def read_cells(path: Path, scratch: Path) -> pd.DataFrame:
    cells = read_frame(path, scratch)
    if CELL_ID not in cells:
        raise TableError(f"{path.name} has no cell_id column.")
    cells[CELL_ID] = cell_keys(cells[CELL_ID].to_numpy())
    return cells


def xenium_cells_zarr(path: Path) -> pd.DataFrame:
    """Cell IDs and `cell_summary` of a Xenium cells.zarr.zip, named like cells.parquet."""
    from spatialdata_io.readers.xenium import cell_id_str_from_prefix_suffix_uint32

    group, store = _zip_group(path)
    try:
        raw = group["cell_id"][:]
        summary = group["cell_summary"][:]
        names = list(group["cell_summary"].attrs["column_names"])
    finally:
        store.close()
    if raw.ndim != 2:
        raise TableError(f"{path.name} stores integer cell IDs (XOA < 1.3) that can't be matched to the matrix.")
    df = pd.DataFrame(summary, columns=names).rename(columns={"cell_centroid_x": "x_centroid",
                                                              "cell_centroid_y": "y_centroid"})
    df.insert(0, CELL_ID, text(cell_id_str_from_prefix_suffix_uint32(raw[:, 0], raw[:, 1])))
    return df


def new_table_group(out: Path) -> zarr.Group:
    shutil.rmtree(out, ignore_errors=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(out), mode="w", zarr_format=3)
    group.attrs.update(ANNDATA_ATTRS)
    return group


def dropped_features(var: pd.DataFrame, keep: np.ndarray) -> str:
    types = var["feature_types"] if "feature_types" in var else var.get("feature_type")
    if types is None or keep.all():
        return ""
    dropped = np.asarray(types)[~keep]
    counts = pd.Series(dropped).value_counts()
    protein = [t for t in counts.index if "protein" in str(t).lower()]
    parts = []
    for t in protein:                                   # protein first, by name: easy to miss among controls
        names = list(np.asarray(var.index)[~keep][dropped == t])
        parts.append(f"{counts[t]} {t} ({', '.join(map(str, names[:10]))}{', …' if len(names) > 10 else ''})")
    parts += [f"{n} {t}" for t, n in counts.items() if t not in protein]
    return ", ".join(parts)


def write_source(m: MatrixFile, rows: np.ndarray | None, obs: pd.DataFrame, obsm: dict, sid: str,
                 out: Path, scratch: Path) -> TableSource:
    """The matrix's rows (None: all) and Gene Expression columns, streamed into an AnnData zarr at `out`."""
    keep, var = symbol_var(m.var, sid)
    dropped = dropped_features(m.var, keep)
    if dropped:
        logger.info("%s: only Gene Expression features kept; left out: %s", sid, dropped)
    if not keep.any():
        raise TableError(f"{sid}: the count matrix has no Gene Expression features.")
    columns = np.flatnonzero(keep)
    group = new_table_group(out)
    row_mask = None
    if rows is not None:
        row_mask = np.zeros(m.shape[0], dtype=bool)
        row_mask[rows] = True
        if row_mask.all():
            row_mask = None

    def bands() -> Iterator[sp.csr_matrix]:
        start = 0
        for band in m.bands(scratch):
            stop = start + band.shape[0]
            if row_mask is not None:
                band = band[row_mask[start:stop]]
            start = stop
            yield band[:, columns] if len(columns) < m.shape[1] else band
        if start != m.shape[0]:
            raise TableError(f"{sid}: the count matrix ended after {start} of {m.shape[0]} cells.")

    n_obs, _ = write_csr(group, bands(), len(columns), X_DTYPE)
    if n_obs != len(obs):
        raise TableError(f"{sid}: {n_obs} matrix rows written but obs has {len(obs)} rows.")
    write_elem(group, "obs", obs)
    write_elem(group, "var", var)
    write_elem(group, "obsm", {k: np.asarray(v) for k, v in obsm.items()})
    return TableSource(out, obs, n_obs, len(columns))


HAS_POSITION = "has_position"


def _barcode_keys(barcodes: pd.Index, like: pd.Series) -> np.ndarray:
    if pd.api.types.is_integer_dtype(like):
        try:
            return barcodes.astype(np.int64).to_numpy()
        except (TypeError, ValueError):
            pass
    return np.asarray(barcodes.astype(str), dtype=object)


def align(barcodes: pd.Index, cells: pd.DataFrame, id_column: str = CELL_ID) -> tuple[pd.DataFrame, int]:
    """One obs row per matrix barcode; returns the obs and how many barcodes had no row in the cells table."""
    ids = pd.Index(cells[id_column].astype(str))
    if not ids.is_unique:
        cells = cells.loc[~ids.duplicated()]
        ids = pd.Index(cells[id_column].astype(str))
    where = ids.get_indexer(barcodes.astype(str))
    found = where >= 0
    obs = cells.iloc[np.where(found, where, 0)].reset_index(drop=True)
    if not found.all():
        obs = obs.astype({c: "float64" for c in obs.columns
                          if pd.api.types.is_integer_dtype(obs[c]) or pd.api.types.is_bool_dtype(obs[c])})
        obs.loc[~found, [c for c in obs.columns if c != id_column]] = np.nan
    # matched rows have cell_id == barcode, so the barcode (in the table's key type) is every row's cell ID
    obs[id_column] = _barcode_keys(barcodes, cells[id_column])
    obs.index = pd.Index(barcodes.astype(str))
    return obs, int((~found).sum())


def xenium_table(m: MatrixFile, cells: pd.DataFrame, sid: str, out: Path, scratch: Path,
                 notes: list[str]) -> TableSource:
    """Every matrix cell is kept; cells without a position get NaN centroids and has_position False."""
    missing = [c for c in POSITION if c not in cells]
    if missing:
        raise TableError(f"{sid}: the cells table has no {', '.join(missing)} column.")
    obs, _ = align(m.barcodes, cells)
    spatial = obs[list(POSITION)].to_numpy(dtype=np.float64)
    placed = np.isfinite(spatial).all(axis=1)
    if not placed.any():
        raise TableError(f"{sid}: no cell of the count matrix has a position in the cells table (the files "
                         "belong to different cell sets).")
    obs = obs.drop(columns=list(POSITION))
    obs[HAS_POSITION] = placed
    return write_source(m, None, obs, {"spatial": spatial}, sid, out, scratch)


def atera_table(m: MatrixFile, sid: str, out: Path, scratch: Path) -> TableSource:
    obs = m.obs
    missing = [c for c in ATERA_POSITION if c not in obs]
    if missing:
        raise TableError(f"{sid}: the matrix obs has no {', '.join(missing)} column (cell positions).")
    obsm = dict(m.obsm)
    obsm["spatial"] = obs[list(ATERA_POSITION)].to_numpy(dtype=np.float64)
    obs = obs.copy()
    obs[HAS_POSITION] = np.isfinite(obsm["spatial"]).all(axis=1)
    return write_source(m, None, obs, obsm, sid, out, scratch)
