"""Finds the samples in the user's inputs: every listed path (folder, archive, file, glob) becomes Xenium
bundles, Atera bundles and SpatialData objects (`contracts.Candidate`), plus a list of what was left out and why
(`contracts.LeftOut`). Nothing is extracted or read beyond file listings and a few small zarr metadata files.

What counts as a sample
- A folder holding `experiment.xenium` (Xenium) or `experiment.spatial` (Atera), plain or `.gz`: the whole
  folder is one bundle. A folder whose marker stands alone (no data files beside it) but holds bundles further
  down uses those bundles instead (a download folder with the marker fetched separately).
- A folder of loose files without a marker: files are grouped by the prefix in front of a known 10x name
  (`<dataset>_experiment.xenium`, `GSM123_<title>_cells.parquet.gz`, `MB263-cells.parquet`, or no prefix at all);
  each group with a count matrix, or with a marker and data, is one sample. Several prefixes are several samples.
  Without a marker the kind is inferred from the files (Atera-only files -> Atera, else Xenium).
- A zarr group that is a SpatialData object (root `spatialdata_attrs`, or element groups `images|labels|shapes|
  points|tables|table` as old stores have them), as a folder or inside a `.zip`/`.zarr.zip`.
- `.zip`, `.tar`, `.tar.gz`, `.tgz` archives are listed (not extracted) and searched the same way, at any depth up
  to MAX_DEPTH; every bundle root in an archive is its own sample.

Junk (macOS `__MACOSX/`, `._*`, `.DS_Store`, Xenium Ranger pipestance internals, ...) is skipped silently.
Everything else that is not a sample is left out with a reason; only when NO sample is found is that an error.

Candidate.files maps canonical 10x names (prefix stripped; `.gz` noted in Member.gz where the readers need the
file decompressed) to where the files really are. See `canonical_key` for the naming rules."""
from __future__ import annotations

import glob
import json
import logging
import os
import re
import tarfile
import zipfile
from zlib import error as zlib_error
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import config
from contracts import NAMES, Candidate, Discovery, LeftOut, Member

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
BROWSER_COPY = re.compile(r"^(?P<stem>.+?) \((?P<n>\d+)\)(?P<ext>(\.[A-Za-z0-9]+)+)$")   # "cells (2).parquet"


class InputError(ValueError):
    """Inputs that can't be used as given (missing path, duplicate sample IDs, nothing usable at all)."""


# ---------------------------------------------------------------------------------------------------- names

def is_junk(name: str) -> bool:
    """macOS/Windows litter, hidden files, and Xenium/Space Ranger (martian) pipestance internals."""
    return (name in JUNK or name.startswith(".") or name.startswith("_") or bool(RANGER_DIR.match(name))
            or name.endswith(".mro") or name.endswith(".mri.tgz"))


def is_archive(name: str) -> bool:
    return name.lower().endswith(ARCHIVES)


def archive_stem(name: str) -> str:
    """`X_outs.zip` -> `X_outs`, `a.tar.gz` -> `a`, `s.zarr.zip` -> `s.zarr` (cleaned further by `clean_name`)."""
    lower = name.lower()
    for suffix in ARCHIVES:
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return name


def canonical_key(canon: str, gz: bool) -> tuple[str, bool]:
    """Canonical key and whether the file must be gunzipped when staged. Files 10x itself ships gzipped
    (`*.csv.gz`, MEX `*.tsv.gz`/`matrix.mtx.gz`) keep `.gz` in their name; other `.gz` (GEO's per-file gzip of
    `.parquet`, `.h5`, `.json`, `.ome.tif`, `experiment.xenium`) are named without it and marked gz."""
    name = canon + ".gz" if gz and canon.endswith((".csv", ".tsv", ".mtx")) else canon
    gz = gz and not name.endswith(".gz")
    if canon in MTX:
        name = "cell_feature_matrix/" + name
    elif canon.startswith("morphology_focus_"):
        name = "morphology_focus/" + name
    return name, gz


def plain_key(name: str) -> tuple[str, bool]:
    """A file name without known meaning: `.gz` dropped (and marked) unless it is read gzipped (see KEEP_GZ)."""
    if name.endswith(".gz") and not name.endswith(KEEP_GZ):
        return name[:-3], True
    return name, False


def clean_name(name: str) -> str:
    """A folder or archive name as a sample-ID candidate: `.zarr`/`.sdata` and a trailing `_outs` dropped."""
    for suffix in (".zarr", ".sdata"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    return re.sub(r"[_\- ]outs$", "", name, flags=re.IGNORECASE)


def name_id(names: Iterable[str]) -> str:
    """The first meaningful name walking up from a sample's root (generic names like `outs`, `data` skipped)."""
    for name in names:
        name = clean_name(name)
        if name and name.lower() not in GENERIC and not GENERIC_RE.match(name):
            return name
    return "sample"


def sanitize(sid: str) -> tuple[str, str | None]:
    """A valid SpatialData element-name ID, and a note when characters had to be replaced."""
    if SAMPLE_ID.match(sid):
        return sid, None
    fixed = re.sub(r"[^A-Za-z0-9_.-]+", "_", sid).strip("_") or "sample"
    return fixed, f"sample ID {sid!r} became {fixed!r} (only letters, digits, '_', '.' and '-' are allowed)"


# ------------------------------------------------------------------------------------------ file trees

class Node:
    """A file or folder, on disk or inside an archive: the one interface the search walks over."""
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
        """Names from this node upward (for sample IDs): inner folders, the archive's stem, disk folders."""
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
    """An archive's member listing as a folder tree (read once; tar.gz listing reads the whole stream). Small
    zarr metadata files are kept from that one pass so SpatialData roots can be recognised without re-reading."""

    def __init__(self, path: Path):
        self.path = path
        self.dirs: dict[str, dict[str, tuple[bool, int]]] = {"": {}}
        self.meta: dict[str, bytes] = {}
        self.zip = zipfile.is_zipfile(path)
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


# ------------------------------------------------------------------------------------------ zarr / SpatialData

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
    """"spatialdata" for a SpatialData store, a left-out reason for another zarr, None for no zarr at all."""
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
        return "an AnnData .zarr (a table with no spatial elements), not a SpatialData object"
    if ".zarray" in kids or v3.get("node_type") == "array":
        return "a zarr array, not a SpatialData object"
    return "a zarr store without SpatialData elements (no images/labels/points/shapes/tables groups)"


# ------------------------------------------------------------------------------------------ grouping loose files

@dataclass
class Group:
    """Loose files sharing one prefix in front of known 10x names (prefix "" = unprefixed)."""
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
    """A sample: a marker with some data beside it, or a count matrix with a cells table (even an empty one).
    A lone matrix (or lone cells table) is left out: scattered single files are not samples."""
    if _marker_kind(keys) is not None:
        return _has_data(keys)
    return _has_matrix(keys) and bool(keys & CELLS_KEYS)


def infer_kind(keys: Iterable[str]) -> str:
    keys = set(keys)
    if keys & ATERA_ONLY or any(k.startswith(("morphology_2d/", "morphology_3d/")) for k in keys):
        return "atera"
    return "xenium"


def split_name(name: str) -> tuple[str, str, str, bool] | None:
    """(normalised prefix, raw prefix, canonical key, gz) for a loose file with a known 10x name, else None."""
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
    """The longest sample prefix `name` starts with (followed by a separator), and the rest of the name."""
    for prefix in sorted(prefixes, key=len, reverse=True):
        if prefix and name.startswith(prefix) and len(name) > len(prefix) and name[len(prefix)] in SEPARATORS:
            rest = name[len(prefix):].lstrip(SEPARATORS)
            if rest:
                return prefix, rest
    return None


# ------------------------------------------------------------------------------------------ the search

@dataclass
class Found:
    candidates: list[tuple[Candidate, str]] = field(default_factory=list)   # (candidate, root identity)
    left_out: list[LeftOut] = field(default_factory=list)

    def extend(self, other: "Found") -> None:
        self.candidates += other.candidates
        self.left_out += other.left_out


def _files_under(node: Node, rel: str = "", depth: int = 0) -> Iterator[tuple[str, Node]]:
    for child in node.children():
        if is_junk(child.name):
            continue
        path = f"{rel}{child.name}"
        if child.is_dir:
            if depth < MAX_DEPTH:
                yield from _files_under(child, path + "/", depth + 1)
        else:
            yield path, child


def bundle_files(node: Node, skip: set[str] = frozenset()) -> tuple[dict[str, Member], int]:
    """Every file of a bundle folder under its canonical key (top-level names in `skip` left out: they belong
    to other samples), and how many need gunzipping."""
    group = Group("")
    for rel, child in _files_under(node):
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
    """One discovery run: walks each entry, collecting candidates and left-out items."""

    def __init__(self):
        self.archives: dict[Path, ArchiveIndex] = {}

    # -- helpers
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

    # -- folders
    def scan(self, node: Node, depth: int = 0, recurse: bool = True, archives: bool = True) -> Found:
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
            found.candidates.append(self._folder_bundle(node, taken))
            for group in others:
                found.candidates.append(self._group_bundle(node, group, []))
            return found
        return self._loose(node, groups, unknown, dirs, depth, recurse, archives)

    def _marker_folder(self, node: Node, plain: Group, dirs: list[Node], depth: int, recurse: bool,
                       archives: bool) -> Found:
        """A folder with experiment.xenium/.spatial: one bundle, unless the marker stands alone above bundles."""
        found = Found()
        own = _has_data(plain.keys) or any(d.name in DATA_DIRS for d in dirs)
        if own:
            found.candidates.append(self._folder_bundle(node, set()))
            return found
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

    def _folder_bundle(self, node: Node, taken: set[str]) -> tuple[Candidate, str]:
        files, n_gz = bundle_files(node, taken)
        kind = _marker_kind(files) or infer_kind(files)
        notes = []
        if not _marker_kind(files):
            notes.append(f"no experiment file: kind inferred from the files ({NAMES[kind]})")
        if n_gz:
            notes.append(f"{n_gz} gzipped file(s) are decompressed when staged")
        return self._candidate(node, kind, name_id(node.up_names()), files, notes)

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

    # -- archives
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
        if path.name.lower().endswith(".zip") and not zipfile.is_zipfile(path):
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
            # one line for the archive, not one per member; a lone zarr's verdict says the most
            reasons = {lo.reason for lo in inner.left_out}
            zarr = len(reasons) == 1 and "zarr" in next(iter(reasons))
            found.left_out.append(LeftOut(node.display, next(iter(reasons)) if zarr else
                                          "holds no Xenium/Atera bundle and no SpatialData object"))
            return found
        roots = [c for c, _ in inner.candidates]
        if len(roots) > 1:
            logger.info("%s holds %d samples: %s", node.display, len(roots), ", ".join(c.sid for c in roots))
        found.extend(inner)
        return found


def _sidecars(found: Found, loose: list[Node]) -> list[tuple[str, set[str], str]]:
    """GEO-style extra files beside an archive that held a sample, named with the archive's prefix
    (`GSM…_Rep1_gene_panel.json.gz` next to `GSM…_Rep1_outs.zip`): one line instead of one per file."""
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
    """A reason when a folder holds browser-renamed copies of 10x files (`cells (2).parquet`): several samples'
    files saved into one folder under the same names can't be told apart."""
    copies = [f.name for f in files if (m := BROWSER_COPY.match(f.name)) and split_name(m["stem"] + m["ext"])]
    if not copies:
        return None
    shown = ", ".join(sorted(copies)[:3])
    return (f"holds {len(copies)} browser-renamed copies of 10x file names ({shown} …): the files of several "
            "samples were saved into one folder and can't be told apart; put each sample's files in its own folder")


def _summarise(node: Node, loose: list[Node], groups: dict[str, Group], samples: list[str]) -> list[LeftOut]:
    """Left-out loose files of a folder: incomplete prefix groups with what they lack, other files listed
    (summarised past MAX_LISTED)."""
    out: list[LeftOut] = []
    names = {n.name for n in loose}
    for prefix, group in sorted(groups.items()):
        if prefix in samples:
            continue
        mine = [n.name for n, _ in group.files.values() if n.name in names]
        if not mine:
            continue
        names -= set(mine)
        keys = ", ".join(sorted(group.keys))
        if _marker_kind(group.keys):
            what = "no data files beside the experiment file"
        elif _has_matrix(group.keys):
            what = "no cells table (cells.parquet / cells.csv.gz) and no experiment.xenium"
        else:
            what = "no count matrix (cell_feature_matrix.h5 / .zarr.zip or matrix.mtx) and no experiment.xenium"
        label = f"{node.display}/{group.raw and sorted(group.raw)[0] or ''}*"
        out.append(LeftOut(label, f"10x files ({keys}) but {what}"))
    rest = sorted(names)
    if len(rest) > MAX_LISTED:
        out.append(LeftOut(node.display, f"{len(rest)} files that belong to no sample (e.g. {', '.join(rest[:3])})"))
    else:
        out += [LeftOut(f"{node.display}/{n}", "not part of a bundle or SpatialData object") for n in rest]
    return out


# ------------------------------------------------------------------------------------------ entries

def _expand(entry: str) -> list[Path]:
    """The paths one list item names: `~` expanded, globs matched, relative paths also tried under /data."""
    text = os.path.expanduser(entry.strip())
    if glob.has_magic(text):
        matches = sorted(glob.glob(text))
        if not matches and not os.path.isabs(text):
            matches = sorted(glob.glob(str(config.DATA_DIR / text)))
        if not matches:
            raise InputError(f"Nothing matches {entry!r}" + ("" if os.path.isabs(text) else
                                                           f" (also looked under {config.DATA_DIR})") + ".")
        return [Path(m).absolute() for m in matches]
    path = config.in_data(Path(text))
    if not path.exists():
        tried = "" if path.is_absolute() else f" (also looked for {config.DATA_DIR / text})"
        raise InputError(f"Input not found: {entry}{tried}")
    return [path.absolute()]


def _inside(member: Member, path: Path) -> bool:
    return member.container == path or path in member.container.parents


def _holder(search: Search, path: Path) -> Found | None:
    """The bundle (up to 3 folders up) that contains a file or folder the user pointed at inside it."""
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
        return _holder(search, path) or Found(left_out=[LeftOut(
            str(path), "a single file that is not part of a bundle; give the folder that holds the sample's files")])
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
    """Drop repeats (the same root reached twice; an archive next to its own extracted copy) and refuse
    sample-ID clashes."""
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
    """The samples in the user's list of paths, in input order, and what was left out. Raises InputError for a
    missing path, a glob matching nothing, duplicate sample IDs, or when nothing usable was found at all.
    `scratch` is not written to (archives are only listed); it is accepted for the stage that follows."""
    if not entries:
        raise InputError("No inputs given: fill in the list of paths to samples.")
    search = Search()
    found = Found()
    seen: set[Path] = set()
    for entry in entries:
        for path in _expand(entry):
            if path in seen:
                continue
            seen.add(path)
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
