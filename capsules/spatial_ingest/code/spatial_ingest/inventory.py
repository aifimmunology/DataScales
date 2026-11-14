"""What a Xenium or Atera bundle holds, decided before anything is read in full: each file's role, which
essentials are missing (a count matrix; cell IDs with x/y positions that match the matrix), which optional
parts are missing, and the matrix's raw features. Only small reads: the matrix's barcodes and features,
ID columns, file headers, and the first rows of transcripts and boundaries.

Files are chosen by cell-ID consistency with the matrix, not by name: a transcripts or boundaries file is
taken only when (nearly) all its assigned cell IDs are matrix barcodes, so a canonical `transcripts.parquet`
next to a `reseg_transcripts.parquet` (GSE352490) can't pair a table with another segmentation's points."""
from __future__ import annotations

import json
import logging
import re
import tarfile
import zipfile
from pathlib import Path

import pandas as pd

import stage
from contracts import Candidate, Inventory, Member
from readers import table as tables

logger = logging.getLogger(__name__)

SAMPLE_ROWS = 1_000_000          # rows of transcripts / boundaries checked against the matrix barcodes
MIN_ASSIGNED_MATCH = 0.99        # share of a file's assigned cell IDs that must be matrix barcodes
MIN_PLACED = 0.5                 # share of matrix cells that must have a position in the cells table
DEFAULT_PIXEL_SIZE = 0.2125      # µm; every XOA output seen
MATRIX_TEXT = "count matrix (cell_feature_matrix.h5, cell_feature_matrix.zarr.zip or matrix.mtx + features.tsv + barcodes.tsv)"
CELLS_TEXT = "cell positions (cells.parquet or cells.csv with cell_id, x_centroid, y_centroid)"
TABULAR = (".parquet", ".csv", ".csv.gz", ".parquet.gz")


def stage_dir(scratch: Path, sid: str) -> Path:
    return scratch / "stage" / sid


def _base(key: str) -> str:
    return key[:-3] if key.endswith(".gz") and not key.endswith((".csv.gz", ".tsv.gz", ".mtx.gz")) else key


def is_empty(member: Member) -> bool:
    """A 0-byte file (GEO failed uploads) counts as missing."""
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
    """A candidate's non-empty files by canonical key, with lookups by name pattern."""

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
        """Keys named `<stem><ext>` first, then `<anything>[_-.]<stem><ext>` (e.g. reseg_transcripts), parquet
        before csv. Files inside folders other than the bundle root are not considered."""
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
        """matrix.mtx + features.tsv (or genes.tsv) + barcodes.tsv in one folder."""
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
    """Stages the few files it reads under `scratch/stage/<sid>/` (reused by `readers.load`)."""

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

    def path(self, key: str) -> Path:
        return stage.fetch(self.c, [key], self.dest)[key]

    def result(self) -> Inventory:
        if self.files.empty:
            self.notes.append(f"0-byte files ignored: {', '.join(self.files.empty)}")
        return Inventory(self.c.sid, self.c.kind, self.chosen, self.missing_essential, self.missing_optional,
                         self.var, self.n_cells, self.notes)

    # ---- matrix and cells

    def matrix(self) -> tables.MatrixFile | None:
        mtx = self.files.mtx()
        options = [{"matrix": k} for k in (self.files.first("cell_feature_matrix.h5"),) if k]
        options += [mtx] if mtx else []
        options += [{"matrix": k} for k in (self.files.first("cell_feature_matrix.zarr.zip"),) if k]
        errors = []
        for roles in options:
            try:
                m = tables.open_matrix({r: self.path(k) for r, k in roles.items()}, self.c.kind)
            except (tables.TableError, OSError) as exc:
                errors.append(str(exc))
                continue
            self.chosen.update(roles)
            self.var, self.barcodes = m.var, m.barcodes
            return m
        reason = f" ({'; '.join(errors)})" if errors else ""
        self.missing_essential.append(MATRIX_TEXT + reason)
        return None

    def cells(self, m: tables.MatrixFile) -> None:
        if m.kind == "atera_zarr":
            missing = [c for c in tables.ATERA_POSITION if c not in m.obs]
            if missing:
                self.missing_essential.append(f"cell positions ({', '.join(missing)} in the matrix obs)")
                return
            self.chosen["cells"] = self.chosen["matrix"]
            self.n_cells = len(m.barcodes)
            return
        options = self.files.variants("cells")
        zarr_key = self.files.first("cells.zarr.zip")
        reasons = []
        for key in options + ([zarr_key] if zarr_key else []):
            try:
                ids = self._cell_ids(key)
            except (tables.TableError, OSError, ValueError, KeyError) as exc:
                reasons.append(f"{key}: {exc}")
                continue
            placed = int(m.barcodes.isin(ids).sum())
            if placed < MIN_PLACED * len(m.barcodes):
                reasons.append(f"{key}: only {placed} of the matrix's {len(m.barcodes)} cells are in it")
                continue
            self.chosen["cells"] = key
            self.n_cells = placed
            if placed < len(m.barcodes):
                self.notes.append(f"{len(m.barcodes) - placed} of {len(m.barcodes)} matrix cells have no row in {key} "
                                  "and are left out")
            if not options:
                self.notes.append(f"cell positions read from {key} (no cells.parquet / cells.csv)")
            elif key != options[0]:
                self.notes.append(f"cell positions read from {key} (not {'; '.join(reasons)})")
            return
        why = f" ({'; '.join(reasons)})" if reasons else ""
        self.missing_essential.append(CELLS_TEXT + why)

    def _cell_ids(self, key: str) -> pd.Index:
        path = self.path(key)
        if key.endswith(".zarr.zip"):
            return pd.Index(tables.xenium_cells_zarr(path)[tables.CELL_ID])
        columns = tables.frame_columns(path, self.scratch)
        missing = [c for c in (tables.CELL_ID, *tables.POSITION) if c not in columns]
        if missing:
            raise tables.TableError(f"no {', '.join(missing)} column")
        ids = tables.read_frame(path, self.scratch, columns=[tables.CELL_ID])[tables.CELL_ID]
        return pd.Index(tables.text(ids.to_numpy()))

    # ---- optional parts

    def assigned_match(self, key: str) -> tuple[float, int]:
        """Share of the file's assigned cell IDs (first rows) that are matrix barcodes, and how many were checked."""
        path = self.path(key)
        ids = tables.read_frame(path, self.scratch, columns=[tables.CELL_ID], nrows=SAMPLE_ROWS)[tables.CELL_ID]
        ids = pd.Index(tables.text(ids.to_numpy()))
        ids = ids[~ids.isin(list(tables.UNASSIGNED))]
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
            except (tables.TableError, OSError, ValueError, KeyError) as exc:
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
        if stack:                                  # the z-stack (GEO's usual only image), read lazily as 3D
            self.chosen["morphology_3d"] = stack
            if "morphology" not in self.chosen:
                self.notes.append("image: the morphology.ome.tif z-stack only (no 2D focus/mip image)")
        elif "morphology" not in self.chosen:
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
        if f.folder("morphology_3d"):
            self.chosen["morphology_3d"] = "morphology_3d/"
        if "experiment" not in self.chosen and "morphology" not in self.chosen:
            self.notes.append("no experiment.spatial and no images: pixel size unknown")


def inspect(candidate: Candidate, scratch: Path) -> Inventory:
    """The bundle's inventory (see the module docstring). Never raises for a bundle's content: what's
    unusable ends up in `missing_essential` / `missing_optional` with the reason."""
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
    """One line for the per-sample warning: what's missing (optional) and the decisions taken."""
    parts = []
    if inv.missing_optional:
        parts.append("not available: " + "; ".join(inv.missing_optional))
    parts += inv.notes
    return ". ".join(parts)
