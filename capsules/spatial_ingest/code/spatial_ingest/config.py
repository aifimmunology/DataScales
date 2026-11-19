"""Run settings from the environment and the app panel, logging, and the records passed between steps."""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import pandas as pd

logger = logging.getLogger("spatial_ingest.config")


class UserError(Exception):
    """A failure the user can fix: logged as one line, exit code 1."""


PANEL = ("store", "samples", "celltypist_pkl")
OLD_PANEL = ("store", "input_type", "samples", "celltypist_pkl")
# Labels of the removed Input type field.
OLD_INPUT_TYPES = ("Xenium bundle (folder or .zip)", "Atera bundle (folder or .zip)", "Existing SpatialData (.zarr)")
DATA_DIR = Path("/data")
DEFAULT_SAMPLES = "/data/samples"
_BOOLS = {"1": True, "true": True, "yes": True, "y": True, "on": True,
          "0": False, "false": False, "no": False, "n": False, "off": False}
_LIST_SEP = re.compile(r"[,;\n]+")
_PAIRS = {("[", "]"), ("(", ")"), ('"', '"'), ("'", "'")}
IGNORED_NOTE = ("The Input type setting ({value!r}) is no longer needed and is ignored: each sample's kind is "
                "detected from its files.")


class ConfigError(UserError, ValueError):
    """A setting that can't be read as its type."""


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


_OLD_LABELS = {_key(label) for label in OLD_INPUT_TYPES}


def is_old_input_type(text: str) -> bool:
    """A label of the removed Input type field (a bare `xenium` could be a path, so it isn't one)."""
    return _key(text) in _OLD_LABELS


def unwrap(text: str) -> str:
    """Strips spaces and wrapping quotes/brackets, and stray ones left by splitting a list."""
    text, previous = text.strip(), None
    while text != previous:
        previous = text
        if len(text) >= 2 and (text[0], text[-1]) in _PAIRS:
            text = text[1:-1].strip()
        for opener, closer in (("[", "]"), ("(", ")")):     # unbalanced: left over from a split list
            if text.startswith(opener) and text.count(opener) > text.count(closer):
                text = text[1:].strip()
            if text.endswith(closer) and text.count(closer) > text.count(opener):
                text = text[:-1].strip()
        for quote in "'\"":
            if text.count(quote) % 2 and (text.startswith(quote) or text.endswith(quote)):
                text = text[1:].strip() if text.startswith(quote) else text[:-1].strip()
    return text


def split_list(raw: str) -> list[str]:
    """Items of a comma, semicolon or newline separated list."""
    return [item for item in (unwrap(part) for part in _LIST_SEP.split(raw)) if item]


def clean_path(raw: str) -> str:
    path = unwrap(raw)
    if path.startswith("~"):
        path = os.path.expanduser(path)
    return path.rstrip("/") if len(path) > 1 and not path.endswith("://") else path


def in_data(path: Path) -> Path:
    """A relative path that doesn't exist here is looked up under /data, where data assets mount."""
    if not path.is_absolute() and not path.exists() and (DATA_DIR / path).exists():
        return DATA_DIR / path
    return path


@dataclass(frozen=True)
class Settings:
    store: str = "/results/spatialdata.zarr"
    samples: str = DEFAULT_SAMPLES
    celltypist_pkl: str = ""
    sample_column: str = "sample_id"
    create: bool = False
    create_roots: str = "/results,/scratch"
    scratch_dir: str = "/scratch"
    results_dir: str = "/results"
    write_cpus: int = 8
    s3_workers: int = 16
    s3_lock_stale_minutes: float = 10.0

    def __post_init__(self):
        object.__setattr__(self, "store", clean_path(self.store))
        object.__setattr__(self, "celltypist_pkl", clean_path(self.celltypist_pkl))

    @property
    def remote(self) -> bool:
        return self.store.startswith("s3://")

    @property
    def work_dir(self) -> Path:
        return Path(self.scratch_dir) / "spatial_ingest"

    @property
    def default_samples(self) -> bool:
        return split_list(self.samples) == [DEFAULT_SAMPLES]

    def sample_entries(self) -> list[str]:
        return split_list(self.samples)

    def model_file(self) -> Path | None:
        if not self.celltypist_pkl:
            return None
        path = in_data(Path(self.celltypist_pkl))
        if path.suffix.lower() != ".pkl":
            raise ConfigError(f"Custom CellTypist model {self.celltypist_pkl!r} isn't a .pkl file; give the path to "
                              "a CellTypist model (.pkl), or leave the field empty for no labels.")
        return path

    def create_root_list(self) -> list[Path]:
        return [Path(r).resolve() for r in split_list(self.create_roots)]

    def may_create(self, location: str) -> bool:
        """New stores go at an s3:// prefix, inside a create root (not the root itself), or anywhere with `create`."""
        if self.create or "://" in location:
            return True
        path = Path(location).resolve()
        return any(path != root and path.is_relative_to(root) for root in self.create_root_list())


_TYPES = {f.name: type(f.default) for f in fields(Settings)}


def _cast(name: str, raw: str):
    kind = _TYPES[name]
    try:
        return _BOOLS[raw.strip().lower()] if kind is bool else kind(raw)
    except (KeyError, ValueError):
        raise ConfigError(f"{name.upper()}={raw!r} is not a valid {kind.__name__}.") from None


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Add Xenium/Atera bundles and SpatialData objects to one SpatialData .zarr.")
    for name in (*PANEL, "create"):
        p.add_argument(f"--{name.replace('_', '-')}", dest=name)
    p.add_argument("--input-type", dest="input_type", help=argparse.SUPPRESS)  # removed, still accepted
    return p


def panel_values(values: list[str]) -> dict[str, str]:
    """The app panel's positional values as settings; 4 values are the older panel with an input type."""
    if len(values) > len(OLD_PANEL):
        raise ConfigError(f"Expected at most {len(PANEL)} app-panel values ({', '.join(PANEL)}; an older panel's 4 "
                          f"with the input type are accepted too), got {len(values)}: {values}.")
    if len(values) == len(OLD_PANEL):
        if unwrap(values[1]):
            logger.info(IGNORED_NOTE.format(value=unwrap(values[1])))
        return {name: value for name, value in zip(OLD_PANEL, values) if name != "input_type"}
    if len(values) == len(PANEL) and not is_old_input_type(unwrap(values[1])):
        return dict(zip(PANEL, values))
    out = {"store": values[0]} if values else {}
    for value in values[1:]:
        text = unwrap(value)
        if not text:
            continue
        if is_old_input_type(text):
            logger.info(IGNORED_NOTE.format(value=text))
        elif text.lower().endswith(".pkl"):
            out["celltypist_pkl"] = value
        else:
            out["samples"] = value
    return out


def load_settings(argv: list[str] | None = None, env: dict[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    if env.get("INPUT_TYPE", "").strip():
        logger.info(IGNORED_NOTE.format(value=unwrap(env["INPUT_TYPE"])))
    from_env = {name: _cast(name, env[name.upper()]) for name in _TYPES if env.get(name.upper())}
    argv = sys.argv[1:] if argv is None else list(argv)
    args = panel_values(argv) if argv and not argv[0].startswith("--") else vars(parser().parse_args(argv))
    old = args.pop("input_type", None)
    if old:
        logger.info(IGNORED_NOTE.format(value=old))
    from_args = {name: _cast(name, value) for name, value in args.items()
                 if value not in (None, "") and unwrap(value)}
    return replace(Settings(), **{**from_env, **from_args})


FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATEFMT = "%H:%M:%S"

# Library loggers: their INFO chatter goes, real warnings stay unless listed at ERROR.
LOGGER_LEVELS = {
    "ome_zarr": logging.ERROR,                 # "no parent found for <Label>" is a WARNING there
    "annizarr.ops._append": logging.ERROR,     # append notes, expected for every sample after the first
    "annizarr": logging.WARNING,               # per-append progress lines (INFO)
    "celltypist": logging.ERROR,               # progress lines per row chunk
    "anndata": logging.WARNING,
    "spatialdata": logging.WARNING,
    "spatialdata_io": logging.WARNING,
    "numcodecs": logging.WARNING,
    "distributed": logging.WARNING,
    "numba": logging.WARNING,
    "matplotlib": logging.WARNING,
    "PIL": logging.WARNING,
    "fsspec": logging.WARNING,
    "s3fs": logging.WARNING,
    "botocore": logging.WARNING,
    "boto3": logging.WARNING,
    "urllib3": logging.WARNING,
    "aiobotocore": logging.WARNING,
    "asyncio": logging.WARNING,
}

# Python warnings that are noise here: (message regex, category, module regex).
NOISE = (
    (r"Object at .* is not recognized as a component of a Zarr hierarchy", "ZarrUserWarning", ""),
    (r"Consolidated metadata is currently not part in the Zarr format 3 specification", "ZarrUserWarning", ""),
    (r".*reading array from closed file", "UserWarning", ""),
    (r".*The index of the dataframe is not monotonic increasing", "UserWarning", ""),
    (r"SpatialData is not stored in the most current format", "UserWarning", ""),
    (r"`?__version__`? is deprecated", "FutureWarning", ""),
    (r"Some cells have zero counts", "UserWarning", ""),
    (r"", "FutureWarning", r"(celltypist|scanpy)(\.|$)"),
)


def _category(name: str) -> type[Warning]:
    if name == "ZarrUserWarning":
        try:
            from zarr.errors import ZarrUserWarning
            return ZarrUserWarning
        except ImportError:
            return UserWarning
    return {"UserWarning": UserWarning, "FutureWarning": FutureWarning}[name]


def quiet_warnings() -> None:
    for message, category, module in NOISE:
        warnings.filterwarnings("ignore", message=message, category=_category(category), module=module)


def _show(message, category, filename, lineno, file=None, line=None) -> None:
    where = filename.rsplit("/", 1)[-1]
    logging.getLogger("py.warnings").warning("%s: %s (%s:%s)", category.__name__, str(message).strip(), where, lineno)


class _Handler(logging.StreamHandler):
    pass


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if not any(isinstance(h, _Handler) for h in root.handlers):
        handler = _Handler(sys.stdout)
        handler.setFormatter(logging.Formatter(FORMAT, DATEFMT))
        root.addHandler(handler)
    root.setLevel(level)
    for name, lvl in LOGGER_LEVELS.items():
        logging.getLogger(name).setLevel(lvl)
    quiet_warnings()
    warnings.showwarning = _show


def block(title: str, lines: list[str]) -> str:
    """A title line and indented items, logged as one record."""
    return "\n".join([title, *(f"  - {line}" for line in lines)])


NAMES = {"xenium": "Xenium bundle", "atera": "Atera bundle", "spatialdata": "SpatialData object"}


@dataclass(frozen=True)
class Member:
    """A file on disk, or a member (`inner`) of an archive; `gz` is a per-file gzip to decompress when staged."""
    container: Path
    inner: str | None = None
    gz: bool = False
    size: int = 0

    @property
    def display(self) -> str:
        return f"{self.container}!{self.inner}" if self.inner else str(self.container)


@dataclass(frozen=True)
class Candidate:
    """A bundle (`files` maps canonical 10x names to where they are) or a SpatialData object (`zarr`)."""
    sid: str
    kind: str                                # xenium, atera or spatialdata
    origin: Path                             # what the user's entry resolved to (folder, archive, .zarr)
    files: dict[str, Member] = field(default_factory=dict)
    zarr: Member | None = None
    notes: tuple[str, ...] = ()              # logged at INFO


@dataclass(frozen=True)
class LeftOut:
    path: str
    reason: str


@dataclass
class Discovery:
    candidates: list[Candidate]
    left_out: list[LeftOut]


@dataclass
class Inventory:
    """What a bundle holds, from file names and small reads only."""
    sid: str
    kind: str
    chosen: dict[str, str]                   # role -> canonical file name in Candidate.files
    missing_essential: list[str]
    missing_optional: list[str]
    var: pd.DataFrame | None                 # raw features, as in the file
    n_cells: int | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Part:
    """One sample of an existing SpatialData object."""
    sid: str
    table: str | None
    regions: tuple[str, ...]
    elements: tuple[tuple[str, str], ...]
    tags: tuple[str, ...]


@dataclass
class Sample:
    sid: str
    kind: str
    candidate: Candidate
    inventory: Inventory | None = None       # bundles
    part: Part | None = None                 # SpatialData samples
    path: Path | None = None                 # SpatialData: the readable (staged) store root
    var: pd.DataFrame | None = None          # Gene Expression var, indexed by symbol
    species: str | None = None


@dataclass
class Plan:
    samples: list[Sample]
    present: list[str]
    left_out: list[LeftOut]
    union_genes: list[str]                   # joint table's genes after this run
    panels: dict[str, dict]                  # sid -> {"genes_in_panel", "genes_shared", "species"}


@dataclass
class TableSource:
    """A sample's counts as a CSR AnnData zarr on scratch, with its obs in memory."""
    path: Path
    obs: pd.DataFrame
    n_obs: int
    n_vars: int


@dataclass
class Loaded:
    elements: dict[str, dict[str, object]]
    table: TableSource
    regions: list[str]
    region_key: str | None
    instance_key: str | None
    tags: tuple[str, ...] = ()


@dataclass
class Prepared:
    sid: str
    elements: dict[str, dict[str, object]]
    table_path: Path
    obs: pd.DataFrame
    var: pd.DataFrame
    fingerprint: str
    region_key: str | None
    stats: dict = field(default_factory=dict)
