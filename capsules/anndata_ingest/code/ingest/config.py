"""Run settings: environment variables (the field names in upper case), overridden by app-panel values;
and the run's logging."""
from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass, fields, replace
from pathlib import Path

SAMPLE_SUFFIXES = (".h5", ".h5ad", ".h5mu")
SAMPLE_KINDS = ".h5/.h5ad files or AnnData/SpatialData .zarr folders"
# The app panel's parameters, in its order (Code Ocean passes their values positionally in this order).
PANEL = ("input_type", "samples", "store", "origin", "store_format", "celltypist_models", "celltypist_pkl")
INPUT_TYPES = {
    "Raw samples (QC, CellTypist, Scrublet)": "raw",
    "Existing AnnData or SpatialData table (as is)": "existing",
    "raw": "raw",
    "existing": "existing",
}
# The app panel's New store format dropdown; only used when the store doesn't exist yet.
STORE_FORMATS = {
    "Auto (.zarr path is zarr, else icechunk)": "auto",
    "Icechunk": "icechunk",
    "Zarr": "zarr",
    "auto": "auto",
    "icechunk": "icechunk",
    "zarr": "zarr",
}
# The default Storage Location; a run refuses it until it's replaced with a real store location.
PLACEHOLDER_STORE = "s3://bucket/prefix"
BUNDLED_MODELS = Path(__file__).resolve().parents[1] / "models"
DATA_DIR = Path("/data")
AIFI_L1 = "pbmc_clean_AIFI_L1_2024"
AIFI_MODELS = f"{AIFI_L1},pbmc_clean_AIFI_L2_2024,pbmc_clean_AIFI_L3_2024"
IMMUNE_ALL = "Immune_All_High,Immune_All_Low"
# The app panel's Celltypist dropdown; matched ignoring case, spaces and punctuation.
CELLTYPIST_PRESETS = {
    "AIFI L1, L2, L3 celltypes": AIFI_MODELS,
    "Only AIFI L1 celltypes": AIFI_L1,
    "CellTypist Immune_All High and Low labels": IMMUNE_ALL,
    "High and low AIFI cell labels": IMMUNE_ALL,
    "None": "",
}
_BOOLS = {"1": True, "true": True, "yes": True, "y": True, "on": True,
          "0": False, "false": False, "no": False, "n": False, "off": False}
_LIST_SEP = re.compile(r"[,;\n]+")
_STRIP = " \t\r'\"[]()"


class ConfigError(ValueError):
    """A setting that can't be read as its type."""


def _as_bool(raw: str) -> bool:
    return _BOOLS[raw.strip().lower()]


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


_PRESETS = {_key(label): models for label, models in CELLTYPIST_PRESETS.items()}
_INPUT_KEYS = {_key(label) for label in INPUT_TYPES}
_FORMATS = {_key(label): kind for label, kind in STORE_FORMATS.items()}


def split_list(raw: str) -> list[str]:
    """Items of a comma, semicolon or newline separated list, with quotes, brackets and spaces stripped."""
    return [item for item in (part.strip(_STRIP) for part in _LIST_SEP.split(raw)) if item]


def clean_path(raw: str) -> str:
    path = raw.strip(_STRIP)
    return path.rstrip("/") if len(path) > 1 and not path.endswith("://") else path


def _in_data(path: Path) -> Path:
    """A relative path that doesn't exist here is looked up under /data, where data assets mount."""
    if not path.is_absolute() and not path.exists() and (DATA_DIR / path).exists():
        return DATA_DIR / path
    return path


def _is_zarr(path: Path) -> bool:
    return path.is_dir() and (path.suffix.lower() == ".zarr" or (path / "zarr.json").is_file()
                              or (path / ".zgroup").is_file())


def _is_sample(path: Path) -> bool:
    if path.name.startswith("."):
        return False
    return _is_zarr(path) or (path.is_file() and path.suffix.lower() in SAMPLE_SUFFIXES)


def _expand(entry: str) -> list[Path]:
    if glob.has_magic(entry):
        matches = glob.glob(entry) or ([] if Path(entry).is_absolute() else glob.glob(str(DATA_DIR / entry)))
        found = [p for p in sorted(map(Path, matches)) if _is_sample(p)]
        if not found:
            raise ConfigError(f"No {SAMPLE_KINDS} match {entry!r}.")
        return found
    path = _in_data(Path(entry))
    if _is_zarr(path):
        return [path]
    if path.is_dir():
        found = sorted(p for p in path.iterdir() if _is_sample(p))
        if not found:
            raise ConfigError(f"The folder {path} holds no {SAMPLE_KINDS} (other subfolders aren't searched).")
        return found
    if path.suffix.lower() not in SAMPLE_SUFFIXES:
        raise ConfigError(f"{entry!r} is not a .h5/.h5ad file, an AnnData/SpatialData .zarr, or a folder of them.")
    return [path]


@dataclass(frozen=True)
class Settings:
    samples: str = "/data/samples"
    store: str = PLACEHOLDER_STORE
    origin: str = ""
    store_format: str = "auto"  # kind of a new store: auto, icechunk or zarr; an existing store keeps its kind
    input_type: str = "raw"
    sample_column: str = "sample_id"
    create: bool = False
    create_roots: str = "/results,/scratch"
    celltypist_models: str = AIFI_MODELS
    celltypist_pkl: str = ""  # custom CellTypist .pkl file(s); used instead of celltypist_models when given
    models_dir: str = str(BUNDLED_MODELS)
    run_scrublet: bool = True
    run_celltypist: bool = True
    random_seed: int = 6767
    chunk_cells: int = 5000
    write_cpus: int = 8
    staging_dir: str = "/scratch/staging"
    results_dir: str = "/results"

    def __post_init__(self):
        for name in ("chunk_cells", "write_cpus"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name.upper()}={getattr(self, name)} must be 1 or more.")
        object.__setattr__(self, "store", clean_path(self.store))
        object.__setattr__(self, "origin", clean_path(self.origin))
        kinds = {_key(label): kind for label, kind in INPUT_TYPES.items()}
        if _key(self.input_type) not in kinds:
            raise ConfigError(f"Input type {self.input_type!r} is not one of {list(INPUT_TYPES)[:2]}.")
        object.__setattr__(self, "input_type", kinds[_key(self.input_type)])
        if _key(self.store_format) not in _FORMATS:
            raise ConfigError(f"Store format {self.store_format!r} is not one of {list(STORE_FORMATS)[:3]}.")
        object.__setattr__(self, "store_format", _FORMATS[_key(self.store_format)])

    @property
    def existing(self) -> bool:
        return self.input_type == "existing"

    def sample_files(self) -> list[Path]:
        """Every listed file, folder (its .h5/.h5ad files) or glob, in order and without repeats."""
        files: dict[Path, None] = {}
        for entry in split_list(self.samples):
            for path in _expand(entry):
                files.setdefault(path, None)
        return list(files)

    def model_names(self) -> list[str]:
        """Custom .pkl files when given; else a dropdown label means its models, and anything else is a
        list of model names or .pkl paths."""
        if not self.run_celltypist:
            return []
        raw = self.celltypist_pkl or _PRESETS.get(_key(self.celltypist_models), self.celltypist_models)
        return [m for m in split_list(raw) if _key(m) != "none"]

    def model_files(self) -> list[Path]:
        """CellTypist models in order: a name means `<models_dir>/<name>.pkl`; a .pkl path is used as is."""
        return [_in_data(Path(m)) if m.endswith(".pkl") else Path(self.models_dir) / f"{m}.pkl"
                for m in self.model_names()]

    def may_create(self, location: str) -> bool:
        """New stores go at a remote URL (its write is the check), inside the create roots (not a root
        itself), or anywhere when `create` allows it."""
        if self.create or "://" in location:
            return True
        path = Path(location).resolve()
        roots = [Path(r).resolve() for r in split_list(self.create_roots)]
        return any(path != root and path.is_relative_to(root) for root in roots)


_TYPES = {f.name: type(f.default) for f in fields(Settings)}


def _cast(name: str, raw: str):
    kind = _TYPES[name]
    try:
        return _as_bool(raw) if kind is bool else kind(raw)
    except (KeyError, ValueError):
        raise ConfigError(f"{name.upper()}={raw!r} is not a valid {kind.__name__}.") from None


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Append single-cell samples to an AnnData store (icechunk or zarr).")
    for name in (*PANEL, "create"):
        p.add_argument(f"--{name.replace('_', '-')}", dest=name)
    return p


def panel_values(values: list[str]) -> dict[str, str]:
    """The app panel's positional values (input type, samples, store, origin, new store format, Celltypist
    model, custom .pkl) as settings.

    Code Ocean may leave out an empty optional value, which shifts the rest, so a shorter list is placed by
    what each value looks like: a dropdown label or a .pkl; the remaining paths, in order, are the
    samples, the store and an s3:// origin (a single path is the store)."""
    if len(values) > len(PANEL):
        raise ConfigError(f"Expected at most {len(PANEL)} app-panel values ({', '.join(PANEL)}), got {values}.")
    if len(values) == len(PANEL):
        return dict(zip(PANEL, values))
    out: dict[str, str] = {}
    paths: list[str] = []
    for value in values:
        text = value.strip(_STRIP)
        if not text:
            continue
        if _key(text) in _INPUT_KEYS:
            out["input_type"] = value
        elif _key(text) in _FORMATS:
            out["store_format"] = value
        elif _key(text) in _PRESETS:
            out["celltypist_models"] = value
        elif all(item.lower().endswith(".pkl") for item in split_list(text)):
            out["celltypist_pkl"] = value
        else:
            paths.append(value)
    if len(paths) > 3:
        raise ConfigError(f"Expected samples, store and origin among the app-panel values, got {paths}.")
    names = ("store",) if len(paths) == 1 else ("samples", "store", "origin")
    out.update(zip(names, paths))
    return out


def load_settings(argv: list[str] | None = None, env: dict[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    from_env = {name: _cast(name, env[name.upper()]) for name in _TYPES if env.get(name.upper())}
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and not argv[0].startswith("--"):
        args = panel_values(argv)
    else:
        args = vars(parser().parse_args(argv))
    from_args = {name: _cast(name, value) for name, value in args.items() if value not in (None, "")
                 and value.strip(_STRIP)}
    return replace(Settings(), **{**from_env, **from_args})


LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
LOG_DATEFMT = "%H:%M:%S"
# Known library noise is silenced at its source, by exact logger or message (never a whole warning
# category), so that every WARNING left asks the user to look at something, as in spatial_ingest.
# annizarr's append notes (new categories, sorted-store contiguity) are expected for every sample after
# the first.
LOGGER_LEVELS = {
    "annizarr.ops._append": logging.ERROR,
    "celltypist": logging.ERROR,
    "anndata": logging.WARNING,
    "numba": logging.WARNING,
    "botocore": logging.WARNING,
    "urllib3": logging.WARNING,
    "asyncio": logging.WARNING,
}
# (message regex, category, module regex); ZarrUserWarning subclasses UserWarning.
NOISE = (
    (r"Consolidated metadata is currently not part in the Zarr format 3 specification", UserWarning, ""),
    (r"`?__version__`? is deprecated", FutureWarning, ""),
    (r"Some cells have zero counts", UserWarning, ""),
    (r"", FutureWarning, r"(celltypist|scanpy)(\.|$)"),
)


def _show_warning(message, category, filename, lineno, file=None, line=None) -> None:
    where = filename.rsplit("/", 1)[-1]
    logging.getLogger("py.warnings").warning("%s: %s (%s:%s)", category.__name__, str(message).strip(), where, lineno)


class _Handler(logging.StreamHandler):
    """The run's stdout handler (marked so `setup_logging` adds it once)."""


def setup_logging(level: int = logging.INFO) -> None:
    """One line format on stdout; Python warnings that remain are logged as one line each. Safe to call
    more than once."""
    root = logging.getLogger()
    if not any(isinstance(h, _Handler) for h in root.handlers):
        handler = _Handler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATEFMT))
        root.addHandler(handler)
    root.setLevel(level)
    for name, lvl in LOGGER_LEVELS.items():
        logging.getLogger(name).setLevel(lvl)
    for message, category, module in NOISE:
        warnings.filterwarnings("ignore", message=message, category=category, module=module)
    warnings.showwarning = _show_warning
