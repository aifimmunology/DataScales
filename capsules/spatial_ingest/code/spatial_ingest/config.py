"""Run settings: environment variables (the field names in upper case), overridden by app-panel values.

The app panel has three fields: Storage Location, List of paths to samples, Custom CellTypist .pkl.
What each sample is (Xenium, Atera, SpatialData) is detected from its files, so the old Input type
field, still sent by saved panels and older scripts, is accepted and ignored with one INFO line."""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path

logger = logging.getLogger("spatial_ingest.config")

PANEL = ("store", "samples", "celltypist_pkl")
OLD_PANEL = ("store", "input_type", "samples", "celltypist_pkl")
# Labels and short names of the removed Input type field.
OLD_INPUT_TYPES = ("Xenium bundle (folder or .zip)", "Atera bundle (folder or .zip)", "Existing SpatialData (.zarr)")
OLD_INPUT_NAMES = ("xenium", "atera", "spatialdata", "existing")
DATA_DIR = Path("/data")
DEFAULT_SAMPLES = "/data/samples"
_BOOLS = {"1": True, "true": True, "yes": True, "y": True, "on": True,
          "0": False, "false": False, "no": False, "n": False, "off": False}
_LIST_SEP = re.compile(r"[,;\n]+")
_PAIRS = {("[", "]"), ("(", ")"), ('"', '"'), ("'", "'")}
IGNORED_NOTE = ("The Input type setting ({value!r}) is no longer needed and is ignored: each sample's kind is "
                "detected from its files.")


class ConfigError(ValueError):
    """A setting that can't be read as its type."""


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


_OLD_LABELS = {_key(label) for label in OLD_INPUT_TYPES}


def is_old_input_type(text: str, short_names: bool = False) -> bool:
    """A value of the removed Input type field: a panel label, or (when `short_names`) xenium/atera/…;
    a bare short name in a positional value could be a relative path, so it only counts when asked."""
    key = _key(text)
    return key in _OLD_LABELS or (short_names and key in OLD_INPUT_NAMES)


def unwrap(text: str) -> str:
    """`text` without surrounding spaces, without quotes or brackets that wrap ALL of it (`"a b"`, `(x)`),
    and without a stray opening or closing bracket or quote left by splitting a list (`["a"` -> a); a
    bracket that belongs to a name (`tiny_atera (1)`) stays."""
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


def _shown(value: str) -> str:
    """A value as the user gave it, for messages."""
    return unwrap(value)


def split_list(raw: str) -> list[str]:
    """Items of a comma, semicolon or newline separated list: spaces, and quotes or brackets that wrap the
    whole list or a whole item, removed (`["a", "b"]` -> a, b); brackets inside a name are kept."""
    return [item for item in (unwrap(part) for part in _LIST_SEP.split(raw)) if item]


def clean_path(raw: str) -> str:
    """A path value without quotes and a trailing `/`, with `~` expanded (s3:// URLs untouched)."""
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
    scratch_dir: str = "/scratch/spatial_ingest"
    results_dir: str = "/results"
    n_jobs: int = 4
    write_cpus: int = 8
    s3_workers: int = 16

    def __post_init__(self):
        object.__setattr__(self, "store", clean_path(self.store))
        object.__setattr__(self, "celltypist_pkl", clean_path(self.celltypist_pkl))

    @property
    def remote(self) -> bool:
        return self.store.startswith("s3://")

    @property
    def default_samples(self) -> bool:
        """The samples list was left at its default (a Reproducible Run, or an empty panel field)."""
        return split_list(self.samples) == [DEFAULT_SAMPLES]

    def sample_entries(self) -> list[str]:
        return split_list(self.samples)

    def model_file(self) -> Path | None:
        """The custom CellTypist model's path (`~` expanded, relative paths looked up under /data)."""
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
        """New stores go at an s3:// prefix, inside the create roots (not a root itself), or anywhere
        when `create` allows it."""
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
    p.add_argument("--input-type", dest="input_type", help=argparse.SUPPRESS)  # removed; accepted and ignored
    return p


def panel_values(values: list[str]) -> dict[str, str]:
    """The app panel's positional values (store, samples, custom .pkl) as settings.

    Four values are an older panel's call (store, input type, samples, .pkl): the input type is
    ignored. Three values are the current panel, in order, unless the second is an old input-type
    label. Code Ocean may leave out an empty optional value, which shifts the rest, so otherwise
    each value after the store is placed by what it looks like: an old input-type label (ignored),
    a .pkl (any case), or samples."""
    if len(values) > len(OLD_PANEL):
        raise ConfigError(f"Expected at most {len(PANEL)} app-panel values ({', '.join(PANEL)}; an older panel's 4 "
                          f"with the input type are accepted too), got {len(values)}: {values}.")
    if len(values) == len(OLD_PANEL):
        if unwrap(values[1]):
            logger.info(IGNORED_NOTE.format(value=_shown(values[1])))
        return {name: value for name, value in zip(OLD_PANEL, values) if name != "input_type"}
    if len(values) == len(PANEL) and not is_old_input_type(unwrap(values[1])):
        return dict(zip(PANEL, values))
    out = {"store": values[0]} if values else {}
    for value in values[1:]:
        text = unwrap(value)
        if not text:
            continue
        if is_old_input_type(text):
            logger.info(IGNORED_NOTE.format(value=_shown(value)))
        elif text.lower().endswith(".pkl"):
            out["celltypist_pkl"] = value
        else:
            out["samples"] = value
    return out


def load_settings(argv: list[str] | None = None, env: dict[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    if env.get("INPUT_TYPE", "").strip():
        logger.info(IGNORED_NOTE.format(value=_shown(env["INPUT_TYPE"])))
    from_env = {name: _cast(name, env[name.upper()]) for name in _TYPES if env.get(name.upper())}
    argv = sys.argv[1:] if argv is None else list(argv)
    args = panel_values(argv) if argv and not argv[0].startswith("--") else vars(parser().parse_args(argv))
    old = args.pop("input_type", None)
    if old:
        logger.info(IGNORED_NOTE.format(value=old))
    from_args = {name: _cast(name, value) for name, value in args.items()
                 if value not in (None, "") and unwrap(value)}
    return replace(Settings(), **{**from_env, **from_args})
