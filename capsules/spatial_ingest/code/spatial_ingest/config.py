"""Run settings: environment variables (the field names in upper case), overridden by app-panel values."""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path

PANEL = ("store", "input_type", "samples", "celltypist_pkl")
INPUT_TYPES = {
    "Xenium bundle (folder or .zip)": "xenium",
    "Atera bundle (folder or .zip)": "atera",
    "Existing SpatialData (.zarr)": "spatialdata",
    "xenium": "xenium",
    "atera": "atera",
    "spatialdata": "spatialdata",
    "existing": "spatialdata",
}
DATA_DIR = Path("/data")
_BOOLS = {"1": True, "true": True, "yes": True, "y": True, "on": True,
          "0": False, "false": False, "no": False, "n": False, "off": False}
_LIST_SEP = re.compile(r"[,;\n]+")
_STRIP = " \t\r'\"[]()"


class ConfigError(ValueError):
    """A setting that can't be read as its type."""


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


_INPUT_KEYS = {_key(label): kind for label, kind in INPUT_TYPES.items()}
_PANEL_LABELS = {_key(label) for label in list(INPUT_TYPES)[:3]}


def split_list(raw: str) -> list[str]:
    """Items of a comma, semicolon or newline separated list, with quotes, brackets and spaces stripped."""
    return [item for item in (part.strip(_STRIP) for part in _LIST_SEP.split(raw)) if item]


def clean_path(raw: str) -> str:
    path = raw.strip(_STRIP)
    return path.rstrip("/") if len(path) > 1 and not path.endswith("://") else path


def in_data(path: Path) -> Path:
    """A relative path that doesn't exist here is looked up under /data, where data assets mount."""
    if not path.is_absolute() and not path.exists() and (DATA_DIR / path).exists():
        return DATA_DIR / path
    return path


@dataclass(frozen=True)
class Settings:
    store: str = "/results/spatialdata.zarr"
    input_type: str = "xenium"
    samples: str = "/data/samples"
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
        if _key(self.input_type) not in _INPUT_KEYS:
            raise ConfigError(f"Input type {self.input_type!r} is not one of {list(INPUT_TYPES)[:3]}.")
        object.__setattr__(self, "input_type", _INPUT_KEYS[_key(self.input_type)])

    @property
    def remote(self) -> bool:
        return self.store.startswith("s3://")

    def model_file(self) -> Path | None:
        return in_data(Path(self.celltypist_pkl)) if self.celltypist_pkl else None

    def may_create(self, location: str) -> bool:
        """New stores go at an s3:// prefix, inside the create roots (not a root itself), or anywhere
        when `create` allows it."""
        if self.create or "://" in location:
            return True
        path = Path(location).resolve()
        roots = [Path(r).resolve() for r in split_list(self.create_roots)]
        return any(path != root and path.is_relative_to(root) for root in roots)


_TYPES = {f.name: type(f.default) for f in fields(Settings)}


def _cast(name: str, raw: str):
    kind = _TYPES[name]
    try:
        return _BOOLS[raw.strip().lower()] if kind is bool else kind(raw)
    except (KeyError, ValueError):
        raise ConfigError(f"{name.upper()}={raw!r} is not a valid {kind.__name__}.") from None


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Add Xenium/Atera bundles or SpatialData objects to one SpatialData .zarr.")
    for name in (*PANEL, "create"):
        p.add_argument(f"--{name.replace('_', '-')}", dest=name)
    return p


def panel_values(values: list[str]) -> dict[str, str]:
    """The app panel's positional values (store, input type, samples, custom .pkl) as settings.

    Code Ocean may leave out an empty optional value, which shifts the rest, so after the store
    each value is placed by what it looks like: a panel input-type label (not a short name, which
    could be a path), a .pkl, or samples."""
    if len(values) > len(PANEL):
        raise ConfigError(f"Expected at most {len(PANEL)} app-panel values ({', '.join(PANEL)}), got {values}.")
    if len(values) == len(PANEL):
        return dict(zip(PANEL, values))
    out = {"store": values[0]} if values else {}
    for value in values[1:]:
        text = value.strip(_STRIP)
        if not text:
            continue
        if _key(text) in _PANEL_LABELS:
            out["input_type"] = value
        elif text.lower().endswith(".pkl"):
            out["celltypist_pkl"] = value
        else:
            out["samples"] = value
    return out


def load_settings(argv: list[str] | None = None, env: dict[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    from_env = {name: _cast(name, env[name.upper()]) for name in _TYPES if env.get(name.upper())}
    argv = sys.argv[1:] if argv is None else list(argv)
    args = panel_values(argv) if argv and not argv[0].startswith("--") else vars(parser().parse_args(argv))
    from_args = {name: _cast(name, value) for name, value in args.items()
                 if value not in (None, "") and value.strip(_STRIP)}
    return replace(Settings(), **{**from_env, **from_args})
