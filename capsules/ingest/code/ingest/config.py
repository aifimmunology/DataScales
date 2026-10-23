"""Run settings: environment variables (the field names in upper case), overridden by app-panel flags."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path

SAMPLE_SUFFIXES = (".h5", ".h5ad")
PANEL = ("samples", "store", "origin", "celltypist_models", "create")
BUNDLED_MODELS = Path(__file__).resolve().parents[1] / "models"
AIFI_MODELS = "pbmc_clean_AIFI_L1_2024,pbmc_clean_AIFI_L2_2024,pbmc_clean_AIFI_L3_2024"
_BOOLS = {"1": True, "true": True, "yes": True, "y": True, "on": True,
          "0": False, "false": False, "no": False, "n": False, "off": False}


class ConfigError(ValueError):
    """A setting that can't be read as its type."""


def _as_bool(raw: str) -> bool:
    return _BOOLS[raw.strip().lower()]


@dataclass(frozen=True)
class Settings:
    samples: str = "/data/samples"
    store: str = ""
    origin: str = ""
    create: bool = False
    celltypist_models: str = AIFI_MODELS
    models_dir: str = str(BUNDLED_MODELS)
    run_scrublet: bool = True
    run_celltypist: bool = True
    random_seed: int = 6767
    chunk_cells: int = 5000
    write_cpus: int = 8
    staging_dir: str = "/scratch/staging"
    results_dir: str = "/results"

    def sample_files(self) -> list[Path]:
        entries = [Path(p.strip()) for p in self.samples.split(",") if p.strip()]
        if len(entries) == 1 and entries[0].is_dir():
            return sorted(p for p in entries[0].iterdir() if p.suffix in SAMPLE_SUFFIXES)
        return entries

    def model_files(self) -> list[Path]:
        """CellTypist models in order: a name means `<models_dir>/<name>.pkl`; a .pkl path is used as is."""
        names = [m.strip() for m in self.celltypist_models.split(",") if m.strip()] if self.run_celltypist else []
        return [Path(m) if m.endswith(".pkl") else Path(self.models_dir) / f"{m}.pkl" for m in names]


_TYPES = {f.name: type(f.default) for f in fields(Settings)}


def _cast(name: str, raw: str):
    kind = _TYPES[name]
    try:
        return _as_bool(raw) if kind is bool else kind(raw)
    except (KeyError, ValueError):
        raise ConfigError(f"{name.upper()}={raw!r} is not a valid {kind.__name__}.") from None


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Append single-cell samples to an AnnData store (icechunk or zarr).")
    for name in PANEL:
        p.add_argument(f"--{name.replace('_', '-')}", dest=name)
    return p


def load_settings(argv: list[str] | None = None, env: dict[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    from_env = {name: _cast(name, env[name.upper()]) for name in _TYPES if env.get(name.upper())}
    args = vars(parser().parse_args(argv))
    from_args = {name: _cast(name, value) for name, value in args.items() if value not in (None, "")}
    return replace(Settings(), **{**from_env, **from_args})
