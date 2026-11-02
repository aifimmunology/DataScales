"""Run settings: environment variables (the field names in upper case), overridden by command-line flags."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path

INTEGRATIONS = ("none", "harmony")
OUTPUT_MODES = ("full", "results-only")
_BOOLS = {"1": True, "true": True, "yes": True, "y": True, "on": True,
          "0": False, "false": False, "no": False, "n": False, "off": False}


class ConfigError(ValueError):
    """A setting that can't be read as its type, or a combination that can't run."""


@dataclass(frozen=True)
class Settings:
    input: str = ""
    branch: str = "qc"
    table: str = "table"
    origin: str = ""
    integration: str = "none"
    batch_keys: str = ""
    output_mode: str = "full"
    leiden_resolution: float = 1.1
    output: str = ""
    sample_key: str = "sample_id"
    rows_per_chunk: int = 24_000
    n_top_genes: int = 2000
    n_comps: int = 50
    n_pcs: int = 30
    n_neighbors: int = 20
    neighbors_algorithm: str = "ivfflat"
    n_probes: int = 20
    umap_min_dist: float = 0.45
    leiden_iterations: int = 100
    target_sum: float = 1e4
    random_seed: int = 4242
    results_dir: str = "/results"

    @property
    def batch_key_list(self) -> list[str]:
        return [k.strip() for k in self.batch_keys.split(",") if k.strip()]

    @property
    def integration_keys(self) -> list[str]:
        """The batch keys the integration uses; none without integration."""
        return self.batch_key_list if self.integration != "none" else []

    def output_path(self) -> str:
        """Where a copy goes: `output`, else `<results_dir>/<input stem>.processed.zarr`."""
        if self.output:
            return self.output
        stem = Path(self.input.rstrip("/")).name.removesuffix(".zarr")
        return str(Path(self.results_dir) / f"{stem}.processed.zarr")

    def validate(self) -> Settings:
        if not self.input:
            raise ConfigError("No input given; set it to an icechunk repo, or an AnnData or SpatialData .zarr.")
        if self.integration not in INTEGRATIONS:
            raise ConfigError(f"integration must be one of {', '.join(INTEGRATIONS)}, got {self.integration!r}.")
        if self.output_mode not in OUTPUT_MODES:
            raise ConfigError(f"output mode must be one of {', '.join(OUTPUT_MODES)}, got {self.output_mode!r}.")
        if self.integration != "none" and not self.batch_key_list:
            raise ConfigError(f"integration '{self.integration}' needs batch keys (e.g. chip_id).")
        if self.n_pcs > self.n_comps:
            raise ConfigError(f"n_pcs ({self.n_pcs}) can't exceed n_comps ({self.n_comps}).")
        return self


_TYPES = {f.name: type(f.default) for f in fields(Settings)}


def _cast(name: str, raw: str):
    kind = _TYPES[name]
    try:
        return _BOOLS[raw.strip().lower()] if kind is bool else kind(raw.strip())
    except (KeyError, ValueError):
        raise ConfigError(f"{name.upper()}={raw!r} is not a valid {kind.__name__}.") from None


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HVG, PCA, optional harmony, neighbors, UMAP and leiden on GPU.")
    for f in fields(Settings):
        p.add_argument(f"--{f.name.replace('_', '-')}", dest=f.name, metavar=_TYPES[f.name].__name__.upper(),
                       help=f"default: {f.default!r}" if f.default != "" else None)
    return p


def load_settings(argv: list[str] | None = None, env: dict[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    from_env = {name: _cast(name, env[name.upper()]) for name in _TYPES if env.get(name.upper())}
    args = vars(parser().parse_args(argv))
    from_args = {name: _cast(name, value) for name, value in args.items() if value not in (None, "")}
    return replace(Settings(), **{**from_env, **from_args}).validate()
