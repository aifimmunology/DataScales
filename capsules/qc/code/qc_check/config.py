"""Run settings: environment variables (the field names in upper case), overridden by app-panel flags."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, fields, replace

PANEL = ("input", "decisions", "mode", "target", "origin", "reviewer",
         "group_columns", "label_column", "celltype_qc", "compute_metrics")
MODES = ("report", "apply")
DEFAULT_BRANCH = "qc"
DEFAULT_LABEL = "pbmc_clean_AIFI_L1_2024_label"
_BOOLS = {"1": True, "true": True, "yes": True, "y": True, "on": True,
          "0": False, "false": False, "no": False, "n": False, "off": False}


class ConfigError(ValueError):
    """A setting that can't be read as its type."""


@dataclass(frozen=True)
class Settings:
    input: str = ""
    decisions: str = ""
    mode: str = "report"
    target: str = ""
    origin: str = ""
    reviewer: str = ""
    group_columns: str = "batch_id,pool_id,chip_id"
    label_column: str = DEFAULT_LABEL
    celltype_qc: bool = False
    compute_metrics: bool = False
    table: str = ""
    results_dir: str = "/results"
    seed: int = 4242
    embed_per_sample: int = 0
    embed_max_cells: int = 200_000
    band_cells: int = 100_000

    def __post_init__(self):
        if self.mode not in MODES:
            raise ConfigError(f"mode must be one of {', '.join(MODES)}, got {self.mode!r}.")

    @property
    def apply(self) -> bool:
        return self.mode == "apply"

    def groups(self) -> list[str]:
        return [c.strip() for c in self.group_columns.split(",") if c.strip()]

    def branch(self) -> str:
        return self.target or DEFAULT_BRANCH


_TYPES = {f.name: type(f.default) for f in fields(Settings)}


def _cast(name: str, raw: str):
    kind = _TYPES[name]
    try:
        return _BOOLS[raw.strip().lower()] if kind is bool else kind(raw.strip())
    except (KeyError, ValueError):
        raise ConfigError(f"{name.upper()}={raw!r} is not a valid {kind.__name__}.") from None


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="QC report for an AnnData store; with apply, write the passing cells.")
    for name in PANEL:
        flags = dict.fromkeys([f"--{name}", f"--{name.replace('_', '-')}"])
        p.add_argument(*flags, dest=name)
    return p


def load_settings(argv: list[str] | None = None, env: dict[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    from_env = {name: _cast(name, env[name.upper()]) for name in _TYPES if env.get(name.upper())}
    args = vars(parser().parse_args(argv))
    from_args = {name: _cast(name, value) for name, value in args.items() if value not in (None, "")}
    return replace(Settings(), **{**from_env, **from_args})
