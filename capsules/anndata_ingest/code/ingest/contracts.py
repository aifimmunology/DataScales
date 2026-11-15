"""Records passed between a run's stages, and the run's errors.

    settings ─► precheck.run ─► input paths + loaded models          (nothing touched yet)
    store    ─► precheck.select_* ─► RawInput / ObjectInput          (only the new ones)
    stage.samples / stage.objects ─► Staged                           (a zarr per input, in staging)
    main.write ─► the store: the first Staged seeds a new one, the rest are appended and committed
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class IngestError(RuntimeError):
    """The inputs or the store's state don't allow this run."""


class ContractError(ValueError):
    """A sample the store cannot take as it is."""


@dataclass(frozen=True)
class RawInput:
    """A raw sample (HISE/10x .h5, .h5ad, or a SpatialData table) and the sample ID its file name gives."""
    sample_id: str
    path: Path

    @property
    def name(self) -> str:
        return self.sample_id


@dataclass(frozen=True)
class ObjectInput:
    """An existing AnnData taken as it is: a unique label, the obs column naming its samples (None when
    the file name is the one sample), and its sample IDs."""
    label: str
    path: Path
    column: str | None
    sample_ids: tuple[str, ...]

    @property
    def name(self) -> str:
        return self.label


@dataclass(frozen=True)
class Staged:
    """One input staged as a zarr, ready to seed or append: what it adds and how it was fitted."""
    sample_id: str
    path: Path
    summary: dict
    source: str = ""
    fitted: str = ""
    samples: tuple[str, ...] = ()

    @property
    def line(self) -> str:
        return f"- {self.sample_id}: {self.source}" + (f" ({self.fitted})" if self.fitted else "")
