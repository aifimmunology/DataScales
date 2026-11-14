"""Logging for a run: one line format on stdout, the capsule's messages at INFO, and known library noise
silenced at its source, so that a clean run logs no WARNING line and every WARNING asks the user to
look at something.

Silenced (each by its exact message or logger, never a whole warning category):
- zarr: "Object at … is not recognized as a component of a Zarr hierarchy" (consolidating a store that
  holds SpatialData's parquet folders) and "Consolidated metadata is currently not part in the Zarr
  format 3 specification";
- ome_zarr.reader "no parent found for <Label …>" (reading labels) — logger set to ERROR;
- pims "reading array from closed file" (spatialdata-io's tiff reader);
- pandas/dask "The index of the dataframe is not monotonic increasing" (Atera reader);
- annizarr append notes (new categories, sorted-store contiguity, nullable obs columns): expected
  when appending samples — logger set to ERROR;
- spatialdata "not stored in the most current format" (old SpatialData inputs are read fine);
- scanpy "Some cells have zero counts" (normalize_total before CellTypist; such cells are expected);
- anndata "storing … as categorical" (INFO chatter), FutureWarnings raised inside celltypist and scanpy,
  and celltypist's per-call progress lines.

Python warnings that remain go through logging as one line each (`py.warnings`), never with the
echoed source line."""
from __future__ import annotations

import logging
import sys
import warnings

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATEFMT = "%H:%M:%S"

# Logger -> level. INFO chatter of libraries goes, their real warnings/errors stay unless listed at ERROR.
LOGGER_LEVELS = {
    "ome_zarr": logging.ERROR,                 # "no parent found for <Label>" is a WARNING there
    "annizarr.ops._append": logging.ERROR,     # append notes, expected for every sample after the first
    "celltypist": logging.ERROR,               # progress lines, repeated per row chunk; overlap is prechecked
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

# (message regex, category name, module regex) of Python warnings that are noise here.
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
    """Adds the filters for the known noise (call again after anything resets the filters)."""
    for message, category, module in NOISE:
        warnings.filterwarnings("ignore", message=message, category=_category(category), module=module)


def _show(message, category, filename, lineno, file=None, line=None) -> None:
    """A Python warning as one log line: `Category: message (module.py:line)`."""
    where = filename.rsplit("/", 1)[-1]
    logging.getLogger("py.warnings").warning("%s: %s (%s:%s)", category.__name__, str(message).strip(), where, lineno)


class _Handler(logging.StreamHandler):
    """The run's stdout handler (marked so `setup` adds it once)."""


def setup(level: int = logging.INFO) -> None:
    """Configures logging for a run; safe to call more than once."""
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
    """A multi-line message: a title line and indented items (one log record)."""
    return "\n".join([title, *(f"  - {line}" for line in lines)])
