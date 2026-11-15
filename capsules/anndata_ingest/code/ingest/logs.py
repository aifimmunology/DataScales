"""Logging for a run: one line format on stdout, the capsule's messages at INFO, and known library noise
silenced at its source, so that every WARNING asks the user to look at something (the same scheme as the
spatial_ingest capsule).

Silenced (each by its exact message or logger, never a whole warning category):
- annizarr append notes (new categories, sorted-store contiguity): expected for every sample after the
  first — logger set to ERROR;
- zarr "Consolidated metadata is currently not part in the Zarr format 3 specification";
- anndata "storing … as categorical" (INFO chatter), FutureWarnings raised inside celltypist and scanpy,
  and celltypist's progress lines.

Python warnings that remain go through logging as one line each (`py.warnings`)."""
from __future__ import annotations

import logging
import sys
import warnings

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATEFMT = "%H:%M:%S"

LOGGER_LEVELS = {
    "annizarr.ops._append": logging.ERROR,
    "celltypist": logging.ERROR,
    "anndata": logging.WARNING,
    "numba": logging.WARNING,
    "botocore": logging.WARNING,
    "urllib3": logging.WARNING,
    "asyncio": logging.WARNING,
}

# (message regex, category, module regex) of Python warnings that are noise here.
NOISE = (
    (r"Consolidated metadata is currently not part in the Zarr format 3 specification", UserWarning, ""),
    (r"`?__version__`? is deprecated", FutureWarning, ""),
    (r"Some cells have zero counts", UserWarning, ""),
    (r"", FutureWarning, r"(celltypist|scanpy)(\.|$)"),
)


def quiet_warnings() -> None:
    """Adds the filters for the known noise (ZarrUserWarning subclasses UserWarning)."""
    for message, category, module in NOISE:
        warnings.filterwarnings("ignore", message=message, category=category, module=module)


def _show(message, category, filename, lineno, file=None, line=None) -> None:
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
