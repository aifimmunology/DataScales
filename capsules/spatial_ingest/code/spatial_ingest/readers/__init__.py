"""One sample, read: lazy spatial elements and its count table streamed to scratch (`contracts.Loaded`).

Bundles (Xenium, Atera): the inventory's chosen files are staged under canonical names in
`scratch/stage/<sid>/bundle/`, the count matrix streamed into a CSR AnnData zarr (`readers.table`), and the
elements built from what exists (`readers.elements`). Missing optional parts and the decisions taken are
said in ONE warning per sample. SpatialData samples are read by `existing.load_part`."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from anndata.io import write_elem

import stage
from contracts import NAMES, Inventory, Loaded, Sample
from readers import elements as builders
from readers import table as tables

logger = logging.getLogger(__name__)

REGION_KEY = "region"
INSTANCE_KEY = "cell_id"
# roles that spatialdata-io / the vendored Atera reader open by name, so they're linked under canonical names
BY_NAME = ("experiment", "cells_zarr", "cell_boundaries", "nucleus_boundaries", "transcripts", "morphology",
           "morphology_mip", "morphology_3d", "panel")


class ReadError(ValueError):
    """A sample that can't be read as it is (user-facing; the message says what and why)."""


def _link(target: Path, link: Path) -> Path:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target.resolve(), target_is_directory=target.is_dir())
    return link


def canonical_name(role: str, key: str, kind: str) -> str:
    """Where a chosen file goes in the canonical folder: spatialdata-io's names, whatever it was called."""
    base = key.rsplit("/", 1)[-1]
    if role == "experiment":
        return "experiment.xenium" if kind == "xenium" else "experiment.spatial"
    if base.endswith(".zarr.zip"):
        return {"cells_zarr": "cells.zarr.zip", "transcripts": "transcripts.zarr.zip"}.get(role, base)
    if role in ("cell_boundaries", "nucleus_boundaries", "transcripts", "cells"):
        ext = ".parquet" if ".parquet" in base else ".csv.gz" if base.endswith(".gz") else ".csv"
        return role + ext
    return base


def stage_bundle(sample: Sample, inv: Inventory, scratch: Path) -> tuple[Path, dict[str, Path]]:
    """The chosen files staged (extracted / gunzipped as needed) and linked under canonical names in one
    folder; returns the folder and role -> path."""
    import inventory

    cand = sample.candidate
    dest = inventory.stage_dir(scratch, sample.sid)
    folder = dest / "bundle"
    files: dict[str, Path] = {}
    for role, key in inv.chosen.items():
        if key.endswith("/"):
            staged = stage.fetch(cand, [key], dest)
            for name, path in staged.items():
                _link(path, folder / name)
            files[role] = folder / key.rstrip("/")
            continue
        path = stage.fetch(cand, [key], dest)[key]
        files[role] = _link(path, folder / canonical_name(role, key, cand.kind)) if role in BY_NAME else path
    if inv.chosen.get("matrix", "").endswith(".zarr.zip") and cand.kind == "atera":
        _link(files["matrix"], folder / "cell_feature_matrix.zarr.zip")
    return folder, files


def _obs_with(obs: pd.DataFrame, extra: dict[str, np.ndarray], region: str) -> pd.DataFrame:
    obs = obs.copy()
    for col, values in extra.items():
        obs[col] = values
    obs[REGION_KEY] = pd.Categorical([region] * len(obs))
    if INSTANCE_KEY in obs and obs[INSTANCE_KEY].dtype == object:
        obs[INSTANCE_KEY] = obs[INSTANCE_KEY].astype(str)
    return obs


def invalid_polygons(elements: dict[str, dict[str, object]]) -> list[str]:
    """'nucleus_boundaries: 8,123 of 156,628 polygons are invalid geometries' per shapes element that has any
    (kept as read; early XOA 2.0 nucleus boundaries have ~5 %)."""
    out = []
    for name, shapes in elements.get("shapes", {}).items():
        geometry = getattr(shapes, "geometry", None)
        if geometry is None or not len(geometry) or (geometry.geom_type == "Point").all():
            continue
        bad = int((~geometry.is_valid).sum())
        if bad:
            out.append(f"{name}: {bad:,} of {len(geometry):,} polygons are invalid geometries (kept as read)")
    return out


def warn_once(sample: Sample, inv: Inventory, problems: list[str], table_notes: list[str]) -> None:
    """What the sample lacks and the decisions taken, as one INFO line (the precheck already warned about
    the inventory's missing parts); parts that failed only now, while reading, are ONE warning."""
    import inventory

    line = inventory.describe(Inventory(inv.sid, inv.kind, inv.chosen, [], list(inv.missing_optional), None,
                                        inv.n_cells, list(inv.notes) + table_notes))
    if line:
        logger.info("%s: %s", sample.sid, line)
    if problems:
        logger.warning("%s: %s read without: %s", sample.sid, NAMES.get(sample.kind, sample.kind), "; ".join(problems))


def load_bundle(sample: Sample, scratch: Path, n_jobs: int = 1) -> Loaded:
    import inventory

    inv = sample.inventory or inventory.inspect(sample.candidate, scratch)
    if inv.missing_essential:
        raise ReadError(f"{sample.sid}: can't be read; missing {'; '.join(inv.missing_essential)}.")
    folder, files = stage_bundle(sample, inv, scratch)
    work = scratch / "work" / sample.sid
    out = scratch / "read" / sample.sid / "counts.zarr"
    notes: list[str] = []
    try:
        m = tables.open_matrix({r: files[r] for r in ("matrix", "matrix_features", "matrix_barcodes") if r in files},
                               sample.kind)
        if m.kind == "atera_zarr":
            source = tables.atera_table(m, sample.sid, out, work)
        else:
            cells_path = files["cells"]
            cells = tables.xenium_cells_zarr(cells_path) if cells_path.name.endswith(".zarr.zip") \
                else tables.read_cells(cells_path, work)
            source = tables.xenium_table(m, cells, sample.sid, out, work, notes)
    except tables.TableError as exc:
        raise ReadError(str(exc)) from exc
    spatial = np.asarray(zarr.open_group(str(out), mode="r")["obsm"]["spatial"][:])
    if m.kind == "atera_zarr":
        built = builders.atera(folder, files, source.obs, spatial)
    else:
        built = builders.xenium(folder, files, source.obs, spatial, work)
    source.obs = _obs_with(source.obs, built.obs_columns, built.region)
    group = zarr.open_group(str(out), mode="r+")
    write_elem(group, "obs", source.obs)
    warn_once(sample, inv, built.problems, notes + invalid_polygons(built.elements))
    return Loaded(built.elements, source, [built.region], REGION_KEY, INSTANCE_KEY)


def load(sample: Sample, scratch: Path, n_jobs: int = 1) -> Loaded:
    """The sample's lazy elements and its streamed table (see the module docstring)."""
    if sample.kind == "spatialdata":
        import existing

        return existing.load_part(sample.path, sample.part, scratch)
    return load_bundle(sample, scratch, n_jobs)
