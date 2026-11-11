"""The local SpatialData `.zarr` grown in place: what a location holds, writability, presence of
samples, staged installs and the repair after a killed run (spec §3S.6, §3S.9).

Commit points are `zarr.consolidate_metadata`: an element counts once it is in the root consolidated
metadata, which is what `spatialdata.read_zarr` lists."""
from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import zarr

import existing

logger = logging.getLogger(__name__)

STAGING = ".spatial_ingest_staging"
MARKER = "spatial_ingest"
KINDS = (*existing.ELEMENT_KINDS, "tables")
PROBE = ".write_probe"


class StoreError(RuntimeError):
    """A location this capsule can't use as its store."""


def root_attrs(path: Path) -> dict | None:
    meta = path / "zarr.json"
    if not meta.is_file():
        return None
    return json.loads(meta.read_text())


def kind_of(meta: dict | None) -> str:
    """ours, spatialdata, anndata or other, from a root zarr.json."""
    attrs = (meta or {}).get("attributes", {})
    if "spatialdata_attrs" in attrs:
        return "ours" if MARKER in attrs else "spatialdata"
    return "anndata" if attrs.get("encoding-type") == "anndata" else "other"


def describe(path: Path) -> str:
    """missing, empty, ours (a SpatialData store this capsule created), spatialdata (any other),
    anndata, icechunk or other."""
    if not path.exists():
        return "missing"
    if path.is_dir() and not any(path.iterdir()):
        return "empty"
    kind = kind_of(root_attrs(path))
    if kind != "other":
        return kind
    if (path / "refs").is_dir() and (path / "snapshots").is_dir():
        return "icechunk"
    try:
        from annizarr import Repo

        if Repo.exists(str(path)):
            return "icechunk"
    except Exception:
        pass
    return "other"


def check_usable(path: Path, may_create: bool) -> bool:
    """Refuses a location that isn't (and can't become) a store of this capsule; True when it must be created.

    Only stores this capsule created are grown in place: its repair deletes element folders that the
    consolidated metadata doesn't list, which in another tool's store may be real data."""
    kind = describe(path)
    if kind == "ours":
        probe_writable(path)
        return False
    refuse(path, kind, may_create)
    return True


def refuse(path, kind: str, may_create: bool) -> None:
    """Raises unless `kind` is a location a new store may be created at."""
    if kind in ("missing", "empty"):
        if not may_create:
            raise StoreError(f"No store at {path}. New stores are created only at an s3:// prefix or inside the "
                             "create roots (CREATE_ROOTS, default /results and /scratch); check the path of an "
                             "existing store, or set CREATE=true to create one there.")
        return
    what = {"anndata": "an AnnData .zarr (use the anndata_ingest capsule for it)",
            "icechunk": "an icechunk repo; SpatialData stores here are plain .zarr folders",
            "spatialdata": "a SpatialData store this capsule didn't create; add it as an input (input type "
                           "'Existing SpatialData') into a new store instead",
            "other": "a folder that isn't a SpatialData store"}[kind]
    raise StoreError(f"{path} is {what}; give a new path or a store this capsule built.")


def probe_writable(path: Path) -> None:
    probe = path / PROBE
    try:
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as exc:
        raise StoreError(f"{path} is read-only ({exc.strerror}). A data asset can't be written to: copy the store "
                         "to S3 (or /results) and give that location.") from None


def create(path: Path) -> None:
    from spatialdata import SpatialData

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.rmdir()
    SpatialData().write(path)
    zarr.open_group(path, mode="r+", use_consolidated=False).attrs[MARKER] = {"version": 1}
    consolidate(path)
    logger.warning("Created a new SpatialData store at %s", path)


def consolidate(path: Path) -> None:
    """Commits whatever is in place; an empty staging folder is removed first (it isn't zarr)."""
    if (path / STAGING).is_dir() and not any((path / STAGING).iterdir()):
        (path / STAGING).rmdir()
    zarr.consolidate_metadata(str(path))


def committed(path: Path) -> dict[str, set[str]]:
    """The elements by kind in the root consolidated metadata."""
    meta = (root_attrs(path) or {}).get("consolidated_metadata") or {}
    out: dict[str, set[str]] = {k: set() for k in KINDS}
    for key in meta.get("metadata", {}):
        parts = key.split("/")
        if len(parts) == 2 and parts[0] in out:
            out[parts[0]].add(parts[1])
    return out


def on_disk(path: Path) -> dict[str, set[str]]:
    return {k: {p.name for p in (path / k).iterdir() if p.is_dir()} if (path / k).is_dir() else set() for k in KINDS}


def column_length(node) -> int:
    """Rows of an obs column as stored: a plain array, or a group (nullable or categorical encoding)."""
    if isinstance(node, zarr.Array):
        return node.shape[0]
    return node["values" if "values" in node else "codes"].shape[0]


def n_obs(table: zarr.Group) -> int:
    obs = table["obs"]
    return column_length(obs[obs.attrs["_index"]])


def staging(path: Path) -> Path:
    out = path / STAGING
    out.mkdir(exist_ok=True)
    return out


def install(path: Path, staged: Path) -> list[str]:
    """Moves every element of the staged SpatialData `staged` into the store with `os.rename`."""
    moved = []
    root = zarr.open_group(path, mode="a", use_consolidated=False)
    for kind in KINDS:
        if not (staged / kind).is_dir():
            continue
        root.require_group(kind)
        for element in sorted(p for p in (staged / kind).iterdir() if p.is_dir()):
            target = path / kind / element.name
            if target.exists():
                raise StoreError(f"{target} already exists in the store.")
            os.rename(element, target)
            moved.append(f"{kind}/{element.name}")
    return moved


def remove_uncommitted(path: Path) -> list[str]:
    """Deletes staging and every element folder the consolidated metadata doesn't list (a killed write)."""
    actions = []
    if (path / STAGING).exists():
        shutil.rmtree(path / STAGING)
        actions.append(f"deleted leftover {STAGING}/")
    listed = committed(path)
    for kind, names in on_disk(path).items():
        for name in sorted(names - listed[kind]):
            shutil.rmtree(path / kind / name)
            actions.append(f"deleted uncommitted {kind}/{name}")
    return actions
