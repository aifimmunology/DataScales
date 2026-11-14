"""The local SpatialData `.zarr` grown in place: what a location holds, writability, presence of
samples, staged installs and the repair after a killed run (spec §3S.6, §3S.9).

Commit points are `zarr.consolidate_metadata`: an element counts once it is in the root consolidated
metadata, which is what `spatialdata.read_zarr` lists. A sample goes in with `install_sample`
(elements and its own table moved in, committed by the caller's consolidate); `verify_tables`
checks every table opens the way spatialdata opens it, without loading any matrix."""
from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import zarr

import existing

logger = logging.getLogger(__name__)

STAGING = ".spatial_ingest_staging"
MARKER = "spatial_ingest"
KINDS = (*existing.ELEMENT_KINDS, "tables")
PROBE = ".write_probe"


class StoreError(RuntimeError):
    """A location this capsule can't use as its store."""


def harden_local_writes() -> None:
    """Makes zarr's local chunk writes survive a replayed rename on NFS (idempotent; called before writing).

    zarr's LocalStore writes every chunk to `<key>.<uuid>.partial` and renames it into place. On an NFS
    mount (Code Ocean's /scratch) a RENAME whose reply is lost under load is sent again; the server has
    already done it, so the repeat fails with ENOENT although the chunk is in place. Nothing else ever
    removes a `.partial` name (each is unique to one write), so a missing source with the target
    present is exactly that case and counts as done; every other failure still raises."""
    from zarr.storage import _local

    if getattr(_local._move_with_retry, "spatial_ingest", False):
        return
    original = _local._move_with_retry

    def move_with_retry(tmp_path, path, move):
        def checked(src, dst):
            try:
                return move(src, dst)
            except FileNotFoundError:
                if Path(dst).exists() and not Path(src).exists():
                    logger.debug("rename %s -> %s reported ENOENT but is done (replayed on NFS)", src, dst)
                    return None
                raise
        return original(tmp_path, path, checked)

    move_with_retry.spatial_ingest = True
    _local._move_with_retry = move_with_retry


def retry_once(what: str, step, cleanup):
    """Runs a write step that only writes its own staging (so it can start over), once more after an
    OSError (a shared filesystem under load); a second failure is one StoreError line."""
    try:
        return step()
    except OSError as exc:
        logger.info("Writing %s failed (%s); writing it again", what, exc)
    cleanup()
    try:
        return step()
    except OSError as exc:
        raise StoreError(f"Writing {what} failed twice ({type(exc).__name__}: {exc.strerror or exc}); nothing of it "
                         "was committed, so the store is unchanged. The filesystem may be overloaded: run again.") from None


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
    anndata, icechunk, xenium / atera (a bundle given as the store), file or other."""
    if not path.exists():
        return "missing"
    if path.is_dir() and not any(path.iterdir()):
        return "empty"
    if path.is_file():
        return "file"
    kind = kind_of(root_attrs(path))
    if kind != "other":
        return kind
    bundle = _bundle_kind(path)
    if bundle:
        return bundle
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
    raise StoreError(REFUSALS[kind].format(path=path))


REFUSALS = {
    "anndata": "{path} is an AnnData .zarr, not a SpatialData store: use the anndata_ingest capsule for it, or give a "
               "new Storage Location.",
    "icechunk": "{path} is an icechunk repo; give a new Storage Location (stores are plain SpatialData .zarr folders).",
    "spatialdata": "{path} is a SpatialData store this capsule didn't create, so it isn't grown in place: put it in "
                   "List of paths to samples and give a new Storage Location.",
    "xenium": "{path} is a Xenium bundle: put it in List of paths to samples, and give a new Storage Location.",
    "atera": "{path} is an Atera bundle: put it in List of paths to samples, and give a new Storage Location.",
    "file": "{path} is a file; the Storage Location is a folder (.zarr) or an s3:// prefix. If it holds samples, put "
            "it in List of paths to samples.",
    "other": "{path} is a folder that isn't a SpatialData store; give a new path or a store this capsule built.",
}


def _bundle_kind(path: Path) -> str | None:
    """xenium or atera when the folder (or a folder in it) holds a bundle's experiment file."""
    for pattern, kind in (("experiment.xenium", "xenium"), ("*_experiment.xenium", "xenium"),
                          ("experiment.spatial", "atera")):
        if any(path.glob(pattern)) or any(path.glob(f"*/{pattern}")):
            return kind
    return None


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
    logger.info("Created a new SpatialData store at %s", path)


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


def _move(source: Path, target: Path, staging_dir: Path) -> None:
    """Moves a folder into the store: a rename on the same filesystem, else a copy into staging first,
    so the store never holds a half-copied element under its final name."""
    try:
        os.rename(source, target)
        return
    except OSError:
        pass
    tmp = staging_dir / f".{target.name}.copy"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(source, tmp)
    os.rename(tmp, target)


def install_sample(path: Path, prepared) -> list[str]:
    """Puts a prepared sample (`contracts.Prepared`) into the store: its elements are written (lazily)
    as a SpatialData in staging and renamed into place, and its table group is moved to
    `tables/<sid>_table`. Nothing counts until the caller consolidates; a run killed before that
    leaves folders `remove_uncommitted` deletes."""
    from spatialdata import SpatialData

    import layout

    path = Path(path)
    table = path / "tables" / layout.table_name(prepared.sid)
    if table.exists():
        raise StoreError(f"{table} already exists in the store.")
    staged = staging(path) / f"{prepared.sid}.zarr"
    shutil.rmtree(staged, ignore_errors=True)
    elements = {k: dict(v) for k, v in prepared.elements.items() if k in existing.ELEMENT_KINDS and v}
    harden_local_writes()
    retry_once(f"{prepared.sid}'s elements", lambda: SpatialData(**elements).write(staged, overwrite=True),
               lambda: shutil.rmtree(staged, ignore_errors=True))
    moved = install(path, staged)
    shutil.rmtree(staged, ignore_errors=True)
    zarr.open_group(path, mode="a", use_consolidated=False).require_group("tables")
    _move(Path(prepared.table_path), table, staging(path))
    return moved + [f"tables/{table.name}"]


def _check_table(group: zarr.Group, elements: dict[str, set[str]]) -> None:
    """Raises what spatialdata would refuse in a table, from metadata and obs alone: invalid or
    case-clashing keys, a region link without its column or elements."""
    import anndata as ad
    from anndata.io import read_elem
    from spatialdata._core.validation import validate_table_attr_keys

    lazy = ad.experimental.read_lazy(group)
    validate_table_attr_keys(lazy)
    uns = group.get("uns")
    attrs = dict(read_elem(uns["spatialdata_attrs"])) if uns is not None and "spatialdata_attrs" in uns else {}
    if not attrs and group.attrs.get("region") is not None:
        raise ValueError("its region link is only in the group attributes; spatialdata reads it from "
                         "uns['spatialdata_attrs']")
    region, region_key = attrs.get("region"), attrs.get("region_key")
    if region is not None and not isinstance(region, str):
        region = [str(r) for r in np.atleast_1d(region)]
    if region is None:
        return
    regions = {region} if isinstance(region, str) else {str(r) for r in region}
    if region_key not in group["obs"]:
        raise ValueError(f"its region column {region_key!r} is missing from obs")
    if group["obs"][region_key].attrs.get("encoding-type") != "categorical":
        raise ValueError(f"obs[{region_key!r}] must be categorical")
    found = {str(v) for v in read_elem(group["obs"][region_key]["categories"])}
    if not found <= regions:
        raise ValueError(f"obs[{region_key!r}] holds regions {sorted(found - regions)} its attributes don't list")
    present = set().union(*elements.values())
    if not regions <= present:
        raise ValueError(f"it annotates {sorted(regions - present)}, which the store doesn't hold")
    instance_key = attrs.get("instance_key")
    if instance_key is not None and instance_key not in group["obs"]:
        raise ValueError(f"its instance column {instance_key!r} is missing from obs")


def verify_tables(path: Path) -> int:
    """Opens every committed table as spatialdata would (lazily: no matrix is loaded) and raises a
    StoreError listing those it would refuse, since one bad table makes the whole store unreadable.
    Returns the number of tables checked."""
    import warnings

    path = Path(path)
    listed = committed(path)
    problems = []
    for name in sorted(listed["tables"]):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _check_table(zarr.open_group(path / "tables" / name, mode="r", use_consolidated=False),
                             {k: v for k, v in listed.items() if k != "tables"})
        except Exception as exc:
            problems.append(f"tables/{name}: {type(exc).__name__}: {' '.join(str(exc).split())}")
    if problems:
        raise StoreError(f"{path} holds tables spatialdata can't read: " + "; ".join(problems))
    return len(listed["tables"])
