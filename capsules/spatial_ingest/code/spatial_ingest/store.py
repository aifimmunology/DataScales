"""The store: a SpatialData .zarr grown in place, with a joint table of every sample's cells; and the S3 copy,
lock and sync around it."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import socket
import threading
import time
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
from anndata.io import read_elem, write_elem

import inputs
from config import UserError

logger = logging.getLogger(__name__)

ELEMENT_KINDS = ("images", "labels", "shapes", "points")
STAGING = ".spatial_ingest_staging"
MARKER = "spatial_ingest"
KINDS = (*ELEMENT_KINDS, "tables")
PROBE = ".write_probe"
ELEMENTS_ATTR = "spatial_ingest_elements"     # on a per-sample table: the elements its sample installed
LOCK_SUFFIX = ".spatial_ingest_lock"
LOCK_STALE_MINUTES = 12 * 60                   # a lock from another host older than this is taken over
CREATING = ".creating"


class StoreError(UserError, RuntimeError):
    """A location this capsule can't use as its store."""


def retry_once(what: str, step, cleanup):
    """Runs a step that only writes its own staging once more after an OSError."""
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
    """What is at `path`: missing, empty, ours, spatialdata, anndata, icechunk, xenium, atera, file or other."""
    if not path.exists():
        return "missing"
    if path.is_dir() and not any(path.iterdir()):
        return "empty"
    if path.is_file():
        return "file"
    if abandoned(path):
        return "empty"
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
    """True when the store must be created; refuses anything that isn't (and can't become) one of ours."""
    kind = describe(path)
    if kind == "ours":
        probe_writable(path)
        return False
    refuse(path, kind, may_create)
    return True


def refuse(path, kind: str, may_create: bool) -> None:
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
    """A new empty store, built beside `path` and renamed into place so it always has our marker."""
    from spatialdata import SpatialData

    path = Path(path)
    tmp = path.parent / f".{path.name}{CREATING}"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)
        SpatialData().write(tmp)
    except OSError as exc:
        raise StoreError(f"Can't create a store at {path} ({exc.strerror}); give a writable location such as "
                         "/results/<name>.zarr or /scratch/<name>.zarr.") from None
    zarr.open_group(tmp, mode="r+", use_consolidated=False).attrs[MARKER] = {"version": 1}
    consolidate(tmp)
    if path.exists():
        if any(path.iterdir()) and not abandoned(path):
            shutil.rmtree(tmp, ignore_errors=True)
            raise StoreError(f"{path} appeared while the new store was being created; give it again.")
        shutil.rmtree(path)
    os.rename(tmp, path)
    logger.info("Created a new SpatialData store at %s", path)


def abandoned(path: Path) -> bool:
    """An empty store without our marker, as a creation killed by an earlier version left it."""
    meta = root_attrs(path)
    if meta is None or kind_of(meta) != "spatialdata":
        return False
    if any(committed(path).values()) or any(on_disk(path).values()):
        return False
    allowed = {"zarr.json", "zmetadata", ".zmetadata", ".zgroup", ".zattrs", STAGING, *KINDS}
    return all(p.name in allowed for p in path.iterdir())


def consolidate(path: Path) -> None:
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


def staging(path: Path) -> Path:
    out = path / STAGING
    out.mkdir(exist_ok=True)
    return out


def install(path: Path, staged: Path) -> list[str]:
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
    """Deletes staging and every element folder the consolidated metadata doesn't list."""
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
    """A rename on the same filesystem, else a copy into staging first."""
    try:
        os.rename(source, target)
        return
    except OSError:
        pass
    tmp = staging_dir / f".{target.name}.copy"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(source, tmp)
    os.rename(tmp, target)


SEPARATORS = "_-."
INVALID_CHARS = re.compile(r"[^A-Za-z0-9_.-]")
RESERVED_OBS = ("sample_id", "cell_id", "region", "n_genes_by_counts", "total_counts", "pct_counts_mt",
                "total_counts_mt")


def safe_names(names, reserved=RESERVED_OBS) -> dict[str, str]:
    """old -> new for column names spatialdata would refuse (fine names are left out)."""
    reserved_lower = {r.lower(): r for r in reserved}
    seen: set[str] = set()
    out = {}

    def taken(n: str) -> bool:
        return n.lower() in seen or (n.lower() in reserved_lower and n != reserved_lower[n.lower()])

    for name in names:
        new = INVALID_CHARS.sub("_", str(name)) or "unnamed"
        if new.startswith("__") or new == "_index":
            new = f"source{new}"
        if taken(new):
            new = f"source_{new}"
        base, k = new, 1
        while taken(new):
            new, k = f"{base}_{k}", k + 1
        seen.add(new.lower())
        if new != name:
            out[name] = new
    return out


def strip_tags(name: str, tags: tuple[str, ...]) -> str:
    for tag in tags:
        for s in SEPARATORS:
            if name.startswith(tag + s) and len(name) > len(tag) + 1:
                return name[len(tag) + 1:]
            if name.endswith(s + tag) and len(name) > len(tag) + 1:
                return name[: -len(tag) - 1]
    return name


def element_name(name: str, sid: str, tags: tuple[str, ...] = ()) -> str:
    """`<sid>_<name>`, without doubling the sample's ID or tag."""
    if name in (sid, *tags):
        return sid
    return f"{sid}_{strip_tags(name, (sid, *tags))}"


def system_name(system: str, sid: str, tags: tuple[str, ...] = ()) -> str:
    """`global` (or the sample's own system) becomes `<sid>`, any other `<sid>_<system>`."""
    base = strip_tags(system, (sid, *tags))
    if system in (sid, *tags) or base == "global":
        return sid
    return f"{sid}_{base}"


def table_name(sid: str) -> str:
    return f"{sid}_table"


def install_sample(path: Path, prepared) -> list[str]:
    """Writes a prepared sample's elements and moves its table in; nothing counts until consolidated."""
    from spatialdata import SpatialData

    path = Path(path)
    table = path / "tables" / table_name(prepared.sid)
    if table.exists():
        raise StoreError(f"{table} already exists in the store.")
    staged = staging(path) / f"{prepared.sid}.zarr"
    shutil.rmtree(staged, ignore_errors=True)
    elements = {k: dict(v) for k, v in prepared.elements.items() if k in ELEMENT_KINDS and v}
    inputs.harden_local_writes()

    def write() -> None:
        SpatialData().write(staged, overwrite=True)
        sdata = SpatialData(**elements)
        sdata.path = staged
        for kind, items in elements.items():
            for name, element in items.items():
                logger.info("%s: writing %s %s (%s)", prepared.sid, kind.rstrip("s"), name, _size(kind, element))
                sdata.write_element(name)

    retry_once(f"{prepared.sid}'s elements", write, lambda: shutil.rmtree(staged, ignore_errors=True))
    moved = install(path, staged)
    shutil.rmtree(staged, ignore_errors=True)
    zarr.open_group(path, mode="a", use_consolidated=False).require_group("tables")
    zarr.open_group(prepared.table_path, mode="r+", use_consolidated=False).attrs[ELEMENTS_ATTR] = moved
    _move(Path(prepared.table_path), table, staging(path))
    return moved + [f"tables/{table.name}"]


def _size(kind: str, element) -> str:
    try:
        if kind in ("images", "labels"):
            arr = element
            if hasattr(element, "children") and element.children:      # multiscale: its full resolution
                scale = next(iter(element.children.values())).ds
                arr = scale[next(iter(scale.data_vars))]
            return f"{' x '.join(map(str, arr.shape))}, {arr.data.nbytes / 1e9:.2f} GB"
        if kind == "shapes":
            return f"{len(element)} shapes"
        return f"{element.npartitions} partition(s)"
    except Exception:
        return "size unknown"


def _sample_elements(path: Path, sid: str) -> list[str]:
    table = path / "tables" / table_name(sid)
    try:
        recorded = zarr.open_group(table, mode="r", use_consolidated=False).attrs.get(ELEMENTS_ATTR)
    except Exception:
        recorded = None
    if recorded is not None:
        return [str(e) for e in recorded]
    others = [t.name.removesuffix("_table") for t in (path / "tables").iterdir()
              if t.name.endswith("_table") and t.name != table.name] if (path / "tables").is_dir() else []
    longer = [o for o in others if o.startswith(sid) and len(o) > len(sid)]
    out = []
    for kind, names in on_disk(path).items():
        if kind == "tables":
            continue
        for name in names:
            if (name == sid or name.startswith(f"{sid}_")) and not any(name == o or name.startswith(f"{o}_")
                                                                       for o in longer):
                out.append(f"{kind}/{name}")
    return sorted(out)


def uninstall_sample(path: Path, sid: str) -> list[str]:
    path = Path(path)
    removed = []
    for item in _sample_elements(path, sid):
        if (path / item).is_dir():
            shutil.rmtree(path / item)
            removed.append(item)
    table = path / "tables" / table_name(sid)
    if table.is_dir():
        shutil.rmtree(table)
        removed.append(f"tables/{table.name}")
    return removed


def _check_table(group: zarr.Group, elements: dict[str, set[str]]) -> None:
    """Raises what spatialdata would refuse in a table, from its metadata and obs alone."""
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
    """Opens every committed table as spatialdata would; one bad table makes the whole store unreadable."""
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


def lock_path(path: Path) -> Path:
    path = Path(path)
    return path.parent / f".{path.name}{LOCK_SUFFIX}"


def _alive(info: dict) -> bool:
    if info.get("host") == socket.gethostname():
        try:
            os.kill(int(info.get("pid", -1)), 0)
        except ProcessLookupError:
            return False
        except (PermissionError, ValueError, OverflowError):
            return True
        return True
    return time.time() - float(info.get("started", 0)) < LOCK_STALE_MINUTES * 60


def acquire_lock(path: Path) -> Path:
    """Takes the store's lock file; a lock its run left behind is taken over with a warning."""
    lock = lock_path(path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps({"host": socket.gethostname(), "pid": os.getpid(), "started": time.time()})
    for attempt in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                info = json.loads(lock.read_text() or "{}")
            except (OSError, ValueError):
                info = {}
            if attempt or _alive(info):
                started = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(info.get("started", 0))))
                raise StoreError(f"Another run is adding to {path} (host {info.get('host', '?')}, pid "
                                 f"{info.get('pid', '?')}, started {started}); wait for it to finish, or delete "
                                 f"{lock} if that run is gone.") from None
            logger.warning("Taking over the lock of %s left by a run that is gone (host %s, pid %s)", path,
                           info.get("host", "?"), info.get("pid", "?"))
            lock.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "w") as f:
            f.write(body)
        return lock
    raise StoreError(f"Couldn't lock {path}.")


def release_lock(lock: Path | None) -> None:
    if lock is not None:
        Path(lock).unlink(missing_ok=True)


JOINT_NAME = "table"
UNS_KEY = "spatial_ingest"
MEASURED = "measured_genes"
N_MEASURED = "n_samples_measured"
LEADING = ("sample_id", "cell_id", "n_genes_by_counts", "total_counts", "pct_counts_mt")
EXCLUDED = {"region", "cell_labels", "label_id", "barcode", "total_counts_mt"}
TABLE_ATTRS = {"spatialdata-encoding-type": "ngff:regions_table", "version": "0.2"}
STRINGLIKE = ("categorical", "string-array", "nullable-string-array")


class JointError(UserError, ValueError):
    """Rows the joint table can't take."""


def x_info(group_path: Path) -> tuple[tuple[int, int], np.dtype, int]:
    """Shape, dtype and nonzeros of a CSR X, from metadata and indptr's end."""
    x = zarr.open_group(Path(group_path) / "X", mode="r", use_consolidated=False)
    shape = (int(x.attrs["shape"][0]), int(x.attrs["shape"][1]))
    nnz = int(x["indptr"][-1]) if x["indptr"].shape[0] else 0
    return shape, x["data"].dtype, nnz


@dataclass
class Schema:
    """What the joint table holds."""
    genes: list[str]
    gene_ids: list[str]
    x_dtype: str
    obs: dict[str, str]
    panel_source: str
    celltypist_model: str = ""
    species: str = ""
    samples: list[str] = field(default_factory=list)
    measured: dict[str, list[str]] = field(default_factory=dict)
    intact: bool = True                       # False: the record is missing (a killed rewrite)

    def var(self) -> pd.DataFrame:
        index = pd.Index(np.asarray(self.genes, dtype=object))
        counts = np.zeros(len(index), dtype=np.int64)
        for sid in self.samples:
            measured = self.measured.get(sid)
            counts += 1 if measured is None else index.isin(measured)
        return pd.DataFrame({"gene_ids": np.asarray(self.gene_ids, dtype=object), "feature_types": inputs.GEX,
                             N_MEASURED: counts}, index=index)

    def record(self) -> dict:
        """The schema as stored in `uns/spatial_ingest` (genes and ids live in var)."""
        out = {k: v for k, v in asdict(self).items() if k not in ("genes", "gene_ids", "measured", "intact")}
        out["obs"] = json.dumps(self.obs)
        out["samples"] = np.asarray(self.samples, dtype=object)
        out[MEASURED] = {sid: np.asarray(g, dtype=object) for sid, g in self.measured.items()}
        return out


def joint_path(store: Path) -> Path:
    return Path(store) / "tables" / JOINT_NAME


def _group(path: Path, mode: str = "r") -> zarr.Group:
    return zarr.open_group(path, mode=mode, use_consolidated=False)


def read_schema(store: Path) -> Schema | None:
    """The joint table's schema, or None without one; never raises (a broken table has `intact=False`)."""
    path = joint_path(store)
    if not (path / "zarr.json").exists():
        return None
    try:
        return _read_schema(path)
    except Exception as exc:
        logger.debug("joint table record unreadable: %s", exc)
        return Schema([], [], "float32", {}, "", intact=False)


def _read_schema(path: Path) -> Schema:
    group = _group(path)
    var = read_elem(group["var"]) if "var" in group else pd.DataFrame(index=pd.Index([], dtype=object))
    ids = list(var["gene_ids"].astype(str)) if "gene_ids" in var else [""] * len(var)
    if "uns" not in group or UNS_KEY not in group["uns"]:
        dtype = str(group["X"]["data"].dtype) if "X" in group else "float32"
        return Schema(list(var.index), ids, dtype, {}, "", intact=False)
    record = dict(read_elem(group["uns"][UNS_KEY]))
    measured = {str(k): [str(g) for g in v] for k, v in dict(record.get(MEASURED, {})).items()}
    return Schema(list(var.index), ids, str(record["x_dtype"]), json.loads(record.get("obs", "{}")),
                  str(record.get("panel_source", "")), str(record.get("celltypist_model", "")),
                  str(record.get("species", "")), [str(s) for s in np.atleast_1d(record.get("samples", []))],
                  measured)


def _replace(parent: Path, name: str, value, staging: Path) -> None:
    """Rewrites element `name` in staging, then swaps it in by two renames."""
    tmp = staging / f".{parent.name}.{name}.new"
    old = staging / f".{parent.name}.{name}.old"
    for p in (tmp, old):
        shutil.rmtree(p, ignore_errors=True)
    _write(zarr.open_group(tmp, mode="w", zarr_format=3), name, value)
    if (parent / name).exists():
        os.rename(parent / name, old)
    os.rename(tmp / name, parent / name)
    shutil.rmtree(old, ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)


def _write(group: zarr.Group, name: str, value) -> None:
    """write_elem with every 1-d array in OBS_CHUNK-long chunks."""
    if isinstance(value, dict):
        write_elem(group, name, {})
        for key, item in value.items():
            _write(group[name], key, item)
    elif isinstance(value, pd.DataFrame) or (isinstance(value, np.ndarray) and value.ndim == 1):
        write_elem(group, name, value, dataset_kwargs={"chunks": (inputs.OBS_CHUNK,)})
    else:
        write_elem(group, name, value)


def write_schema(path: Path, schema: Schema, staging: Path) -> None:
    _replace(path, "var", schema.var(), staging)
    if not (path / "uns").exists():
        _group(path, "r+").require_group("uns")
    _replace(path / "uns", UNS_KEY, schema.record(), staging)


def gene_union(store_genes: Sequence[str], store_ids: Sequence[str],
               new_vars: dict[str, tuple[Sequence[str], Sequence[str]]]) -> tuple[list[str], dict[str, dict],
                                                                                  dict[str, list[str]]]:
    """The joint genes after adding samples, as `add_joint_rows` maps them, with per-sample stats and split notes."""
    schema = Schema(list(store_genes), list(store_ids) or [""] * len(store_genes), "", {}, "")
    origin = {g: "the store" for g in schema.genes}
    mapped: dict[str, list[str]] = {}
    added: dict[str, int] = {}
    splits: dict[str, list[str]] = {}
    for sid, (names, ids) in new_vars.items():
        names, ids = [str(g) for g in names], [str(i) for i in ids]
        held = dict(zip(schema.genes, schema.gene_ids))
        try:
            targets, fresh = _targets(schema, pd.Index(np.asarray(names, dtype=object)), ids)
        except JointError:
            targets, fresh = names, [(g, i) for g, i in zip(names, ids) if g not in held]
        splits[sid] = [_split(g, inputs.plain_id(i), t, held, origin) for g, i, t in zip(names, ids, targets)
                       if t not in held and t == f"{g}-{inputs.plain_id(i)}"]
        schema.genes += [g for g, _ in fresh]
        schema.gene_ids += [i for _, i in fresh]
        origin.update({g: f"sample {sid}" for g, _ in fresh})
        mapped[sid], added[sid] = targets, len(fresh)
    stats = {}
    for sid, targets in mapped.items():
        own = set(targets)
        others = set(store_genes).union(*(set(t) for s, t in mapped.items() if s != sid))
        shared = len(own & others) if others else len(own)
        stats[sid] = {"genes_in_panel": len(own), "genes_shared": shared,
                      "pct_shared": round(100.0 * shared / len(own), 1) if own else 0.0,
                      "genes_added": added[sid], "genes_missing": len(schema.genes) - len(own),
                      "genes_split": len(splits[sid])}
    return schema.genes, stats, splits


def _split(symbol: str, gene_id: str, target: str, held: dict[str, str], origin: dict[str, str]) -> str:
    other = symbol if symbol in held else next((g for g in held if g.startswith(f"{symbol}-")), symbol)
    return (f"{symbol} has gene_id {gene_id} here and {held.get(other) or '?'} in {origin.get(other, 'the store')}; "
            f"kept as separate genes {other} and {target}")


def _note(n_new: int, n_union: int, n_missing: int) -> str:
    parts = []
    if n_new:
        parts.append(f"{n_new} new gene(s) added to the joint table (now {n_union})")
    if n_missing:
        parts.append(f"{n_missing} joint gene(s) not in its panel (zero in its rows, not counted in "
                     f"var['{N_MEASURED}'])")
    return "; ".join(parts)


def obs_kinds(obs: pd.DataFrame) -> dict[str, str]:
    out = {}
    for c, dtype in obs.dtypes.items():
        if isinstance(dtype, pd.CategoricalDtype):
            out[c] = "category"
        elif pd.api.types.is_bool_dtype(dtype):
            out[c] = "bool"
        elif pd.api.types.is_integer_dtype(dtype):
            out[c] = "int"
        elif pd.api.types.is_float_dtype(dtype):
            out[c] = "float"
        else:
            out[c] = "str"
    return out


def _as_str(values: pd.Series) -> pd.Series:
    objects = values.astype(object)
    return pd.Series([None if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v) for v in objects],
                     index=values.index, dtype="str")


def normalized(obs: pd.DataFrame) -> pd.DataFrame:
    """obs in encodings an in-place append can extend."""
    out = {}
    for c in obs.columns:
        v = obs[c]
        if isinstance(v.dtype, pd.CategoricalDtype):
            out[c] = pd.Categorical(_as_str(v))
        elif pd.api.types.is_bool_dtype(v.dtype):
            out[c] = v.astype("boolean") if v.isna().any() else v.astype(bool)
        elif pd.api.types.is_integer_dtype(v.dtype):
            out[c] = v.astype("Int64") if v.isna().any() else v.astype(np.int64)
        elif pd.api.types.is_float_dtype(v.dtype):
            out[c] = v.astype(np.float64)
        else:
            out[c] = _as_str(v)
    return pd.DataFrame(out, index=obs.index)


def store_obs(path: Path) -> dict[str, tuple[str, np.dtype | None]]:
    """The joint obs columns by (family, dtype), from their encodings alone."""
    obs = _group(path / "obs")
    out = {}
    for c in obs.attrs.get("column-order", []):
        node = obs[c]
        if node.attrs.get("encoding-type") in STRINGLIKE:
            out[c] = ("str", None)
            continue
        values = node if isinstance(node, zarr.Array) else node["values"]
        out[c] = ({"b": "bool", "i": "int", "u": "int", "f": "float"}.get(values.dtype.kind, "other"), values.dtype)
    return out


def _fit_column(v: pd.Series, family: str, dtype: np.dtype | None):
    """A sample's column in the store column's family and dtype, or None when that would change its values."""
    mine = obs_kinds(v.to_frame(name="c"))["c"]
    if family == "str":
        return v if mine in ("category", "str") else pd.Categorical(_as_str(v))
    if mine in ("category", "str") or family == "other":
        return None
    if family == "float":
        return v.astype(dtype or np.float64)
    if family == "bool":
        return (v.astype("boolean") if v.isna().any() else v.astype(bool)) if mine == "bool" else None
    finite = v.dropna().astype(np.float64)
    if mine == "float" and not np.all(np.mod(finite, 1) == 0):
        return None
    dtype = np.dtype(dtype or np.int64)
    return v.astype(f"Int{dtype.itemsize * 8}") if v.isna().any() else v.astype(dtype)


def rows_obs(obs: pd.DataFrame, sid: str, region_key: str | None) -> pd.DataFrame:
    """A per-sample table's obs as joint rows, indexed `<sid>:<cell_id>`."""
    cell_ids = (obs["cell_id"] if "cell_id" in obs else pd.Series(obs.index, index=obs.index)).astype(str)
    names = pd.Index([f"{sid}:{c}" for c in cell_ids], dtype=object)
    if not names.is_unique and region_key in obs:
        names = pd.Index([f"{sid}:{r}:{c}" for r, c in zip(obs[region_key].astype(str), cell_ids)], dtype=object)
    if not names.is_unique:
        raise JointError(f"{sid}: cell IDs repeat within the sample, so its joint cell names wouldn't be unique.")
    keep = [c for c in obs.columns if c not in EXCLUDED and c != region_key and not str(c).startswith("leiden")
            and c not in ("sample_id", "cell_id")]
    rest = normalized(obs[keep].rename(columns=safe_names(keep, reserved=LEADING)))
    head = pd.DataFrame({"sample_id": pd.Categorical([sid] * len(obs)), "cell_id": _as_str(cell_ids)},
                        index=obs.index)
    out = pd.concat([head, rest], axis=1)
    out = out[[c for c in LEADING if c in out] + [c for c in out.columns if c not in LEADING]]
    out.index = names
    return out


def fit_obs(obs: pd.DataFrame, target: Path, sid: str, warn: bool = True) -> pd.DataFrame:
    """Joint rows' obs fitted to the joint table's columns; a column that can't fit is left out with a warning."""
    out, dropped = obs.copy(), []
    obs_group = _group(target / "obs")
    for c, (family, dtype) in store_obs(target).items():
        if c not in out:
            if obs_group[c].attrs.get("encoding-type") == "string-array":
                out[c] = ""
            continue
        fitted = _fit_column(out[c], family, dtype)
        if fitted is None:
            out = out.drop(columns=c)
            dropped.append(c)
        else:
            out[c] = fitted
    if dropped and warn:
        logger.warning("%s: obs columns %s are left out of the joint table: their values don't fit the type the joint "
                       "table already holds them in (the sample's own table keeps them).", sid, ", ".join(dropped))
    return out


def _stage_rows(staged: Path, table_path: Path, obs: pd.DataFrame, union_genes: Sequence[str],
                targets: Sequence[str], dtype: np.dtype, sid: str) -> None:
    var = pd.DataFrame(index=pd.Index(np.asarray(union_genes, dtype=object)))
    columns = var.index.get_indexer(pd.Index(np.asarray(targets, dtype=object))).astype(np.int32)
    writer = inputs.TableWriter(staged, obs, var, dtype)
    for _, chunk in inputs.iter_rows(table_path):
        if not inputs.holds_counts(chunk):
            raise JointError(f"{sid}: its table's X holds non-count values; the joint table takes raw counts only.")
        moved = sp.csr_matrix((chunk.data, columns[chunk.indices], chunk.indptr), shape=(chunk.shape[0], len(var)))
        try:
            writer.add(inputs.as_dtype(moved, dtype))
        except inputs.PanelError as exc:
            raise JointError(f"{sid}: {exc}") from None
    writer.close()


def _extend_var(target: Path, schema: Schema, staging: Path) -> None:
    """New genes go at the end of var, so stored CSR rows stay valid."""
    _replace(target, "var", schema.var(), staging)
    x = _group(target / "X", "r+")
    x.attrs["shape"] = [int(x.attrs["shape"][0]), len(schema.genes)]


def _fix_chunks(group_path: Path, chunk: int, staging: Path) -> None:
    """Rewrites 1-d arrays to `chunk`-long chunks, so appends touch a few files and stay readable mid-sync."""
    group = zarr.open_array(group_path, mode="r") if (group_path / "zarr.json").is_file() and \
        json.loads((group_path / "zarr.json").read_text()).get("node_type") == "array" else _group(group_path)
    nodes = [group] if isinstance(group, zarr.Array) else [n for _, n in group.members(max_depth=None)]
    for node in nodes:
        if not isinstance(node, zarr.Array) or node.ndim != 1 or (node.shards or node.chunks)[0] == chunk:
            continue
        where = group_path / node.path[len(group.path):].lstrip("/") if not isinstance(group, zarr.Array) \
            else group_path
        tmp, old = staging / ".rechunk.new", staging / ".rechunk.old"
        for p in (tmp, old):
            shutil.rmtree(p, ignore_errors=True)
        zarr.create_array(store=str(tmp), data=node[:], chunks=(chunk,), fill_value=node.fill_value,
                          attributes=dict(node.attrs), zarr_format=3)
        os.rename(where, old)
        os.rename(tmp, where)
        shutil.rmtree(old, ignore_errors=True)


def _commit(table: Path) -> None:
    """Consolidates the joint table's own metadata, so readers of the table alone see it whole."""
    zarr.consolidate_metadata(str(table))


def annizarr_config(cpus: int):
    from annizarr import AppConfig
    from annizarr.config import ChunkConfig, IOConfig

    return AppConfig(io=IOConfig(overwrite=True), chunks=ChunkConfig(cpus=cpus))


def _targets(schema: Schema, sample_genes: pd.Index, gene_ids: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Where each of a sample's genes goes in the joint var, and the genes it adds; matched by gene_id, else symbol."""
    plain = inputs.plain_id
    by_id = {plain(i): g for g, i in zip(schema.genes, schema.gene_ids) if i}
    ids = {g: plain(i) for g, i in zip(schema.genes, schema.gene_ids)}
    suffixed = {g.removesuffix(f"-{i}") for g, i in zip(schema.genes, schema.gene_ids) if i and g.endswith(f"-{i}")}
    out, fresh = [], []
    for g, i in zip(sample_genes, gene_ids):
        i = plain(i) if i else i
        if i and i in by_id:
            t = by_id[i]
        elif i and ((g in ids and ids[g] and ids[g] != i) or g in suffixed):
            t = f"{g}-{i}"
        else:
            t = g
        if t not in ids:
            fresh.append((t, i))
            ids[t] = i
            if i:
                by_id[i] = t
        out.append(t)
    if len(set(out)) != len(out):
        dup = pd.Index(out)[pd.Index(out).duplicated()][:5]
        raise JointError(f"several of the sample's genes map to the same joint gene ({', '.join(map(str, dup))}); "
                         "check its gene names and gene_ids.")
    return out, fresh


def joint_dtype(dtype) -> np.dtype:
    """At least 32 bits, so a later sample with larger counts still fits."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.unsignedinteger):
        return np.promote_types(dtype, np.uint32)
    if np.issubdtype(dtype, np.integer):
        return np.promote_types(dtype, np.int32)
    if np.issubdtype(dtype, np.floating):
        return np.promote_types(dtype, np.float32)
    return dtype


@dataclass
class _Rows:
    schema: Schema
    rows: pd.DataFrame
    targets: list[str]
    n_new: int
    n_missing: int
    seed: bool


def _plan_rows(local: Path, sid: str, table_path: Path, celltypist_model: str, warn: bool = True) -> _Rows:
    group = _group(table_path)
    var = read_elem(group["var"])
    obs = read_elem(group["obs"])
    (n_obs, n_vars), x_dtype, _ = x_info(table_path)
    if (n_obs, n_vars) != (len(obs), len(var)):
        raise JointError(f"{sid}: its table's X is {n_obs} x {n_vars}, but obs/var hold {len(obs)} x {len(var)}.")
    if not var.index.is_unique:
        raise JointError(f"{sid}: its table's gene names repeat; they must be unique to place them in the joint table.")
    target = local / "tables" / JOINT_NAME
    schema = read_schema(local)
    if schema is not None and not schema.intact:
        raise JointError(f"{target} is incomplete (a killed run); the repair rebuilds it before samples are added.")
    if schema is not None and sid in schema.samples:
        raise JointError(f"{sid} is already in the joint table.")
    sample_genes = pd.Index(np.asarray(var.index, dtype=object))
    gene_ids = [str(i) if not pd.isna(i) else "" for i in var["gene_ids"]] if "gene_ids" in var else [""] * len(var)
    rows = rows_obs(obs, sid, group.attrs.get("region_key"))
    if schema is None:
        schema = Schema(list(sample_genes), gene_ids, str(joint_dtype(x_dtype)), obs_kinds(rows), sid, celltypist_model,
                        inputs.species(var) or "")
        return _Rows(schema, rows, list(sample_genes), 0, 0, True)
    targets, fresh = _targets(schema, sample_genes, gene_ids)
    n_missing = len(schema.genes) - (len(targets) - len(fresh))
    schema.genes += [g for g, _ in fresh]
    schema.gene_ids += [i for _, i in fresh]
    rows = fit_obs(rows, target, sid, warn)
    return _Rows(schema, rows, targets, len(fresh), n_missing, False)


def _encodable(obs: pd.DataFrame, sid: str) -> None:
    try:
        write_elem(zarr.open_group(zarr.storage.MemoryStore(), mode="w", zarr_format=3), "obs", obs)
    except Exception as exc:
        raise JointError(f"{sid}: its obs can't be stored in the joint table ({type(exc).__name__}: {exc}).") from None


def check_joint_rows(local: Path, sid: str, table_path: Path) -> None:
    """Raises JointError for anything `add_joint_rows` would refuse, without writing."""
    local, table_path = Path(local), Path(table_path)
    planned = _plan_rows(local, sid, table_path, "", warn=False)
    _encodable(planned.rows, sid)
    dtype = np.dtype(planned.schema.x_dtype)
    for _, chunk in inputs.iter_rows(table_path):
        if not inputs.holds_counts(chunk):
            raise JointError(f"{sid}: its table's X holds non-count values; the joint table takes raw counts only.")
        try:
            inputs.as_dtype(chunk, dtype)
        except inputs.PanelError as exc:
            raise JointError(f"{sid}: {exc}") from None


def add_joint_rows(local: Path, sid: str, table_path: Path, staging: Path, cpus: int, celltypist_model: str = "") -> str:
    """Seeds the joint table with the sample's rows, or extends var and appends them in place."""
    import annizarr

    local, table_path, staging = Path(local), Path(table_path), Path(staging)
    staging.mkdir(parents=True, exist_ok=True)
    planned = _plan_rows(local, sid, table_path, celltypist_model)
    schema, rows, target = planned.schema, planned.rows, local / "tables" / JOINT_NAME
    staged = staging / f"{sid}.joint.zarr"
    retry_once(f"{sid}'s joint rows", lambda: _stage_rows(staged, table_path, rows, schema.genes,
                                                          planned.targets, np.dtype(schema.x_dtype), sid),
               lambda: shutil.rmtree(staged, ignore_errors=True))
    if planned.seed:
        schema.samples, schema.measured = [sid], {sid: list(planned.targets)}
        _group(staged, "r+").attrs.update(TABLE_ATTRS)
        write_schema(staged, schema, staging)
        _commit(staged)
        _group(local, "a").require_group("tables")
        os.rename(staged, target)
    else:
        if planned.n_new:
            _extend_var(target, schema, staging)
        _fix_chunks(target / "obs", inputs.OBS_CHUNK, staging)
        _commit(target)                                  # annizarr reads the store through this metadata
        annizarr.append(str(target), cells=str(staged), include_new_obs=True, cfg=annizarr_config(cpus))
        _fix_chunks(target / "obs", inputs.OBS_CHUNK, staging)       # columns annizarr just added or rewrote whole
        _fix_chunks(target / "X" / "indptr", inputs.INDPTR_CHUNK, staging)   # annizarr rewrites it in one chunk
        schema.samples.append(sid)
        schema.measured[sid] = list(planned.targets)
        schema.obs = {**obs_kinds(rows), **schema.obs}
        write_schema(target, schema, staging)
        _commit(target)
    shutil.rmtree(staged, ignore_errors=True)
    return _note(planned.n_new, len(schema.genes), planned.n_missing)


def repair_joint(local: Path, present: list[str], staging: Path, cpus: int, models: dict[str, str] | None = None) -> list[str]:
    """Rebuilds the joint table from the per-sample tables when it is broken, appends missing samples,
    and takes out a sample it can't take."""
    from annizarr.errors import AnzError

    local, staging = Path(local), Path(staging)
    models = models or {}
    actions: list[str] = []
    order: list[str] = []
    has_joint = (joint_path(local) / "zarr.json").exists()
    counts = sample_counts(local) if has_joint else pd.Series(dtype=np.int64)
    sizes = {sid: inputs.n_obs(_group(local / "tables" / f"{sid}_table")) for sid in present}
    if counts is None or any(sizes.get(sid) != n for sid, n in counts.items()):
        schema = read_schema(local)
        recorded = schema.samples if schema is not None else []
        order = [sid for sid in recorded if sid in sizes]
        drop_joint(local)
        counts = pd.Series(dtype=np.int64)
        actions.append("the joint table was inconsistent (a killed append or sync); rebuilt from the per-sample tables")
    missing = [sid for sid in present if sid not in counts.index and sid not in order]
    for sid in order + missing:
        try:
            add_joint_rows(local, sid, local / "tables" / f"{sid}_table", staging, cpus, models.get(sid, ""))
        except _data_errors() as exc:
            if sample_counts(local) is None and joint_path(local).exists():
                return actions + _quarantine_and_rebuild(local, sid, exc, present, staging, cpus, models)
            uninstall_sample(local, sid)
            logger.warning("%s is taken out of the store: its rows can't go into the joint table (%s). Add it "
                           "again once that is fixed.", sid, exc)
            actions.append(f"took {sid} out of the store (its rows can't go into the joint table)")
            continue
        except (StoreError, OSError, AnzError) as exc:
            detail = str(exc).split(";")[0] if isinstance(exc, StoreError) else f"{type(exc).__name__}: {exc}"
            detail = " ".join(detail.split())
            raise StoreError(f"Repairing the joint table failed while adding {sid}'s rows ({detail}); nothing was "
                             "removed from the store; run again.") from None
        if sid in missing:
            actions.append(f"appended present sample {sid}, missing from the joint table")
    return actions


def _data_errors() -> tuple[type[Exception], ...]:
    from annizarr.errors import ConversionError, ValidationError

    return JointError, inputs.PanelError, ValidationError, ConversionError


def _quarantine_and_rebuild(local: Path, sid: str, exc: Exception, present: list[str], staging: Path, cpus: int,
                            models: dict[str, str]) -> list[str]:
    uninstall_sample(local, sid)
    logger.warning("%s is taken out of the store: its rows can't go into the joint table (%s). Add it again once "
                   "that is fixed.", sid, exc)
    drop_joint(local)
    rest = [s for s in present if s != sid]
    return [f"took {sid} out of the store (its rows can't go into the joint table)"] + \
        repair_joint(local, rest, staging, cpus, models)


def sample_counts(store: Path) -> pd.Series | None:
    """Cells per sample in the joint table, or None when it is inconsistent."""
    path = joint_path(store)
    try:
        group = _group(path)
        sample = read_elem(group["obs"]["sample_id"])
        n_index = inputs.n_obs(group)
        shape = group["X"].attrs["shape"]
        indptr = group["X"]["indptr"].shape[0]
        var = group["var"]
        n_var = inputs.column_length(var[var.attrs["_index"]])
        obs = group["obs"]
        lengths = {inputs.column_length(obs[c]) for c in obs.attrs.get("column-order", [])}
        schema = read_schema(store)
    except Exception:
        return None
    if not (len(sample) == n_index == shape[0] == indptr - 1) or n_var != shape[1] or lengths - {n_index}:
        return None
    counts = pd.Series(np.asarray(sample, dtype=object)).value_counts()
    if schema is None or not schema.intact or set(counts.index) != set(schema.samples):
        return None
    return counts


def drop_joint(store: Path) -> None:
    shutil.rmtree(joint_path(store), ignore_errors=True)


PART_SIZE = 8 * 1024 * 1024
PART_THREADS = 4                    # threads per multipart transfer (boto3 default 10)
ROOT_META = "zarr.json"
POOL_MIN = 32
CONTROL = ".spatial_ingest_"                  # control objects at the prefix root, never part of the store
LOCK = ".spatial_ingest_lock"
S3_CREATING = ".spatial_ingest_creating"
ROOT_ETAG = ".root_etag"                      # key in download()'s result: the root zarr.json's ETag
S3_LOCK_STALE_MINUTES = 10
WORKERS = 16
HEARTBEAT_MIN = 5.0                           # seconds between lock refreshes, at least
ACCESS = {"AccessDenied": "access denied for this capsule's AWS identity",
          "AllAccessDisabled": "access denied for this capsule's AWS identity",
          "NoCredentialsError": "this capsule has no AWS credentials",
          "InvalidAccessKeyId": "this capsule's AWS credentials aren't valid",
          "SignatureDoesNotMatch": "this capsule's AWS credentials aren't valid",
          "InvalidToken": "this capsule's AWS credentials aren't valid",
          "ExpiredToken": "this capsule's AWS credentials have expired"}
ATTACH = ("Attach an AWS secret with read/write access to this bucket in Code Ocean (capsule › Environment › "
          "Secrets / cloud credentials), or use a path under /results for a one-off store.")
METADATA = ("zarr.json", ".zarray", ".zgroup", ".zattrs", ".zmetadata", "zmetadata")

_HELD: dict[str, "S3Lock"] = {}               # url -> this process's lock on it


class S3Error(UserError, RuntimeError):
    """An S3 location that can't be read or written."""


def split(url: str) -> tuple[str, str]:
    """(bucket, prefix) of an s3:// store URL; a whole bucket is refused."""
    bucket, _, prefix = url.removeprefix("s3://").partition("/")
    if not bucket:
        raise S3Error(f"{url} names no bucket; give s3://bucket/prefix.")
    if not prefix.strip("/"):
        raise S3Error(f"{url} is a whole bucket; give a prefix inside it for the store, e.g. "
                      f"s3://{bucket}/spatialdata.zarr.")
    return bucket, prefix.strip("/")


def client(workers: int = WORKERS):
    """An S3 client whose connection pool fits the parallel transfers."""
    import boto3
    from botocore.config import Config

    return boto3.client("s3", config=Config(max_pool_connections=max(POOL_MIN, PART_THREADS * workers)))


def failures() -> tuple[type[Exception], ...]:
    from boto3.exceptions import Boto3Error
    from botocore.exceptions import BotoCoreError, ClientError

    return BotoCoreError, ClientError, Boto3Error, OSError


def _code_and_operation(exc: Exception) -> tuple[str, str]:
    error = getattr(exc, "response", None) or {}
    code = error.get("Error", {}).get("Code", "")
    operation = getattr(exc, "operation_name", "")
    if not code:
        found = re.search(r"An error occurred \(([^)]+)\) when calling the (\w+) operation", str(exc))
        if found:
            code, operation = found.groups()
    return code, operation


def error(exc: Exception, url: str, doing: str, state: str = "") -> S3Error:
    """One clear S3Error for a failed S3 call, without the policy text."""
    code, operation = _code_and_operation(exc)
    message = (getattr(exc, "response", None) or {}).get("Error", {}).get("Message", "") if code else str(exc)
    detail = " ".join(message.split())[:200] if code not in ACCESS else ""
    code = code or type(exc).__name__
    raw = f"({code}{f' on {operation}' if operation else ''})"
    state = f" {state}" if state else ""
    if code in ACCESS:
        return S3Error(f"Can't {doing} {url}: {ACCESS[code]}. {ATTACH}{state}\n{raw}")
    if code == "NoSuchBucket":
        bucket = url.removeprefix("s3://").split("/", 1)[0]
        return S3Error(f"Can't {doing} {url}: the bucket {bucket} doesn't exist; check the Storage Location.{state}"
                       f"\n{raw}")
    return S3Error(f"Can't {doing} {url}{f': {detail}' if detail else ''}.{state}\n{raw}")


def transfer_config():
    from boto3.s3.transfer import TransferConfig

    return TransferConfig(multipart_threshold=PART_SIZE, multipart_chunksize=PART_SIZE,
                          max_concurrency=PART_THREADS)


def _key(prefix: str, rel: str) -> str:
    return f"{prefix}/{rel}" if prefix else rel


def _code(exc: Exception) -> str:
    return (getattr(exc, "response", None) or {}).get("Error", {}).get("Code", "")


def remote_files(s3, bucket: str, prefix: str, control: bool = False) -> dict[str, str]:
    """Relative path -> ETag of every object under the prefix."""
    out = {}
    base = f"{prefix}/" if prefix else ""
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=base):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(base):]
            if not obj["Key"].endswith("/") and (control or not rel.startswith(CONTROL)):
                out[rel] = obj["ETag"].strip('"')
    return out


def local_files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def etag(path: Path) -> str:
    """The ETag S3 gives this file when uploaded with `transfer_config`."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        if size < PART_SIZE:
            return hashlib.md5(f.read()).hexdigest()
        digests = [hashlib.md5(chunk).digest() for chunk in iter(lambda: f.read(PART_SIZE), b"")]
    return f"{hashlib.md5(b''.join(digests)).hexdigest()}-{len(digests)}"


@dataclass
class S3Lock:
    url: str
    etag: str
    body: dict
    stale_minutes: float
    lost: bool = False
    stop: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    @property
    def location(self) -> tuple[str, str]:
        bucket, prefix = split(self.url)
        return bucket, _key(prefix, LOCK)


def _put_lock(s3, bucket: str, key: str, body: dict, **condition) -> str:
    return s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(body).encode(), **condition)["ETag"]


def _read_lock(s3, bucket: str, key: str) -> tuple[dict, str] | None:
    try:
        held = s3.get_object(Bucket=bucket, Key=key)
    except Exception as exc:
        if _code(exc) in ("NoSuchKey", "404", "NotFound"):
            return None
        raise
    return json.loads(held["Body"].read() or b"{}"), held["ETag"]


def _ours(body: dict, info: dict) -> bool:
    return all(info.get(k) == body[k] for k in ("host", "pid", "started"))


def acquire_s3_lock(url: str, stale_minutes: float = S3_LOCK_STALE_MINUTES) -> S3Lock:
    """Takes the store's S3 lock, kept fresh by a heartbeat; a stale lock is taken over with a warning."""
    s3 = client()
    bucket, prefix = split(url)
    key = _key(prefix, LOCK)
    stale = float(stale_minutes)
    now = time.time()
    body = {"host": socket.gethostname(), "pid": os.getpid(), "started": now, "heartbeat": now}
    try:
        tag = _put_lock(s3, bucket, key, body, IfNoneMatch="*")
    except failures() as exc:
        if _code(exc) not in ("PreconditionFailed", "ConditionalRequestConflict"):
            raise error(exc, url, "write to", "Nothing was written.") from None
        tag = _take_over(s3, bucket, key, url, body, stale)
    lock = S3Lock(url, tag, body, stale)
    lock.thread = threading.Thread(target=_heartbeat, args=(lock,), daemon=True, name="s3-lock-heartbeat")
    lock.thread.start()
    _HELD[url] = lock
    return lock


def _dead_here(info: dict) -> bool:
    if info.get("host") != socket.gethostname():
        return False
    try:
        os.kill(int(info.get("pid", -1)), 0)
    except ProcessLookupError:
        return True
    except (PermissionError, ValueError, OverflowError):
        return False
    return False


def _take_over(s3, bucket: str, key: str, url: str, body: dict, stale: float) -> str:
    try:
        held = s3.get_object(Bucket=bucket, Key=key)
        info, old = json.loads(held["Body"].read() or b"{}"), held["ETag"]
    except Exception:
        info, old = {}, None
    age = (time.time() - float(info.get("heartbeat", 0))) / 60
    if old is not None and age < stale and not _dead_here(info):
        started = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(float(info.get("started", 0))))
        raise S3Error(f"Another run is adding to {url} (host {info.get('host', '?')}, pid {info.get('pid', '?')}, "
                      f"started {started}, last heartbeat {age:.0f} min ago); wait for it to finish. A lock without "
                      f"heartbeat for {stale:.0f} min is taken over.")
    why = "is no longer running on this host" if _dead_here(info) else f"sent no heartbeat for {age:.0f} min"
    logger.warning("Taking over the lock of %s: its run (host %s, pid %s) %s", url, info.get("host", "?"),
                   info.get("pid", "?"), why)
    try:
        return _put_lock(s3, bucket, key, body, **({"IfMatch": old} if old else {"IfNoneMatch": "*"}))
    except failures() as exc:
        if _code(exc) not in ("PreconditionFailed", "ConditionalRequestConflict"):
            raise error(exc, url, "write to", "Nothing was written.") from None
        raise S3Error(f"Another run took the lock of {url} at the same time; run again later.") from None


def _heartbeat(lock: S3Lock) -> None:
    """Lost only when another run holds the lock, or no refresh got through for half the stale limit."""
    interval = max(HEARTBEAT_MIN, lock.stale_minutes * 60 / 4)
    s3 = client()
    bucket, key = lock.location
    refreshed = lock.body["heartbeat"]
    while not lock.stop.wait(interval):
        lock.body["heartbeat"] = time.time()
        try:
            lock.etag = _put_lock(s3, bucket, key, lock.body, IfMatch=lock.etag)
            refreshed = lock.body["heartbeat"]
            continue
        except Exception as exc:
            why = _code(exc) or type(exc).__name__
            refused = why in ("PreconditionFailed", "ConditionalRequestConflict")
        if refused:
            try:
                held = _read_lock(s3, bucket, key)
            except Exception as exc:
                held, why = False, f"{why}, then {_code(exc) or type(exc).__name__} reading it back"
            if held is None or (held and not _ours(lock.body, held[0])):
                lock.lost = True
                logger.warning("Lost the lock of %s (another run took it over or removed it); this run won't sync to "
                               "it", lock.url)
                return
            if held:
                lock.etag = held[1]
                refreshed = float(held[0].get("heartbeat", refreshed))
                continue
        if time.time() - refreshed > lock.stale_minutes * 60 / 2:
            lock.lost = True
            logger.warning("Lost the lock of %s: its heartbeat couldn't be refreshed for %.0f min (%s); this run won't "
                           "sync to it", lock.url, (time.time() - refreshed) / 60, why)
            return
        logger.info("Refreshing the lock of %s failed (%s); trying again in %.0f s", lock.url, why, interval)


def release_s3_lock(lock: S3Lock | None) -> None:
    if lock is None:
        return
    lock.stop.set()
    if lock.thread is not None:
        lock.thread.join(timeout=30)
    _HELD.pop(lock.url, None)
    s3 = client()
    bucket, key = lock.location
    try:
        held = _read_lock(s3, bucket, key)
        if held is None or not _ours(lock.body, held[0]):
            return
        try:
            s3.delete_object(Bucket=bucket, Key=key, IfMatch=held[1])
        except Exception as exc:
            if _code(exc) in ("PreconditionFailed", "NoSuchKey", "404"):
                return
            s3.delete_object(Bucket=bucket, Key=key)       # endpoints without conditional delete
    except Exception:
        logger.warning("Couldn't remove the lock of %s; it expires after %.0f min", lock.url, lock.stale_minutes)


def _check_lock(url: str) -> None:
    lock = _HELD.get(url)
    if lock is not None and lock.lost:
        raise S3Error(f"This run lost its lock on {url} (another run took it over); nothing was synced. Run again.")


def _root_etag(s3, bucket: str, prefix: str) -> str | None:
    try:
        return s3.head_object(Bucket=bucket, Key=_key(prefix, ROOT_META))["ETag"].strip('"')
    except Exception as exc:
        if _code(exc) in ("404", "NoSuchKey", "NotFound"):
            return None
        raise error(exc, f"s3://{bucket}/{_key(prefix, ROOT_META)}", "read", "Nothing was synced; run again.") \
            from None


def _delete(s3, bucket: str, prefix: str, rels: list[str]) -> None:
    for start in range(0, len(rels), 1000):
        batch = [{"Key": _key(prefix, rel)} for rel in rels[start:start + 1000]]
        if not batch:
            continue
        refused = s3.delete_objects(Bucket=bucket, Delete={"Objects": batch, "Quiet": True}).get("Errors") or []
        if refused:
            keys = ", ".join(e.get("Key", "?") for e in refused[:3]) + (", …" if len(refused) > 3 else "")
            raise S3Error(f"S3 refused to delete {len(refused)} file(s) under s3://{bucket}/{prefix} "
                          f"({refused[0].get('Code', '?')}): {keys}")


def root_meta(url: str, stale_minutes: float = S3_LOCK_STALE_MINUTES) -> dict | None:
    """The store's root zarr.json: None for no store, {} for no root; a killed first sync is cleared."""
    s3 = client()
    bucket, prefix = split(url)
    try:
        files = remote_files(s3, bucket, prefix, control=True)
        store = [rel for rel in files if not rel.startswith(CONTROL)]
        if not store:
            return None
        if ROOT_META in files:
            return json.loads(s3.get_object(Bucket=bucket, Key=_key(prefix, ROOT_META))["Body"].read())
    except failures() as exc:
        raise error(exc, url, "read", "Nothing was written.") from None
    if S3_CREATING not in files:
        return {}
    if LOCK in files and url not in _HELD:
        try:
            info = json.loads(s3.get_object(Bucket=bucket, Key=_key(prefix, LOCK))["Body"].read() or b"{}")
        except Exception:
            info = {}
        if (time.time() - float(info.get("heartbeat", 0))) / 60 < stale_minutes:
            raise S3Error(f"Another run is creating the store at {url}; wait for it to finish.")
    logger.warning("%s holds a store whose creation was killed (%d files, no root zarr.json); cleared to create "
                   "it again", url, len(store))
    try:
        _delete(s3, bucket, prefix, store + [S3_CREATING])
    except failures() as exc:
        raise error(exc, url, "write to", "Nothing else was changed; run again.") from None
    return None


def stat(path: Path) -> tuple[int, int]:
    info = path.stat()
    return info.st_size, info.st_mtime_ns


def download(url: str, local: Path, workers: int) -> dict:
    """Copies the store to `local`; returns each file's size and mtime, and the root's ETag."""
    s3 = client(workers)
    bucket, prefix = split(url)
    cfg = transfer_config()

    def get(rel: str) -> None:
        target = local / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, _key(prefix, rel), str(target), Config=cfg)

    try:
        files = remote_files(s3, bucket, prefix)
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(get, files))
    except failures() as exc:
        raise error(exc, url, "read", "Nothing was changed in the S3 store; run again.") from None
    logger.info("Copied %d files from %s to %s", len(files), url, local)
    out: dict = {rel: stat(local / rel) for rel in files}
    out[ROOT_ETAG] = files.get(ROOT_META)
    return out


def _phase(name: str) -> None:
    """A hook for fault-injection tests."""
    logger.debug("sync phase done: %s", name)


def _is_meta(rel: str) -> bool:
    return rel.rsplit("/", 1)[-1] in METADATA


def sync(local: Path, url: str, workers: int, downloaded: dict | None = None) -> dict[str, int]:
    """Uploads new data, changed data, metadata deepest first, the root, then deletes, so the store stays readable."""
    s3 = client(workers)
    bucket, prefix = split(url)
    _check_lock(url)
    downloaded = downloaded or {}
    expected = downloaded.get(ROOT_ETAG)
    current = _root_etag(s3, bucket, prefix)
    if current != expected:
        raise S3Error(f"The store at {url} changed during this run (another run wrote to it); nothing was synced. "
                      "Run again: the samples will be added to the current store.")
    creating = current is None
    before = ("The S3 prefix holds part of the new store, which the next run clears; run again." if creating else
              "The S3 store still holds the previous version and is readable; run again.")
    try:
        remote = remote_files(s3, bucket, prefix)
    except failures() as exc:
        raise error(exc, url, "read", "Nothing was synced; run again.") from None
    files = local_files(local)

    def unchanged(rel: str) -> bool:
        if rel not in remote:
            return False
        if downloaded.get(rel) == stat(local / rel):
            return True
        return remote[rel] == etag(local / rel)

    changed = [rel for rel in files if not unchanged(rel)]
    gone = sorted(set(remote) - set(files))
    cfg = transfer_config()

    def put(rel: str) -> None:
        s3.upload_file(str(local / rel), bucket, _key(prefix, rel), Config=cfg)

    def upload(rels: list[str]) -> None:
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(put, rels))

    try:
        if creating:
            s3.put_object(Bucket=bucket, Key=_key(prefix, S3_CREATING), Body=b"")
        data = [rel for rel in changed if not _is_meta(rel)]
        upload([rel for rel in data if rel not in remote])
        _phase("new data")
        upload([rel for rel in data if rel in remote])
        _phase("changed data")
        meta = [rel for rel in changed if _is_meta(rel) and rel != ROOT_META]
        for depth in sorted({rel.count("/") for rel in meta}, reverse=True):
            upload([rel for rel in meta if rel.count("/") == depth])
        _phase("metadata")
        _check_lock(url)
        if ROOT_META in changed:
            put(ROOT_META)
    except failures() as exc:
        raise error(exc, url, "write to", before) from None
    _phase("root")
    try:
        _delete(s3, bucket, prefix, [rel for rel in gone if rel != ROOT_META])
        if creating:
            s3.delete_object(Bucket=bucket, Key=_key(prefix, S3_CREATING))
    except (S3Error, *failures()) as exc:
        state = ("The S3 store already holds this run's samples and is readable; files it no longer needs are left "
                 "until the next run removes them.")
        if isinstance(exc, S3Error):
            raise S3Error(f"{exc}. {state}") from None
        raise error(exc, url, "delete from", state) from None
    _phase("deletes")
    result = {"uploaded": len(changed), "deleted": len(gone), "unchanged": len(files) - len(changed)}
    logger.info("Synced %s to %s: %s", local, url, result)
    return result
