"""User labelsets: anndata categorical columns on the store's obs (categorical/0.2.0 with
int codes, -1 = unlabeled, string-array categories). Any categorical column is edited in
place; edited columns are tagged `datavis-labelset`. Views read every column from the
root except VIEW_NATIVE (each view's own clustering), which is edited inside the view.
Assignments arrive as barcodes, so labels made inside a GPU view land on the root store.
"""

import re
import shutil
import threading
from pathlib import Path

from fastapi import HTTPException

from . import storage
from .config import SOURCE

_lock = threading.Lock()
_barcode_index: dict[str, dict[str, int]] = {}  # group -> barcode -> row
VIEW_NATIVE = "leiden"

CODES_CHUNK = 375_000
STAGING = ".datavis_staging"  # leading dot: never served or mirrored
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,59}")


def _open_obs(group: str):
    import zarr

    if SOURCE is None:
        raise HTTPException(503, "no store configured")
    return zarr.open_group(SOURCE.uri, mode="r+", use_consolidated=False)[_obs_rel(group)]


def _obs_rel(group: str) -> str:
    return f"{group}/obs" if group else "obs"


def _rows_for(obs, group: str, barcodes: list[str]):
    import numpy as np

    if group not in _barcode_index:
        names = obs[obs.attrs.get("_index", "_index")][:]
        _barcode_index[group] = {str(b): i for i, b in enumerate(names)}
    idx = _barcode_index[group]
    try:
        return np.fromiter((idx[b] for b in barcodes), dtype=np.int64, count=len(barcodes))
    except KeyError:
        missing = sum(1 for b in barcodes if b not in idx)
        raise HTTPException(400, f"{missing} barcodes not found in the store")


def _write_categorical(obs, rel: str, name: str, codes, cats: list[str]) -> None:
    """Local stores get the new column built aside and renamed in, so readers and failed
    saves never see a half-written column; a GCS store is rewritten in place."""
    import zarr

    attrs = {**(dict(obs[name].attrs) if name in obs else {}), "encoding-type": "categorical",
             "encoding-version": "0.2.0", "ordered": False, "datavis-labelset": True}
    if SOURCE.gcs:
        g = obs.require_group(name)
        for key in ("codes", "categories"):
            if key in g:
                del g[key]
        _fill(g, codes, cats, attrs)
        return
    staging = Path(SOURCE.uri, STAGING)
    staging.mkdir(exist_ok=True)
    tag = rel.replace("/", "~")
    new, old, final = staging / f"{tag}~{name}.new", staging / f"{tag}~{name}.old", Path(SOURCE.uri, rel, name)
    for p in (new, old):
        shutil.rmtree(p, ignore_errors=True)
    _fill(zarr.open_group(str(new), mode="w"), codes, cats, attrs)
    if final.exists():
        final.rename(old)
    new.rename(final)
    shutil.rmtree(old, ignore_errors=True)


def _fill(g, codes, cats: list[str], attrs: dict) -> None:
    import numpy as np
    from zarr.core.dtype import VariableLengthUTF8

    g.update_attributes(attrs)
    codes = codes.astype(np.min_scalar_type(-max(len(cats), 1)))
    c = g.create_array("codes", shape=codes.shape, dtype=codes.dtype, fill_value=-1,
                       chunks=(min(len(codes), CODES_CHUNK),))
    c[:] = codes
    c.update_attributes({"encoding-type": "array", "encoding-version": "0.2.0"})
    cat = g.create_array("categories", shape=(len(cats),), dtype=VariableLengthUTF8())
    cat[:] = np.asarray(cats, dtype=object)
    cat.update_attributes({"encoding-type": "string-array", "encoding-version": "0.2.0"})


def _baseline(obs, name: str):
    import numpy as np

    if name not in obs:
        n = obs[obs.attrs.get("_index", "_index")].shape[0]
        return np.full(n, -1, dtype=np.int32), []
    g = obs[name]
    if g.attrs.get("encoding-type") != "categorical":
        raise HTTPException(409, f"obs column '{name}' exists and is not categorical")
    return g["codes"][:].astype(np.int32), [str(c) for c in g["categories"][:]]


def _drop_empty(codes, cats: list[str]):
    import numpy as np

    counts = np.bincount(codes[codes >= 0], minlength=len(cats))
    keep = np.flatnonzero(counts)
    remap = np.full(len(cats) + 1, -1, dtype=np.int32)
    remap[keep] = np.arange(len(keep), dtype=np.int32)
    return remap[codes], [cats[k] for k in keep]


def save_labels(payload: dict) -> dict:
    name = str(payload.get("name", "")).strip()
    if not NAME.fullmatch(name) or name == "zarr.json":
        raise HTTPException(400, "labelset name must be alphanumeric/_/./- (max 60 chars)")
    assignments = payload.get("assignments") or []
    if not assignments:
        raise HTTPException(400, "no label assignments")
    group = str(payload.get("group") or "")
    if group:
        if name != VIEW_NATIVE:
            raise HTTPException(400, f"only '{VIEW_NATIVE}' is edited inside a view; other columns live on the root")
        if not group.startswith("umap_views/") or group.split("/", 1)[1] not in storage.list_children("umap_views"):
            raise HTTPException(404, "unknown view")
    rel = _obs_rel(group)

    with _lock:
        obs = _open_obs(group)
        codes, cats = _baseline(obs, name)
        for a in assignments:
            label = str(a.get("label", "")).strip()[:80]
            bcs = a.get("barcodes") or []
            if not label or not bcs:
                continue
            if label not in cats:
                cats.append(label)
            codes[_rows_for(obs, group, bcs)] = cats.index(label)
        codes, cats = _drop_empty(codes, cats)
        _write_categorical(obs, rel, name, codes, cats)
        cols = list(obs.attrs.get("column-order", []))
        if name not in cols:
            obs.update_attributes({"column-order": cols + [name]})
        try:
            storage.remirror(f"{rel}/{name}")
            storage.mirror(f"{rel}/zarr.json")
        except storage.MirrorError as e:
            raise HTTPException(502, f"labelset saved to the store but not mirrored: {e}")
    return {"name": name, "categories": cats, "labeled": int((codes >= 0).sum())}
