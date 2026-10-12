"""User labelsets: anndata categorical columns on the store's obs (categorical/0.2.0 with
int codes, -1 = unlabeled, string-array categories). Assignments arrive as barcodes so
labels made inside a GPU view land on the root store.
"""

import re
import threading

from fastapi import HTTPException

from . import storage
from .config import SOURCE

_lock = threading.Lock()
_barcode_index = None

CODES_CHUNK = 375_000
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,59}")


def _open_obs():
    import zarr

    if SOURCE is None:
        raise HTTPException(503, "no store configured")
    return zarr.open_group(SOURCE.uri, mode="r+", use_consolidated=False)["obs"]


def _rows_for(obs, barcodes: list[str]):
    import numpy as np

    global _barcode_index
    if _barcode_index is None:
        names = obs[obs.attrs.get("_index", "_index")][:]
        _barcode_index = {str(b): i for i, b in enumerate(names)}
    idx = _barcode_index
    try:
        return np.fromiter((idx[b] for b in barcodes), dtype=np.int64, count=len(barcodes))
    except KeyError:
        missing = sum(1 for b in barcodes if b not in idx)
        raise HTTPException(400, f"{missing} barcodes not found in the store")


def _write_categorical(obs, name: str, codes, cats: list[str]) -> None:
    import numpy as np
    from zarr.core.dtype import VariableLengthUTF8

    g = obs.require_group(name)
    g.update_attributes({"encoding-type": "categorical", "encoding-version": "0.2.0",
                         "ordered": False, "datavis-labelset": True})
    for key in ("codes", "categories"):
        if key in g:
            del g[key]
    c = g.create_array("codes", shape=codes.shape, dtype=codes.dtype,
                       chunks=(min(len(codes), CODES_CHUNK),))
    c[:] = codes
    c.update_attributes({"encoding-type": "array", "encoding-version": "0.2.0"})
    cat = g.create_array("categories", shape=(len(cats),), dtype=VariableLengthUTF8())
    cat[:] = np.asarray(cats, dtype=object)
    cat.update_attributes({"encoding-type": "string-array", "encoding-version": "0.2.0"})


def _baseline(obs, name: str, seed: str):
    import numpy as np

    if name in obs:
        g = obs[name]
        if not g.attrs.get("datavis-labelset"):
            raise HTTPException(409, f"obs column '{name}' exists and is not a datavis labelset")
    elif seed:
        if seed not in obs:
            raise HTTPException(400, f"seed column '{seed}' not found in obs")
        g = obs[seed]
        if g.attrs.get("encoding-type") != "categorical":
            raise HTTPException(400, f"seed column '{seed}' is not categorical")
    else:
        n = obs[obs.attrs.get("_index", "_index")].shape[0]
        return np.full(n, -1, dtype=np.int16), []
    return g["codes"][:].astype(np.int16), [str(c) for c in g["categories"][:]]


def save_labels(payload: dict) -> dict:
    name = str(payload.get("name", "")).strip()
    if not NAME.fullmatch(name):
        raise HTTPException(400, "labelset name must be alphanumeric/_/./- (max 60 chars)")
    assignments = payload.get("assignments") or []
    seed = str(payload.get("seed") or "")
    if not assignments and not seed:
        raise HTTPException(400, "no label assignments")

    with _lock:
        obs = _open_obs()
        codes, cats = _baseline(obs, name, seed)
        for a in assignments:
            label = str(a.get("label", "")).strip()[:80]
            bcs = a.get("barcodes") or []
            if not label or not bcs:
                continue
            if label not in cats:
                cats.append(label)
            codes[_rows_for(obs, bcs)] = cats.index(label)
        _write_categorical(obs, name, codes, cats)
        cols = list(obs.attrs.get("column-order", []))
        if name not in cols:
            obs.update_attributes({"column-order": cols + [name]})
        try:
            storage.mirror(f"obs/{name}", "obs/zarr.json")
        except storage.MirrorError as e:
            raise HTTPException(502, f"labelset saved to the store but not mirrored: {e}")
    return {"name": name, "categories": cats, "labeled": int((codes >= 0).sum())}
