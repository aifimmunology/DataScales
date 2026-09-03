"""User labelsets: saved onto the root store's obs as anndata categorical columns
(encoding verified against vendored anndata 0.12.19 methods.py — categorical/0.2.0
group, int codes array/0.2.0 with -1 = unlabeled, categories string-array/0.2.0).
Assignments arrive as barcodes so labels made inside a GPU view land on the root
store; the barcode->row index is built once and cached for the process.
"""

import re
import threading

from fastapi import HTTPException

from .config import DATA_DIR

_lock = threading.Lock()
_barcode_index = None  # {barcode: root row} built once; root obs is effectively static

CODES_CHUNK = 375_000  # matches the store's obs column chunking


def _open_obs():
    import zarr

    root = zarr.open_group(DATA_DIR, mode="r+", use_consolidated=False)
    return root["obs"]


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
    for key in ("codes", "categories"):  # category count changes shape: delete + recreate
        if key in g:
            del g[key]
    c = g.create_array("codes", shape=codes.shape, dtype=codes.dtype,
                       chunks=(min(len(codes), CODES_CHUNK),))
    c[:] = codes
    c.update_attributes({"encoding-type": "array", "encoding-version": "0.2.0"})
    cat = g.create_array("categories", shape=(len(cats),), dtype=VariableLengthUTF8())
    cat[:] = np.asarray(cats, dtype=object)
    cat.update_attributes({"encoding-type": "string-array", "encoding-version": "0.2.0"})


def save_labels(payload: dict) -> dict:
    name = str(payload.get("name", "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,59}", name):
        raise HTTPException(400, "labelset name must be alphanumeric/_/./- (max 60 chars)")
    assignments = payload.get("assignments") or []
    seed = str(payload.get("seed") or "")
    if not assignments and not seed:
        raise HTTPException(400, "no label assignments")

    import numpy as np

    with _lock:
        obs = _open_obs()
        if name in obs:
            g = obs[name]
            if not g.attrs.get("datavis-labelset"):
                raise HTTPException(409, f"obs column '{name}' exists and is not a datavis labelset")
            codes = g["codes"][:].astype(np.int16)
            cats = [str(c) for c in g["categories"][:]]
        elif seed:
            # fork-to-edit: copy the seed column server-side as the baseline (the
            # seeded labels are too many to ride the request payload)
            if seed not in obs:
                raise HTTPException(400, f"seed column '{seed}' not found in obs")
            sg = obs[seed]
            if sg.attrs.get("encoding-type") != "categorical":
                raise HTTPException(400, f"seed column '{seed}' is not categorical")
            codes = sg["codes"][:].astype(np.int16)
            cats = [str(c) for c in sg["categories"][:]]
        else:
            n = obs[obs.attrs.get("_index", "_index")].shape[0]
            codes = np.full(n, -1, dtype=np.int16)
            cats = []
        for a in assignments:  # applied in order: later assignments win on overlap
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
    return {"name": name, "categories": cats, "labeled": int((codes >= 0).sum())}
