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
_barcode_index = None  # (sorted names, argsort order); root obs is effectively static

CODES_CHUNK = 375_000  # matches the store's obs column chunking


def _open_obs():
    import zarr

    root = zarr.open_group(DATA_DIR, mode="r+", use_consolidated=False)
    return root["obs"]


def _rows_for(obs, barcodes: list[str]):
    import numpy as np

    global _barcode_index
    if _barcode_index is None:
        names = np.asarray(obs[obs.attrs.get("_index", "_index")][:], dtype=object)
        order = np.argsort(names)
        _barcode_index = (names[order], order)
    sorted_names, order = _barcode_index
    q = np.asarray(barcodes, dtype=object)
    pos = np.clip(np.searchsorted(sorted_names, q), 0, len(sorted_names) - 1)
    ok = sorted_names[pos] == q
    if not bool(ok.all()):
        raise HTTPException(400, f"{int((~ok).sum())} barcodes not found in the store")
    return order[pos]


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
    if not assignments:
        raise HTTPException(400, "no label assignments")

    import numpy as np

    with _lock:
        obs = _open_obs()
        n = obs[obs.attrs.get("_index", "_index")].shape[0]
        if name in obs:
            g = obs[name]
            if not g.attrs.get("datavis-labelset"):
                raise HTTPException(409, f"obs column '{name}' exists and is not a datavis labelset")
            codes = g["codes"][:].astype(np.int16)
            cats = [str(c) for c in g["categories"][:]]
        else:
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
