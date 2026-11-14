"""Labelsets are categorical obs columns edited in place (-1 = unlabeled). Views read every column from the
root except VIEW_NATIVE, which is edited inside the view."""

import re
import threading

import numpy as np
from fastapi import HTTPException

from . import selection, store

_lock = threading.Lock()
VIEW_NATIVE = "leiden"

CODES_CHUNK = 375_000
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,59}")


def _write_categorical(obs_attrs: dict, write_to, codes, cats: list[str]) -> None:
    attrs = {**obs_attrs, "encoding-type": "categorical", "encoding-version": "0.2.0",
             "ordered": False, "datavis-labelset": True}

    def fill(g) -> None:
        from zarr.core.dtype import VariableLengthUTF8

        g.update_attributes(attrs)
        dtype = np.min_scalar_type(-max(len(cats), 1))
        c = g.create_array("codes", shape=codes.shape, dtype=dtype, fill_value=-1,
                           chunks=(min(len(codes), CODES_CHUNK),))
        c[:] = codes.astype(dtype)
        c.update_attributes({"encoding-type": "array", "encoding-version": "0.2.0"})
        cat = g.create_array("categories", shape=(len(cats),), dtype=VariableLengthUTF8())
        cat[:] = np.asarray(cats, dtype=object)
        cat.update_attributes({"encoding-type": "string-array", "encoding-version": "0.2.0"})

    write_to(fill)


def _baseline(obs, name: str):
    if name not in obs:
        n = obs[obs.attrs.get("_index", "_index")].shape[0]
        return np.full(n, -1, dtype=np.int32), []
    g = obs[name]
    if g.attrs.get("encoding-type") != "categorical":
        raise HTTPException(409, f"obs column '{name}' exists and is not categorical")
    return g["codes"][:].astype(np.int32), [str(c) for c in g["categories"][:]]


def _drop_empty(codes, cats: list[str]):
    keep = np.flatnonzero(np.bincount(codes[codes >= 0], minlength=len(cats)))
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
    native = bool(group) and name == VIEW_NATIVE
    target = group if native else ""
    rel = f"{target}/obs" if target else "obs"

    with _lock:
        root = selection.open_root()
        rows = [(str(a.get("label", "")).strip()[:80],
                 selection.resolve_rows({"group": group, "n": payload.get("n"), "indices": a.get("indices")},
                                        root, to_root=not native))
                for a in assignments]
        import zarr

        obs = zarr.open_group(store.STORE.uri, mode="r+", use_consolidated=False)[rel]
        before, cats = _baseline(obs, name)
        codes = before.copy()
        for label, idx in rows:
            if not label:
                continue
            if label not in cats:
                cats.append(label)
            codes[idx] = cats.index(label)
        codes, cats = _drop_empty(codes, cats)
        attrs = dict(obs[name].attrs) if name in obs else {}
        _write_categorical(attrs, lambda fill: store.STORE.replace_group(f"{rel}/{name}", fill), codes, cats)
        cols = list(obs.attrs.get("column-order", []))
        if name not in cols:
            obs.update_attributes({"column-order": cols + [name]})
        if store.MIRROR:
            try:
                store.MIRROR.replace(f"{rel}/{name}")
                store.MIRROR.copy(f"{rel}/zarr.json")
            except store.MirrorError as e:
                raise HTTPException(502, f"labelset saved to the store but not mirrored: {e}")
    return {"name": name, "categories": cats, "labeled": int((codes >= 0).sum())}
