"""Selection payload: {"group": "" | "umap_views/<slug>", "n": cells in that group, "indices": [...]}.
`n` refuses selections made against a stale client; a view's rows map to root rows via obs/root_row."""

import numpy as np
import zarr
from fastapi import HTTPException

from . import store


def open_root():
    if store.STORE is None:
        raise HTTPException(503, "no store configured")
    return zarr.open_group(store.STORE.uri, mode="r", use_consolidated=False)


def _check_group(group: str) -> None:
    if group and (not group.startswith("umap_views/") or group.split("/", 1)[1] not in store.STORE.list_children("umap_views")):
        raise HTTPException(404, "unknown view")


def group_size(root, group: str) -> int:
    obs = root[f"{group}/obs" if group else "obs"]
    return obs[obs.attrs.get("_index", "_index")].shape[0]


def resolve_rows(payload: dict, root=None, to_root: bool = True) -> np.ndarray:
    group = str(payload.get("group") or "")
    _check_group(group)
    root = root if root is not None else open_root()
    raw = payload.get("indices")
    if not isinstance(raw, list) or not raw:
        raise HTTPException(400, "empty selection")
    n = group_size(root, group)
    if payload.get("n") != n:
        raise HTTPException(409, f"the client has {payload.get('n')} cells but '{group or 'the store'}' has {n}: reload")
    try:
        idx = np.asarray(raw, dtype=np.int64)
    except (TypeError, ValueError):
        raise HTTPException(400, "indices must be integers")
    if idx.ndim != 1 or idx.min() < 0 or idx.max() >= n:
        raise HTTPException(400, "indices out of range")
    if group and to_root:
        idx = root[f"{group}/obs/root_row"][:].astype(np.int64)[idx]
    return np.unique(idx)
