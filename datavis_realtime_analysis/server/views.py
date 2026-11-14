"""A view is listed only once LABEL_ATTR exists: the pipeline writes it last, so half-written views never show."""

from concurrent.futures import ThreadPoolExecutor

from fastapi import HTTPException
from store_meta import LABEL_ATTR, PARAMS_ATTR

from . import gpu, store

ROOT_VIEW = {"id": "main", "label": "Full store", "path": ""}
READ_THREADS = 16


def list_views() -> list[dict]:
    slugs = store.STORE.list_children("umap_views")
    with ThreadPoolExecutor(READ_THREADS) as pool:
        entries = list(pool.map(_entry, slugs))
    return [ROOT_VIEW, *(e for e in entries if e)]


def _entry(slug: str) -> dict | None:
    attrs = store.STORE.read_attrs(f"umap_views/{slug}")
    if not attrs or LABEL_ATTR not in attrs:
        return None
    return {"id": slug, "label": attrs[LABEL_ATTR], "path": f"umap_views/{slug}",
            "params": attrs.get(PARAMS_ATTR)}


def delete_view(view_id: str) -> dict:
    if view_id not in store.STORE.list_children("umap_views"):
        raise HTTPException(404, "unknown view")
    if any(j["slug"] == view_id and j["status"] not in gpu.FINISHED for j in gpu.JOBS.values()):
        raise HTTPException(409, "this view's run is still in progress")
    rel = f"umap_views/{view_id}"
    if store.MIRROR:
        store.MIRROR.delete_prefix(rel)
    return {"deleted": view_id, "objects": store.STORE.delete_prefix(rel)}
