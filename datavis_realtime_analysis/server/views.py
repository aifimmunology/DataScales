"""Views: the store's umap_views/ children, listed from the store on every request.
Only complete views are listed: the pipeline sets LABEL_ATTR after everything else is written."""

from fastapi import HTTPException
from store_meta import LABEL_ATTR, PARAMS_ATTR

from . import gpu, storage

ROOT_VIEW = {"id": "main", "label": "Full store", "path": ""}


def list_views() -> list[dict]:
    entries = (_entry(s) for s in storage.list_children("umap_views"))
    return [ROOT_VIEW, *(e for e in entries if e)]


def _entry(slug: str) -> dict | None:
    attrs = storage.read_attrs(f"umap_views/{slug}")
    if LABEL_ATTR not in attrs:
        return None
    return {"id": slug, "label": attrs[LABEL_ATTR], "path": f"umap_views/{slug}",
            "params": attrs.get(PARAMS_ATTR)}


def delete_view(view_id: str) -> dict:
    if view_id not in storage.list_children("umap_views"):
        raise HTTPException(404, "unknown view")
    if any(j["slug"] == view_id and j["status"] not in gpu.FINISHED for j in gpu.JOBS.values()):
        raise HTTPException(409, "this view's run is still in progress")
    return {"deleted": view_id, "objects": storage.delete_prefix(f"umap_views/{view_id}")}
