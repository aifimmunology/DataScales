"""Views: the store's umap_views/ children, listed from the store on every request (no
listing object is kept, nothing cached). A view's display label lives in its root
zarr.json attributes, set by the pipeline."""

from fastapi import HTTPException

from . import storage

ROOT_VIEW = {"id": "main", "label": "Full store", "path": ""}
LABEL_ATTR = "datavis-label"


def list_views() -> list[dict]:
    return [ROOT_VIEW, *(_entry(s) for s in storage.list_children("umap_views"))]


def _entry(slug: str) -> dict:
    attrs = storage.read_attrs(f"umap_views/{slug}")
    return {"id": slug, "label": attrs.get(LABEL_ATTR, slug), "path": f"umap_views/{slug}"}


def delete_view(view_id: str) -> dict:
    if view_id not in storage.list_children("umap_views"):
        raise HTTPException(404, "unknown view")
    return {"deleted": view_id, "objects": storage.delete_prefix(f"umap_views/{view_id}")}
