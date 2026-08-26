"""View lifecycle: register/delete entries in the store's groups.json."""

import threading

from fastapi import HTTPException

from . import storage
from .config import SOURCE

# register (job worker thread) and delete (request thread) both read-modify-write
# groups.json; unserialized, a finishing job can resurrect a just-deleted view
_groups_lock = threading.Lock()

_DEFAULT = [{"id": "main", "label": "Full store", "path": ""}]


def register_view(slug: str, label: str) -> str:
    if not SOURCE["gcs"]:
        raise RuntimeError("view management requires a gs:// store")
    path = f"umap_views/{slug}"
    with _groups_lock:
        groups = storage.read_json("groups.json") or list(_DEFAULT)
        groups.append({"id": slug, "label": label, "path": path})
        storage.write_json("groups.json", groups)
    return path


def delete_view(view_id: str) -> dict:
    if not SOURCE["gcs"]:
        raise HTTPException(400, "view management requires a gs:// store")
    path = None
    with _groups_lock:
        groups = storage.read_json("groups.json")
        if groups:
            entry = next((g for g in groups if g.get("id") == view_id and g.get("path")), None)
            if entry is not None:
                path = entry["path"]
                # unregister first: an interrupted delete then leaves unlisted orphan
                # objects, never a listed-but-broken view
                storage.write_json("groups.json", [g for g in groups if g is not entry])
    # sweep even when unlisted, so retrying an interrupted delete clears the orphans
    n = storage.delete_prefix(path or f"umap_views/{view_id}")
    return {"deleted": view_id, "objects": n}
