"""Store access: the zarr proxy (GET/HEAD/Range) + small JSON-object helpers.

Missing objects must 404 (zarrita reads a 404 as "chunk is all fill-value").
HEAD + Range are required: zarrita reads sharded arrays via partial requests.
"""

import json
import re
import threading
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, Response

from .config import DATA_DIR, SOURCE

# Client created on first use so /api/health works without credentials.
_gcs_client = None
_client_lock = threading.Lock()
_size_cache: dict[str, int] = {}


def bucket():
    global _gcs_client
    with _client_lock:
        if _gcs_client is None:
            from google.cloud import storage

            try:
                _gcs_client = storage.Client()
            except Exception as e:
                raise HTTPException(503, f"GCS auth failed: {e}")
    return _gcs_client.bucket(SOURCE["bucket"])


def key(rel: str) -> str:
    return f"{SOURCE['prefix']}/{rel}" if SOURCE["prefix"] else rel


def _media_type(path: str) -> str:
    return "application/json" if path.endswith(".json") else "application/octet-stream"


def serve(path: str, request: Request):
    if not DATA_DIR:
        raise HTTPException(503, "DATA_DIR is not set")
    if not path or path.startswith("/") or ".." in path.split("/"):
        raise HTTPException(404, "Not found")

    if SOURCE["gcs"]:
        return _gcs_response(path, request)

    file = (Path(SOURCE["root"]) / path).resolve()
    if not file.is_relative_to(Path(SOURCE["root"]).resolve()) or not file.is_file():
        raise HTTPException(404, "Not found")
    # groups.json mutates on register/delete — a cached copy resurrects deleted views
    headers = {"Cache-Control": "no-store"} if path.endswith("groups.json") else None
    return FileResponse(file, media_type=_media_type(path), headers=headers)


def _gcs_response(path: str, request: Request) -> Response:
    from google.api_core.exceptions import NotFound

    b = bucket()
    k = key(path)
    media = _media_type(path)

    if request.method == "HEAD":
        size = _size_cache.get(k)
        if size is None:
            blob = b.get_blob(k)
            if blob is None:
                raise HTTPException(404, "Not found")
            size = _size_cache[k] = blob.size
        headers = {"Content-Length": str(size), "Accept-Ranges": "bytes"}
        return Response(headers=headers, media_type=media)

    # bytes=A-B / bytes=A- / bytes=-N; GCS serves a negative start as a suffix range.
    start = end = None
    rng = re.fullmatch(r"bytes=(\d+)-(\d*)|bytes=-(\d+)", request.headers.get("range", ""))
    if rng:
        if rng[3]:
            start = -int(rng[3])
        else:
            start, end = int(rng[1]), int(rng[2]) if rng[2] else None

    headers = {"Accept-Ranges": "bytes"}
    if start is not None:
        # ranged reads are small (shard indexes) — buffered is fine
        try:
            payload = b.blob(k).download_as_bytes(start=start, end=end)
        except NotFound:
            raise HTTPException(404, "Not found")
        if start >= 0:
            headers["Content-Range"] = f"bytes {start}-{start + len(payload) - 1}/*"
        return Response(payload, status_code=206, media_type=media, headers=headers)

    # full-object GET. Buffered on purpose: zarr chunks compress to KBs (a 3M-row
    # dense uint16 column is ~4 KB), so streaming buys nothing — benchmarked via
    # both BlobReader and a download_to_file relay, no measurable win.
    try:
        payload = b.blob(k).download_as_bytes()
    except NotFound:
        raise HTTPException(404, "Not found")
    if path.endswith("groups.json"):  # mutable listing — never let a cache serve it
        headers["Cache-Control"] = "no-store"
    return Response(payload, media_type=media, headers=headers)


# ── JSON-object helpers (GCS store only): the jobs queue + groups.json ─────────


def read_json(rel: str):
    from google.api_core.exceptions import NotFound

    try:
        return json.loads(bucket().blob(key(rel)).download_as_bytes())
    except NotFound:
        return None


def write_json(rel: str, obj) -> None:
    bucket().blob(key(rel)).upload_from_string(
        json.dumps(obj, indent=1), content_type="application/json"
    )


def delete_object(rel: str) -> None:
    from google.api_core.exceptions import NotFound

    try:
        bucket().blob(key(rel)).delete()
    except NotFound:
        pass


def delete_prefix(rel: str) -> int:
    """Delete every object under rel/ (batched: 100 per round-trip); returns count."""
    b = bucket()
    prefix = key(rel).rstrip("/") + "/"
    blobs = list(b.list_blobs(prefix=prefix))
    for i in range(0, len(blobs), 100):
        with _gcs_client.batch(raise_exception=False):
            for blob in blobs[i:i + 100]:
                blob.delete()
    for k in [k for k in _size_cache if k.startswith(prefix)]:
        del _size_cache[k]
    return len(blobs)
