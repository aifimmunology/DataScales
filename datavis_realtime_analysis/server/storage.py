"""Store access: the zarr proxy (GET/HEAD/Range), child listing, attribute reads,
prefix deletes, and the mirror that copies local writes to GCS_DATA.

Missing objects must 404 (zarrita reads a 404 as "chunk is all fill-value").
HEAD + Range are required: zarrita reads sharded arrays via partial requests.
"""

import hashlib
import json
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, Response

from .config import MIRROR, SOURCE, Location

MIRROR_THREADS = 16
EAGER_BYTES = 8 << 20

_client = None
_client_lock = threading.Lock()
_size_cache: dict[str, int] = {}
_LOCAL_ROOT = Path(SOURCE.uri).resolve() if SOURCE and not SOURCE.gcs else None
_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")


class MirrorError(Exception):
    pass


def gcs_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                from google.cloud import storage
                from requests.adapters import HTTPAdapter

                try:
                    client = storage.Client()
                except Exception as e:
                    raise HTTPException(503, f"GCS auth failed: {e}")
                client._http.mount("https://", HTTPAdapter(pool_connections=64, pool_maxsize=64))
                _client = client
    return _client


def bucket(loc: Location):
    return gcs_client().bucket(loc.bucket)


def _media_type(path: str) -> str:
    return "application/json" if path.endswith(".json") else "application/octet-stream"


def serve(path: str, request: Request):
    if SOURCE is None:
        raise HTTPException(503, "no store configured: set LOCAL_DATA and/or GCS_DATA")
    if not path or path.startswith("/") or ".." in path.split("/"):
        raise HTTPException(404, "Not found")
    if SOURCE.gcs:
        return _gcs_response(path, request)
    if not all(_SAFE_SEGMENT.fullmatch(s) for s in path.split("/")):
        raise HTTPException(404, "Not found")
    try:
        file = (_LOCAL_ROOT / path).resolve()
        ok = file.is_relative_to(_LOCAL_ROOT) and file.is_file()
    except (OSError, ValueError):
        ok = False
    if not ok:
        raise HTTPException(404, "Not found")
    try:
        st = file.stat()
    except FileNotFoundError:
        raise HTTPException(404, "Not found")
    headers = {"ETag": _etag(st), "Cache-Control": "no-cache"}
    if headers["ETag"] in request.headers.get("if-none-match", ""):
        return Response(status_code=304, headers=headers)
    if st.st_size <= EAGER_BYTES and "range" not in request.headers:
        try:
            return Response(file.read_bytes(), media_type=_media_type(path), headers=headers)
        except FileNotFoundError:
            raise HTTPException(404, "Not found")
    return FileResponse(file, media_type=_media_type(path), headers=headers, stat_result=st)


def _etag(st) -> str:
    return f'"{hashlib.md5(f"{st.st_mtime_ns}-{st.st_size}".encode(), usedforsecurity=False).hexdigest()}"'


def _download(loc: Location, rel: str, start=None, end=None) -> bytes:
    from google.api_core.exceptions import NotFound

    try:
        return bucket(loc).blob(loc.key(rel)).download_as_bytes(start=start, end=end)
    except NotFound:
        raise HTTPException(404, "Not found")


def _gcs_response(path: str, request: Request) -> Response:
    k = SOURCE.key(path)
    media = _media_type(path)

    if request.method == "HEAD":
        size = _size_cache.get(k)
        if size is None:
            blob = bucket(SOURCE).get_blob(k)
            if blob is None:
                raise HTTPException(404, "Not found")
            size = blob.size
            if not path.endswith(".json"):
                _size_cache[k] = size
        return Response(headers={"Content-Length": str(size), "Accept-Ranges": "bytes"}, media_type=media)

    start = end = None
    rng = re.fullmatch(r"bytes=(\d+)-(\d*)|bytes=-(\d+)", request.headers.get("range", ""))
    if rng:
        if rng[3]:
            start = -int(rng[3])
        else:
            start, end = int(rng[1]), int(rng[2]) if rng[2] else None

    headers = {"Accept-Ranges": "bytes", "Cache-Control": "no-cache"}
    if start is not None:
        payload = _download(SOURCE, path, start=start, end=end)
        if start >= 0:
            headers["Content-Range"] = f"bytes {start}-{start + len(payload) - 1}/*"
        return Response(payload, status_code=206, media_type=media, headers=headers)
    return Response(_download(SOURCE, path), media_type=media, headers=headers)


def list_children(rel: str) -> list[str]:
    if SOURCE.gcs:
        prefix = SOURCE.key(rel).rstrip("/") + "/"
        pages = bucket(SOURCE).list_blobs(prefix=prefix, delimiter="/").pages
        return sorted(p[len(prefix):].rstrip("/") for page in pages for p in page.prefixes)
    d = _LOCAL_ROOT / rel
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.is_dir() else []


def read_attrs(rel: str) -> dict:
    try:
        if SOURCE.gcs:
            raw = _download(SOURCE, f"{rel}/zarr.json")
        else:
            raw = (_LOCAL_ROOT / rel / "zarr.json").read_bytes()
        return json.loads(raw).get("attributes", {})
    except (HTTPException, OSError, ValueError):
        return {}


def delete_prefix(rel: str) -> int:
    """Mirror first: if it fails, the local copy is still listed and the delete can be retried."""
    if MIRROR:
        _delete_gcs_prefix(MIRROR, rel)
    return _delete_gcs_prefix(SOURCE, rel) if SOURCE.gcs else _delete_local_prefix(rel)


def _delete_local_prefix(rel: str) -> int:
    d = _LOCAL_ROOT / rel
    n = sum(1 for p in d.rglob("*") if p.is_file()) if d.is_dir() else 0
    shutil.rmtree(d, ignore_errors=True)
    return n


def _delete_gcs_prefix(loc: Location, rel: str) -> int:
    prefix = loc.key(rel).rstrip("/") + "/"
    blobs = list(bucket(loc).list_blobs(prefix=prefix))
    for i in range(0, len(blobs), 100):
        with gcs_client().batch(raise_exception=False):
            for blob in blobs[i:i + 100]:
                blob.delete()
    for k in [k for k in _size_cache if k.startswith(prefix)]:
        del _size_cache[k]
    return len(blobs)


def mirror(*rels: str) -> int:
    """Copy local files or directories at `rels` to MIRROR; blocks, raises MirrorError.
    Chunks go before metadata, deepest zarr.json first, so a bucket reader never finds
    a node whose data hasn't landed."""
    if MIRROR is None:
        return 0
    files = [f for rel in rels for f in _local_files(rel)]
    chunks = [f for f in files if f.name != "zarr.json"]
    metas = sorted((f for f in files if f.name == "zarr.json"), key=lambda f: -len(f.parts))
    try:
        b = bucket(MIRROR)

        def upload(f: Path) -> None:
            rel = f.relative_to(_LOCAL_ROOT).as_posix()
            b.blob(MIRROR.key(rel)).upload_from_filename(str(f), content_type=_media_type(rel))

        with ThreadPoolExecutor(MIRROR_THREADS) as pool:
            list(pool.map(upload, chunks))
        for f in metas:
            upload(f)
    except Exception as e:
        raise MirrorError(f"{len(files)} files -> {MIRROR.uri}: {e}") from e
    return len(files)


def remirror(rel: str) -> int:
    """Replace MIRROR's copy of the local directory `rel`: upload, then drop objects it no longer has."""
    if MIRROR is None:
        return 0
    n = mirror(rel)
    keep = {MIRROR.key(f.relative_to(_LOCAL_ROOT).as_posix()) for f in _local_files(rel)}
    try:
        stale = [blob for blob in bucket(MIRROR).list_blobs(prefix=MIRROR.key(rel).rstrip("/") + "/")
                 if blob.name not in keep]
        for blob in stale:
            blob.delete()
    except Exception as e:
        raise MirrorError(f"pruning {MIRROR.uri}/{rel}: {e}") from e
    return n


def _local_files(rel: str) -> list[Path]:
    p = _LOCAL_ROOT / rel
    if p.is_dir():
        return [q for q in p.rglob("*") if q.is_file()]
    return [p] if p.is_file() else []
