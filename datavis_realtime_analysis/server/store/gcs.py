import json
import re
import threading
from collections.abc import Callable

from fastapi import HTTPException, Request
from fastapi.responses import Response

from ..config import Location
from .base import Store

_client = None
_client_lock = threading.Lock()


def media_type(path: str) -> str:
    return "application/json" if path.endswith(".json") else "application/octet-stream"


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


def delete_gcs_prefix(loc: Location, rel: str, client=None) -> list[str]:
    client = client or gcs_client()
    prefix = loc.key(rel).rstrip("/") + "/"
    blobs = list(client.bucket(loc.bucket).list_blobs(prefix=prefix))
    for i in range(0, len(blobs), 100):
        with client.batch(raise_exception=False):
            for blob in blobs[i:i + 100]:
                blob.delete()
    return [b.name for b in blobs]


class GcsStore(Store):
    is_gcs = True

    def __init__(self, loc: Location, client=None):
        self.loc, self.uri = loc, loc.uri
        self._client = client
        self._size_cache: dict[str, int] = {}

    def _bucket(self):
        return (self._client or gcs_client()).bucket(self.loc.bucket)

    def _download(self, rel: str, start=None, end=None) -> bytes:
        from google.api_core.exceptions import NotFound

        try:
            return self._bucket().blob(self.loc.key(rel)).download_as_bytes(start=start, end=end)
        except NotFound:
            raise HTTPException(404, "Not found")

    def serve(self, path: str, request: Request) -> Response:
        k, media = self.loc.key(path), media_type(path)
        if request.method == "HEAD":
            size = self._size_cache.get(k)
            if size is None:
                blob = self._bucket().get_blob(k)
                if blob is None:
                    raise HTTPException(404, "Not found")
                size = blob.size
                if not path.endswith(".json"):
                    self._size_cache[k] = size
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
            payload = self._download(path, start=start, end=end)
            if start >= 0:
                headers["Content-Range"] = f"bytes {start}-{start + len(payload) - 1}/*"
            return Response(payload, status_code=206, media_type=media, headers=headers)
        return Response(self._download(path), media_type=media, headers=headers)

    def list_children(self, rel: str) -> list[str]:
        prefix = self.loc.key(rel).rstrip("/") + "/"
        pages = self._bucket().list_blobs(prefix=prefix, delimiter="/").pages
        return sorted(p[len(prefix):].rstrip("/") for page in pages for p in page.prefixes)

    def read_attrs(self, rel: str) -> dict | None:
        try:
            raw = self._download(f"{rel}/zarr.json")
        except HTTPException as e:
            if e.status_code == 404:
                return None
            raise
        except Exception as e:
            raise HTTPException(503, f"could not read {rel}: {e}")
        return json.loads(raw).get("attributes", {})

    def delete_prefix(self, rel: str) -> int:
        gone = delete_gcs_prefix(self.loc, rel, self._client)
        prefix = self.loc.key(rel).rstrip("/") + "/"
        for k in [k for k in self._size_cache if k.startswith(prefix)]:
            del self._size_cache[k]
        return len(gone)

    def replace_group(self, rel: str, write: Callable) -> None:
        import zarr

        self.delete_prefix(rel)
        write(zarr.open_group(f"{self.uri}/{rel}", mode="w"))
