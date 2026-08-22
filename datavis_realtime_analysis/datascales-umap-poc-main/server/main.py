"""Zarr data proxy: /api/data serves DATA_DIR (gene vis); /api/rapids-data serves RAPIDS_DIR."""

import os
import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response

DATA_DIR = os.environ.get("DATA_DIR", "")
RAPIDS_DIR = os.environ.get("RAPIDS_DIR", "") or DATA_DIR
SIMULATE_SCRIPT = Path(__file__).parent / "simulate_gpu.sh"
MAX_JOBS = 50

app = FastAPI()

JOBS: dict[str, dict] = {}


def _parse_dir(d: str) -> dict:
    if d.startswith("gs://"):
        bucket, _, prefix = d[len("gs://"):].partition("/")
        return {"gcs": True, "bucket": bucket, "prefix": prefix.rstrip("/")}
    return {"gcs": False, "root": d}


_SOURCES = {"data": _parse_dir(DATA_DIR), "rapids-data": _parse_dir(RAPIDS_DIR)}

# Client created on first data request so /api/health works without credentials.
_gcs_client = None
_size_cache: dict[str, int] = {}


def _bucket(name: str):
    global _gcs_client
    if _gcs_client is None:
        from google.cloud import storage

        try:
            _gcs_client = storage.Client()
        except Exception as e:
            raise HTTPException(503, f"GCS auth failed: {e}")
    return _gcs_client.bucket(name)


def _media_type(path: str) -> str:
    return "application/json" if path.endswith(".json") else "application/octet-stream"


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def config():
    # store: what the app reads (gene vis); rapids_store: what submissions reference
    return {"store": DATA_DIR, "rapids_store": RAPIDS_DIR}


# Missing objects must 404 (zarrita reads a 404 as "chunk is all fill-value").
# HEAD + Range are required: zarrita reads sharded arrays via partial requests.
@app.api_route("/api/data/{path:path}", methods=["GET", "HEAD"])
def data(path: str, request: Request):
    return _serve("data", path, request)


@app.api_route("/api/rapids-data/{path:path}", methods=["GET", "HEAD"])
def rapids_data(path: str, request: Request):
    return _serve("rapids-data", path, request)


def _serve(source: str, path: str, request: Request):
    src = _SOURCES[source]
    if not DATA_DIR:
        raise HTTPException(503, "DATA_DIR is not set")
    if not path or path.startswith("/") or ".." in path.split("/"):
        raise HTTPException(404, "Not found")

    if src["gcs"]:
        return _gcs_response(src, path, request)

    file = (Path(src["root"]) / path).resolve()
    if not file.is_relative_to(Path(src["root"]).resolve()) or not file.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(file, media_type=_media_type(path))


def _gcs_response(src: dict, path: str, request: Request) -> Response:
    from google.api_core.exceptions import NotFound

    bucket = _bucket(src["bucket"])
    key = f"{src['prefix']}/{path}" if src["prefix"] else path
    media = _media_type(path)

    if request.method == "HEAD":
        cache_key = f"{src['bucket']}/{key}"
        size = _size_cache.get(cache_key)
        if size is None:
            blob = bucket.get_blob(key)
            if blob is None:
                raise HTTPException(404, "Not found")
            size = _size_cache[cache_key] = blob.size
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
            payload = bucket.blob(key).download_as_bytes(start=start, end=end)
        except NotFound:
            raise HTTPException(404, "Not found")
        if start >= 0:
            headers["Content-Range"] = f"bytes {start}-{start + len(payload) - 1}/*"
        return Response(payload, status_code=206, media_type=media, headers=headers)

    # full-object GET. Buffered on purpose: zarr chunks compress to KBs (a 3M-row
    # dense uint16 column is ~4 KB), so streaming buys nothing — benchmarked via
    # both BlobReader and a download_to_file relay, no measurable win.
    try:
        payload = bucket.blob(key).download_as_bytes()
    except NotFound:
        raise HTTPException(404, "Not found")
    return Response(payload, media_type=media, headers=headers)


def _run_job(job_id: str):
    try:
        rc = subprocess.run(["bash", str(SIMULATE_SCRIPT), job_id]).returncode
        JOBS[job_id]["status"] = "done" if rc == 0 else "failed"
    except Exception:
        JOBS[job_id]["status"] = "failed"


@app.post("/api/submit")
def submit(artifact: dict):
    job_id = uuid.uuid4().hex[:8]
    cells = len(artifact.get("indices", []))
    print(f"[submit] job {job_id}: {cells} cells, group '{artifact.get('group', '')}'",
          file=sys.stderr, flush=True)
    while len(JOBS) >= MAX_JOBS:
        del JOBS[next(iter(JOBS))]
    JOBS[job_id] = {
        "id": job_id,
        "cells": cells,
        "group": artifact.get("group", ""),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
    return {"job_id": job_id, "status": "submitted"}


@app.get("/api/jobs")
def jobs():
    return list(JOBS.values())[::-1]
