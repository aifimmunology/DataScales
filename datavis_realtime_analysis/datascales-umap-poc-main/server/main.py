"""Zarr data proxy: /api/data serves DATA_DIR (gene vis); /api/rapids-data serves RAPIDS_DIR."""

import json
import os
import queue
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

# Real GPU runs happen when GPU_INSTANCE is set (gcloud compute ssh); else the
# sleep-script simulator handles submits.
GPU_INSTANCE = os.environ.get("GPU_INSTANCE", "")
GPU_ZONE = os.environ.get("GPU_ZONE", "us-central1-c")
GPU_DATA = os.environ.get("GPU_DATA", "/mnt/subset3M_megazarr_v1.0.zarr")
GPU_PIXI_DIR = os.environ.get("GPU_PIXI_DIR", "/mnt/DataScales/rapids_user_notebook")
RERUN_SCRIPT = Path(__file__).resolve().parents[2] / "rerun_umap_on_selection.py"

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

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


# ── Real GPU queue: one job at a time on the box, driven over gcloud ssh ──────

JOB_QUEUE: queue.Queue = queue.Queue()
_worker_lock = threading.Lock()
_worker_started = False


def _ssh_cmd(remote: str) -> list[str]:
    return ["gcloud", "compute", "ssh", GPU_INSTANCE, f"--zone={GPU_ZONE}",
            "--command", remote]


def _scp_cmd(locals_: list[str], remote: str) -> list[str]:
    return ["gcloud", "compute", "scp", *locals_, f"{GPU_INSTANCE}:{remote}",
            f"--zone={GPU_ZONE}"]


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])}… failed: {r.stderr.strip()[-300:]}")


def _register_view(slug: str, label: str) -> None:
    src = _SOURCES["rapids-data"]
    if not src["gcs"]:
        raise RuntimeError("GPU view return requires a gs:// RAPIDS_DIR store")
    bucket = _bucket(src["bucket"])
    key = f"{src['prefix']}/groups.json" if src["prefix"] else "groups.json"
    blob = bucket.blob(key)
    groups = (json.loads(blob.download_as_bytes()) if blob.exists()
              else [{"id": "main", "label": "Full store", "path": ""}])
    groups.append({"id": slug, "label": label, "path": f"umap_views/{slug}"})
    blob.upload_from_string(json.dumps(groups, indent=1), content_type="application/json")


def _run_gpu_job(job: dict, slug: str, sel_path: str) -> None:
    # two remote invocations total: one scp (selection + script), one ssh that
    # chains pipeline -> GCS upload -> cleanup, streaming stage lines back
    rid = job["id"]
    sel_remote = f"/tmp/{Path(sel_path).name}"
    out_remote = f"/tmp/datavis_view_{rid}"
    job["stage"] = "shipping selection"
    _run(_scp_cmd([sel_path, str(RERUN_SCRIPT)], "/tmp/"))

    job["stage"] = "starting pipeline"
    dest = f"{RAPIDS_DIR.rstrip('/')}/umap_views/{slug}"
    env = (f"RERUN_DATA={GPU_DATA} RERUN_SELECTION={sel_remote} "
           f"RERUN_OUT={out_remote} RERUN_GPUS=0")
    remote = (f"cd {GPU_PIXI_DIR} && {env} ~/.pixi/bin/pixi run python "
              f"/tmp/{RERUN_SCRIPT.name} && "
              f"echo 'stage: uploading view to GCS' && "
              f"gcloud -q storage rsync -r {out_remote} {dest} && "
              f"rm -rf {out_remote} {sel_remote}")
    log = open(f"/tmp/datavis_job_{rid}.log", "w")
    proc = subprocess.Popen(_ssh_cmd(remote), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        log.write(line)
        log.flush()
        if line.startswith("stage: "):
            job["stage"] = line[7:].strip()
    log.close()
    if proc.wait() != 0:
        raise RuntimeError(f"pipeline exited {proc.returncode} during '{job['stage']}' "
                           f"(log: /tmp/datavis_job_{rid}.log)")

    job["stage"] = "registering view"
    _register_view(slug, job["name"])
    job["view"] = f"umap_views/{slug}"
    job["stage"] = "done"


def _gpu_worker() -> None:
    while True:
        job_id, slug, sel_path = JOB_QUEUE.get()
        job = JOBS.get(job_id)
        if job is None:
            continue
        job["status"] = "running"
        try:
            _run_gpu_job(job, slug, sel_path)
            job["status"] = "done"
        except Exception as e:
            job["stage"] = str(e)[:200]
            job["status"] = "failed"
        finally:
            Path(sel_path).unlink(missing_ok=True)


def _ensure_worker() -> None:
    global _worker_started
    with _worker_lock:
        if not _worker_started:
            threading.Thread(target=_gpu_worker, daemon=True).start()
            _worker_started = True


@app.post("/api/submit")
def submit(artifact: dict):
    job_id = uuid.uuid4().hex[:8]
    cells = len(artifact.get("indices", []))
    name = str(artifact.get("name") or f"view {job_id}").strip()[:60]
    print(f"[submit] job {job_id} '{name}': {cells} cells, group '{artifact.get('group', '')}'",
          file=sys.stderr, flush=True)
    while len(JOBS) >= MAX_JOBS:
        del JOBS[next(iter(JOBS))]
    JOBS[job_id] = {
        "id": job_id,
        "name": name,
        "cells": cells,
        "group": artifact.get("group", ""),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "status": "queued" if GPU_INSTANCE else "running",
        "stage": "queued" if GPU_INSTANCE else "",
        "view": None,
    }
    if GPU_INSTANCE:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_").lower() or "view"
        slug = f"{slug}_{job_id}"
        sel_path = f"/tmp/datavis_selection_{job_id}.json"
        with open(sel_path, "w") as fh:
            json.dump(artifact, fh)
        _ensure_worker()
        JOB_QUEUE.put((job_id, slug, sel_path))
    else:
        threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
    return {"job_id": job_id, "status": "submitted"}


@app.get("/api/jobs")
def jobs():
    return list(JOBS.values())[::-1]


@app.delete("/api/views/{view_id}")
def delete_view(view_id: str):
    src = _SOURCES["rapids-data"]
    if not src["gcs"]:
        raise HTTPException(400, "view management requires a gs:// rapids store")
    bucket = _bucket(src["bucket"])
    key = f"{src['prefix']}/groups.json" if src["prefix"] else "groups.json"
    blob = bucket.blob(key)
    if not blob.exists():
        raise HTTPException(404, "no views registered")
    groups = json.loads(blob.download_as_bytes())
    entry = next((g for g in groups if g.get("id") == view_id and g.get("path")), None)
    if entry is None:
        raise HTTPException(404, f"view '{view_id}' not found")
    prefix = f"{src['prefix']}/{entry['path']}/" if src["prefix"] else f"{entry['path']}/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    for i in range(0, len(blobs), 100):
        bucket.delete_blobs(blobs[i:i + 100])
    blob.upload_from_string(
        json.dumps([g for g in groups if g is not entry], indent=1),
        content_type="application/json",
    )
    return {"deleted": view_id, "objects": len(blobs)}
