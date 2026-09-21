"""GPU jobs: in-memory queue, dispatched to a warm pipeline process
(gpu/worker.py) that holds imports + CUDA context between jobs and exits after
WORKER_IDLE_S idle. Stage updates stream over the worker's stdout; the store
only sees the finished view and one jobs/history/<id>.json record per job.
"""

import json
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException

from . import storage, views
from .config import DATA_DIR, MAX_JOBS, SOURCE

WORKER_SCRIPT = Path(__file__).resolve().parents[1] / "gpu" / "worker.py"

JOBS: dict[str, dict] = {}
JOB_QUEUE: queue.Queue = queue.Queue()
_worker_lock = threading.Lock()
_worker_started = False

_proc: subprocess.Popen | None = None
_proc_lock = threading.Lock()


class _Cancelled(Exception):
    pass


def _pipeline_proc() -> subprocess.Popen:
    global _proc
    with _proc_lock:
        if _proc is None or _proc.poll() is not None:
            _proc = subprocess.Popen(
                [sys.executable, "-u", str(WORKER_SCRIPT)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        return _proc


def _send(req: dict) -> subprocess.Popen:
    global _proc
    for attempt in (0, 1):  # one respawn retry: the idle worker may have just exited
        proc = _pipeline_proc()
        try:
            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.flush()
            return proc
        except (BrokenPipeError, OSError):
            with _proc_lock:
                _proc = None
            if attempt:
                raise RuntimeError("pipeline worker won't start (docker compose logs backend)")


def _run_gpu_job(job: dict, artifact: dict) -> None:
    rid = job["id"]
    sel = Path(f"/tmp/datavis_job_{rid}.json")
    sel.write_text(json.dumps(artifact))
    proc = _send({"id": rid, "store": DATA_DIR, "selection": str(sel),
                  "out": f"{DATA_DIR.rstrip('/')}/umap_views/{job['slug']}"})
    try:
        while True:
            line = proc.stdout.readline()
            if not line:
                if job["status"] == "cancelling":
                    raise _Cancelled()
                raise RuntimeError(f"pipeline worker died (rc={proc.poll()})")
            line = line.strip()
            if line.startswith("stage: "):
                job["stage"] = line[len("stage: "):]
            elif line == f"done: {rid}":
                return
            elif line.startswith("error: "):
                raise RuntimeError(line[len("error: "):])
    finally:
        sel.unlink(missing_ok=True)


def _record(job: dict, duration_s: float) -> None:
    try:
        storage.write_json(f"jobs/history/{job['id']}.json",
                           {**job, "duration_s": round(duration_s, 2)})
    except Exception:
        pass


def _worker() -> None:
    while True:
        job_id, artifact = JOB_QUEUE.get()
        job = JOBS.get(job_id)
        if job is None or job["status"] == "cancelled":
            continue
        job["status"] = "running"
        t0 = time.monotonic()
        try:
            _run_gpu_job(job, artifact)
            job["view"] = views.register_view(job["slug"], job["name"])
            job["stage"] = "done"
            job["status"] = "done"
        except _Cancelled:
            job["stage"] = "cancelled"
            job["status"] = "cancelled"
        except Exception as e:
            job["stage"] = str(e)[:200]
            job["status"] = "failed"
            _invalidate_probe()
        _record(job, time.monotonic() - t0)


def _ensure_worker() -> None:
    global _worker_started
    with _worker_lock:
        if not _worker_started:
            threading.Thread(target=_worker, daemon=True).start()
            _worker_started = True


def _config_problem() -> str | None:
    if not SOURCE["gcs"]:
        return "GPU jobs require a gs:// DATA_DIR store"
    return None


def submit(artifact: dict) -> dict:
    problem = _config_problem()
    if problem:
        raise HTTPException(400, problem)
    job_id = uuid.uuid4().hex[:8]
    cells = len(artifact.get("indices", []))
    name = str(artifact.get("name") or f"view {job_id}").strip()[:60]
    print(f"[submit] job {job_id} '{name}': {cells} cells, group '{artifact.get('group', '')}'",
          file=sys.stderr, flush=True)
    # evict only finished jobs — dropping a queued entry would orphan its run
    while len(JOBS) >= MAX_JOBS:
        victim = next((k for k, j in JOBS.items()
                       if j["status"] in ("done", "failed", "cancelled")), None)
        if victim is None:
            break
        del JOBS[victim]
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_").lower() or "view"
    JOBS[job_id] = {
        "id": job_id,
        "name": name,
        "slug": f"{slug}_{job_id}",
        "cells": cells,
        "group": artifact.get("group", ""),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "status": "queued",
        "stage": "queued",
        "view": None,
    }
    _ensure_worker()
    JOB_QUEUE.put((job_id, artifact))
    return {"job_id": job_id, "status": "submitted"}


def cancel(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    if job["status"] == "queued":
        job["status"] = "cancelled"
        job["stage"] = "cancelled"
    elif job["status"] == "running":
        job["status"] = "cancelling"
        job["stage"] = "cancelling"
        with _proc_lock:
            if _proc is not None and _proc.poll() is None:
                _proc.kill()  # warm state is lost; the next job respawns the worker
    else:
        raise HTTPException(409, f"job is {job['status']}")
    return {"id": job_id, "status": job["status"]}


def list_jobs() -> list[dict]:
    return list(JOBS.values())[::-1]


# ── GPU access probe ──────────────────────────────────────────────────────────
# Store access (submits/views/labels ride it) + GPU visibility in this container.

PROBE_TTL_S = 60

_probe_lock = threading.Lock()
_probe_state: dict = {"status": "checking", "checked_at": None}
_probe_ts = 0.0
_probing = False

_GPU_FIX = [
    "on the VM: sudo apt install nvidia-container-toolkit",
    "sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker",
    "docker compose up -d",
]


def _verdict(status: str, problem: str | None = None, summary: str = "",
             detail: str = "", fix: list[str] | None = None) -> dict:
    return {"status": status, "problem": problem, "summary": summary,
            "detail": detail, "fix": fix or [],
            "checked_at": datetime.now(timezone.utc).isoformat()}


def _probe_gpu() -> dict:
    try:  # backend credential → store: views/labels/history ride these writes
        storage.bucket().get_blob(storage.key("zarr.json"))
    except Exception as e:
        s, low = str(e), str(e).lower()
        if "403" in s or "does not have" in low or "denied" in low:
            return _verdict("error", "bucket",
                            "The backend can reach GCS but lacks bucket access.", s[-400:],
                            [f"grant the VM's service account roles/storage.objectAdmin "
                             f"on gs://{SOURCE['bucket']}"])
        return _verdict("error", "adc",
                        "The backend has no working GCS credential. On the GPU VM this "
                        "comes from the metadata server — check the VM's service account "
                        "and access scopes. On a laptop, run `gcloud auth "
                        "application-default login` and restart the backend.", s[-400:])
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        return _verdict("error", "gpu",
                        "No GPU in this container (nvidia-smi missing) — submits will fail.",
                        "", _GPU_FIX)
    except subprocess.TimeoutExpired:
        return _verdict("error", "gpu", "nvidia-smi timed out.", "", _GPU_FIX)
    if r.returncode != 0:
        return _verdict("error", "gpu", "The container cannot see a GPU.",
                        (r.stderr or r.stdout)[-400:], _GPU_FIX)
    n = len(r.stdout.strip().splitlines())
    return _verdict("ok", None, f"store access + {n} GPU{'s' if n != 1 else ''} visible")


def _probe_worker() -> None:
    global _probing, _probe_ts
    try:
        v = _probe_gpu()
    except Exception as e:  # never leave the checking flag stuck
        v = _verdict("error", "probe", f"probe crashed: {e}")
    with _probe_lock:
        _probe_state.clear()
        _probe_state.update(v)
        _probe_ts = time.monotonic()
        _probing = False


def _invalidate_probe() -> None:
    global _probe_ts
    with _probe_lock:
        _probe_ts = 0.0


def health(refresh: bool = False) -> dict:
    """Cached probe verdict; kicks a background re-probe when stale or forced."""
    problem = _config_problem()
    if problem:
        return {"status": "unconfigured", "summary": problem, "checking": False}
    global _probing
    with _probe_lock:
        fresh = _probe_ts and time.monotonic() - _probe_ts < PROBE_TTL_S
        if not _probing and (refresh or not fresh):
            _probing = True
            threading.Thread(target=_probe_worker, daemon=True).start()
        return {**_probe_state, "checking": _probing}
