"""GPU jobs: in-memory queue and history, dispatched to a warm pipeline process
(gpu/worker.py) that keeps imports + CUDA context between jobs and exits after
WORKER_IDLE_S idle. Stage updates stream over the worker's stdout; the store only
sees the finished view (mirrored to GCS_DATA when configured).
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

from . import storage
from .config import MAX_JOBS, MIRROR, SOURCE, Location
from .embedding import RunParams

WORKER_SCRIPT = Path(__file__).resolve().parents[1] / "gpu" / "worker.py"
PROBE_TTL_S = 60
FINISHED = ("done", "failed", "cancelled")

JOBS: dict[str, dict] = {}
JOB_QUEUE: queue.Queue = queue.Queue()
_worker_lock = threading.Lock()
_worker_started = False
_proc: subprocess.Popen | None = None
_proc_lock = threading.Lock()


class _Cancelled(Exception):
    pass


# ── worker process ─────────────────────────────────────────────────────────────


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
    for attempt in (0, 1):
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
    view = f"umap_views/{job['slug']}"
    proc = _send({"id": rid, "store": SOURCE.uri, "selection": str(sel), "label": job["name"],
                  "out": f"{SOURCE.uri}/{view}", "params": job["params"]})
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
                break
            elif line.startswith("error: "):
                raise RuntimeError(line[len("error: "):])
    except BaseException:
        _discard(view)
        raise
    finally:
        sel.unlink(missing_ok=True)
    if MIRROR:
        job["stage"] = "mirroring view to GCS"
        try:
            storage.mirror(view)
        except storage.MirrorError as e:
            raise RuntimeError(f"view written but not mirrored: {e}")


def _discard(view: str) -> None:
    """A failed or cancelled run leaves no partial view behind."""
    try:
        storage.delete_prefix(view)
    except Exception as e:
        print(f"[jobs] could not remove partial {view}: {e}", file=sys.stderr, flush=True)


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
            if job["status"] == "cancelling":
                _discard(f"umap_views/{job['slug']}")
                raise _Cancelled()
            job["view"] = f"umap_views/{job['slug']}"
            job["stage"] = job["status"] = "done"
        except _Cancelled:
            job["stage"] = job["status"] = "cancelled"
        except Exception as e:
            job["stage"] = str(e)[:200]
            job["status"] = "failed"
            _invalidate_probe()
        job["duration_s"] = round(time.monotonic() - t0, 2)


def _ensure_worker() -> None:
    global _worker_started
    with _worker_lock:
        if not _worker_started:
            threading.Thread(target=_worker, daemon=True).start()
            _worker_started = True


# ── API ────────────────────────────────────────────────────────────────────────


def _config_problem() -> str | None:
    return None if SOURCE else "no store configured: set LOCAL_DATA and/or GCS_DATA"


def submit(artifact: dict) -> dict:
    problem = _config_problem()
    if problem:
        raise HTTPException(400, problem)
    try:
        params = RunParams.from_dict(artifact.get("params")).to_dict()
    except ValueError as e:
        raise HTTPException(400, str(e))
    job_id = uuid.uuid4().hex[:8]
    cells = len(artifact.get("indices", []))
    name = str(artifact.get("name") or f"view {job_id}").strip()[:60]
    print(f"[submit] job {job_id} '{name}': {cells} cells, group '{artifact.get('group', '')}'",
          file=sys.stderr, flush=True)
    while len(JOBS) >= MAX_JOBS:
        victim = next((k for k, j in JOBS.items() if j["status"] in FINISHED), None)
        if victim is None:
            break
        del JOBS[victim]
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_").lower() or "view"
    JOBS[job_id] = {
        "id": job_id, "name": name, "slug": f"{slug}_{job_id}", "cells": cells,
        "group": artifact.get("group", ""), "params": params,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "status": "queued", "stage": "queued", "view": None,
    }
    _ensure_worker()
    JOB_QUEUE.put((job_id, artifact))
    return {"job_id": job_id, "status": "submitted"}


def cancel(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    if job["status"] == "queued":
        job["status"] = job["stage"] = "cancelled"
    elif job["status"] == "running":
        job["status"] = job["stage"] = "cancelling"
        with _proc_lock:
            if _proc is not None and _proc.poll() is None:
                _proc.kill()
    else:
        raise HTTPException(409, f"job is {job['status']}")
    return {"id": job_id, "status": job["status"]}


def list_jobs() -> list[dict]:
    return list(JOBS.values())[::-1]


# ── access probe: store, bucket credential, GPU visible in this container ──────

_probe_lock = threading.Lock()
_probe_state: dict = {"status": "checking", "checked_at": None}
_probe_ts = 0.0
_probing = False

_GPU_FIX = [
    "on the VM: sudo apt install nvidia-container-toolkit",
    "sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker",
    "docker compose up -d",
]
_ADC_FIX = [
    "on the VM: gcloud auth application-default login --no-launch-browser",
    "docker compose restart backend",
]


def _verdict(status: str, problem: str | None = None, summary: str = "",
             detail: str = "", fix: list[str] | None = None) -> dict:
    return {"status": status, "problem": problem, "summary": summary, "detail": detail,
            "fix": fix or [], "checked_at": datetime.now(timezone.utc).isoformat()}


def _probe_local() -> dict | None:
    if SOURCE.gcs or (Path(SOURCE.root) / "zarr.json").is_file():
        return None
    return _verdict("error", "store", f"No zarr group at LOCAL_DATA={SOURCE.uri} (zarr.json missing).",
                    "", ["check the /data volume mount in docker-compose.yml",
                         "the scratch disk is wiped on stop/start: deploy/restage_store.sh"])


def _probe_bucket(loc: Location, what: str) -> dict | None:
    try:
        storage.bucket(loc).get_blob(loc.key("zarr.json"))
        return None
    except Exception as e:
        s, low = str(e), str(e).lower()
        if "403" in s or "does not have" in low or "denied" in low:
            return _verdict("error", "bucket", f"The backend reaches GCS but lacks access to the {what} bucket.",
                            s[-400:], [f"your account needs roles/storage.objectAdmin on gs://{loc.bucket}"])
        return _verdict("error", "adc", f"No working GCS credential for the {what} (missing or expired ADC).",
                        s[-400:], _ADC_FIX)


def _probe_nvidia() -> dict | None:
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        return _verdict("error", "gpu", "No GPU in this container (nvidia-smi missing).", "", _GPU_FIX)
    except subprocess.TimeoutExpired:
        return _verdict("error", "gpu", "nvidia-smi timed out.", "", _GPU_FIX)
    if r.returncode != 0:
        return _verdict("error", "gpu", "The container cannot see a GPU.", (r.stderr or r.stdout)[-400:], _GPU_FIX)
    return None


def _probe() -> dict:
    checks = [_probe_local()]
    if SOURCE.gcs:
        checks.append(_probe_bucket(SOURCE, "store"))
    if MIRROR:
        checks.append(_probe_bucket(MIRROR, "mirror"))
    checks.append(_probe_nvidia())
    failed = next((c for c in checks if c), None)
    if failed:
        return failed
    n = len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines())
    access = ("bucket store" if SOURCE.gcs else "local store") + (" + GCS mirror" if MIRROR else "")
    return _verdict("ok", None, f"{access} + {n} GPU{'s' if n != 1 else ''} visible")


def _probe_worker() -> None:
    global _probing, _probe_ts
    try:
        v = _probe()
    except Exception as e:
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
