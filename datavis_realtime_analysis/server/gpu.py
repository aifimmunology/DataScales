"""GPU jobs: in-memory queue and history, dispatched to a warm pipeline process
(gpu/worker.py) that keeps imports + CUDA context between jobs. Stage updates stream
over the worker's stdout; the store only sees the finished view (mirrored when configured).
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
from run_params import RunParams

from . import storage
from .config import MAX_JOBS, MIRROR, SOURCE, Location

WORKER_SCRIPT = Path(__file__).resolve().parents[1] / "gpu" / "worker.py"
PROBE_TTL_S = 60
FINISHED = ("done", "failed")

JOBS: dict[str, dict] = {}
JOB_QUEUE: queue.Queue = queue.Queue()
_worker_lock = threading.Lock()
_worker_started = False
_proc: subprocess.Popen | None = None
_proc_lock = threading.Lock()


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


def _follow(proc: subprocess.Popen, job: dict) -> None:
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(f"pipeline worker died (rc={proc.poll()})")
        line = line.strip()
        if line.startswith("stage: "):
            job["stage"] = line[len("stage: "):]
        elif line == f"done: {job['id']}":
            return
        elif line.startswith("error: "):
            raise RuntimeError(line[len("error: "):])


def _run_gpu_job(job: dict, barcodes: list[str]) -> None:
    sel = Path(f"/tmp/datavis_job_{job['id']}.json")
    sel.write_text(json.dumps({"barcodes": barcodes}))
    view = f"umap_views/{job['slug']}"
    try:
        _follow(_send({"id": job["id"], "store": SOURCE.uri, "selection": str(sel), "label": job["name"],
                       "out": f"{SOURCE.uri}/{view}", "params": job["params"]}), job)
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
    try:
        storage.delete_prefix(view)
    except Exception as e:
        print(f"[jobs] could not remove partial {view}: {e}", file=sys.stderr, flush=True)


def _worker() -> None:
    while True:
        job_id, barcodes = JOB_QUEUE.get()
        job = JOBS.get(job_id)
        if job is None:
            continue
        job["status"] = "running"
        t0 = time.monotonic()
        try:
            _run_gpu_job(job, barcodes)
            job["view"] = f"umap_views/{job['slug']}"
            job["stage"] = job["status"] = "done"
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


def _config_problem() -> str | None:
    return None if SOURCE else "no store configured: set LOCAL_DATA and/or GCS_DATA"


def submit(payload: dict) -> dict:
    """Queue a re-embedding of the cells named by payload["barcodes"]."""
    problem = _config_problem()
    if problem:
        raise HTTPException(400, problem)
    barcodes = [str(b) for b in payload.get("barcodes") or []]
    if not barcodes:
        raise HTTPException(400, "empty selection")
    try:
        params = RunParams.from_dict(payload.get("params")).to_dict()
    except ValueError as e:
        raise HTTPException(400, str(e))
    job_id = uuid.uuid4().hex[:8]
    name = str(payload.get("name") or f"view {job_id}").strip()[:60]
    print(f"[submit] job {job_id} '{name}': {len(barcodes)} cells", file=sys.stderr, flush=True)
    while len(JOBS) >= MAX_JOBS:
        victim = next((k for k, j in JOBS.items() if j["status"] in FINISHED), None)
        if victim is None:
            break
        del JOBS[victim]
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_").lower() or "view"
    JOBS[job_id] = {
        "id": job_id, "name": name, "slug": f"{slug}_{job_id}", "cells": len(barcodes), "params": params,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "status": "queued", "stage": "queued", "view": None,
    }
    _ensure_worker()
    JOB_QUEUE.put((job_id, barcodes))
    return {"job_id": job_id, "status": "submitted"}


def list_jobs() -> list[dict]:
    return list(JOBS.values())[::-1]


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
    if SOURCE.gcs or (Path(SOURCE.uri) / "zarr.json").is_file():
        return None
    return _verdict("error", "store", f"No zarr group at LOCAL_DATA={SOURCE.uri} (zarr.json missing).",
                    "", ["check the /data volume mount in docker-compose.yml",
                         "the scratch disk is wiped on stop/start: copy the store back to LOCAL_DATA"])


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


def _probe_nvidia() -> tuple[dict | None, int]:
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        return _verdict("error", "gpu", "No GPU in this container (nvidia-smi missing).", "", _GPU_FIX), 0
    except subprocess.TimeoutExpired:
        return _verdict("error", "gpu", "nvidia-smi timed out.", "", _GPU_FIX), 0
    if r.returncode != 0:
        return _verdict("error", "gpu", "The container cannot see a GPU.", (r.stderr or r.stdout)[-400:], _GPU_FIX), 0
    return None, len(r.stdout.strip().splitlines())


def _probe() -> dict:
    gpu_problem, n = _probe_nvidia()
    checks = [_probe_local()]
    if SOURCE.gcs:
        checks.append(_probe_bucket(SOURCE, "store"))
    if MIRROR:
        checks.append(_probe_bucket(MIRROR, "mirror"))
    checks.append(gpu_problem)
    failed = next((c for c in checks if c), None)
    if failed:
        return failed
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
    """Cached store/credential/GPU probe; a stale or forced check runs in the background."""
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
