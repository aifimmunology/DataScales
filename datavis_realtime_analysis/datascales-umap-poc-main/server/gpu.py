"""GPU jobs, cold per run (cluster-style): the job json goes to the store's queue
(jobs/submitted/<id>.json), one ssh dispatches gpu_job.sh on the box — fresh env
setup, rerun_umap_on_selection.py, view upload — and the script reports progress
to jobs/status/<id>.json, which the worker polls to drive the app's job status.
"""

import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import storage, views
from .config import (DATA_DIR, GPU_INSTANCE, GPU_PIXI_DIR, GPU_ZONE, JOB_SCRIPT,
                     MAX_JOBS, RERUN_SCRIPT, SIMULATE_SCRIPT, SOURCE)

JOBS: dict[str, dict] = {}
JOB_QUEUE: queue.Queue = queue.Queue()
_worker_lock = threading.Lock()
_worker_started = False
POLL_S = 1.5


def _ssh_cmd(remote: str) -> list[str]:
    return ["gcloud", "compute", "ssh", GPU_INSTANCE, f"--zone={GPU_ZONE}",
            "--command", remote]


def _ship(job_id: str) -> None:
    # fresh scripts every job: the repo stays the source of truth on the box
    r = subprocess.run(["gcloud", "compute", "scp", str(JOB_SCRIPT), str(RERUN_SCRIPT),
                        f"{GPU_INSTANCE}:/tmp/", f"--zone={GPU_ZONE}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"scp to {GPU_INSTANCE} failed: {r.stderr.strip()[-300:]}")


def _run_gpu_job(job: dict, slug: str) -> None:
    rid = job["id"]
    _ship(rid)
    job["stage"] = "dispatching to GPU"
    remote = f"bash /tmp/{JOB_SCRIPT.name} {DATA_DIR} {rid} {slug} {GPU_PIXI_DIR}"
    log = open(f"/tmp/datavis_job_{rid}.log", "ab")
    proc = subprocess.Popen(_ssh_cmd(remote), stdout=log, stderr=subprocess.STDOUT)

    # the job script owns the truth: poll its status object until terminal
    dead_polls = 0
    while True:
        st = storage.read_json(f"jobs/status/{rid}.json")
        if st:
            job["stage"] = st.get("stage", "")
            if st.get("status") == "done":
                proc.wait(timeout=120)
                return
            if st.get("status") == "failed":
                proc.wait(timeout=120)
                raise RuntimeError(f"gpu job failed: {job['stage']} "
                                   f"(box log: /tmp/datavis_job_{rid}.log)")
        if proc.poll() is not None:
            dead_polls += 1  # one grace poll: the final status write may still land
            if dead_polls > 1:
                raise RuntimeError(f"ssh exited rc={proc.returncode} before the job "
                                   f"finished (log: /tmp/datavis_job_{rid}.log)")
        time.sleep(POLL_S)


def _worker() -> None:
    while True:
        job_id, slug = JOB_QUEUE.get()
        job = JOBS.get(job_id)
        if job is None:
            continue
        job["status"] = "running"
        try:
            _run_gpu_job(job, slug)
            job["view"] = views.register_view(slug, job["name"])
            job["stage"] = "done"
            job["status"] = "done"
        except Exception as e:
            job["stage"] = str(e)[:200]
            job["status"] = "failed"
            try:  # keep the store's status history consistent with what the UI saw
                storage.write_json(f"jobs/status/{job_id}.json",
                                   {"status": "failed", "stage": job["stage"]})
            except Exception:
                pass
        finally:
            try:
                storage.delete_object(f"jobs/submitted/{job_id}.json")
            except Exception:
                pass


def _ensure_worker() -> None:
    global _worker_started
    with _worker_lock:
        if not _worker_started:
            threading.Thread(target=_worker, daemon=True).start()
            _worker_started = True


def _simulate(job_id: str) -> None:
    try:
        rc = subprocess.run(["bash", str(SIMULATE_SCRIPT), job_id]).returncode
        JOBS[job_id]["status"] = "done" if rc == 0 else "failed"
    except Exception:
        JOBS[job_id]["status"] = "failed"


def submit(artifact: dict) -> dict:
    job_id = uuid.uuid4().hex[:8]
    cells = len(artifact.get("indices", []))
    name = str(artifact.get("name") or f"view {job_id}").strip()[:60]
    print(f"[submit] job {job_id} '{name}': {cells} cells, group '{artifact.get('group', '')}'",
          file=sys.stderr, flush=True)
    # evict only finished jobs — dropping a queued entry would orphan its GPU run
    while len(JOBS) >= MAX_JOBS:
        victim = next((k for k, j in JOBS.items() if j["status"] in ("done", "failed")), None)
        if victim is None:
            break
        del JOBS[victim]
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
        if not SOURCE["gcs"]:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["stage"] = "GPU jobs require a gs:// DATA_DIR store"
            return {"job_id": job_id, "status": "failed"}
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_").lower() or "view"
        slug = f"{slug}_{job_id}"
        storage.write_json(f"jobs/submitted/{job_id}.json", artifact)
        _ensure_worker()
        JOB_QUEUE.put((job_id, slug))
    else:
        threading.Thread(target=_simulate, args=(job_id,), daemon=True).start()
    return {"job_id": job_id, "status": "submitted"}


def list_jobs() -> list[dict]:
    return list(JOBS.values())[::-1]
