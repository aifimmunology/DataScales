"""GPU jobs, cluster-style: submit writes the job json to the store's queue
(jobs/submitted/<id>.json); the watcher on the GPU host (gpu/job_watcher.sh,
systemd datavis-watcher) claims it and runs gpu/gpu_job.sh in the rapids pixi
env. The job script reports each stage to jobs/status/<id>.json, which the
worker here polls to drive the app's job status. No ssh and no user
credentials — the backend only reads/writes the store (the VM's service
account when deployed).
"""

import queue
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone

from fastapi import HTTPException

from . import storage, views
from .config import MAX_JOBS, SOURCE

JOBS: dict[str, dict] = {}
JOB_QUEUE: queue.Queue = queue.Queue()
_worker_lock = threading.Lock()
_worker_started = False
POLL_S = 2.0
WATCHER_STALE_S = 120  # heartbeat (jobs/watcher.json) older than this = watcher down
_STALE_CHECK_POLLS = 15  # queued jobs re-check the heartbeat every ~30 s, not every poll

_WATCHER_FIX = [
    "on the GPU host:",
    "  sudo systemctl restart datavis-watcher",
    "  journalctl -u datavis-watcher -f",
]


def _watcher_age() -> float | None:
    """Seconds since the watcher's last heartbeat; None = never seen/unreadable."""
    hb = storage.read_json("jobs/watcher.json")
    try:
        ts = datetime.fromisoformat(hb["ts"].replace("Z", "+00:00"))
    except Exception:
        return None
    return (datetime.now(timezone.utc) - ts).total_seconds()


def _run_gpu_job(job: dict) -> None:
    rid = job["id"]
    vanished = 0
    polls = 0
    while True:
        st = storage.read_json(f"jobs/status/{rid}.json")
        if st:
            vanished = 0
            job["stage"] = st.get("stage", "")
            if st.get("status") == "done":
                return
            if st.get("status") == "failed":
                raise RuntimeError(f"gpu job failed: {job['stage']} "
                                   f"(log: /tmp/datavis_job_{rid}.log on the GPU host)")
        elif storage.read_json(f"jobs/submitted/{rid}.json") is None:
            vanished += 1  # claimed but not yet reporting — grace for the first status write
            if vanished > 5:
                raise RuntimeError("job left the queue without reporting a status — "
                                   "check the GPU host (journalctl -u datavis-watcher)")
        else:
            job["stage"] = "queued — waiting for the GPU host watcher"
            if polls % _STALE_CHECK_POLLS == _STALE_CHECK_POLLS - 1:
                age = _watcher_age()
                if age is None or age > WATCHER_STALE_S:
                    raise RuntimeError("the GPU job watcher is not running — the job "
                                       "would queue forever (systemctl status "
                                       "datavis-watcher on the GPU host)")
        polls += 1
        time.sleep(POLL_S)


def _worker() -> None:
    while True:
        job_id = JOB_QUEUE.get()
        job = JOBS.get(job_id)
        if job is None:
            continue
        job["status"] = "running"
        try:
            _run_gpu_job(job)
            job["view"] = views.register_view(job["slug"], job["name"])
            job["stage"] = "done"
            job["status"] = "done"
        except Exception as e:
            job["stage"] = str(e)[:200]
            job["status"] = "failed"
            _invalidate_probe()  # a failed job often means the watcher/SA needs a look
            try:  # keep the store's status history consistent with what the UI saw
                storage.write_json(f"jobs/status/{job_id}.json",
                                   {"status": "failed", "stage": job["stage"]})
            except Exception:
                pass
            try:  # unqueue: a returning watcher must not run a job the UI marked failed
                storage.delete_object(f"jobs/submitted/{job_id}.json")
            except Exception:
                pass


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
    # evict only finished jobs — dropping a queued entry would orphan its GPU run
    while len(JOBS) >= MAX_JOBS:
        victim = next((k for k, j in JOBS.items() if j["status"] in ("done", "failed")), None)
        if victim is None:
            break
        del JOBS[victim]
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_").lower() or "view"
    slug = f"{slug}_{job_id}"
    JOBS[job_id] = {
        "id": job_id,
        "name": name,
        "slug": slug,
        "cells": cells,
        "group": artifact.get("group", ""),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "status": "queued",
        "stage": "queued",
        "view": None,
    }
    # id/slug ride the queue object so the watcher can dispatch without a side-channel
    storage.write_json(f"jobs/submitted/{job_id}.json",
                       {**artifact, "id": job_id, "slug": slug})
    _ensure_worker()
    JOB_QUEUE.put(job_id)
    return {"job_id": job_id, "status": "submitted"}


def list_jobs() -> list[dict]:
    return list(JOBS.values())[::-1]


# ── GPU access probe ──────────────────────────────────────────────────────────
# Two cheap store reads: the backend credential → store (submits ride these
# writes), and the watcher heartbeat → is anything on the GPU host consuming
# the queue.

PROBE_TTL_S = 60

_probe_lock = threading.Lock()
_probe_state: dict = {"status": "checking", "checked_at": None}
_probe_ts = 0.0
_probing = False


def _verdict(status: str, problem: str | None = None, summary: str = "",
             detail: str = "", fix: list[str] | None = None) -> dict:
    return {"status": status, "problem": problem, "summary": summary,
            "detail": detail, "fix": fix or [],
            "checked_at": datetime.now(timezone.utc).isoformat()}


def _probe_gpu() -> dict:
    try:  # backend credential → store: submits queue jobs through these writes
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
    age = _watcher_age()
    if age is None:
        return _verdict("error", "watcher",
                        "No job-watcher heartbeat on this store — submits will queue "
                        "but never run.", "",
                        ["on the GPU host:",
                         "  sudo systemctl enable --now datavis-watcher",
                         "  journalctl -u datavis-watcher -f"])
    if age > WATCHER_STALE_S:
        return _verdict("error", "watcher_stale",
                        f"The job watcher last reported {int(age)}s ago — it looks down.",
                        "", _WATCHER_FIX)
    return _verdict("ok", None,
                    f"store access + job watcher verified (heartbeat {int(age)}s ago)")


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
