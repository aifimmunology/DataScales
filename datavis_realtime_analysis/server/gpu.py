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

from fastapi import HTTPException

from . import storage, views
from .config import (DATA_DIR, GPU_INSTANCE, GPU_PIXI_DIR, GPU_ZONE, JOB_SCRIPT,
                     MAX_JOBS, RERUN_SCRIPT, SOURCE)

JOBS: dict[str, dict] = {}
JOB_QUEUE: queue.Queue = queue.Queue()
_worker_lock = threading.Lock()
_worker_started = False
POLL_S = 1.5


# gcloud -q: non-interactive — on a fresh machine the first ssh/scp auto-generates
# the ssh keypair (no passphrase prompt) and waits for it to propagate
def _ssh_cmd(remote: str) -> list[str]:
    return ["gcloud", "-q", "compute", "ssh", GPU_INSTANCE, f"--zone={GPU_ZONE}",
            "--command", remote]


def _ship() -> None:
    # fresh scripts every job: the repo stays the source of truth on the box
    r = subprocess.run(["gcloud", "-q", "compute", "scp", str(JOB_SCRIPT), str(RERUN_SCRIPT),
                        f"{GPU_INSTANCE}:/tmp/", f"--zone={GPU_ZONE}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"scp to {GPU_INSTANCE} failed: {r.stderr.strip()[-300:]}")


def _run_gpu_job(job: dict, slug: str) -> None:
    rid = job["id"]
    _ship()
    job["stage"] = "dispatching to GPU"
    remote = f"bash /tmp/{JOB_SCRIPT.name} {DATA_DIR} {rid} {slug} {GPU_PIXI_DIR}"
    with open(f"/tmp/datavis_job_{rid}.log", "ab") as log:  # child keeps its own fd
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
            _invalidate_probe()  # a failed job is often an expired credential
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


def _config_problem() -> str | None:
    missing = [k for k, v in (("GPU_INSTANCE", GPU_INSTANCE), ("GPU_ZONE", GPU_ZONE),
                              ("GPU_PIXI_DIR", GPU_PIXI_DIR)) if not v]
    if missing:
        return f"GPU config missing in .env: {', '.join(missing)}"
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
    JOBS[job_id] = {
        "id": job_id,
        "name": name,
        "cells": cells,
        "group": artifact.get("group", ""),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "status": "queued",
        "stage": "queued",
        "view": None,
    }
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_").lower() or "view"
    slug = f"{slug}_{job_id}"
    storage.write_json(f"jobs/submitted/{job_id}.json", artifact)
    _ensure_worker()
    JOB_QUEUE.put((job_id, slug))
    return {"job_id": job_id, "status": "submitted"}


def list_jobs() -> list[dict]:
    return list(JOBS.values())[::-1]


# ── GPU access probe ──────────────────────────────────────────────────────────
# TEMPORARY while dispatch rides ssh: one cached round trip that exercises the
# exact permissions a job needs — backend ADC → store, host gcloud cred → ssh to
# the box, box gcloud → store write. Drop this whole section when jobs move to
# an HPC-style queue instead of a direct ssh connection.

PROBE_TTL_S = 300
PROBE_TIMEOUT_S = 60
_PROBE_OK = "DATAVIS_PROBE_OK"
_PROBE_STORE_FAIL = "DATAVIS_PROBE_STORE_FAIL"

_probe_lock = threading.Lock()
_probe_state: dict = {"status": "checking", "checked_at": None}
_probe_ts = 0.0
_probing = False

# The org session policy expires user credentials (~weekly). The CLI credential
# (`gcloud auth login`, drives compute ssh) and ADC (`application-default`, drives
# GCS reads) are separate tokens — one can expire while the other still works.
# Both live on the machine running the app; the container copies them at start,
# so a backend restart is required after reauth.
_HOST_REAUTH_FIX = [
    "on this host machine:",
    "  gcloud auth login",
    "  gcloud auth application-default login",
    "  docker compose restart backend",
]


def _verdict(status: str, problem: str | None = None, summary: str = "",
             detail: str = "", fix: list[str] | None = None) -> dict:
    return {"status": status, "problem": problem, "summary": summary,
            "detail": detail, "fix": fix or [],
            "checked_at": datetime.now(timezone.utc).isoformat()}


def _instance_fix() -> list[str]:
    return [f"gcloud compute instances describe {GPU_INSTANCE} --zone={GPU_ZONE} "
            "--format='value(status)'",
            f"gcloud compute instances start {GPU_INSTANCE} --zone={GPU_ZONE}"]


def _classify_ssh_failure(rc: int, err: str) -> dict:
    low, detail = err.lower(), err.strip()[-400:]
    if any(t in low for t in ("reauth", "invalid_grant", "invalid_rapt",
                              "credentials have expired", "do not currently have an active account")):
        return _verdict("error", "host_auth",
                        "This machine's gcloud ssh credential expired (org policy reauths "
                        "~weekly) GPU submits are blocked. It's a separate token from the "
                        "GCS one, which is why the store still loads.", detail, _HOST_REAUTH_FIX)
    if "not found" in low:
        return _verdict("error", "config",
                        f"GPU instance '{GPU_INSTANCE}' (zone '{GPU_ZONE}') was not found — "
                        "check GPU_INSTANCE/GPU_ZONE in .env.", detail)
    if any(t in low for t in ("timed out", "connection refused", "unable to connect")):
        return _verdict("error", "unreachable",
                        f"GPU instance '{GPU_INSTANCE}' is unreachable — it may be stopped.",
                        detail, _instance_fix())
    if "permission" in low:
        return _verdict("error", "ssh_perms",
                        "Your account can't ssh to the GPU instance.", detail,
                        ["ask an admin for roles/compute.osLogin (or instanceAdmin.v1) "
                         "on the instance",
                         "plus roles/iam.serviceAccountUser on the instance's service account"])
    return _verdict("error", "ssh", f"ssh to the GPU box failed (rc={rc}).", detail,
                    ["docker compose logs backend    # full probe stderr"])


def _probe_gpu() -> dict:
    try:  # backend ADC → store: submits queue jobs through these writes
        storage.bucket().get_blob(storage.key("zarr.json"))
    except Exception as e:
        s, low = str(e), str(e).lower()
        if "403" in s or "does not have" in low or "denied" in low:
            return _verdict("error", "host_bucket",
                            "The backend can reach GCS but lacks bucket access.", s[-400:],
                            [f"grant your account roles/storage.objectAdmin on "
                             f"gs://{SOURCE['bucket']}"])
        return _verdict("error", "host_adc",
                        "The backend's GCS credentials on this machine are missing or "
                        "expired (org policy reauths ~weekly).", s[-400:], _HOST_REAUTH_FIX)

    # same first act as gpu_job.sh: a store write from the box via gcloud storage cp
    probe_obj = f"{DATA_DIR.rstrip('/')}/jobs/probe"
    remote = (f"err=$(printf datavis-probe | gcloud -q storage cp - '{probe_obj}' 2>&1) "
              f"&& echo {_PROBE_OK} || "
              f"{{ echo {_PROBE_STORE_FAIL}; printf '%s\\n' \"$err\" | tail -5; }}")
    try:
        r = subprocess.run(_ssh_cmd(remote), capture_output=True, text=True,
                           timeout=PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return _verdict("error", "unreachable",
                        f"Probe timed out after {PROBE_TIMEOUT_S}s — the instance may be "
                        "stopped or ssh is hanging.", "", _instance_fix())
    if _PROBE_OK in r.stdout:
        return _verdict("ok", None, "ssh + store write from the GPU box verified")
    if _PROBE_STORE_FAIL in r.stdout:
        detail = r.stdout.split(_PROBE_STORE_FAIL, 1)[1].strip()[-400:]
        return _verdict("error", "box_gcs",
                        "The GPU box can't write the store — its GCS grant/credential "
                        "is missing or expired. (This machine's access is fine.)", detail,
                        ["Reauth on the gpu:",
                         "  gcloud auth login",
                         "  gcloud auth application-default login"
                        ])
    return _classify_ssh_failure(r.returncode, r.stderr)


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
