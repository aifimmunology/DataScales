"""Backend configuration: the single store + GPU job settings, all env-driven."""

import os
from pathlib import Path

# One store serves everything: proxied reads (coords, obs, layers/gexp), the
# jobs/ queue + status objects, and returned umap_views/.
DATA_DIR = os.environ.get("DATA_DIR", "")

# GPU box for cold per-job runs (from .env); submit rejects jobs if any is unset.
GPU_INSTANCE = os.environ.get("GPU_INSTANCE", "")
GPU_ZONE = os.environ.get("GPU_ZONE", "")
GPU_PIXI_DIR = os.environ.get("GPU_PIXI_DIR", "")

MAX_JOBS = 50

# gpu/ holds the scripts shipped to the box. Compose bind-mounts the DIRECTORY
# (mounting single files pins their inode — edits that replace the file leave the
# container serving stale, truncated content).
APP_ROOT = Path(__file__).resolve().parents[1]
RERUN_SCRIPT = APP_ROOT / "gpu" / "rerun_umap_on_selection.py"
JOB_SCRIPT = APP_ROOT / "gpu" / "gpu_job.sh"


def _parse_dir(d: str) -> dict:
    if d.startswith("gs://"):
        bucket, _, prefix = d[len("gs://"):].partition("/")
        return {"gcs": True, "bucket": bucket, "prefix": prefix.rstrip("/")}
    return {"gcs": False, "root": d}


SOURCE = _parse_dir(DATA_DIR)
