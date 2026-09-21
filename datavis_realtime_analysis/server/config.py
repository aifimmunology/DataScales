"""Backend configuration: the single store, env-driven."""

import os

# One store serves everything: proxied reads (coords, obs, layers/gexp), the
# jobs/ queue + status objects, and returned umap_views/.
DATA_DIR = os.environ.get("DATA_DIR", "")

MAX_JOBS = 50


def _parse_dir(d: str) -> dict:
    if d.startswith("gs://"):
        bucket, _, prefix = d[len("gs://"):].partition("/")
        return {"gcs": True, "bucket": bucket, "prefix": prefix.rstrip("/")}
    return {"gcs": False, "root": d}


SOURCE = _parse_dir(DATA_DIR)
