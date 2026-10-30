"""What the store offers the re-embedding pipeline: its recorded latent space (if any),
the route that implies, and the pipeline's default parameters."""

import sys
from functools import lru_cache
from pathlib import Path

from .config import SOURCE

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpu"))
from run_params import RunParams  # noqa: E402


@lru_cache(maxsize=1)
def describe() -> dict:
    latent = _recorded_latent() if SOURCE else None
    return {"latent": latent, "default": "latent" if latent else "pca", "defaults": RunParams().to_dict()}


def _recorded_latent() -> str | None:
    import zarr
    from anndata.io import read_elem

    root = zarr.open_group(SOURCE.uri, mode="r", use_consolidated=False)
    params = root.get("uns/neighbors/params")
    if params is None or "use_rep" not in params:
        return None
    rep = str(read_elem(params["use_rep"]))
    return rep if rep in root["obsm"] else None
