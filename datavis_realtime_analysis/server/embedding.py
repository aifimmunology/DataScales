"""What the store offers the re-embedding pipeline: its recorded latent space (if any),
the route that implies, and the pipeline's default parameters."""

from functools import lru_cache

from run_params import RunParams
from store_meta import latent_rep

from .config import SOURCE


@lru_cache(maxsize=1)
def describe() -> dict:
    latent = _recorded_latent() if SOURCE else None
    return {"latent": latent, "default": "latent" if latent else "pca", "defaults": RunParams().to_dict()}


def _recorded_latent() -> str | None:
    import zarr

    return latent_rep(zarr.open_group(SOURCE.uri, mode="r", use_consolidated=False))
