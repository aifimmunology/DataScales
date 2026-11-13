from functools import lru_cache

from run_params import RunParams
from store_meta import latent_rep

from . import store


@lru_cache(maxsize=1)
def describe() -> dict:
    latent = _recorded_latent() if store.STORE else None
    return {"latent": latent, "default": "latent" if latent else "pca", "defaults": RunParams().to_dict()}


def _recorded_latent() -> str | None:
    import zarr

    return latent_rep(zarr.open_group(store.STORE.uri, mode="r", use_consolidated=False))
