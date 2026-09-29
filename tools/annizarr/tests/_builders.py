from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from annizarr._ops import convert_adata
from annizarr.config import AppConfig


def make_adata(n_obs: int = 100, n_vars: int = 50, *, seed: int = 0, density: float = 0.3) -> AnnData:
    """A synthetic CSR AnnData with a categorical obs column and an obsm entry."""
    rng = np.random.default_rng(seed)
    x = sp.random(n_obs, n_vars, density=density, format="csr", dtype=np.float32, random_state=seed)
    x.data = np.abs(x.data) + 1.0
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(rng.choice(["a", "b", "c"], n_obs), categories=["a", "b", "c"])},
        index=[f"cell{seed}_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_vars)])
    return AnnData(X=x, obs=obs, var=var, obsm={"X_pca": rng.random((n_obs, 2)).astype(np.float32)})


def make_h5ad(tmp_path, name: str = "in.h5ad", *, adata: AnnData | None = None, **kw):
    """Write a synthetic (or given) AnnData to ``tmp_path/name``; return the path."""
    a = adata if adata is not None else make_adata(**kw)
    path = tmp_path / name
    a.write_h5ad(path)
    return path


def make_store(tmp_path, name: str = "store.zarr", *, adata: AnnData | None = None, cfg: AppConfig | None = None, **kw):
    """Convert a synthetic (or given) AnnData to a zarr store via convert_adata; return the path."""
    a = adata if adata is not None else make_adata(**kw)
    out = tmp_path / name
    convert_adata(a, output=out, cfg=cfg or AppConfig())
    return out
