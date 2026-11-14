import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr
from starlette.requests import Request

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gpu"))

from server import store  # noqa: E402
from server.store import LocalStore  # noqa: E402


def request(method="GET", **headers) -> Request:
    return Request({"type": "http", "method": method, "path": "/", "query_string": b"",
                    "headers": [(k.lower().replace("_", "-").encode(), v.encode()) for k, v in headers.items()]})


def write_adata(path: Path, n=40, *, latent=True, umap=True, X="csr", barcodes=None, extra_obs=None) -> Path:
    ad.settings.zarr_write_format = 3
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({
        "cell_type": pd.Categorical(rng.choice(["T", "B", "NK", "unused"], n, p=[.4, .3, .3, 0]),
                                    categories=["T", "B", "NK", "unused"]),
        "batch": pd.Categorical(rng.choice(["a", "b"], n)),
        "score": rng.random(n).astype("float32"),
        **(extra_obs or {}),
    }, index=pd.Index(barcodes or [f"BC{i:03d}" for i in range(n)], name="barcode"))
    adata = ad.AnnData(
        X={"csr": sp.csr_matrix(rng.poisson(1, (n, 6)).astype("float32")),
           "csc": sp.csc_matrix(rng.poisson(1, (n, 6)).astype("float32")), None: None}[X], obs=obs)
    if X is None:
        adata = ad.AnnData(obs=obs)
    if umap:
        adata.obsm["X_umap"] = rng.random((n, 2)).astype("float32")
    if latent:
        adata.obsm["lat"] = rng.random((n, 4)).astype("float32")
        adata.uns["neighbors"] = {"params": {"use_rep": "lat"}}
    adata.write_zarr(path)
    return path


def write_view(root: Path, slug: str, rows: list[int], label="a view") -> None:
    n_root = zarr.open_group(str(root), mode="r")["obs"]["barcode"].shape[0]
    assert max(rows) < n_root
    obs = pd.DataFrame({"root_row": np.asarray(rows, dtype="int32"),
                        "leiden": pd.Categorical([str(i % 2) for i in range(len(rows))])},
                       index=pd.Index([f"BC{r:03d}" for r in rows], name="barcode"))
    out = root / "umap_views" / slug
    ad.AnnData(obs=obs, obsm={"X_umap": np.zeros((len(rows), 2), "float32")}).write_zarr(out)
    zarr.open_group(str(out), mode="r+").update_attributes({"datavis-label": label})


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    path = write_adata(tmp_path / "store.zarr")
    monkeypatch.setattr(store, "STORE", LocalStore(path))
    monkeypatch.setattr(store, "MIRROR", None)
    return path
