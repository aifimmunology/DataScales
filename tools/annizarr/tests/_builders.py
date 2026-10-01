from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from annizarr.config import AppConfig
from annizarr.ops import convert_adata


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


def make_10x_v3_h5(path, *, include_non_gex: bool = False, int32_data: bool = False):
    """Write a Cell Ranger v3 10x HDF5 file (matrix/{barcodes,data,indices,indptr,shape,features/*}).

    With ``include_non_gex``, appends a 4th feature tagged "Antibody Capture" (not "Gene
    Expression") to pin the ``gex_only`` filter.
    """
    dtype = np.int32 if int32_data else np.float32
    ids = [b"ENSG001", b"ENSG002", b"ENSG003"]
    names = [b"GENEA", b"GENEB", b"GENEC"]
    ftypes = [b"Gene Expression"] * 3
    # dense == [[1, 0, 3], [0, 2, 0]]; the non-GEX case appends a 4th (antibody) column
    # on top of the same 3 GEX columns, so filtering it back out reproduces the base case.
    data, indices, indptr = [1, 3, 2], [0, 2, 1], [0, 2, 3]
    if include_non_gex:
        ids.append(b"AB001")
        names.append(b"ABC1")
        ftypes.append(b"Antibody Capture")
        data, indices, indptr = [1, 3, 9, 2, 5], [0, 2, 3, 1, 3], [0, 3, 5]
    n_features = len(ids)

    with h5py.File(path, "w") as f:
        m = f.create_group("matrix")
        m.create_dataset("barcodes", data=np.array([b"CELL1-1", b"CELL2-1"]))
        m.create_dataset("data", data=np.array(data, dtype=dtype))
        m.create_dataset("indices", data=np.array(indices, dtype=np.int32))
        m.create_dataset("indptr", data=np.array(indptr, dtype=np.int32))
        m.create_dataset("shape", data=np.array([n_features, 2], dtype=np.int32))
        feat = m.create_group("features")
        feat.create_dataset("id", data=np.array(ids))
        feat.create_dataset("name", data=np.array(names))
        feat.create_dataset("feature_type", data=np.array(ftypes))
        feat.create_dataset("genome", data=np.array([b"GRCh38"] * n_features))
    return path


def make_10x_v2_h5(path, *, genomes: tuple[str, ...] = ("GRCh38",)):
    """Write a Cell Ranger v2 10x HDF5 file: one ``<genome>`` group per entry in ``genomes``."""
    with h5py.File(path, "w") as f:
        for genome in genomes:
            g = f.create_group(genome)
            g.create_dataset("data", data=np.array([1.0, 3.0, 2.0], dtype=np.float32))
            g.create_dataset("indices", data=np.array([0, 2, 1], dtype=np.int32))
            g.create_dataset("indptr", data=np.array([0, 2, 3], dtype=np.int32))
            g.create_dataset("shape", data=np.array([3, 2], dtype=np.int32))
            g.create_dataset("barcodes", data=np.array([b"CELL1-1", b"CELL2-1"]))
            g.create_dataset("genes", data=np.array([b"ENSG001", b"ENSG002", b"ENSG003"]))
            g.create_dataset("gene_names", data=np.array([b"GENEA", b"GENEB", b"GENEC"]))
    return path
