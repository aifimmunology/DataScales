"""The GPU pipeline: lazy CSR read, normalize, HVG, PCA, optional harmony, neighbors, UMAP, leiden."""
from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
import zarr
from anndata.io import read_elem

from config import Settings


class PipelineError(RuntimeError):
    """The input can't be processed as given."""


@dataclass
class Embedding:
    highly_variable: np.ndarray
    variance_ratio: np.ndarray
    obsm: dict[str, np.ndarray]
    leiden: pd.Categorical

    @property
    def n_clusters(self) -> int:
        return len(self.leiden.categories)


def to_host(a) -> np.ndarray:
    return a.get() if hasattr(a, "get") else np.asarray(a)


def read_obs(root: zarr.Group, columns: list[str]) -> pd.DataFrame:
    """The obs index and only `columns`, each as a category (NA stays NA)."""
    obs = root["obs"]
    missing = [c for c in columns if c not in obs]
    if missing:
        raise PipelineError(f"obs has no column(s) {missing}; available: {list(obs.attrs['column-order'])}")
    index = pd.Index(read_elem(obs[obs.attrs["_index"]]).astype(str))
    return pd.DataFrame({c: pd.Series(read_elem(obs[c])).astype("string").astype("category").array for c in columns},
                        index=index)


def check_batch_keys(obs: pd.DataFrame, keys: list[str]) -> None:
    missing = {k: int(obs[k].isna().sum()) for k in keys if obs[k].isna().any()}
    if missing:
        raise PipelineError(f"Batch keys have missing values {missing}; fill or drop those cells before integrating.")


def read_var(root: zarr.Group) -> pd.DataFrame:
    return read_elem(root["var"])


def lazy_adata(root: zarr.Group, obs: pd.DataFrame, var: pd.DataFrame, settings: Settings) -> ad.AnnData:
    """X as a Dask CSR array of `rows_per_chunk` rows, cast to float32 and moved to the GPU."""
    import rapids_singlecell as rsc
    from anndata.experimental import read_elem_lazy

    x = root["X"]
    if x.attrs.get("encoding-type") != "csr_matrix":
        raise PipelineError(f"X must be CSR, found {x.attrs.get('encoding-type', 'a dense array')}.")
    n_obs, n_vars = x.attrs["shape"]
    X = read_elem_lazy(x, (min(settings.rows_per_chunk, n_obs), n_vars)).astype(np.float32)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    rsc.get.anndata_to_GPU(adata)
    return adata


def normalize(adata: ad.AnnData, settings: Settings):
    """normalize_total + log1p in place; returns the (lazy) log-normalized matrix over all genes."""
    import rapids_singlecell as rsc

    rsc.pp.normalize_total(adata, target_sum=settings.target_sum)
    rsc.pp.log1p(adata)
    return adata.X


def hvg_pca(adata: ad.AnnData, settings: Settings, n_workers: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seurat HVGs, then scale and PCA on them; returns the HVG mask, X_pca and the variance ratio."""
    import rapids_singlecell as rsc

    rsc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=settings.n_top_genes)
    hvg = np.asarray(adata.var["highly_variable"])
    sub = adata[:, hvg].copy()
    rows = -(-sub.n_obs // max(n_workers, 1))
    sub.X = sub.X.rechunk((rows, sub.n_vars)).persist()
    sub.X = sub.X.astype(np.float64)
    rsc.pp.scale(sub, max_value=10, zero_center=False)
    rsc.pp.pca(sub, n_comps=settings.n_comps, svd_solver="covariance_eigh", rng=settings.random_seed)
    X_pca = to_host(sub.obsm["X_pca"].compute()).astype(np.float32)
    return hvg, X_pca, to_host(sub.uns["pca"]["variance_ratio"])


def integrate(graph: ad.AnnData, settings: Settings) -> str:
    """Adds the integrated basis to `graph.obsm`; returns the basis neighbors should use."""
    import rapids_singlecell as rsc

    if settings.integration == "harmony":
        rsc.pp.harmony_integrate(graph, key=settings.integration_keys, basis="X_pca",
                                 adjusted_basis="X_pca_harmony", rng=settings.random_seed)
        graph.obsm["X_pca_harmony"] = to_host(graph.obsm["X_pca_harmony"]).astype(np.float32)
        return "X_pca_harmony"
    return "X_pca"


def neighbors_umap_leiden(graph: ad.AnnData, basis: str, settings: Settings) -> None:
    import rapids_singlecell as rsc

    rsc.pp.neighbors(graph, n_neighbors=settings.n_neighbors, n_pcs=settings.n_pcs, use_rep=basis,
                     algorithm=settings.neighbors_algorithm, algorithm_kwds={"n_probes": settings.n_probes},
                     rng=settings.random_seed)
    rsc.tl.umap(graph, min_dist=settings.umap_min_dist, init_pos="spectral", rng=settings.random_seed)
    rsc.tl.leiden(graph, resolution=settings.leiden_resolution, n_iterations=settings.leiden_iterations,
                  rng=settings.random_seed)
    graph.obsm["X_umap"] = to_host(graph.obsm["X_umap"]).astype(np.float32)


def graph_adata(obs: pd.DataFrame, X_pca: np.ndarray) -> ad.AnnData:
    """A matrix-free AnnData for the single-GPU graph steps, so the workers' X can be released."""
    return ad.AnnData(obs=obs.copy(), obsm={"X_pca": X_pca})


def embedding(graph: ad.AnnData, hvg: np.ndarray, variance_ratio: np.ndarray) -> Embedding:
    leiden = graph.obs["leiden"].astype(str)
    order = sorted(leiden.unique(), key=int)
    return Embedding(
        highly_variable=hvg,
        variance_ratio=variance_ratio,
        obsm={k: v for k, v in graph.obsm.items() if k in ("X_pca", "X_pca_harmony", "X_umap")},
        leiden=pd.Categorical(leiden, categories=order),
    )
