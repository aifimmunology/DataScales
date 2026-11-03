"""The GPU steps, in the order of the out-of-core notebook and the RAPIDS benchmark."""
from __future__ import annotations

import sys

import anndata as ad
import numpy as np
import pandas as pd
from anndata.io import read_elem

from config import Config
from store import InputError


def to_host(a) -> np.ndarray:
    return a.get() if hasattr(a, "get") else np.asarray(a)


def read_obs(group, columns: list[str], required: list[str]) -> pd.DataFrame:
    """The obs index and those of `columns` the store has; every column in `required` must be there."""
    obs = group["obs"]
    missing = [c for c in required if c not in obs]
    if missing:
        raise InputError(f"obs has no column {', '.join(missing)}; it has {', '.join(obs.attrs['column-order'])}")
    keep = [c for c in dict.fromkeys(columns + required) if c in obs]
    return pd.DataFrame({c: read_elem(obs[c]) for c in keep}, index=read_elem(obs[obs.attrs["_index"]]))


def check_batch(obs: pd.DataFrame, keys: list[str]) -> None:
    """Batch keys as categories; a cell without one (missing, or "" as ingest writes it) is an error."""
    for key in keys:
        obs[key] = obs[key].astype("string").replace("", pd.NA).astype("category")
        n = int(obs[key].isna().sum())
        if n:
            raise InputError(f"{n:,} of {len(obs):,} cells have no {key} (missing or \"\"); "
                             "choose another batch key, or a subset without them")


def subset_rows(obs: pd.DataFrame, column: str, values: list[str]) -> np.ndarray:
    rows = np.flatnonzero(obs[column].astype("string").isin(values).to_numpy())
    if rows.size == 0:
        found = ", ".join(map(str, obs[column].astype("string").unique()[:50]))
        raise InputError(f"no cells have {column} in {values}; it has {found}")
    return rows


def spans(rows: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, end) runs of `rows`: one per value on a store sorted by the subset column."""
    breaks = np.flatnonzero(np.diff(rows) > 1) + 1
    starts, ends = np.r_[0, breaks], np.r_[breaks, rows.size]
    return [(int(rows[s]), int(rows[e - 1]) + 1) for s, e in zip(starts, ends)]


def slice_rows(x, rows: np.ndarray | None):
    """Rows of the lazy X by slicing contiguous spans, so dask reads only the chunks that hold them."""
    import dask.array as da

    if rows is None:
        return x
    parts = [x[s:e] for s, e in spans(rows)]
    return parts[0] if len(parts) == 1 else da.concatenate(parts, axis=0)


def lazy_counts(group, chunk_rows: int):
    from anndata.experimental import read_elem_lazy

    n_cells, n_genes = group["X"].attrs["shape"]
    return read_elem_lazy(group["X"], (min(chunk_rows, n_cells), n_genes))


def lognormalized(x, obs: pd.DataFrame | None, var: pd.DataFrame | None, target_sum: float) -> ad.AnnData:
    """Counts on the GPU (lazy), normalized per cell to `target_sum` and log1p'd."""
    import rapids_singlecell as rsc

    adata = ad.AnnData(X=x.astype(np.float32), obs=obs, var=var)
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.normalize_total(adata, target_sum=target_sum)
    rsc.pp.log1p(adata)
    return adata


def highly_variable(adata: ad.AnnData, cfg: Config) -> np.ndarray:
    import rapids_singlecell as rsc

    rsc.pp.highly_variable_genes(adata, n_top_genes=cfg.n_top_genes, flavor=cfg.hvg_flavor,
                                 batch_key=cfg.hvg_batch_key or None)
    return adata.var["highly_variable"].to_numpy()


def persist_bands(adata: ad.AnnData, n_workers: int) -> None:
    n_rows, n_cols = adata.shape
    adata.X = adata.X.rechunk((-(-n_rows // max(n_workers, 1)), n_cols)).persist()
    adata.X.compute_chunk_sizes()


def scale_pca(adata: ad.AnnData, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    import rapids_singlecell as rsc

    rsc.pp.scale(adata, max_value=cfg.scale_max_value, zero_center=False)
    rsc.pp.pca(adata, n_comps=cfg.n_comps, svd_solver="covariance_eigh", rng=cfg.random_seed)
    pcs = adata.obsm["X_pca"]
    pcs = pcs.compute() if hasattr(pcs, "compute") else pcs
    return to_host(pcs).astype(np.float32), to_host(adata.uns["pca"]["variance_ratio"])


def harmony(graph: ad.AnnData, keys: list[str], cfg: Config) -> None:
    import rapids_singlecell as rsc

    rsc.pp.harmony_integrate(graph, key=keys[0] if len(keys) == 1 else keys, basis="X_pca",
                             adjusted_basis="X_pca_harmony", max_iter_harmony=cfg.harmony_max_iter,
                             theta=cfg.harmony_theta, alpha=cfg.harmony_alpha, rng=cfg.random_seed)
    graph.obsm["X_pca_harmony"] = to_host(graph.obsm["X_pca_harmony"]).astype(np.float32)


def train_scvi(counts, obs: pd.DataFrame, genes: pd.Index, keys: list[str], cfg: Config, model_dir: str) -> np.ndarray:
    """scVI on the raw counts of the HVGs (in host memory), on the GPU through torch. Runs after the Dask-CUDA
    cluster is closed, so torch has the GPU memory to itself."""
    import scvi

    scvi.settings.seed = cfg.random_seed
    data = ad.AnnData(X=counts, obs=obs[keys].copy(), var=pd.DataFrame(index=genes.astype(str)))
    scvi.model.SCVI.setup_anndata(data, batch_key=keys[0] if keys else None,
                                  categorical_covariate_keys=keys[1:] or None)
    model = scvi.model.SCVI(data, n_latent=cfg.scvi_n_latent, n_layers=cfg.scvi_n_layers, n_hidden=cfg.scvi_n_hidden,
                            dropout_rate=cfg.scvi_dropout_rate, gene_likelihood=cfg.scvi_gene_likelihood)
    model.train(max_epochs=cfg.scvi_max_epochs or None, batch_size=cfg.scvi_batch_size, train_size=cfg.scvi_train_size,
                early_stopping=cfg.scvi_early_stopping, early_stopping_patience=cfg.scvi_early_stopping_patience,
                accelerator=cfg.scvi_accelerator, plan_kwargs={"lr": cfg.scvi_lr})
    latent = np.asarray(model.get_latent_representation(), dtype=np.float32)
    model.save(model_dir, save_anndata=False)
    del model
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return latent


def neighbors(graph: ad.AnnData, basis: str, cfg: Config) -> None:
    import rapids_singlecell as rsc

    n_pcs = min(cfg.n_pcs, graph.obsm[basis].shape[1]) if basis.startswith("X_pca") else None
    rsc.pp.neighbors(graph, n_neighbors=cfg.n_neighbors, n_pcs=n_pcs, use_rep=basis,
                     algorithm=cfg.neighbors_algorithm, rng=cfg.random_seed)


def umap(graph: ad.AnnData, cfg: Config) -> None:
    import rapids_singlecell as rsc

    rsc.tl.umap(graph, min_dist=cfg.umap_min_dist, init_pos="spectral", n_components=cfg.umap_n_components,
                rng=cfg.random_seed)
    graph.obsm["X_umap"] = to_host(graph.obsm["X_umap"]).astype(np.float32)


def leiden(graph: ad.AnnData, cfg: Config) -> None:
    """One leiden per resolution, all on the one neighbor graph."""
    import rapids_singlecell as rsc

    for key, resolution in cfg.leiden_keys.items():
        rsc.tl.leiden(graph, resolution=resolution, key_added=key, n_iterations=cfg.leiden_iterations,
                      rng=cfg.random_seed)


def full_rows(values: np.ndarray, rows: np.ndarray | None, n: int) -> np.ndarray:
    if rows is None:
        return values
    out = np.full((n, *values.shape[1:]), np.nan, dtype=np.float32)
    out[rows] = values
    return out


def full_column(values: pd.Series, rows: np.ndarray | None, n: int) -> pd.Categorical:
    values = pd.Categorical(values)
    if rows is None:
        return values
    codes = np.full(n, -1, dtype=values.codes.dtype)
    codes[rows] = values.codes
    return pd.Categorical.from_codes(codes, categories=values.categories)
