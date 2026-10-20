"""pca (run 2): dev_tools/rapids-benchmark `run_pipeline` steps on the subset rows, then Harmony.

load (t_load): lazy zarr X in (chunk_rows, all genes) dask blocks, rows picked inside each block (stores are
not sorted by cell type, so rapids-benchmark's span slicing would build ~10^5 tasks); no X is read yet.
run (t_pre): to GPU -> qc -> normalize -> log1p -> HVG seurat -> persist -> scale (float32) -> PCA
covariance_eigh -> Harmony on batch (alpha/theta/max_iter from params). The read happens inside t_pre.
Cluster: rapids-benchmark capacity preset (tcp + RMM managed memory), 1 GPU.
"""
import contextlib

import numpy as np

from .. import config

USES_RMM = True


def needs_cluster(ctx):
    return True


def _rb_cfg():
    import rapids_benchmark as rb
    return rb.Config(gpus="0")


@contextlib.contextmanager
def cluster(ctx):
    import rapids_benchmark as rb
    from dask.distributed import Client
    cfg = _rb_cfg()
    protocol, rmm_mode = cfg.resolve_cluster()
    with rb.make_cluster(cfg, protocol, rmm_mode) as cl, Client(cl) as client:
        client.run(rb._set_zarr_config, cfg.zarr_concurrency, cfg.zarr_max_workers)
        rb._set_zarr_config(cfg.zarr_concurrency, cfg.zarr_max_workers)
        rb.init_client_rmm(cfg.gpu_ids, rmm_mode)
        yield client


def load(ctx):
    import anndata as ad
    from anndata.experimental import read_elem_lazy
    X = ctx.root["X"]
    n, g = (int(v) for v in X.attrs["shape"])
    Xd = read_elem_lazy(X, (ctx.params["pca"]["chunk_rows"], g))
    if np.issubdtype(Xd.dtype, np.integer):
        Xd = Xd.astype(np.float32)
    rows = np.asarray(ctx.rows)
    if len(rows) < n:
        Xd = Xd[rows]
    obs = ctx.obs.iloc[rows].copy()
    for c in obs.select_dtypes("category").columns:
        obs[c] = obs[c].cat.remove_unused_categories()
    ctx.data = ad.AnnData(X=Xd, obs=obs, var=ad.io.read_elem(ctx.root["var"]))


def harmony(adata, basis, adjusted, p):
    import rapids_singlecell as rsc
    h = p["harmony"]
    adata.obs[h["key"]] = adata.obs[h["key"]].astype("category").cat.remove_unused_categories()
    rsc.pp.harmony_integrate(adata, key=h["key"], basis=basis, adjusted_basis=adjusted,
                             alpha=config.harmony_alpha(p), theta=h["theta"], max_iter_harmony=h["max_iter"])
    return adjusted


def run(ctx):
    import rapids_singlecell as rsc
    p, c = ctx.params, ctx.params["pca"]
    adata, ctx.data = ctx.data, None
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.calculate_qc_metrics(adata)
    rsc.pp.normalize_total(adata)
    rsc.pp.log1p(adata)
    rsc.pp.highly_variable_genes(adata, flavor=p["hvg"]["pca_flavor"], n_top_genes=p["hvg"]["pca_n_top_genes"])
    adata = adata[:, np.asarray(adata.var["highly_variable"])].copy()
    n_rows, n_cols = adata.shape
    adata.X = adata.X.rechunk((n_rows, n_cols)).persist()
    adata.X.compute_chunk_sizes()
    rsc.pp.scale(adata, max_value=10, zero_center=False)
    rsc.pp.pca(adata, n_comps=c["n_comps"], svd_solver=c["svd_solver"], random_state=c["random_seed"])
    adata.obsm["X_pca"] = adata.obsm["X_pca"].persist()
    adata.obsm["X_pca"].compute_chunk_sizes()
    adata.obsm["X_pca"] = adata.obsm["X_pca"].compute()
    rep = harmony(adata, "X_pca", "X_pca_harmony", p)
    ctx.n_pcs = c["n_comps"]
    ctx.arrays["X_pca_unintegrated"] = adata.obsm["X_pca"]
    return adata.obsm[rep], rep
