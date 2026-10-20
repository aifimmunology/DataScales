"""Shared tail: neighbors -> UMAP (-> leiden when tail.leiden), each a separate timed Stage."""
import anndata as ad
import numpy as np


def run_tail(rec, emb, rep, n_pcs, p):
    """emb: cupy (n, d) float32. Returns (umap float32 (n, 2), leiden int32 labels or None)."""
    import cupy as cp
    import rapids_singlecell as rsc

    t = p["tail"]
    seed = t.get("random_seed", 0)
    adata = ad.AnnData(X=None, shape=(emb.shape[0], 0))
    adata.obsm[rep] = cp.asarray(emb, dtype=cp.float32)
    with rec.stage("t_neighbors"):
        rsc.pp.neighbors(adata, n_neighbors=t["n_neighbors"], n_pcs=n_pcs, use_rep=rep,
                         algorithm=t["algorithm"], random_state=seed)
    with rec.stage("t_umap"):
        rsc.tl.umap(adata, min_dist=t["min_dist"], init_pos=t["init_pos"], n_components=2, random_state=seed)
    labels = None
    if t.get("leiden", True):
        with rec.stage("t_leiden"):
            rsc.tl.leiden(adata, resolution=t["resolution"], n_iterations=t["n_iterations"], random_state=seed)
        labels = adata.obs["leiden"].astype(int).to_numpy().astype(np.int32)
    umap = adata.obsm["X_umap"]
    umap = umap.get() if hasattr(umap, "get") else np.asarray(umap)
    return umap.astype(np.float32), labels
