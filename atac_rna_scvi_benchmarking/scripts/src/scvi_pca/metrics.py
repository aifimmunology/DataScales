"""Run-2 metric inputs: evaluation cells, embeddings, uncorrected baselines and kNN purity (default env, GPU).

Every strategy of one (ds, subset) is scored on the same cells: the subset's labelled cells, or a fixed
sample-stratified EVAL_N of them when larger. scib-metrics itself runs in the `metrics` env
(scripts/21_scib_worker.py) on job files written here.
"""
import contextlib
import fcntl
import json

import numpy as np
import pandas as pd

from . import config, data

GPU_LOCK = "/mnt/scvi_pca/gpu.lock"
EVAL_N = 500_000
PURITY_K = 20
UNLABELLED = {"nan", "NA", "None", ""}


@contextlib.contextmanager
def gpu_lock():
    with open(GPU_LOCK, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def cache_dir():
    return config.out_root() / "cache" / "eval"


def ok_runs(ds):
    """{(subset, strategy): run_id}, rep 2 preferred."""
    r = pd.read_parquet(config.REPO / "results" / "runs.parquet")
    r = r[(r.ds == ds) & (r.status == "ok")].sort_values("rep")
    return {(x.subset, x.strategy): x.run_id for x in r.itertuples()}


def labels(ds, root, rows):
    """(coarse, fine) label arrays (object, unlabelled -> None) for the store rows."""
    cfg = config.dataset(ds)
    obs = data.label_obs(ds, root)
    out = []
    for c in cfg["plot_labels"][1:3]:
        v = obs[c].astype(object).to_numpy()[rows]
        out.append(np.array([None if (x is None or str(x) in UNLABELLED) else str(x) for x in v], dtype=object))
    return tuple(out)


def eval_positions(ds, subset, root, rows, coarse, fine):
    """Positions (into `rows`) of the evaluation cells: cells with both label levels, stratified to EVAL_N by
    sample (cached), so precomputed kNN graphs hold for every metric."""
    path = cache_dir() / f"{ds}_{subset}.npy"
    if path.exists():
        return np.load(path)
    pos = np.flatnonzero(pd.notna(coarse) & pd.notna(fine))
    if len(pos) > EVAL_N:
        col = config.dataset(ds)["sample_col"]
        samp = data.read_obs_cols(root, [col])[col].astype(str).to_numpy()[rows][pos]
        rng = np.random.default_rng(0)
        keep = []
        for s in np.unique(samp):
            p = pos[samp == s]
            k = int(round(len(p) * EVAL_N / len(pos)))
            keep.append(rng.choice(p, size=min(k, len(p)), replace=False))
        pos = np.sort(np.concatenate(keep))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, pos)
    return pos


def run_embedding(run_id, pos):
    """The run's embedding at `pos`, cut to the dims the tail used (n_pcs)."""
    rid = config.RunId.parse(run_id)
    meta = json.loads((rid.repo_dir() / "run.json").read_text())
    emb = np.load(rid.out_dir() / "embedding.npy", mmap_mode="r")
    n = int(meta.get("n_pcs") or emb.shape[1])
    return np.ascontiguousarray(emb[pos, :n], dtype=np.float32)


def unintegrated(ds, subset, root, obs, rows, pos, runs):
    """Uncorrected baseline for scib's PCR: RNA = the pca run's pre-Harmony PCA of the subset;
    ATAC = binarised TF-IDF -> 50-component LSI (first dropped) of the evaluation cells."""
    if config.dataset(ds)["modality"] == "rna":
        rid = config.RunId.parse(runs[(subset, "pca")])
        return np.ascontiguousarray(np.load(rid.out_dir() / "X_pca_unintegrated.npy", mmap_mode="r")[pos], dtype=np.float32)
    import cupyx.scipy.sparse as sp
    from . import lsi
    host = data.load_X(root, obs, rows[pos], eager=True).X
    host.data = (host.data > 0).astype(np.float32)
    host.eliminate_zeros()
    with gpu_lock():
        emb, _ = lsi.randomized_svd(lsi.tfidf(sp.csr_matrix(host)), 51)
        return emb[:, 1:].get().astype(np.float32)


def knn(emb, k):
    """Exact (brute, GPU) kNN incl. self: (indices int32, distances float32), shape (n, k)."""
    import cupy as cp
    from cuml.neighbors import NearestNeighbors
    with gpu_lock():
        x = cp.asarray(emb)
        d, i = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(x).kneighbors(x)
        out = cp.asnumpy(i).astype(np.int32), cp.asnumpy(d).astype(np.float32)
        del x, d, i
        cp.get_default_memory_pool().free_all_blocks()
    return out


def purity(emb, coarse, fine, batch, idx=None):
    """kNN (k=20, brute, GPU) purity per label level (cell-weighted and per-type mean) + same-batch share."""
    if idx is None:
        idx = knn(emb, PURITY_K + 1)[0]
    idx = idx[:, :PURITY_K + 1]
    self_hit = idx == np.arange(len(idx))[:, None]
    nbr = np.take_along_axis(idx, np.argsort(self_hit, axis=1, kind="stable")[:, :PURITY_K], axis=1)
    out = {}
    for name, lab in (("coarse", coarse), ("fine", fine)):
        ok = pd.notna(lab)
        same = (lab[nbr] == lab[:, None]).mean(1)[ok]
        out[f"purity_{name}"] = float(same.mean())
        out[f"purity_macro_{name}"] = float(pd.Series(same).groupby(pd.Series(lab[ok])).mean().mean())
    out["same_batch"] = float((batch[nbr] == batch[:, None]).mean())
    out["same_batch_random"] = float((pd.Series(batch).value_counts(normalize=True) ** 2).sum())
    return out
