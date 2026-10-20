"""scib-metrics Benchmarker worker (metrics env, JAX): 21_scib_worker.py <job.npz> ... -> <job>.json.

Standard BioConservation() + BatchCorrection() sets, JAX GPU kNN, uncorrected baseline for PCR. scib's
silhouette_samples is swapped for the chunked GPU version (gpu_silhouette, equal to 1e-4) when called with
the default euclidean/nearest arguments, otherwise scib's own runs. kNN graphs are precomputed (exact, cuml) by
20_metrics.py and served to Benchmarker.prepare by matching the embedding (JAX approx_min_k at 500k cells
stalled ptxas for >30 min).
"""
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scib_metrics.metrics._isolated_labels as _iso
import scib_metrics.metrics._silhouette as _sil
from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation
from scib_metrics.nearest_neighbors import NeighborsResults

from scvi_pca import gpu_silhouette

_orig = _sil.silhouette_samples


def fast_silhouette(X, labels, chunk_size=256, metric="euclidean", between_cluster_distances="nearest"):
    if metric == "euclidean" and between_cluster_distances == "nearest":
        return gpu_silhouette.silhouette_samples(np.asarray(X, np.float32), pd.factorize(np.asarray(labels))[0])
    return _orig(X, labels, chunk_size=chunk_size, metric=metric, between_cluster_distances=between_cluster_distances)


_sil.silhouette_samples = fast_silhouette
_iso.silhouette_samples = fast_silhouette


def precomputed(z, keys, X, k):
    """The job's kNN graph for embedding X (matched by content), cut to k neighbours."""
    for key in keys + ["unintegrated"]:
        e = z[key]
        if e.shape == X.shape and np.array_equal(e[:16], np.asarray(X[:16])):
            name = key[4:] if key.startswith("emb_") else key
            return NeighborsResults(indices=z[f"idx_{name}"][:, :k], distances=z[f"dist_{name}"][:, :k])
    raise KeyError("no precomputed kNN graph matches this embedding")


def run(job):
    z = np.load(job, allow_pickle=True)
    keys = [k for k in z.files if k.startswith("emb_")]
    out = {}
    for level in ("coarse", "fine"):
        lab = z[f"label_{level}"].astype(object)
        ok = pd.notna(lab)
        assert ok.all(), "evaluation cells must all be labelled (precomputed graphs index them)"
        a = ad.AnnData(obs=pd.DataFrame({"batch": z["batch"][ok].astype(str), "label": lab[ok].astype(str)}))
        a.obs = a.obs.astype("category")
        for k in keys + ["unintegrated"]:
            a.obsm[k] = np.ascontiguousarray(z[k][ok], dtype=np.float32)
        t0 = time.perf_counter()
        bm = Benchmarker(a, batch_key="batch", label_key="label", embedding_obsm_keys=keys + ["unintegrated"],
                         bio_conservation_metrics=BioConservation(), batch_correction_metrics=BatchCorrection(),
                         pre_integrated_embedding_obsm_key="unintegrated", progress_bar=False)
        bm.prepare(neighbor_computer=lambda X, k: precomputed(z, keys, X, k))
        bm.benchmark()
        res = bm.get_results(min_max_scale=False)
        out[level] = {"n_eval": int(ok.sum()), "seconds": time.perf_counter() - t0,
                      "table": res.reset_index().astype(str).to_dict(orient="records")}
        print(f"{job.name} {level}: {ok.sum()} cells, {time.perf_counter() - t0:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    for j in sys.argv[1:]:
        p = Path(j)
        p.with_suffix(".json").write_text(json.dumps(run(p)))
