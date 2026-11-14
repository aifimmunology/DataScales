"""Optional pipeline for the App Panel / a Reproducible Run: the out-of-core notebook's steps, non-interactively.

load -> h2d -> normalize/log1p -> HVG -> PCA path: rechunk/persist -> scale -> PCA -> harmony (with a batch key)
                                      -> scVI path: gather the HVG counts -> close the cluster -> train scVI
-> neighbors -> UMAP -> leiden (one per resolution) -> save (outputs.py: a copy in /results, or an external location)

dask-cuda spawns its workers, and spawn re-imports __main__ in each: keep module level free of work.
"""
from __future__ import annotations

import os
import sys

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from dataclasses import asdict, dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

from config import Config, ConfigError, load_config  # noqa: E402

USAGE = """\
This capsule is mainly a RAPIDS GPU environment with two notebooks (code/notebooks/); open them in
JupyterLab or VS Code. The pipeline that runs here is optional, and no input was given, so nothing ran.

To run it, open the App Panel, set "Input store" to an AnnData .zarr, a SpatialData .zarr or an icechunk
repo in an attached data asset (e.g. /data/<asset>/<name>.zarr), pick a method (PCA or scVI (train)),
and press Run. From a terminal: code/run --input /data/<asset>/<name>.zarr [--flag value ...]; code/run --help
lists every setting, and each one can also be set as an upper-case environment variable.

The results go in a new copy in /results (<name>.processed.zarr, or an icechunk repo for an icechunk input),
or to an external location: an icechunk input's own S3 repo (one commit), or a new or existing zarr.
See code/README.md, "Optional: run from the App Panel or as a Reproducible Run".
"""


@dataclass
class Loaded:
    src: object
    dest: object
    obs: object
    var: object
    counts: object
    x: object
    rows: object
    n_cells: int


def load(cfg: Config) -> Loaded:
    """Everything that can fail on the input is checked here, before any GPU work."""
    from anndata.io import read_elem

    import steps
    from cluster import set_zarr_config
    import outputs
    from store import open_source

    set_zarr_config(cfg.zarr_concurrency, cfg.zarr_max_workers)
    dest = outputs.plan(cfg)
    src = open_source(cfg.input, cfg.table, cfg.branch, dest.location if dest.mode == "commit" else "")
    outputs.check(dest, src)
    group = src.group
    n_cells, n_genes = group["X"].attrs["shape"]
    keys = cfg.batch_keys + ([cfg.hvg_batch_key] if cfg.hvg_batch_key else [])
    obs = steps.read_obs(group, [cfg.sample_key, *cfg.color_columns], keys + ([cfg.subset_column] if cfg.subset_column else []))
    rows = steps.subset_rows(obs, cfg.subset_column, cfg.subset) if cfg.subset_column else None
    if rows is not None:
        obs = obs.iloc[rows].copy()
        for c in obs.select_dtypes("category"):
            obs[c] = obs[c].cat.remove_unused_categories()
    steps.check_batch(obs, keys)
    counts = steps.lazy_counts(group, cfg.chunk_rows)
    print(f"{len(obs):,} of {n_cells:,} cells x {n_genes:,} genes from {src.path}"
          + (f" ({src.table})" if src.table else "") + (f", snapshot {src.snapshot[:12]}" if src.repo else "") + f"; results: {dest.describe()}")
    return Loaded(src, dest, obs, read_elem(group["var"]), counts, steps.slice_rows(counts, rows), rows, n_cells)


def embed(cfg: Config, data: Loaded, gpu, report) -> tuple:
    """normalize -> HVG -> PCA (+harmony) or scVI; returns the graph AnnData, its basis, lognorm, HVGs and uns."""
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp

    import steps
    from store import free_path

    keys = cfg.batch_keys
    with report.step("h2d_normalize"):
        adata = steps.lognormalized(data.x, data.obs, data.var, cfg.target_sum)
        lognorm = adata.X if data.rows is None else steps.lognormalized(data.counts, None, None, cfg.target_sum).X
    with report.step("hvg"):
        hv = steps.highly_variable(adata, cfg)
        adata = adata[:, hv].copy()
        if cfg.method == "pca":
            steps.persist_bands(adata, gpu.n_workers)

    uns = {}
    if cfg.method == "pca":
        with report.step("scale_pca"):
            X_pca, uns["pca_variance_ratio"] = steps.scale_pca(adata, cfg)
        del adata
        graph = ad.AnnData(obs=data.obs, obsm={"X_pca": X_pca})
        if not keys:
            return graph, "X_pca", lognorm, hv, uns
        with report.step("harmony"):
            steps.harmony(graph, keys, cfg)
        return graph, "X_pca_harmony", lognorm, hv, uns

    del adata
    with report.step("gather_counts"):
        counts = sp.csr_matrix(data.x[:, hv].compute(), dtype=np.float32)
    gpu.close()
    uns["scvi_model"] = model_dir = free_path(cfg.model_dir.rstrip("/"))
    with report.step("scvi"):
        latent = steps.train_scvi(counts, data.obs, data.var.index[hv], keys, cfg, model_dir)
    return ad.AnnData(obs=data.obs, obsm={"X_scVI": latent}), "X_scVI", lognorm, hv, uns


def write(cfg: Config, data: Loaded, graph, basis: str, lognorm, hv, uns: dict) -> str:
    import numpy as np

    import steps
    import outputs
    from store import PROCESS_OBSM, Results

    leiden = cfg.leiden_keys
    clusters = ", ".join(f"{k}: {graph.obs[k].nunique()} clusters" for k in leiden)
    results = Results(
        lognorm=lognorm, target_sum=cfg.target_sum, highly_variable=hv,
        obsm={k: steps.full_rows(v, data.rows, data.n_cells) for k, v in graph.obsm.items() if k in PROCESS_OBSM},
        obs={k: steps.full_column(graph.obs[k], data.rows, data.n_cells) for k in leiden},
        uns={**uns, "input": cfg.input, "table": data.src.table, "snapshot": data.src.snapshot, "method": cfg.method,
             "basis": basis, "batch_keys": np.array(cfg.batch_keys, dtype=str), "target_sum": cfg.target_sum,
             "leiden": dict(leiden), "n_cells": graph.n_obs,
             "subset": {"column": cfg.subset_column, "values": np.array(cfg.subset, dtype=str)},
             "settings": asdict(cfg)},
    )
    written = outputs.save(data.dest, data.src, results.write,
                           f"Process: {cfg.method} ({basis}), {clusters} over {graph.n_obs:,} cells", cfg.branch,
                           {"method": cfg.method, "basis": basis, "leiden": dict(leiden), "n_cells": graph.n_obs})
    print(f"{clusters}; {written}")
    return written


def plots(graph, cfg: Config) -> None:
    """umap.png, one panel per colour key the store has ("leiden" is every cluster column), and cluster_summary csvs."""
    import pandas as pd
    import scanpy as sc

    results = Path(cfg.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    keys = [k for name in cfg.color_columns for k in (cfg.leiden_keys if name == "leiden" else [name]) if k in graph.obs]
    if keys:
        fig = sc.pl.umap(graph, color=list(dict.fromkeys(keys)), ncols=2, return_fig=True, show=False)
        fig.savefig(results / "umap.png", dpi=150, bbox_inches="tight")
    sample = cfg.sample_key in graph.obs
    for key in cfg.leiden_keys:
        labels = graph.obs[key]
        summary = pd.crosstab(labels, graph.obs[cfg.sample_key]) if sample else pd.DataFrame(index=labels.cat.categories)
        summary.insert(0, "n_cells", labels.value_counts())
        summary.to_csv(results / ("cluster_summary.csv" if key == "leiden" else f"cluster_summary_{key}.csv"))


def run(cfg: Config) -> str:
    import steps
    from cluster import GpuCluster
    from report import Report, header

    report, gpu = Report(cfg.sample_interval_s, tuple(cfg.gpu_ids)), GpuCluster(cfg)
    with report.step("load"):
        data = load(cfg)
    try:
        with report.step("start_cluster"):
            gpu.start()
        n_gpus = gpu.n_workers
        print(header(cfg, n_gpus))
        graph, basis, lognorm, hv, uns = embed(cfg, data, gpu, report)
        with report.step("neighbors"):
            steps.neighbors(graph, basis, cfg)
        with report.step("umap"):
            steps.umap(graph, cfg)
        with report.step("leiden"):
            steps.leiden(graph, cfg)
        with report.step("write"):
            written = write(cfg, data, graph, basis, lognorm, hv, uns)
        if cfg.plots:
            with report.step("plots"):
                plots(graph, cfg)
    finally:
        gpu.close()
    print("\n" + report.table())
    report.write(Path(cfg.results_dir) / "process_report.txt", header(cfg, n_gpus, written))
    return written


def main(argv: list[str] | None = None) -> int:
    try:
        cfg = load_config(argv)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not cfg.input:
        print(USAGE)
        return 0
    if cfg.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
    from store import InputError

    try:
        run(cfg)
    except InputError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
