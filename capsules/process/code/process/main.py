"""Embeddings and clusters on GPU for a curated store.

Reads X lazily onto a Dask-CUDA cluster, then normalize, HVG, PCA, optional harmony, neighbors,
UMAP and leiden. The results and a `lognorm` layer go back to the store (icechunk commit or plain
zarr in place), or to a new zarr when the input is read-only. For a SpatialData store, that's its table.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
from annizarr.errors import AnzError

import cluster
import steps
import store as stores
from config import ConfigError, Settings, load_settings

logger = logging.getLogger("process")

USER_ERRORS = (ConfigError, stores.StoreError, steps.PipelineError, AnzError)
PACKAGES = ("rapids-singlecell-cu13", "cuml-cu13", "cugraph-cu13", "cuvs-cu13", "dask-cuda", "anndata", "annizarr",
            "icechunk", "zarr", "numpy", "scanpy")


class Timer:
    def __init__(self):
        self.steps: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str):
        logger.info("%s ...", name)
        start = time.perf_counter()
        yield
        cluster.sync()
        self.steps[name] = round(time.perf_counter() - start, 2)
        logger.info("%s done in %.1fs", name, self.steps[name])


def versions() -> dict[str, str]:
    found = {}
    for name in PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            continue
    return found


def obs_columns(settings: Settings, root) -> list[str]:
    sample = [settings.sample_key] if settings.sample_key in root["obs"] else []
    return list(dict.fromkeys(sample + settings.integration_keys))


def uns_process(settings: Settings, target, emb: steps.Embedding, n_gpus: int, packages: dict) -> dict:
    params = {k: v for k, v in asdict(settings).items() if k not in ("input", "origin", "output", "results_dir")}
    params["batch_keys"] = settings.integration_keys
    return {
        "params": params,
        "n_gpus": n_gpus,
        "versions": packages,
        "pca_variance_ratio": np.asarray(emb.variance_ratio, dtype=np.float64),
        "source": target.source(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def compute(root, obs: pd.DataFrame, settings: Settings, n_gpus: int, timer: Timer):
    """Runs every GPU step; returns the embedding and the lazy lognorm matrix."""
    adata = steps.lazy_adata(root, obs, steps.read_var(root), settings)
    lognorm = steps.normalize(adata, settings)
    with timer("read_normalize_hvg_pca"):
        hvg, X_pca, variance_ratio = steps.hvg_pca(adata, settings, n_gpus)
    del adata
    graph = steps.graph_adata(obs, X_pca)
    with timer(f"integration_{settings.integration}"):
        basis = steps.integrate(graph, settings)
    with timer("neighbors_umap_leiden"):
        steps.neighbors_umap_leiden(graph, basis, settings)
    return steps.embedding(graph, hvg, variance_ratio), lognorm


def cluster_summary(emb: steps.Embedding, obs: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    leiden = pd.Series(emb.leiden, index=obs.index, name="leiden")
    table = pd.crosstab(leiden, obs[settings.sample_key]) if settings.sample_key in obs else pd.DataFrame(index=leiden.cat.categories)
    table.insert(0, "n_cells", leiden.value_counts().reindex(table.index))
    return table


def write_run_json(settings: Settings, report: dict) -> None:
    results = Path(settings.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    (results / "run.json").write_text(json.dumps(report, indent=2, default=str))


def run(settings: Settings) -> dict:
    started = datetime.now(timezone.utc)
    Path(settings.results_dir).mkdir(parents=True, exist_ok=True)
    (Path(settings.results_dir) / "cluster_summary.csv").unlink(missing_ok=True)
    packages = versions()
    target = stores.open_target(settings)
    logger.info("Input %s (%s)", target.location, target.kind)
    root = target.read_root()
    obs = steps.read_obs(root, obs_columns(settings, root))
    steps.check_batch_keys(obs, settings.integration_keys)
    logger.info("%d cells x %d genes", len(obs), root["X"].attrs["shape"][1])

    timer = Timer()
    with cluster.MemoryMonitor() as memory, cluster.gpu_cluster() as client:
        n_gpus = cluster.n_workers(client)
        logger.info("Dask-CUDA cluster with %d GPU worker(s)", n_gpus)
        emb, lognorm = compute(root, obs, settings, n_gpus, timer)
        with timer("write"):
            written = target.write(emb, uns_process(settings, target, emb, n_gpus, packages), lognorm, settings)

    report = {
        "status": "ok",
        "input": target.source(),
        "written": written,
        "n_cells": len(obs),
        "n_clusters": emb.n_clusters,
        "n_gpus": n_gpus,
        "timings_s": timer.steps,
        **memory.summary(),
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": packages,
        "settings": asdict(settings),
    }
    cluster_summary(emb, obs, settings).to_csv(Path(settings.results_dir) / "cluster_summary.csv")
    write_run_json(settings, report)
    logger.info("Done: %d leiden clusters over %d cells -> %s", emb.n_clusters, len(obs), written)
    return report


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    settings = None
    try:
        settings = load_settings(argv)
        run(settings)
    except BaseException as exc:
        if settings is not None:
            write_run_json(settings, {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "started": started,
                                      "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                      "settings": asdict(settings)})
        if not isinstance(exc, USER_ERRORS):
            raise
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
